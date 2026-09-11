"""Persistent DATA worker for the FaceTime voice loop — REAL session mode.

2026-09-07 latency rewrite: the old path called hermes_cli.oneshot._run_agent
per turn, which rebuilt the agent, system prompt, toolset, memory context and
skill scan EVERY turn — ~9.5s of pure overhead per turn (measured: 10.5s for a
3-char reply vs 1.0s raw API). Now the worker holds ONE AIAgent for its whole
life and calls run_conversation() per turn: system prompt + toolset are
byte-stable across turns (provider prompt-cache friendly), memories load once,
and voice_agent.py drives it turn by turn over stdin/stdout.

2026-09-07 latency pass #2 — TWO-TIER TOOL POLICY
-------------------------------------------------
Measured on an M-series Mac, same model, same prompt, first-token latency
across four consecutive turns:

    with the 6 production toolsets (10 tools, 6046 tokens of schema):
        2.08s  4.69s  3.59s  0.68s
    with no toolsets:
        0.40s  0.65s  0.87s  1.62s

The tool schema is the single largest per-turn LLM cost on a call, and it is
paid on EVERY turn including "you there?" — which is most of them. Worse, a
turn that actually calls a tool emits no content deltas at all, so the caller
hears nothing until the whole agent loop (LLM -> tool -> LLM) finishes.

So the worker now keeps TWO agents that share nothing but the process:

  FAST  — no toolsets. Handles conversation. Streams from the first token.
  FULL  — the production toolsets. Handles anything that needs DATA to act.

Routing is a local keyword/heuristic pass over the user's line (no extra model
call — that would cost more than it saves). When a turn routes to FULL, the
worker immediately emits a spoken filler so the caller hears acknowledgement
instead of dead air while the tool loop runs.

Both agents share the conversation transcript, so DATA does not lose the thread
when a turn switches tier.

stdin:  {"prompt": "...", "stream": true} lines
stdout: {"status":"ready"} then {"delta": "..."} / {"filler": "..."} /
        {"content": "...", "tier": "fast|full"} lines
"""
import json
import os
import re
import sys
import time

os.environ.setdefault("HERMES_HOME", os.path.expanduser("~/.hermes"))
os.environ.setdefault("HERMES_YOLO_MODE", "1")
os.environ.setdefault("HERMES_ACCEPT_HOOKS", "1")
sys.path.insert(0, os.path.expanduser("~/.hermes/hermes-agent"))

import logging
logging.disable(logging.CRITICAL)

_fast_agent = None
_full_agent = None

# The voice loop lives next to this file. The old hardcoded absolute path broke
# on any host that did not use that exact directory, and the resulting
# ImportError surfaced as DATA silently saying nothing.
DFV_DIR = os.path.dirname(os.path.abspath(__file__))

TOOL_TOOLSETS = ["file", "web", "search", "terminal", "session_search", "memory"]

# Reasoning effort is set PER TIER. On a reasoning model, thinking tokens are
# emitted before any content, so they are pure dead air on a phone call.
# Measured on the raw API with an open-ended conversational prompt:
#
#     effort unset    -> 418 chars of thinking, first content 0.74s
#     effort low      -> 237 chars,             first content 0.55s
#     effort none     ->   0 chars,             first content 0.25s
#
# Chat gains nothing from deliberation, so the fast tier turns it off outright.
# The tool tier keeps a little, where it genuinely helps pick the right tool.
# 2026-09-11: "none" makes glm-5.3-flash (ollama-cloud) emit its reasoning as
# untagged CONTENT — DATA would speak his thoughts aloud. "low" is separated
# cleanly by the provider. Measured on DATA's host; see voice_agent.py.
FAST_REASONING_EFFORT = os.environ.get("DFV_REASONING_EFFORT_FAST", "low")
FULL_REASONING_EFFORT = os.environ.get(
    "DFV_REASONING_EFFORT", os.environ.get("DFV_REASONING_EFFORT_FULL", "low"))

# Spoken the instant a turn routes to the tool-enabled agent, so the caller
# hears acknowledgement rather than dead air while the tool loop runs.
FILLER_LINE = os.environ.get("DFV_TOOL_FILLER", "Let me check that, Captain.")

# Routing to the tool tier. Deliberately a cheap local heuristic: an extra
# classifier model call would cost more latency than the tool schema it avoids.
_ACTION_PATTERNS = re.compile(
    r"\b("
    r"search|google|look ?up|find out|check|weather|forecast|"
    r"calendar|schedule|appointment|meeting|remind|reminder|"
    r"note|notes|write down|log|track|fitness|workout|weight|"
    r"email|message|text|send|"
    r"github|repo|issue|pull request|pr\b|commit|deploy|build|"
    r"file|execute|terminal|command|script|"
    r"remember|recall|memory|last time|we talked|you said|"
    r"news|price|stock|score|time in"
    r")\b",
    re.IGNORECASE,
)
# Words that only mean "use a tool" in a command context. Bare `run` used to
# live in the list above and sent "heading out for a run this evening" to the
# tool tier — a 6.5s turn for small talk. Same trap for read/open/date.
_ACTION_PHRASES = re.compile(
    r"\b(run|rerun|re-run)\s+(the\s+|a\s+|that\s+|this\s+|it\b)?"
    r"(command|script|test|tests|build|make|query|it\b)"
    r"|\b(read|open)\s+(the\s+|my\s+|that\s+)?(file|log|logs|readme|config|doc)"
    r"|\bwhat'?s?\s+(the\s+)?date\b|\btoday'?s\s+date\b",
    re.IGNORECASE,
)
# Conversational openers that would otherwise trip the keyword list.
_SMALLTALK = re.compile(
    r"^\s*(hey|hi|hello|yo|good (morning|afternoon|evening)|thanks|thank you|"
    r"ok(ay)?|cool|nice|got it|never ?mind|nothing|no|yes|yeah|yep|nope|"
    r"how are you|what'?s up|you there|are you there)\b[\s.,!?]*$",
    re.IGNORECASE,
)


def needs_tools(prompt: str) -> bool:
    """True when this turn should run on the tool-enabled agent."""
    if os.environ.get("DFV_TOOL_TIER", "auto").lower() == "always":
        return True
    if os.environ.get("DFV_TOOL_TIER", "auto").lower() == "never":
        return False
    text = (prompt or "").strip()
    if not text or _SMALLTALK.match(text):
        return False
    return bool(_ACTION_PATTERNS.search(text) or _ACTION_PHRASES.search(text))


def _voice_system_prompt() -> str:
    """DATA's voice-call system context (voice_persona is the one source)."""
    if DFV_DIR not in sys.path:
        sys.path.insert(0, DFV_DIR)
    try:
        from voice_persona import SYSTEM_CONTEXT
        return SYSTEM_CONTEXT + VOICE_BREVITY_PROMPT
    except Exception:
        # voice_persona has no deps, but if the import fails for any reason the
        # import fails. Never let that take the whole worker down silently —
        # a degraded persona beats a call where DATA never speaks.
        return (
            "You are DATA, the right-hand AI agent for Captain Kirk (Spencer), "
            "speaking over a live FaceTime Audio call through a realtime voice "
            "pipeline. Calm, precise, dry wit. 'Sir' or 'Captain' at most once "
            "per reply. Voice-friendly: no markdown, no code, no URLs. SHORT: "
            "1-3 sentences by default."
        ) + VOICE_BREVITY_PROMPT


# 2026-09-09 (call-4): DATA's replies ran 25-38s of continuous TTS before
# Captain could barge in. Persona rules ('1-3 sentences by default') were not
# enough — the model reads 'by default' as negotiable on any substantive
# question. This suffix is unconditional and applies to EVERY voice turn on
# both tiers (it rides _voice_system_prompt, which both agents share).
VOICE_BREVITY_PROMPT = (
    " You are on a LIVE PHONE CALL. Reply in AT MOST 2 short sentences. "
    "No lists, no markdown, no tool talk. Stop talking after the answer; "
    "the Captain will ask if he wants more."
)


def _build_agent(toolsets, reasoning_effort):
    from run_agent import AIAgent
    from hermes_cli.config import load_config
    from hermes_cli.runtime_provider import resolve_runtime_provider

    cfg = load_config()
    # Voice latency: thinking tokens are emitted before any content, so on a
    # phone call they are pure dead air.
    #
    # 2026-09-07 BUGFIX: this used to be
    #     cfg["agent"]["reasoning_effort"] = ...
    # which mutated a dict that was never passed to AIAgent. load_config()
    # returns a FRESH dict on every call (verified), so the mutation was
    # discarded and DFV_REASONING_EFFORT had no effect whatsoever — every voice
    # turn has been running at the config default ("medium") this whole time.
    # The real knob is the reasoning_config constructor argument.
    try:
        from hermes_constants import parse_reasoning_effort
        reasoning_config = parse_reasoning_effort(reasoning_effort)
    except Exception:
        reasoning_config = None
    info = resolve_runtime_provider()
    model_cfg = cfg.get("model") or {}
    model_name = (
        model_cfg.get("default") or model_cfg.get("model") or ""
        if isinstance(model_cfg, dict)
        else str(model_cfg)
    )
    return AIAgent(
        model=model_name,
        provider=info.get("provider"),
        base_url=info.get("base_url"),
        api_key=info.get("api_key"),
        api_mode=info.get("api_mode"),
        credential_pool=info.get("credential_pool"),
        enabled_toolsets=toolsets,
        reasoning_config=reasoning_config,
        ephemeral_system_prompt=_voice_system_prompt(),
        skip_context_files=True,
        load_soul_identity=True,
        quiet_mode=True,
        save_trajectories=False,
        platform="voice",
        user_name="Captain Spencer",
    )


def _ensure_fast():
    global _fast_agent
    if _fast_agent is None:
        _fast_agent = _build_agent([], FAST_REASONING_EFFORT)
    return _fast_agent


def _ensure_full():
    global _full_agent
    if _full_agent is None:
        _full_agent = _build_agent(TOOL_TOOLSETS, FULL_REASONING_EFFORT)
    return _full_agent


def _mirror_turn(agent, user_text: str, reply: str) -> None:
    """Give the OTHER tier this turn so DATA keeps one continuous thread.

    Without this, asking DATA to check the calendar (FULL) and then saying
    "what did you just say?" (FAST) would hit an agent that never heard the
    exchange. Best-effort: transcript shapes differ across Hermes versions, so
    a failure here degrades continuity, never the call.
    """
    if agent is None or not reply:
        return
    try:
        history = getattr(agent, "conversation_history", None)
        if isinstance(history, list):
            history.append({"role": "user", "content": user_text})
            history.append({"role": "assistant", "content": reply})
    except Exception:
        pass


def process_request(request: dict) -> dict:
    prompt = request.get("prompt", "")
    if not prompt:
        return {"error": "empty prompt"}
    t0 = time.perf_counter()
    use_tools = needs_tools(prompt)
    tier = "full" if use_tools else "fast"
    try:
        if use_tools:
            # The tool loop streams nothing until it finishes, so give the
            # caller something to hear right now.
            sys.stdout.write(json.dumps({"filler": FILLER_LINE}) + "\n")
            sys.stdout.flush()
            agent = _ensure_full()
            other = _fast_agent
        else:
            agent = _ensure_fast()
            other = _full_agent

        if request.get("stream"):
            def _cb(delta: str) -> None:
                sys.stdout.write(json.dumps({"delta": delta}) + "\n")
                sys.stdout.flush()
            text = agent.chat(prompt, stream_callback=_cb)
        else:
            text = agent.chat(prompt)
        text = text if isinstance(text, str) else str(text)
        _mirror_turn(other, prompt, text)
        return {"content": text, "tier": tier,
                "elapsed": round(time.perf_counter() - t0, 3)}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}", "tier": tier,
                "elapsed": round(time.perf_counter() - t0, 3)}


def main():
    # Pre-warm BOTH agents BEFORE announcing ready — construction + memory load
    # + toolset build happen here, not on the Captain's first words. (Measured
    # 2026-09-11: with 'ready' printed first, the first turn paid 7-25s.)
    errors = {}
    for name, fn in (("fast", _ensure_fast), ("full", _ensure_full)):
        try:
            fn()
        except Exception as e:
            errors[name] = f"{type(e).__name__}: {e}"
    sys.stdout.write(json.dumps({"status": "ready", "errors": errors}) + "\n")
    sys.stdout.flush()
    while True:
        line = sys.stdin.readline()
        if not line:
            break
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except json.JSONDecodeError as e:
            sys.stdout.write(json.dumps({"error": f"invalid JSON: {e}"}) + "\n")
            sys.stdout.flush()
            continue
        response = process_request(request)
        sys.stdout.write(json.dumps(response) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
