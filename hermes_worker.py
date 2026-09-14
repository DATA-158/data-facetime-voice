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

stdin:  {"prompt": "...", "stream": true, "image": "/path.jpg"?} lines
        {"refresh": true}                       reload cross-channel context
        {"followup": true, "prompt": "..."}     post-call completion (text mode)
stdout: {"status":"ready"} then {"delta": "..."} / {"filler": "..."} /
        {"progress": "...", "tool": "..."} / {"content": "...", "tier": "fast|full"} lines
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
# 2026-09-12, Captain's call: tool turns think at MEDIUM (the filler covers
# the extra seconds; multi-step tasks — find/price/remind, calendar/text —
# benefit). Conversation stays LOW: measured on the same prompts, medium
# added 2-6 s of silence before the first word for identical answers.
FULL_REASONING_EFFORT = os.environ.get(
    "DFV_REASONING_EFFORT", os.environ.get("DFV_REASONING_EFFORT_FULL", "medium"))

# Spoken the instant a turn routes to the tool-enabled agent, so the caller
# hears acknowledgement rather than dead air while the tool loop runs.
FILLER_LINE = os.environ.get("DFV_TOOL_FILLER", "Let me check that, Captain.")

# 2026-09-13: progress past the filler. Tool turns run 7-150 s (live call
# 10:40 today: 146 s), and the model streams no content while it works, so
# after the filler the Captain heard only the hum — he hung up at 22 s. The
# FULL agent's tool_start_callback now emits one short spoken line per tool
# call; the voice loop rate-limits them (see voice_agent.PROGRESS_EVERY_S).
_PROGRESS_LINES = {
    "web_search": "Searching the web.",
    "web_extract": "Reading a page.",
    "web_fetch": "Reading a page.",
    "browser": "Opening a page.",
    "terminal": "Running a command.",
    "execute_code": "Running a command.",
    "read_file": "Reading a file.",
    "write_file": "Writing that down.",
    "patch": "Writing that down.",
    "search_files": "Searching files.",
    "memory": "Making a note of that.",
    "session_search": "Checking our past conversations.",
}
_PROGRESS_DEFAULT = "Still working on it."


def _progress_line(tool_name: str, args) -> str:
    name = (tool_name or "").lower()
    for key, line in _PROGRESS_LINES.items():
        if name.startswith(key):
            if key == "web_search" and isinstance(args, dict):
                q = str(args.get("query") or "").strip()
                if 0 < len(q.split()) <= 8:
                    return f"Searching for {q.rstrip('.?!')}."
            return line
    return _PROGRESS_DEFAULT


def _on_tool_start(tool_call_id, function_name, display_args=None, *_a, **_k):
    sys.stdout.write(json.dumps({"progress": _progress_line(function_name, display_args),
                                 "tool": str(function_name)}) + "\n")
    sys.stdout.flush()


# Post-call completion. The Captain's rule (2026-09-13): "if I say 'do this
# thing' and hang up, DATA should proceed to do that thing completely, finish
# it, and then iMessage me when it is finished." The voice loop lets the
# in-flight turn run to the end, then sends this follow-up on the FULL agent
# with the voice rules lifted; the reply is delivered as an iMessage.
FOLLOWUP_SYSTEM_SUFFIX = (
    "\n\nTHE CALL HAS ENDED. You are no longer speaking aloud: your next reply is "
    "delivered to the Captain as an iMessage. The voice rules above (no URLs, "
    "1-3 sentences) do NOT apply to it. Write plain text without markdown, up to "
    "about eight short lines, links welcome."
)

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
    r"news|price|prices|pricing|stock|score|time in|"
    # 2026-09-12: the Captain's typical call tasks — shop for something and
    # remind him with a link; read the calendar and text someone a window.
    r"buy|purchase|order|cheapest|cheaper|option|options|find me|look into|"
    r"let (?:him|her|them|\w+) know|tell \w+ (?:that|when|i)|"
    r"available|availability|free (?:time|slot|window)|when (?:i'?m|am i) free"
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
    ,
    re.IGNORECASE,
)
# The date/time used to route to the tool tier ("what's the date" -> terminal,
# 13.8 s on the 2026-09-11 16:44 call). Every prompt now carries a timestamp
# (see _stamp), so the FAST tier answers it.
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


# ---------------------------------------------------------------------------
# Continuity with DATA's other channels (2026-09-11 16:45 call: "we talked
# about it literally earlier today" — the answer was in the iMessage session
# at 13:02, in ~/.hermes/state.db, but the voice agent starts with an empty
# window and session_search picked the wrong query). On every call start the
# voice loop asks for a refresh; the last 24 h of user/assistant turns from
# the other channels are appended to the ephemeral system prompt, which
# Hermes re-reads on each request.
# ---------------------------------------------------------------------------
RECENT_HOURS = float(os.environ.get("DFV_RECENT_HOURS", "24"))
RECENT_MAX_MSGS = int(os.environ.get("DFV_RECENT_MAX_MSGS", "40"))
RECENT_MAX_CHARS = int(os.environ.get("DFV_RECENT_MAX_CHARS", "240"))
_EXCLUDED_SOURCES = ("cron", "subagent", "voice")


def _recent_context() -> str:
    """Compact transcript of DATA's recent conversations with the Captain."""
    import sqlite3
    db = os.path.join(os.environ["HERMES_HOME"], "state.db")
    if not os.path.exists(db):
        return ""
    try:
        con = sqlite3.connect(f"file:{db}?mode=ro", uri=True, timeout=2)
        since = time.time() - RECENT_HOURS * 3600
        # Sessions first (small table), then messages by session id (indexed):
        # the naive join scanned the whole messages table — 5-7 s on this host.
        sessions = con.execute(
            "SELECT id, source FROM sessions WHERE source NOT IN (%s) "
            "AND started_at > ? ORDER BY started_at DESC LIMIT 20"
            % ",".join("?" * len(_EXCLUDED_SOURCES)),
            (*_EXCLUDED_SOURCES, since - 7 * 86400),
        ).fetchall()
        rows = []
        for sid, source in sessions:
            rows += [(source, r, ts, c) for r, ts, c in con.execute(
                "SELECT role, timestamp, content FROM messages WHERE session_id = ? "
                "AND timestamp > ? AND role IN ('user','assistant') ORDER BY timestamp DESC LIMIT ?",
                (sid, since, RECENT_MAX_MSGS)).fetchall()]
        rows.sort(key=lambda r: r[2], reverse=True)
        rows = rows[:RECENT_MAX_MSGS]
        con.close()
    except Exception as e:
        sys.stderr.write(f"recent-context read failed: {e}\n")
        return ""
    lines = []
    for source, role, ts, content in reversed(rows):
        if not isinstance(content, str):
            continue
        text = " ".join(content.split())
        if not text or text.startswith("{") or text.startswith("<untrusted"):
            continue
        if len(text) > RECENT_MAX_CHARS:
            text = text[:RECENT_MAX_CHARS - 1] + "…"
        who = "Captain" if role == "user" else "DATA"
        lines.append(f"[{time.strftime('%a %H:%M', time.localtime(ts))} {source}] {who}: {text}")
    if not lines:
        return ""
    return ("\n\nRECENT CONVERSATIONS WITH THE CAPTAIN on other channels (last "
            f"{RECENT_HOURS:.0f}h, oldest first). This is what he means by 'earlier today' — "
            "answer from it directly before reaching for session_search:\n" + "\n".join(lines))


def _refresh_context() -> int:
    """Re-read recent context into both agents' ephemeral prompt. Returns line count."""
    ctx = _recent_context()
    base = _voice_system_prompt()
    for agent in (_fast_agent, _full_agent):
        if agent is not None:
            agent.ephemeral_system_prompt = base + ctx
    return ctx.count("\n") if ctx else 0


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
        ephemeral_system_prompt=_voice_system_prompt() + _recent_context(),
        skip_context_files=True,
        load_soul_identity=True,
        quiet_mode=True,
        save_trajectories=False,
        platform="voice",
        user_name="Captain Spencer",
        tool_start_callback=_on_tool_start if toolsets else None,
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


# Conversation memory (2026-09-14). AIAgent keeps NO transcript between
# chat() calls: each turn is a fresh conversation unless the caller passes
# conversation_history= and takes result["messages"] back (that is what the
# Hermes CLI does). Verified on this host: "my favorite number is 17" /
# "what's my favorite number?" -> "you've never told me one". So every voice
# turn until today stood alone; the old _mirror_turn wrote to an attribute
# that does not exist. Now: one history per tier, threaded through each turn,
# the other tier receives the user/assistant text of the exchange (its tool
# messages would not be valid for a no-tool agent), kept across calls and
# trimmed to HISTORY_MAX messages.
HISTORY_MAX = int(os.environ.get("DFV_HISTORY_MAX", "30"))
_histories: dict = {"fast": [], "full": []}


def _remember(tier: str, messages, user_text: str, reply: str) -> None:
    if isinstance(messages, list):
        _histories[tier] = messages[-HISTORY_MAX:]
    other = "fast" if tier == "full" else "full"
    if reply:
        _histories[other] += [{"role": "user", "content": user_text},
                              {"role": "assistant", "content": reply}]
        _histories[other] = _histories[other][-HISTORY_MAX:]


# Camera frames (2026-09-14). The frame rides the turn as a native image part
# (glm-5.3-flash is vision-capable per Hermes' catalog; image_input_mode auto
# resolves to "native"). After the turn the base64 is dropped from the
# history — a 1280 px frame is ~200 KB of prompt on every later turn — and
# replaced by a one-line note, so DATA remembers he was shown something
# without re-sending the pixels for the rest of the call.
CAMERA_NOTE = ("(The Captain is on FaceTime VIDEO and is showing you his camera. The "
               "attached frame is what he is pointing at right now; answer about it.)")


def _with_image(prompt: str, image: str):
    from agent.image_routing import build_native_content_parts
    parts, skipped = build_native_content_parts(f"{CAMERA_NOTE} {prompt}", [image])
    if skipped or not any(p.get("type") == "image_url" for p in parts):
        sys.stderr.write(f"image not attached: {skipped}\n")
        return prompt
    return parts


def _drop_image_from_history(tier: str, prompt: str) -> None:
    for msg in reversed(_histories[tier]):
        if msg.get("role") == "user" and isinstance(msg.get("content"), list):
            msg["content"] = f"{prompt} [showed a camera frame]"
            return


def _stamp(prompt: str) -> str:
    """Prefix the local date/time so time questions never need a tool."""
    return f"[{time.strftime('%A %Y-%m-%d %H:%M %Z')}] {prompt}"


def process_request(request: dict) -> dict:
    prompt = request.get("prompt", "")
    if not prompt:
        return {"error": "empty prompt"}
    prompt = _stamp(prompt)
    image = request.get("image")
    message = _with_image(prompt, image) if image else prompt
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
        else:
            agent = _ensure_fast()

        cb = None
        if request.get("stream"):
            def cb(delta: str) -> None:
                sys.stdout.write(json.dumps({"delta": delta}) + "\n")
                sys.stdout.flush()
        result = agent.run_conversation(message, conversation_history=list(_histories[tier]) or None,
                                        stream_callback=cb)
        text = result.get("final_response") or ""
        text = text if isinstance(text, str) else str(text)
        _remember(tier, result.get("messages"), prompt, text)
        if image:
            _drop_image_from_history(tier, prompt)
        return {"content": text, "tier": tier,
                "elapsed": round(time.perf_counter() - t0, 3)}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}", "tier": tier,
                "elapsed": round(time.perf_counter() - t0, 3)}


def process_followup(request: dict) -> dict:
    """Finish the Captain's last request after he hung up; return the iMessage text."""
    prompt = request.get("prompt", "")
    if not prompt:
        return {"error": "empty prompt"}
    t0 = time.perf_counter()
    agent = _ensure_full()
    voice_prompt = agent.ephemeral_system_prompt
    try:
        agent.ephemeral_system_prompt = (voice_prompt or _voice_system_prompt()) + FOLLOWUP_SYSTEM_SUFFIX
        result = agent.run_conversation(_stamp(prompt), conversation_history=list(_histories["full"]) or None)
        text = result.get("final_response") or ""
        text = text if isinstance(text, str) else str(text)
        _remember("full", result.get("messages"), prompt, text)
        return {"content": text, "tier": "full", "elapsed": round(time.perf_counter() - t0, 3)}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}", "tier": "full",
                "elapsed": round(time.perf_counter() - t0, 3)}
    finally:
        agent.ephemeral_system_prompt = voice_prompt


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
        if request.get("refresh"):
            n = _refresh_context()
            sys.stdout.write(json.dumps({"refreshed": n}) + "\n")
            sys.stdout.flush()
            continue
        if request.get("followup"):
            response = process_followup(request)
        else:
            response = process_request(request)
        sys.stdout.write(json.dumps(response) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
