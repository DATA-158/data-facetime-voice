"""Persistent DATA worker for the FaceTime voice loop — REAL session mode.

2026-09-07 latency rewrite: the old path called hermes_cli.oneshot._run_agent
per turn, which rebuilt the agent, system prompt, toolset, memory context and
skill scan EVERY turn — ~9.5s of pure overhead per turn (measured: 10.5s for a
3-char reply vs 1.0s raw API). Now the worker holds ONE AIAgent for its whole
life and calls run_conversation() per turn: system prompt + toolset are
byte-stable across turns (provider prompt-cache friendly), memories load once,
and voice_loop.py owns the dialogue history (append-only, passed each turn).

stdin:  {"prompt": "...", "history": [{"role","content"}, ...]} lines
stdout: {"status":"ready"} then {"content": "..."} lines
"""
import json
import os
import sys
import time

os.environ.setdefault("HERMES_HOME", os.path.expanduser("~/.hermes"))
os.environ.setdefault("HERMES_YOLO_MODE", "1")
os.environ.setdefault("HERMES_ACCEPT_HOOKS", "1")
sys.path.insert(0, os.path.expanduser("~/.hermes/hermes-agent"))

import logging
logging.disable(logging.CRITICAL)

_agent = None
DFV_DIR = "/Users/data/Documents/Programming/Datas_Projects/data-facetime-voice"


def _voice_system_prompt() -> str:
    """DATA's voice-call system context, imported from voice_loop (one source)."""
    if DFV_DIR not in sys.path:
        sys.path.insert(0, DFV_DIR)
    import voice_loop as vl
    return vl.SYSTEM_CONTEXT


def _ensure_agent():
    global _agent
    if _agent is not None:
        return _agent
    from run_agent import AIAgent
    from hermes_cli.config import load_config
    from hermes_cli.runtime_provider import resolve_runtime_provider

    cfg = load_config()
    # Voice latency: reasoning_effort max costs seconds per turn on the phone.
    agent_cfg = cfg.get("agent")
    if isinstance(agent_cfg, dict):
        agent_cfg["reasoning_effort"] = os.environ.get("DFV_REASONING_EFFORT", "low")
    info = resolve_runtime_provider()
    model_cfg = cfg.get("model") or {}
    model_name = (
        model_cfg.get("default") or model_cfg.get("model") or ""
        if isinstance(model_cfg, dict)
        else str(model_cfg)
    )
    _agent = AIAgent(
        model=model_name,
        provider=info.get("provider"),
        base_url=info.get("base_url"),
        api_key=info.get("api_key"),
        api_mode=info.get("api_mode"),
        credential_pool=info.get("credential_pool"),
        enabled_toolsets=["file", "web", "search", "terminal", "session_search", "memory"],
        ephemeral_system_prompt=_voice_system_prompt(),
        skip_context_files=True,
        load_soul_identity=True,
        quiet_mode=True,
        save_trajectories=False,
        platform="voice",
        user_name="Captain Spencer",
    )
    return _agent


def process_request(request: dict) -> dict:
    prompt = request.get("prompt", "")
    if not prompt:
        return {"error": "empty prompt"}
    t0 = time.perf_counter()
    try:
        agent = _ensure_agent()
        # The persistent agent owns the conversation (session history lives in
        # the agent in-process — same as a long-lived gateway session, which is
        # what keeps the provider prompt cache warm). One call per turn.
        if request.get("stream"):
            def _cb(delta: str) -> None:
                sys.stdout.write(json.dumps({"delta": delta}) + "\n")
                sys.stdout.flush()
            text = agent.chat(prompt, stream_callback=_cb)
        else:
            text = agent.chat(prompt)
        text = text if isinstance(text, str) else str(text)
        return {"content": text, "elapsed": round(time.perf_counter() - t0, 3)}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}", "elapsed": round(time.perf_counter() - t0, 3)}


def main():
    sys.stdout.write(json.dumps({"status": "ready"}) + "\n")
    sys.stdout.flush()
    # Pre-warm the agent once — construction + memory load + toolset build
    # happen HERE, not on Captain's first words.
    try:
        _ensure_agent()
    except Exception:
        pass
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