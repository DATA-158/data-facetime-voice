#!/usr/bin/env python3
"""End-to-end turn latency harness — measures the hot path without FaceTime.

Simulates one conversational turn exactly as the call loop runs it:

    utterance audio -> STT -> persistent Hermes worker (streaming)
                    -> first sentence popped off the delta stream
                    -> TTS -> first audio ready to hand the bridge

and reports TIME TO FIRST AUDIO, which is the number the caller actually feels
as dead air. Endpointing (the VAD silence window) is added as a constant since
it is a fixed configured cost with no audio to measure it against here.

Usage:
    python bench_turn.py                      # default conversational turns
    python bench_turn.py "check the weather"  # force a specific line
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).parent))
import tts_engine
from stt_engine import STT

HERMES_VENV = os.environ.get(
    "HERMES_VENV", os.path.expanduser("~/.hermes/hermes-agent/venv/bin/python"))
WORKER = str(Path(__file__).parent / "hermes_worker.py")
SILENCE_MS = int(os.environ.get("DFV_SILENCE_MS", "700"))

TURNS = [
    "Hey DATA, you there?",
    "What do you think about heading out for a run this evening?",
    "Fair enough. What's the weather going to be like around six?",
]


def make_utterance(text: str) -> np.ndarray:
    """Render a spoken utterance to feed the STT stage."""
    path = tempfile.mktemp(suffix=".wav")
    subprocess.run(["say", "-o", path, "--file-format=WAVE",
                    "--data-format=LEI16@16000", "--", text],
                   check=True, capture_output=True)
    audio, _sr = sf.read(path, dtype="float32", always_2d=False)
    os.unlink(path)
    return audio.mean(axis=1) if audio.ndim > 1 else audio


# Use the REAL splitter from voice_loop so this harness measures the same
# chunking the call loop performs, including the first-chunk clause break.
from voice_loop import _pop_sentences


def pop_sentence(buf: str) -> tuple[str | None, str]:
    chunks, rest = _pop_sentences(buf, allow_clause=True)
    if not chunks:
        return None, rest
    return chunks[0].strip(), "".join(chunks[1:]) + rest


def main() -> int:
    turns = [" ".join(sys.argv[1:])] if len(sys.argv) > 1 else TURNS

    print("warming up (this is service-start cost, not per-turn)…")
    t0 = time.perf_counter()
    stt = STT()
    tts_engine.init()
    print(f"  STT={stt.engine_name()}  TTS={tts_engine.engine_name()}  "
          f"({time.perf_counter()-t0:.1f}s)")

    env = dict(os.environ)
    env["HERMES_HOME"] = os.path.expanduser("~/.hermes")
    env["HERMES_YOLO_MODE"] = "1"
    env["HERMES_ACCEPT_HOOKS"] = "1"
    t0 = time.perf_counter()
    proc = subprocess.Popen([HERMES_VENV, WORKER],
                            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.DEVNULL, text=True, bufsize=1, env=env)
    ready = proc.stdout.readline()
    if not ready or json.loads(ready).get("status") != "ready":
        print("worker failed to start", file=sys.stderr)
        return 2
    print(f"  worker ready ({time.perf_counter()-t0:.1f}s)\n")

    rows = []
    for text in turns:
        audio = make_utterance(text)

        t_stt = time.perf_counter()
        transcript = stt.transcribe(audio)
        stt_ms = (time.perf_counter() - t_stt) * 1000

        t_llm = time.perf_counter()
        proc.stdin.write(json.dumps({"prompt": transcript, "stream": True}) + "\n")
        proc.stdin.flush()

        # Read to the turn's TERMINAL message every time. Stopping at the first
        # sentence leaves the rest of this turn's lines in the pipe, and the
        # next turn then reads them as its own — which is exactly how this
        # harness first reported a bogus 0ms turn. Timing is captured when the
        # first sentence appears; the loop keeps draining after that.
        buf, sentence, tier = "", None, "fast"
        first_tok_ms = llm_ms = None
        final_answer, final_ms = "", None
        while True:
            line = proc.stdout.readline()
            if not line:
                break
            msg = json.loads(line)
            if "filler" in msg:            # tool turn: filler IS the first audio
                tier = "full"
                if sentence is None:
                    sentence = msg["filler"]
                    llm_ms = (time.perf_counter() - t_llm) * 1000
                continue
            if "delta" in msg:
                if first_tok_ms is None:
                    first_tok_ms = (time.perf_counter() - t_llm) * 1000
                buf += msg["delta"]
                if sentence is None:
                    sentence, buf = pop_sentence(buf)
                    if sentence is not None:
                        llm_ms = (time.perf_counter() - t_llm) * 1000
            elif "content" in msg or "error" in msg:
                tier = msg.get("tier", tier)
                final_answer = (msg.get("content") or "").strip()
                final_ms = (time.perf_counter() - t_llm) * 1000
                if msg.get("error"):
                    final_answer = f"[ERROR] {msg['error']}"
                if sentence is None:
                    sentence = final_answer or None
                    llm_ms = final_ms
                break
        if llm_ms is None:
            llm_ms = (time.perf_counter() - t_llm) * 1000

        t_tts = time.perf_counter()
        tts_engine.synthesize(sentence or "I lost that one.")
        tts_ms = (time.perf_counter() - t_tts) * 1000

        pipeline = stt_ms + llm_ms + tts_ms
        rows.append((text, tier, stt_ms, first_tok_ms, llm_ms, tts_ms, pipeline))

        print(f'  "{text}"')
        print(f'    heard : "{transcript}"')
        print(f'    says  : "{(sentence or "")[:70]}"  [{tier} tier]')
        if final_answer and final_answer != (sentence or ""):
            print(f'    then  : "{final_answer[:70]}"  (full answer at '
                  f'{0 if final_ms is None else final_ms:.0f}ms)')
        print(f"    STT {stt_ms:6.0f}ms | LLM->1st sentence {llm_ms:6.0f}ms"
              f"{'' if first_tok_ms is None else f' (1st token {first_tok_ms:.0f}ms)'}"
              f" | TTS {tts_ms:5.0f}ms")
        print(f"    PIPELINE {pipeline:6.0f}ms  + {SILENCE_MS}ms endpointing "
              f"= {pipeline + SILENCE_MS:6.0f}ms of dead air\n")

    # Drain the worker's final line for the last turn if we broke early.
    try:
        proc.stdin.close()
        proc.wait(timeout=5)
    except Exception:
        proc.kill()

    if rows:
        avg = sum(r[6] for r in rows) / len(rows)
        worst = max(r[6] for r in rows)
        print(f"  {'-'*62}")
        print(f"  mean  time-to-first-audio: {avg + SILENCE_MS:6.0f}ms")
        print(f"  worst time-to-first-audio: {worst + SILENCE_MS:6.0f}ms")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
