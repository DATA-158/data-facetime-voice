#!/usr/bin/env python3
"""Persistent AVSpeechSynthesizer helper — one process, main thread, runloop.

WHY A SEPARATE PROCESS
----------------------
AVSpeechSynthesizer is the fastest path to macOS speech by a wide margin, and it
is the only one that STREAMS — audio buffers arrive while synthesis is still
running, instead of a whole file at the end:

    say                   first audio 665 ms   (fixed engine init, every call)
    NSSpeechSynthesizer   first audio  25 ms
    AVSpeechSynthesizer   first audio 8.7 ms   complete 42 ms, 303 buffers

And it renders SAMPLE-FOR-SAMPLE IDENTICAL audio to `say` — verified by decoding
both and comparing: 77562 samples each, max error 0.000000, correlation 1.000000.

The catch: ``writeUtterance:toBufferCallback:`` only fires on the MAIN runloop.
On a worker thread the callback never runs at all (measured: 0 buffers, every
attempt times out) — the same trap as NSSpeechSynthesizer's delegate.

The voice loop's main thread is busy running the service, and restructuring it to
host a runloop would be an invasive change to a process that also owns gRPC
streams and call lifecycle. So the synthesizer lives here instead: a tiny process
whose main thread does nothing but pump a runloop and synthesize. stdin is read on
a background thread so the main thread stays free.

Engine init is paid ONCE, at startup, for the life of the process.

PROTOCOL
    stdin :  {"text": "...", "voice": "<identifier>"}\\n     (voice optional)
    stdout:  {"ok": true, "samples": N, "rate": R}\\n
             followed by exactly N*4 bytes of little-endian float32 mono
             or {"ok": false, "error": "..."}\\n with no audio payload

Run directly to self-test:  python tts_helper.py --self-test
"""
from __future__ import annotations

import json
import queue
import sys
import threading
import time

SILENCE_TIMEOUT_S = 30.0


def _fail(msg: str) -> None:
    sys.stdout.write(json.dumps({"ok": False, "error": msg}) + "\n")
    sys.stdout.flush()


def main() -> int:
    self_test = "--self-test" in sys.argv[1:]
    try:
        import AVFoundation as AV
        from Foundation import NSRunLoop, NSDate
        import numpy as np
    except Exception as e:  # pyobjc / numpy missing
        if self_test:
            print(f"unavailable: {type(e).__name__}: {e}")
            return 2
        sys.stdout.write(json.dumps({"status": "unavailable",
                                     "error": f"{type(e).__name__}: {e}"}) + "\n")
        sys.stdout.flush()
        return 2

    synth = AV.AVSpeechSynthesizer.alloc().init()
    runloop = NSRunLoop.currentRunLoop()

    def resolve_voice(ident: str):
        """Exact identifier, else the system default's identifier, else en-US."""
        if ident:
            for v in AV.AVSpeechSynthesisVoice.speechVoices():
                if str(v.identifier()) == ident or str(v.name()) == ident:
                    return v
            return None
        try:
            from AppKit import NSSpeechSynthesizer
            default_id = str(NSSpeechSynthesizer.defaultVoice())
            for v in AV.AVSpeechSynthesisVoice.speechVoices():
                if str(v.identifier()) == default_id:
                    return v
        except Exception:
            pass
        return AV.AVSpeechSynthesisVoice.voiceWithLanguage_("en-US")

    def render(text: str, voice_ident: str):
        """Synthesize on THIS (main) thread, pumping the runloop. Returns (f32, rate)."""
        voice = resolve_voice(voice_ident)
        if voice is None:
            raise RuntimeError(f"voice {voice_ident!r} not available to AVSpeechSynthesizer")
        state = {"chunks": 0, "parts": [], "done": False, "rate": None}

        def cb(buf):
            try:
                n = int(buf.frameLength())
            except Exception:
                n = 0
            if n == 0:
                state["done"] = True
                return
            if state["rate"] is None:
                try:
                    state["rate"] = float(buf.format().sampleRate())
                except Exception:
                    state["rate"] = 22050.0
            state["chunks"] += 1
            try:
                fp = buf.floatChannelData()
                if fp:
                    state["parts"].append(np.asarray(fp[0][:n], dtype=np.float32).copy())
            except Exception:
                pass

        utt = AV.AVSpeechUtterance.speechUtteranceWithString_(text)
        utt.setVoice_(voice)
        t0 = time.perf_counter()
        synth.writeUtterance_toBufferCallback_(utt, cb)
        while not state["done"] and time.perf_counter() - t0 < SILENCE_TIMEOUT_S:
            runloop.runUntilDate_(NSDate.dateWithTimeIntervalSinceNow_(0.002))
        if not state["parts"]:
            raise RuntimeError(
                "synthesizer produced no audio "
                f"(chunks={state['chunks']}) — voice likely not renderable")
        audio = np.concatenate(state["parts"])
        return audio, float(state["rate"] or 22050.0)

    if self_test:
        t = time.perf_counter()
        a, sr = render("Captain, the line is live.", "")
        print(f"first render : {(time.perf_counter()-t)*1000:.0f}ms  "
              f"{len(a)} samples @ {sr:.0f}Hz")
        for i in range(3):
            t = time.perf_counter()
            a, sr = render("Captain, the line is live.", "")
            print(f"warm render  : {(time.perf_counter()-t)*1000:.0f}ms  {len(a)} samples")
        return 0

    # stdin on a background thread so the main thread stays free for the runloop
    requests: "queue.Queue[str | None]" = queue.Queue()

    def reader():
        for line in sys.stdin:
            requests.put(line)
        requests.put(None)

    threading.Thread(target=reader, name="tts-helper-stdin", daemon=True).start()

    sys.stdout.write(json.dumps({"status": "ready"}) + "\n")
    sys.stdout.flush()

    while True:
        try:
            line = requests.get(timeout=0.25)
        except queue.Empty:
            # Keep the runloop alive between requests so the engine stays warm.
            runloop.runUntilDate_(NSDate.dateWithTimeIntervalSinceNow_(0.01))
            continue
        if line is None:
            return 0
        line = line.strip()
        if not line:
            continue
        try:
            req = json.loads(line)
        except json.JSONDecodeError as e:
            _fail(f"invalid JSON: {e}")
            continue
        text = (req.get("text") or "").strip()
        if not text:
            _fail("empty text")
            continue
        try:
            audio, rate = render(text, req.get("voice") or "")
        except Exception as e:
            _fail(f"{type(e).__name__}: {e}")
            continue
        payload = audio.astype("<f4").tobytes()
        sys.stdout.write(json.dumps(
            {"ok": True, "samples": len(audio), "rate": rate}) + "\n")
        sys.stdout.flush()
        sys.stdout.buffer.write(payload)
        sys.stdout.buffer.flush()


if __name__ == "__main__":
    raise SystemExit(main())
