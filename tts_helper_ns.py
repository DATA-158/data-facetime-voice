#!/usr/bin/env python3
"""Persistent NSSpeechSynthesizer helper — the SYSTEM voice (Siri), same protocol
as tts_helper.py.

WHY (2026-09-11, DATA's Mac, macOS 26.6)
---------------------------------------
The Captain's system voice is a Siri voice (com.apple.siri.natural.Aaron,
"Siri Voice 1"). Apple does not expose Siri voices to AVSpeechSynthesizer at
all — speechVoices() lists none, voiceWithIdentifier() returns nil for every
spelling — so tts_helper.py silently fell back to Samantha (the "robotic
woman"). `say` renders the Siri voice but pays ~0.6 s of process/engine init
per sentence.

NSSpeechSynthesizer(voice: nil) renders the system default voice — verified
sample-identical to `say` (max abs diff 0.0, same byte size) — and stays
warm. Measured per render: "Yes." 235 ms; typical sentence 350-650 ms
(~0.2 s per second of audio, neural synthesis ~5x realtime). The one open
question is whether a persistent instance follows a system-voice change at
runtime; restart the voice service after changing the voice.

Output is written to a temp AIFF (startSpeakingString:toURL:) — the only
file/buffer path NSSpeechSynthesizer offers — then decoded. The runloop must
be pumped on the calling (main) thread while it renders.

PROTOCOL (identical to tts_helper.py)
    stdin :  {"text": "...", "voice": "<identifier>"}\n   (voice optional; "" = system)
    stdout:  {"ok": true, "samples": N, "rate": R}\n + N*4 bytes float32 LE mono
             or {"ok": false, "error": "..."}\n
Run directly to self-test:  python tts_helper_ns.py --self-test
"""
from __future__ import annotations

import json
import os
import queue
import sys
import tempfile
import threading
import time

RENDER_TIMEOUT_S = 30.0


def _fail(msg: str) -> None:
    sys.stdout.write(json.dumps({"ok": False, "error": msg}) + "\n")
    sys.stdout.flush()


def main() -> int:
    self_test = "--self-test" in sys.argv[1:]
    try:
        import numpy as np
        import soundfile as sf
        from AppKit import NSSpeechSynthesizer
        from Foundation import NSDate, NSRunLoop, NSURL
    except Exception as e:  # pragma: no cover
        if self_test:
            print(f"unavailable: {type(e).__name__}: {e}")
            return 2
        sys.stdout.write(json.dumps({"status": "unavailable", "error": f"{type(e).__name__}: {e}"}) + "\n")
        sys.stdout.flush()
        return 2

    # Silence NSSpeechSynthesizer's harmless "Error -50" property chatter.
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, 2)

    synth = NSSpeechSynthesizer.alloc().initWithVoice_(None)
    runloop = NSRunLoop.currentRunLoop()
    tmpdir = tempfile.mkdtemp(prefix="dfv-tts-")
    counter = [0]

    def render(text: str, voice_ident: str):
        if voice_ident and str(synth.voice() or "") != voice_ident:
            if not synth.setVoice_(voice_ident):
                raise RuntimeError(f"voice {voice_ident!r} rejected by NSSpeechSynthesizer")
        counter[0] += 1
        path = os.path.join(tmpdir, f"{counter[0]}.aiff")
        t0 = time.perf_counter()
        if not synth.startSpeakingString_toURL_(text, NSURL.fileURLWithPath_(path)):
            raise RuntimeError("startSpeakingString:toURL: refused")
        while synth.isSpeaking() and time.perf_counter() - t0 < RENDER_TIMEOUT_S:
            runloop.runUntilDate_(NSDate.dateWithTimeIntervalSinceNow_(0.003))
        if synth.isSpeaking():
            synth.stopSpeaking()
            raise RuntimeError("render timed out")
        try:
            audio, rate = sf.read(path, dtype="float32", always_2d=False)
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass
        if getattr(audio, "ndim", 1) > 1:
            audio = audio.mean(axis=1)
        if len(audio) == 0:
            raise RuntimeError("synthesizer produced no audio")
        return audio.astype(np.float32), float(rate)

    if self_test:
        print("system voice:", NSSpeechSynthesizer.defaultVoice() or "(default)")
        for label, text in (("first", "Captain, the line is live."), ("warm", "Captain, the line is live."),
                            ("short", "Yes."), ("long", "Day 254 of 365, Captain. September 11th.")):
            t = time.perf_counter()
            a, sr = render(text, "")
            print(f"{label:6s}: {(time.perf_counter()-t)*1000:5.0f}ms  {len(a)/sr:.2f}s audio @ {sr:.0f}Hz")
        return 0

    requests: "queue.Queue[str | None]" = queue.Queue()

    def reader():
        for line in sys.stdin:
            requests.put(line)
        requests.put(None)

    threading.Thread(target=reader, name="tts-helper-stdin", daemon=True).start()
    sys.stdout.write(json.dumps({"status": "ready", "engine": "NSSpeechSynthesizer",
                                 "voice": str(NSSpeechSynthesizer.defaultVoice() or "system-default")}) + "\n")
    sys.stdout.flush()
    while True:
        try:
            line = requests.get(timeout=0.25)
        except queue.Empty:
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
        sys.stdout.write(json.dumps({"ok": True, "samples": len(audio), "rate": rate}) + "\n")
        sys.stdout.flush()
        sys.stdout.buffer.write(payload)
        sys.stdout.buffer.flush()


if __name__ == "__main__":
    raise SystemExit(main())
