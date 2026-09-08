#!/usr/bin/env python3
"""Regression test for speculative STT — drives the real VAD/capture path.

Speculative STT starts transcribing during the endpointing window instead of
after it, which takes STT off the critical path. It is also the most race-prone
code in the voice loop: a transcript is produced on one thread, invalidated from
the audio callback on another, and consumed on a third. This test drives
CallSession.on_capture with synthetic packets and checks the properties that
actually matter on a live call:

  1. the speculation fires, IS used, and produces the same transcript the
     ordinary path would have produced
  2. when the caller resumes speaking, the stale half-utterance result is never
     spoken to (a fresh speculation covering the whole utterance is fine)
  3. DFV_SPECULATIVE_STT_MS=0 disables the feature cleanly

Run:  python test_speculative_stt.py     (exit 0 = pass)
"""
import os, sys, subprocess, tempfile, time, threading
os.environ.setdefault("DFV_CAPTURE_GAIN", "1.0")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, soundfile as sf
import voice_loop as vl

vl.stt = vl.STT()

SR = vl.SAMPLE_RATE_BRIDGE
PKT = int(SR * 0.1)  # 100ms packets, matching the daemon's cadence


def say_audio(text):
    p = tempfile.mktemp(suffix=".wav")
    subprocess.run(["say", "-o", p, "--file-format=WAVE",
                    f"--data-format=LEI16@{SR}", "--", text],
                   check=True, capture_output=True)
    a, _ = sf.read(p, dtype="float32", always_2d=False); os.unlink(p)
    return a.mean(axis=1) if a.ndim > 1 else a


class Pkt:
    def __init__(self, f32): self.pcm16 = vl.f32_to_pcm16(f32)


class Harness(vl.CallSession):
    """CallSession with the network and the LLM stubbed out."""
    def __init__(self):
        super().__init__(stub=None, call_id="test")
        self.turns = []
        self.done = threading.Event()

    def _process_turn(self, f32_16k, spec_epoch=None):
        text = None
        if spec_epoch is not None:
            text = self._resolve_speculation(spec_epoch, len(f32_16k),
                                             vl.SPECULATIVE_WAIT_MS / 1000.0)
        spec = text is not None
        if text is None:
            text = vl.stt.transcribe(f32_16k)
        self.turns.append({"text": text, "speculative": spec,
                           "samples": len(f32_16k)})
        self.done.set()


def feed(sess, audio, label=""):
    for i in range(0, len(audio) - PKT, PKT):
        sess.on_capture(Pkt(audio[i:i + PKT]))
        time.sleep(0.001)


def silence(ms):
    return np.zeros(int(SR * ms / 1000.0), dtype=np.float32)


def run(name, fn):
    print(f"\n--- {name} ---")
    try:
        fn()
    except AssertionError as e:
        print(f"  FAIL: {e}"); return False
    return True


SPEECH = say_audio("What is the weather going to be like this evening?")
print(f"probe utterance: {len(SPEECH)/SR:.2f}s")
LIVE_REF = vl.stt.transcribe(vl.Resampler(up=False).process(SPEECH))
print(f"live-path reference transcript: {LIVE_REF!r}")

ok = True

def t1():
    s = Harness()
    feed(s, SPEECH)
    feed(s, silence(1200))            # full endpointing window
    assert s.done.wait(10), "no turn produced"
    t = s.turns[0]
    print(f"  transcript  : {t['text']!r}")
    print(f"  speculative : {t['speculative']}")
    assert t["speculative"], "speculation was NOT used — no latency win"
    assert t["text"].strip(), "empty transcript"
    # same words as the live path (whitespace/case tolerant)
    norm = lambda x: " ".join(x.lower().replace(",", "").replace(".", "").split())
    assert norm(t["text"]) == norm(LIVE_REF), \
        f"speculative differs from live:\n    spec={t['text']!r}\n    live={LIVE_REF!r}"
    print("  matches the live-path transcript exactly")
ok &= run("1. speculation fires, is used, and is correct", t1)

def t2():
    s = Harness()
    first = say_audio("What is the weather")
    second = say_audio("going to be like this evening?")
    feed(s, first)
    feed(s, silence(400))     # long enough to START speculation (300ms)
    assert s.spec_started, "speculation should have started at 400ms of silence"
    feed(s, second)           # caller resumes — must invalidate
    assert not s.spec_started, "resumed speech did not invalidate the speculation"
    feed(s, silence(1200))
    assert s.done.wait(15), "no turn produced"
    t = s.turns[0]
    print(f"  transcript  : {t['text']!r}")
    print(f"  speculative : {t['speculative']}")
    # The safety property is NOT "no speculation was used" — after the caller
    # stops again a FRESH speculation legitimately starts, covering the whole
    # utterance, and using it is correct. What must never happen is the STALE
    # one (covering only the first half) being spoken to. The transcript is the
    # proof: if the stale result had been used, the second half would be gone.
    low = t["text"].lower()
    assert "evening" in low, f"second half of the utterance was LOST: {t['text']!r}"
    assert "weather" in low, f"first half of the utterance was lost: {t['text']!r}"
    print("  full utterance present — the stale first-half result was not used")
ok &= run("2. resumed speech invalidates the speculation", t2)

def t3():
    os.environ["DFV_SPECULATIVE_STT_MS"] = "0"
    import importlib; importlib.reload(vl)
    vl.stt = vl.STT()
    class H(Harness): pass
    s = H.__new__(H); vl.CallSession.__init__(s, stub=None, call_id="t"); s.turns=[]; s.done=threading.Event()
    s._process_turn = lambda a, p=None: (s.turns.append({"speculative": p is not None}), s.done.set())
    feed(s, SPEECH); feed(s, silence(1200))
    assert s.done.wait(10), "no turn produced with speculation disabled"
    assert not s.turns[0]["speculative"], "speculation ran despite being disabled"
    print("  DFV_SPECULATIVE_STT_MS=0 cleanly disables it")
ok &= run("3. the feature can be turned off", t3)

print("\n" + ("ALL CHECKS PASSED" if ok else "FAILURES ABOVE"))
sys.exit(0 if ok else 1)
