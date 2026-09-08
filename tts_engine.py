#!/usr/bin/env python3
"""macOS TTS for the voice loop — same voice as `say`, without `say`'s startup tax.

WHY THIS EXISTS
---------------
`say` pays a fixed speech-engine initialization on EVERY invocation. Measured on
an M-series Mac, 10 consecutive identical calls:

    687 663 718 677 668 660 664 677 662 654  ms

...and a bare `subprocess.run(["true"])` is 4ms, so that ~665ms is engine init
inside `say`, not fork cost. It is unavoidable per-invocation, it cannot be
prewarmed across processes, and it lands directly on time-to-first-audio for every
sentence the agent speaks.

THREE BACKENDS, TRIED IN ORDER
------------------------------
    backend                per sentence   notes
    NSSpeechSynthesizer        21-37ms    fastest, but cannot load Siri voices
    AVSpeechSynthesizer        62-77ms    reaches premium/Siri voices; streams
    say                      632-675ms    always works; the floor we fall back to

Order matters and is deliberate. NSSpeechSynthesizer is fastest, so it goes
first — but it is the legacy API and cannot load Siri voices, which is exactly
the case on the live host. AVSpeechSynthesizer is a different, newer engine that
CAN reach those voices, so it catches the hosts where NSSpeechSynthesizer fails.
Either way the host lands on something 10-30x faster than `say`.

THE VOICE NEVER CHANGES. That is enforced, not assumed: at startup each candidate
backend renders a probe phrase, `say` renders the same phrase, both are decoded to
raw samples and compared. A backend is used ONLY if its audio is sample-identical
to `say`. Verified for AVSpeechSynthesizer on the development host:

    77562 samples both · max sample error 0.000000 · correlation 1.000000

Comparison is at the SAMPLE level, not a file hash, because the backends emit
different containers (AIFF vs raw float32 buffers) — hashing files would report a
false mismatch on audio that is in fact identical.

Set DFV_TTS_ENGINE to force one of: ns | av | say. Default `auto` tries in order.
"""
from __future__ import annotations

import json
import logging
import os
import queue
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np

log = logging.getLogger("dfv.tts")

BRIDGE_RATE = 24000
_ENGINE_MODE = os.environ.get("DFV_TTS_ENGINE", "auto").lower()
# Empty = system default voice (the Captain's pick; do not change it).
TTS_VOICE = os.environ.get("DFV_TTS_VOICE", "")
HELPER_SCRIPT = str(Path(__file__).parent / "tts_helper.py")

# Sample-level fidelity thresholds. Identical synthesis should be exact; the
# tolerance exists only for float round-trip, not to wave through a near-match.
_MAX_SAMPLE_ERR = 1e-4
_MAX_LEN_DELTA = 8  # samples


def _resample_to_bridge(audio: np.ndarray, sr: float) -> np.ndarray:
    """Linear resample to the bridge's 24 kHz. The synthesizers emit 22050."""
    if abs(sr - BRIDGE_RATE) <= 1:
        return audio.astype(np.float32)
    n_out = int(round(len(audio) * BRIDGE_RATE / sr))
    if n_out <= 1:
        return np.zeros(0, dtype=np.float32)
    idx = np.linspace(0, len(audio) - 1, n_out)
    return np.interp(idx, np.arange(len(audio)), audio).astype(np.float32)


def _read_audio(path: str) -> tuple[np.ndarray, float]:
    import soundfile as sf
    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio, float(sr)


# ---------------------------------------------------------------------------
# Backends. Each returns (float32 mono, sample_rate) at its NATIVE rate.
# ---------------------------------------------------------------------------
class _SayBackend:
    """The floor. Always available, always correct, always ~665ms."""

    name = "say"

    def render(self, text: str) -> tuple[np.ndarray, float]:
        path = tempfile.mktemp(suffix=".aiff")
        try:
            cmd = ["say", "-o", path]
            if TTS_VOICE:
                cmd += ["-v", TTS_VOICE]
            cmd += ["--", text]
            subprocess.run(cmd, check=True, capture_output=True, timeout=30)
            return _read_audio(path)
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass

    def close(self) -> None:
        pass


class _AVBackend:
    """AVSpeechSynthesizer, hosted in tts_helper.py.

    The write callback only fires on the MAIN runloop — on a worker thread it
    never runs at all (measured: 0 buffers). The voice loop's main thread is busy
    running the service, so the synthesizer lives in its own small process whose
    main thread does nothing but pump a runloop. See tts_helper.py.
    """

    name = "AVSpeechSynthesizer"

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._proc = subprocess.Popen(
            [sys.executable, HELPER_SCRIPT],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, bufsize=0,
        )
        line = self._proc.stdout.readline()
        if not line:
            raise RuntimeError("tts helper did not start")
        status = json.loads(line.decode())
        if status.get("status") != "ready":
            raise RuntimeError(status.get("error") or "tts helper unavailable")

    def render(self, text: str) -> tuple[np.ndarray, float]:
        with self._lock:
            if self._proc.poll() is not None:
                raise RuntimeError("tts helper died")
            req = json.dumps({"text": text, "voice": TTS_VOICE}) + "\n"
            self._proc.stdin.write(req.encode())
            self._proc.stdin.flush()
            header = self._proc.stdout.readline()
            if not header:
                raise RuntimeError("tts helper closed mid-request")
            meta = json.loads(header.decode())
            if not meta.get("ok"):
                raise RuntimeError(meta.get("error") or "helper render failed")
            n = int(meta["samples"])
            need = n * 4
            buf = bytearray()
            while len(buf) < need:
                chunk = self._proc.stdout.read(need - len(buf))
                if not chunk:
                    raise RuntimeError("tts helper truncated audio")
                buf.extend(chunk)
            audio = np.frombuffer(bytes(buf), dtype="<f4").astype(np.float32)
            return audio, float(meta["rate"])

    def close(self) -> None:
        try:
            self._proc.stdin.close()
            self._proc.wait(timeout=3)
        except Exception:
            try:
                self._proc.kill()
            except Exception:
                pass


class _NSBackend:
    """NSSpeechSynthesizer on a dedicated worker thread.

    Completion is detected by polling isSpeaking(), NOT the delegate: the delegate
    is scheduled on the main runloop and never fires on a worker thread.
    """

    name = "NSSpeechSynthesizer"

    def __init__(self) -> None:
        self._jobs: "queue.Queue" = queue.Queue()
        self._ready = threading.Event()
        self._error: str | None = None
        self._thread = threading.Thread(target=self._run, name="dfv-tts-ns", daemon=True)
        self._thread.start()
        if not self._ready.wait(timeout=20):
            raise RuntimeError("NSSpeechSynthesizer thread did not become ready")
        if self._error:
            raise RuntimeError(self._error)

    def _run(self) -> None:
        try:
            from AppKit import NSSpeechSynthesizer
            from Foundation import NSURL
            if TTS_VOICE:
                voice = None
                for v in NSSpeechSynthesizer.availableVoices():
                    attrs = NSSpeechSynthesizer.attributesForVoice_(v)
                    name = str(attrs.get("VoiceName", "")) if attrs else ""
                    if TTS_VOICE in (str(v), name):
                        voice = v
                        break
                if voice is None:
                    self._error = f"voice {TTS_VOICE!r} not available"
                    self._ready.set()
                    return
            else:
                voice = NSSpeechSynthesizer.defaultVoice()
            synth = NSSpeechSynthesizer.alloc().init()
            if not synth.setVoice_(voice):
                self._error = f"could not select voice {voice!r}"
                self._ready.set()
                return
            self.voice_id = str(voice)
        except Exception as e:
            self._error = f"{type(e).__name__}: {e}"
            self._ready.set()
            return

        self._ready.set()
        while True:
            text, path, result = self._jobs.get()
            if text is None:
                return
            try:
                synth.startSpeakingString_toURL_(text, NSURL.fileURLWithPath_(path))
                t0 = time.perf_counter()
                while time.perf_counter() - t0 < 30:
                    if not synth.isSpeaking():
                        try:
                            if os.path.getsize(path) > 0:
                                break
                        except OSError:
                            pass
                    time.sleep(0.002)
                else:
                    synth.stopSpeaking()
                    result.put(("error", "synthesis timed out"))
                    continue
                result.put(("ok", None))
            except Exception as e:
                result.put(("error", f"{type(e).__name__}: {e}"))

    def render(self, text: str) -> tuple[np.ndarray, float]:
        path = tempfile.mktemp(suffix=".aiff")
        result: "queue.Queue" = queue.Queue()
        try:
            self._jobs.put((text, path, result))
            status, err = result.get(timeout=35)
            if status != "ok":
                raise RuntimeError(err or "native synthesis failed")
            return _read_audio(path)
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass

    def close(self) -> None:
        try:
            self._jobs.put((None, None, None))
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Fidelity gate
# ---------------------------------------------------------------------------
_PROBE = "Captain, the evening hours should be quite agreeable for a run."


def compare_to_say(backend, probe: str = _PROBE) -> tuple[bool, str]:
    """True only if `backend` renders audio sample-identical to `say`.

    Compared as decoded samples rather than file bytes: the backends emit
    different containers (AIFF vs raw float32), so a file hash would report a
    mismatch on audio that is in fact identical.
    """
    try:
        ref, ref_sr = _SayBackend().render(probe)
        got, got_sr = backend.render(probe)
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"

    if abs(got_sr - ref_sr) > 1:
        n = int(round(len(got) * ref_sr / got_sr))
        if n <= 1:
            return False, "empty render"
        got = np.interp(np.linspace(0, len(got) - 1, n),
                        np.arange(len(got)), got).astype(np.float32)

    if abs(len(got) - len(ref)) > _MAX_LEN_DELTA:
        return False, (f"length differs: {len(got)} vs {len(ref)} samples "
                       f"({abs(len(got)-len(ref))} delta)")
    n = min(len(got), len(ref))
    if n == 0:
        return False, "empty render"
    err = float(np.max(np.abs(got[:n] - ref[:n])))
    if err > _MAX_SAMPLE_ERR:
        return False, f"audio differs: max sample error {err:.6f}"
    return True, f"sample-identical ({n} samples, max error {err:.2e})"


# ---------------------------------------------------------------------------
# Engine selection, phrase cache, public API
# ---------------------------------------------------------------------------
_backend = None
_init_lock = threading.Lock()
_initialized = False

# Fixed lines (fillers, fallbacks, greetings) are spoken verbatim many times and
# are often the FIRST audio of a turn — on a tool turn the filler is all the
# caller hears until the agent loop finishes. Synthesizing them once at startup
# takes that cost off the critical path entirely.
_cache: dict[str, np.ndarray] = {}
_cache_lock = threading.Lock()
_CACHE_MAX = int(os.environ.get("DFV_TTS_CACHE_MAX", "64"))


def _candidates() -> list:
    order = {"av": ["av"], "ns": ["ns"], "say": ["say"]}.get(
        _ENGINE_MODE, ["ns", "av", "say"])
    out = []
    for key in order:
        try:
            if key == "av":
                out.append(_AVBackend())
            elif key == "ns":
                out.append(_NSBackend())
            else:
                out.append(_SayBackend())
        except Exception as e:
            log.info("TTS backend %s unavailable: %s", key, e)
    return out


def init() -> None:
    """Select the fastest backend that proves sample-identical to `say`."""
    global _backend, _initialized
    with _init_lock:
        if _initialized:
            return
        _initialized = True
        t0 = time.perf_counter()

        for cand in _candidates():
            if cand.name == "say":
                _backend = cand
                log.info("TTS engine: `say` (the fallback floor, ~665ms/sentence)")
                break
            ok, detail = compare_to_say(cand)
            if ok:
                _backend = cand
                log.info("TTS engine: %s — voice fidelity verified vs `say`: %s",
                         cand.name, detail)
                break
            log.warning("TTS backend %s rejected — %s", cand.name, detail)
            cand.close()

        if _backend is None:
            _backend = _SayBackend()
            log.warning("TTS falling back to `say`; the voice is preserved, the "
                        "speedup is not available on this host")

        try:  # burn the one-off first-call cost here, not on the Captain's turn
            synthesize("Ready.")
        except Exception:
            pass
        log.info("TTS init complete in %.2fs (engine=%s)",
                 time.perf_counter() - t0, engine_name())


def preload(phrases) -> None:
    """Pre-synthesize fixed lines so they cost nothing when first spoken."""
    if not _initialized:
        init()
    done = 0
    for phrase in phrases:
        text = (phrase or "").strip()
        if not text or text in _cache:
            continue
        try:
            audio = _render(text)
            with _cache_lock:
                _cache[text] = audio
            done += 1
        except Exception as e:
            log.warning("preload failed for %r: %s", text[:40], e)
    if done:
        log.info("TTS preloaded %d fixed phrase(s); they now cost ~0ms", done)


def _render(text: str) -> np.ndarray:
    audio, sr = _backend.render(text)
    return _resample_to_bridge(audio, sr)


def synthesize(text: str) -> np.ndarray:
    """Return float32 mono PCM at 24 kHz for `text`."""
    if not _initialized:
        init()
    with _cache_lock:
        hit = _cache.get(text)
    if hit is not None:
        return hit
    try:
        return _render(text)
    except Exception as e:
        # One bad sentence must never take the call down. Fall back for this
        # utterance and keep the fast backend for the next one.
        log.warning("TTS backend %s failed (%s) — `say` for this sentence",
                    engine_name(), e)
        audio, sr = _SayBackend().render(text)
        return _resample_to_bridge(audio, sr)


def engine_name() -> str:
    return getattr(_backend, "name", "say")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    init()
    print(f"\nengine: {engine_name()}")
    for text in ("Here and ready, Captain.",
                 "The evening hours should be quite agreeable for a run."):
        times = []
        for _ in range(5):
            t = time.perf_counter()
            audio = synthesize(text)
            times.append((time.perf_counter() - t) * 1000)
        print(f"  {len(text):3d} chars -> {len(audio)/BRIDGE_RATE:.2f}s audio | "
              f"synth {min(times):.0f}-{max(times):.0f}ms")
    preload(["Let me check that, Captain."])
    t = time.perf_counter()
    synthesize("Let me check that, Captain.")
    print(f"  cached phrase -> {(time.perf_counter()-t)*1000:.2f}ms")
