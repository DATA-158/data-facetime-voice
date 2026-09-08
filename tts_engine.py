#!/usr/bin/env python3
"""Persistent macOS TTS engine — same voice as `say`, without `say`'s startup tax.

WHY THIS EXISTS
---------------
`say` pays a fixed speech-engine initialization on EVERY invocation. Measured on
an M-series Mac, 10 consecutive identical calls:

    687 663 718 677 668 660 664 677 662 654  ms

...and a bare `subprocess.run(["true"])` is 4ms, so that ~665ms is engine init
inside `say`, not fork cost. It is unavoidable per-invocation and it lands
directly on time-to-first-audio for every sentence the agent speaks.

A persistent ``NSSpeechSynthesizer`` initializes ONCE and then synthesizes the
same sentence in 23-30ms (411ms on its first call). That is ~640ms off every
spoken sentence, including the first one of every turn.

THE VOICE DOES NOT CHANGE. ``NSSpeechSynthesizer.defaultVoice()`` is the same
voice `say` uses with no ``-v`` flag, and the rendered audio is byte-identical:

    say        : 159220 bytes  sha256=f0ad702a7dfdf1af9011ab5e...
    persistent : 159220 bytes  sha256=f0ad702a7dfdf1af9011ab5e...

That equality is not assumed — ``verify_voice_fidelity()`` re-proves it at
startup on whatever machine this runs on, and any mismatch (or any failure to
load PyObjC or the configured voice) falls back to `say` permanently. Siri
voices in particular are restricted from NSSpeechSynthesizer, so a host whose
default voice is a Siri voice will fail the check and keep using `say` — the
voice is preserved either way, we just don't get the speedup there.

Set DFV_TTS_ENGINE=say to force the old path, =native to skip the self-test.
"""
from __future__ import annotations

import hashlib
import logging
import os
import queue
import subprocess
import tempfile
import threading
import time

import numpy as np

log = logging.getLogger("dfv.tts")

BRIDGE_RATE = 24000
_ENGINE_MODE = os.environ.get("DFV_TTS_ENGINE", "auto").lower()  # auto|native|say
# Empty = system default voice (the Captain's pick; do not change it).
TTS_VOICE = os.environ.get("DFV_TTS_VOICE", "")


def _resample_to_bridge(audio: np.ndarray, sr: int) -> np.ndarray:
    """Linear resample to the bridge's 24 kHz. NSSpeechSynthesizer emits 22050."""
    if abs(sr - BRIDGE_RATE) <= 1:
        return audio.astype(np.float32)
    n_out = int(round(len(audio) * BRIDGE_RATE / sr))
    if n_out <= 1:
        return np.zeros(0, dtype=np.float32)
    idx = np.linspace(0, len(audio) - 1, n_out)
    return np.interp(idx, np.arange(len(audio)), audio).astype(np.float32)


def _read_audio(path: str) -> tuple[np.ndarray, int]:
    import soundfile as sf
    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    return audio, sr


# ---------------------------------------------------------------------------
# Fallback path: `say` (correct, slow — 665ms/sentence)
# ---------------------------------------------------------------------------
def _say_to_array(text: str) -> np.ndarray:
    """Synthesize via `say` at the bridge rate directly (the legacy path)."""
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
        path = tf.name
    try:
        cmd = ["say", "-o", path, "--file-format=WAVE",
               f"--data-format=LEI16@{BRIDGE_RATE}"]
        if TTS_VOICE:
            cmd += ["-v", TTS_VOICE]
        cmd += ["--", text]
        subprocess.run(cmd, check=True, capture_output=True, timeout=30)
        audio, sr = _read_audio(path)
        return _resample_to_bridge(audio, sr)
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Fast path: one long-lived NSSpeechSynthesizer on a dedicated runloop thread
# ---------------------------------------------------------------------------
class _NativeSynth:
    """Owns one long-lived NSSpeechSynthesizer on a dedicated worker thread.

    Completion is detected by polling ``isSpeaking()``, NOT by the delegate
    callback: NSSpeechSynthesizer schedules its delegate on the MAIN runloop, so
    a delegate never fires on a worker thread and every synthesis times out.
    Polling works off-thread and needs no runloop (measured: 407ms first call,
    then 24-27ms, byte-identical output).

    The synthesizer is not thread-safe and concurrent turns can overlap, so all
    work is serialized through a queue. Synthesis is ~25ms, so callers simply
    block on the result rather than dealing in futures.
    """

    def __init__(self) -> None:
        self._jobs: "queue.Queue[tuple[str, str, queue.Queue]]" = queue.Queue()
        self._ready = threading.Event()
        self._error: str | None = None
        self._thread = threading.Thread(
            target=self._run, name="dfv-tts-synth", daemon=True)
        self._thread.start()
        if not self._ready.wait(timeout=20):
            raise RuntimeError("native synth thread did not become ready")
        if self._error:
            raise RuntimeError(self._error)

    def _run(self) -> None:
        try:
            from AppKit import NSSpeechSynthesizer
            from Foundation import NSURL

            voice = None
            if TTS_VOICE:
                # Accept either an identifier or a display name ("Samantha").
                for v in NSSpeechSynthesizer.availableVoices():
                    attrs = NSSpeechSynthesizer.attributesForVoice_(v)
                    name = str(attrs.get("VoiceName", "")) if attrs else ""
                    if TTS_VOICE in (str(v), name):
                        voice = v
                        break
                if voice is None:
                    self._error = (
                        f"voice {TTS_VOICE!r} not available to NSSpeechSynthesizer "
                        "(Siri voices are restricted); using `say`")
                    self._ready.set()
                    return
            else:
                voice = NSSpeechSynthesizer.defaultVoice()

            synth = NSSpeechSynthesizer.alloc().init()
            if not synth.setVoice_(voice):
                self._error = f"could not select voice {voice!r}; using `say`"
                self._ready.set()
                return

            self.voice_id = str(voice)
        except Exception as e:  # PyObjC missing, voice unavailable, etc.
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
                # Done when the engine stops AND the file has actually landed —
                # isSpeaking() can still read False in the instant between the
                # call and the engine spinning up.
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

    def to_file(self, text: str, path: str) -> None:
        result: "queue.Queue[tuple[str, str | None]]" = queue.Queue()
        self._jobs.put((text, path, result))
        status, err = result.get(timeout=35)
        if status != "ok":
            raise RuntimeError(err or "native synthesis failed")

    def to_array(self, text: str) -> np.ndarray:
        path = tempfile.mktemp(suffix=".aiff")
        try:
            self.to_file(text, path)
            audio, sr = _read_audio(path)
            return _resample_to_bridge(audio, sr)
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Engine selection + the voice-fidelity proof
# ---------------------------------------------------------------------------
_synth: _NativeSynth | None = None
_use_native = False
_init_lock = threading.Lock()
_initialized = False

_PROBE = "Captain, the evening hours should be quite agreeable for a run."


def verify_voice_fidelity(synth: _NativeSynth) -> tuple[bool, str]:
    """Prove the persistent synth renders IDENTICAL audio to `say`.

    Both paths write their native default format (AIFF, no --data-format) so the
    comparison is of the voice itself, not of a resampling stage. Returns
    (identical, detail).
    """
    say_path = tempfile.mktemp(suffix=".aiff")
    nat_path = tempfile.mktemp(suffix=".aiff")
    try:
        cmd = ["say", "-o", say_path]
        if TTS_VOICE:
            cmd += ["-v", TTS_VOICE]
        cmd += ["--", _PROBE]
        subprocess.run(cmd, check=True, capture_output=True, timeout=30)
        synth.to_file(_PROBE, nat_path)

        def sha(p: str) -> str:
            with open(p, "rb") as f:
                return hashlib.sha256(f.read()).hexdigest()

        a, b = sha(say_path), sha(nat_path)
        if a == b:
            return True, f"identical (sha256 {a[:16]})"
        sa = os.path.getsize(say_path)
        sb = os.path.getsize(nat_path)
        return False, f"differ: say {sa}B/{a[:12]} vs native {sb}B/{b[:12]}"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"
    finally:
        for p in (say_path, nat_path):
            try:
                os.unlink(p)
            except OSError:
                pass


def init() -> None:
    """Select the TTS path once, proving voice fidelity before using the fast one."""
    global _synth, _use_native, _initialized
    with _init_lock:
        if _initialized:
            return
        _initialized = True

        if _ENGINE_MODE == "say":
            log.info("TTS engine: `say` (forced by DFV_TTS_ENGINE=say)")
            _say_to_array(".")  # prewarm
            return

        t0 = time.perf_counter()
        try:
            _synth = _NativeSynth()
        except Exception as e:
            log.warning("TTS native engine unavailable (%s) — using `say`", e)
            _synth = None
            _say_to_array(".")
            return

        if _ENGINE_MODE == "native":
            _use_native = True
            log.info("TTS engine: persistent NSSpeechSynthesizer voice=%s "
                     "(self-test skipped by DFV_TTS_ENGINE=native)",
                     getattr(_synth, "voice_id", "?"))
        else:
            identical, detail = verify_voice_fidelity(_synth)
            if identical:
                _use_native = True
                log.info("TTS engine: persistent NSSpeechSynthesizer voice=%s — "
                         "voice fidelity verified vs `say`: %s",
                         getattr(_synth, "voice_id", "?"), detail)
            else:
                _use_native = False
                _synth = None
                log.warning("TTS voice fidelity check FAILED (%s) — keeping `say`. "
                            "The voice is preserved; the ~640ms/sentence speedup "
                            "is not available on this host.", detail)
                _say_to_array(".")

        # Burn the native engine's one-time 411ms first-call cost here, not on
        # the Captain's first turn.
        if _use_native:
            try:
                synthesize("Ready.")
            except Exception:
                pass
        log.info("TTS init complete in %.2fs", time.perf_counter() - t0)


def synthesize(text: str) -> np.ndarray:
    """Return float32 mono PCM at 24 kHz for `text`, using the selected engine."""
    if not _initialized:
        init()
    if _use_native and _synth is not None:
        try:
            return _synth.to_array(text)
        except Exception as e:
            # One bad sentence must never take the call down: fall back for
            # this utterance and keep the fast path for the next one.
            log.warning("native TTS failed (%s) — `say` for this sentence", e)
    return _say_to_array(text)


def engine_name() -> str:
    return "NSSpeechSynthesizer" if _use_native else "say"


if __name__ == "__main__":
    # Self-test / benchmark: python tts_engine.py
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
