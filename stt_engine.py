#!/usr/bin/env python3
"""Speech-to-text for the voice loop — Apple Silicon GPU first, CPU fallback.

WHY THIS EXISTS
---------------
The original path ran faster-whisper (CTranslate2, int8, CPU). Measured on an
M-series Mac against a 6.79s utterance, best of three runs each:

    faster-whisper distil-small.en   cpu_threads=6    2112 ms
    faster-whisper distil-small.en   cpu_threads=10   3167 ms   (oversubscribed)
    faster-whisper distil-medium.en  cpu_threads=6    5849 ms
    mlx-whisper    whisper-base.en   GPU               125 ms
    mlx-whisper    whisper-small.en  GPU               334 ms
    mlx-whisper    distil-large-v3   GPU              1019 ms

whisper-small.en on the GPU is 6.3x faster than the CPU model it replaces AND
transcribed the probe more accurately. Two further benefits: STT stops competing
with TTS and the rest of the pipeline for CPU, and raising cpu_threads — the
obvious "make it faster" knob — actually made the old path 50% SLOWER, so this
removes a live footgun.

PRIVACY: inference is entirely local (Metal GPU, this machine). The only network
access is a one-time model download from HuggingFace on first run, exactly as
faster-whisper already did. No call audio ever leaves the machine.

Env:
  DFV_STT_ENGINE  auto (default) | mlx | faster
  DFV_STT_MODEL   MLX repo or faster-whisper name; sensible default per engine
  DFV_STT_THREADS faster-whisper CPU threads (default 6; do not raise, see above)
"""
from __future__ import annotations

import logging
import os
import threading
import time

import numpy as np

log = logging.getLogger("dfv.stt")

SAMPLE_RATE = 16000
_ENGINE = os.environ.get("DFV_STT_ENGINE", "auto").lower()
_MLX_DEFAULT = "mlx-community/whisper-small.en-mlx"
_FASTER_DEFAULT = "distil-small.en"


class _MLXBackend:
    """whisper on the Apple Silicon GPU via MLX."""

    name = "mlx"

    def __init__(self, model: str):
        import mlx_whisper  # noqa: F401  (import proves availability)
        self.model = model
        self._mlx_whisper = mlx_whisper

    def transcribe(self, f32_16k: np.ndarray) -> str:
        result = self._mlx_whisper.transcribe(
            f32_16k,
            path_or_hf_repo=self.model,
            language="en",
            fp16=True,
            condition_on_previous_text=False,
        )
        return (result.get("text") or "").strip()


class _FasterBackend:
    """faster-whisper on CPU — the original path, kept as fallback."""

    name = "faster-whisper"

    def __init__(self, model: str, threads: int):
        from faster_whisper import WhisperModel
        self.model = model
        self._m = WhisperModel(
            model,
            device="auto",
            compute_type="int8",
            cpu_threads=threads,
            download_root=os.path.expanduser("~/.cache/dfv-whisper"),
        )

    def transcribe(self, f32_16k: np.ndarray) -> str:
        segments, _info = self._m.transcribe(
            f32_16k, language="en", vad_filter=False, beam_size=1,
            condition_on_previous_text=False,
        )
        return " ".join(s.text.strip() for s in segments).strip()


class STT:
    """Picks the fastest available backend and prewarms it.

    Construct once (prewarms), then call ``transcribe(f32_16k) -> str``.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.backend = None
        model_env = os.environ.get("DFV_STT_MODEL", "")

        if _ENGINE in ("auto", "mlx"):
            try:
                t0 = time.perf_counter()
                self.backend = _MLXBackend(model_env or _MLX_DEFAULT)
                log.info("STT engine: MLX GPU model=%s (init %.2fs)",
                         self.backend.model, time.perf_counter() - t0)
            except Exception as e:
                if _ENGINE == "mlx":
                    raise
                log.warning("MLX STT unavailable (%s) — falling back to "
                            "faster-whisper on CPU (~6x slower)", e)

        if self.backend is None:
            threads = int(os.environ.get("DFV_STT_THREADS", "6"))
            t0 = time.perf_counter()
            self.backend = _FasterBackend(model_env or _FASTER_DEFAULT, threads)
            log.info("STT engine: faster-whisper CPU model=%s threads=%d (init %.2fs)",
                     self.backend.model, threads, time.perf_counter() - t0)

        self._prewarm()

    def _prewarm(self) -> None:
        """Burn the first-call cost (MLX: ~7s of model load + graph build) now.

        Without this the Captain's FIRST utterance of the day pays it, which is
        exactly the 'why was the first reply so slow' complaint.
        """
        try:
            t0 = time.perf_counter()
            silence = np.zeros(SAMPLE_RATE, dtype=np.float32)
            self.backend.transcribe(silence)
            log.info("STT prewarmed in %.2fs", time.perf_counter() - t0)
        except Exception as e:
            log.warning("STT prewarm failed: %s", e)

    def transcribe(self, f32_16k: np.ndarray) -> str:
        # Concurrent turns must not enter the same model at once.
        with self._lock:
            try:
                return self.backend.transcribe(f32_16k)
            except Exception:
                log.exception("STT failed")
                return ""

    def engine_name(self) -> str:
        return getattr(self.backend, "name", "?")


if __name__ == "__main__":
    # Self-test / benchmark: python stt_engine.py
    import subprocess
    import tempfile
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    text = ("Hey DATA, I was thinking about heading out for a run this evening, "
            "can you check what the weather is going to be like around six?")
    path = tempfile.mktemp(suffix=".wav")
    subprocess.run(["say", "-o", path, "--file-format=WAVE",
                    f"--data-format=LEI16@{SAMPLE_RATE}", "--", text],
                   check=True, capture_output=True)
    import soundfile as sf
    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    os.unlink(path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    stt = STT()
    print(f"\nengine: {stt.engine_name()}   utterance {len(audio)/sr:.2f}s")
    for i in range(3):
        t = time.perf_counter()
        out = stt.transcribe(audio)
        print(f"  run {i+1}: {(time.perf_counter()-t)*1000:6.0f}ms")
    print(f"  -> {out}")
