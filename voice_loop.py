#!/usr/bin/env python3
"""
DATA FaceTime Voice — call handler for the facetime-bridge daemon.

Answers the Captain's FaceTime Audio call and runs the voice loop:
  bridge CAPTURE (24 kHz s16le) -> resample 16k -> Silero VAD end-of-speech
  -> faster-whisper STT -> persistent Hermes worker (true DATA, warm ~1.7s)
  -> sentence-chunked Siri system-voice TTS -> resample 24k -> bridge PLAYBACK.

Barge-in: while TTS is playing, VAD watches the caller's stream; on speech,
send a CLEAR packet (flush daemon playback queue) and cancel in-flight work.

Concurrency model: one thread per call. gRPC bidi Audio stream is driven by a
reader thread (CAPTURE -> VAD buffer) and a writer queue thread (PLAYBACK).
The bridge daemon enforces single-call semantics on its side.

Exit code 0 on clean hangup; nonzero on bridge errors (launchd RestartAlways
brings the handler back for the next call).
"""
from __future__ import annotations

import json
import logging
import os
import queue
import signal
import subprocess
import sys
import threading
import time
import wave
from pathlib import Path

import grpc
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
import facetime_media_pb2 as pb
import facetime_media_pb2_grpc as pb_grpc
import tts_engine

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
SOCKET_PATH = os.environ.get(
    "FACETIME_BRIDGE_SOCKET", os.path.expanduser("~/.facetime-bridge/bridge.sock")
)
RECORDINGS_DIR = os.environ.get("FACETIME_BRIDGE_RECORDINGS_DIR", "")
LOG_PATH = os.environ.get(
    "DFV_LOG_PATH",
    str(Path(__file__).parent / "logs" / "facetime_voice.log"),
)
# The worker runs under the HERMES venv (needs the hermes-agent tree); the
# voice loop itself runs under the AUDIO venv (grpc/mlx/whisper/pyobjc). They
# are usually different interpreters, so this must be host-configurable rather
# than a hardcoded path that silently breaks the worker on any other machine.
HERMES_VENV = os.environ.get(
    "HERMES_VENV", os.path.expanduser("~/agent-calling/.venv-native-tts/bin/python"))
WORKER_SCRIPT = str(Path(__file__).parent / "hermes_worker.py")

SAMPLE_RATE_BRIDGE = 24000  # daemon contract
SAMPLE_RATE_STT = 16000      # whisper + VAD
CHUNK_MS = 20                # legacy label; actual packet cadence measured per packet
MIN_SILENCE_MS = int(os.environ.get("DFV_SILENCE_MS", "700"))  # end-of-utterance
SPEECH_START_MS = int(os.environ.get("DFV_SPEECH_START_MS", "60"))  # consecutive speech packets to open
# Speculative STT: begin transcribing once the caller has been quiet this long,
# rather than waiting out the full endpointing window first. See
# _maybe_start_speculative_stt for why this is safe. 0 disables.
SPECULATIVE_STT_MS = int(os.environ.get("DFV_SPECULATIVE_STT_MS", "300"))
# How long a finalized turn will wait for an in-flight speculation before giving
# up and transcribing itself. Sized above a normal STT decode so a slower host
# still collects the work it already started rather than paying for it twice.
SPECULATIVE_WAIT_MS = int(os.environ.get("DFV_SPECULATIVE_WAIT_MS", "700"))
# Outbound trigger: the gateway (or any local client) drops a JSON file here
# to have the warm voice service place an authorized call. Polled at TRIGGER_POLL_S
TRIGGER_PATH = os.environ.get(
    "DFV_TRIGGER_PATH", os.path.expanduser("~/.facetime-bridge/outbound.trigger")
)
# 2026-09-07: was 2.0s. This interval is dead time between "DATA, call me" and
# the dial actually starting, on every outbound call, for no benefit — the poll
# is a single stat() on a path that almost never exists.
TRIGGER_POLL_S = float(os.environ.get("DFV_TRIGGER_POLL_S", "0.25"))
# The only number this system will ever dial or answer (fail-closed, mirrors
# the bridge daemon's FACETIME_BRIDGE_AUTHORIZED_CALLER_E164 contract).
#
# 2026-09-07 BUGFIX: this was a double lookup —
#     os.environ.get(os.environ.get("FACETIME_BRIDGE_AUTHORIZED_CALLER_E164", ""))
# which reads the env var *named by* the phone number and therefore always
# evaluated to None. Consequences, both confirmed: the dial URL became the
# literal "facetime-audio://None", and _call_timer_running() passed None into
# subprocess(env=...) which raises TypeError, was swallowed by its bare except,
# and so ALWAYS reported "not connected" — forcing every outbound call through
# the blind 60s/90s timeout branches. That was the "calling takes minutes" bug.
AUTHORIZED_E164 = os.environ.get("FACETIME_BRIDGE_AUTHORIZED_CALLER_E164", "").strip()
MAX_UTTERANCE_MS = 30_000
# Voice selection now lives in tts_engine (it owns synthesis). Empty = the
# system default voice — the Captain's pick, and not to be changed.
TTS_VOICE = tts_engine.TTS_VOICE
MAX_TURNS = 200

# Spoken when a turn yields nothing sayable. A live call must never go silent:
# the caller cannot see the log, so an audible failure beats dead air.
TURN_FAILED_LINE = os.environ.get(
    "DFV_TURN_FAILED_LINE",
    "Captain, I lost that one. Say again?",
)
# Spoken when an outbound call connects, so the Captain does not answer to dead air.
GREETING_LINE = os.environ.get(
    "DFV_GREETING_LINE",
    "Captain, DATA here. The line is live — go ahead.",
)
# Must match hermes_worker's DFV_TOOL_FILLER — the worker chooses when to say it,
# but the voice loop is what synthesizes it, so it preloads it here.
TOOL_FILLER_LINE = os.environ.get("DFV_TOOL_FILLER", "Let me check that, Captain.")

# Fixed lines are spoken verbatim over and over, and are often the FIRST audio of
# a turn — on a tool turn the filler is all the caller hears until the agent loop
# finishes. Pre-synthesizing them takes that cost off the critical path.
PRELOAD_PHRASES = [TURN_FAILED_LINE, GREETING_LINE, TOOL_FILLER_LINE]

# Silence RMS floor applied to utterance audio before STT (caller holds-open
# mic + BlackHole loop can carry a faint DC/noise floor).
UTTERANCE_RMS_GATE = 0.004
# Caller capture gain (2026-09-07 15:01 call: voice arrived at RMS 0.001-0.005
# with peaks 0.295 — 20-30x below the agent's own level; the 0.012 VAD trigger
# never fired). Applied to capture packets before VAD/STT; recordings stay raw.
CAPTURE_GAIN = float(os.environ.get("DFV_CAPTURE_GAIN", "10.0"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stderr)],
)
log = logging.getLogger("dfv")


def _ensure_log_dir() -> None:
    p = Path(LOG_PATH)
    p.parent.mkdir(parents=True, exist_ok=True)
    try:
        h = logging.FileHandler(p)
        h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        logging.getLogger().addHandler(h)
    except Exception:
        pass


# ----------------------------------------------------------------------------
# Audio helpers
# ----------------------------------------------------------------------------
def pcm16_to_f32(pcm: bytes) -> np.ndarray:
    return np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0


def f32_to_pcm16(x: np.ndarray) -> bytes:
    return (np.clip(x, -1.0, 1.0) * 32767.0).astype(np.int16).tobytes()


class Resampler:
    """Linear resampler — cheap, fine for 16k<->24k speech."""

    def __init__(self, up: bool):
        self.up = up  # True: 16k -> 24k
        self.ratio = SAMPLE_RATE_BRIDGE / SAMPLE_RATE_STT if up else SAMPLE_RATE_STT / SAMPLE_RATE_BRIDGE
        self._carry = np.zeros(0, dtype=np.float32)

    def process(self, x: np.ndarray) -> np.ndarray:
        buf = np.concatenate([self._carry, x])
        n_out = int(len(buf) * self.ratio)
        if n_out <= 1:
            self._carry = buf
            return np.zeros(0, dtype=np.float32)
        idx = np.arange(n_out) / self.ratio
        out = np.interp(idx, np.arange(len(buf)), buf).astype(np.float32)
        consumed = int(idx[-1]) + 1
        self._carry = buf[consumed:]
        return out


# ----------------------------------------------------------------------------
# STT — MLX GPU whisper, faster-whisper CPU fallback (see stt_engine.py)
# ----------------------------------------------------------------------------
# 2026-09-07 latency pass #2: the CPU model took 2112ms on a 6.8s utterance,
# fully serial ahead of the LLM. The same work on the Apple Silicon GPU is
# 333ms and transcribes more accurately. Inference stays entirely local.
from stt_engine import STT  # noqa: E402


# ----------------------------------------------------------------------------
# LLM — persistent Hermes worker (true DATA: persona, memory, tools)
# ----------------------------------------------------------------------------
# DATA's persona lives in voice_persona (dependency-free) so hermes_worker
# can import it under a different interpreter without dragging in the audio
# stack. Re-exported here so existing references keep working.
from voice_persona import SYSTEM_CONTEXT  # noqa: E402,F401

_worker_proc = None
_worker_lock = threading.Lock()
_dialogue: list[dict] = []          # [{"role","content"}] for this call
_dialogue_lock = threading.Lock()


def _ensure_worker() -> None:
    global _worker_proc
    if _worker_proc is not None and _worker_proc.poll() is None:
        return
    env = dict(os.environ)
    env["HERMES_HOME"] = os.path.expanduser("~/.hermes")
    env["HERMES_YOLO_MODE"] = "1"
    env["HERMES_ACCEPT_HOOKS"] = "1"
    _worker_proc = subprocess.Popen(
        [HERMES_VENV, WORKER_SCRIPT],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL, text=True, bufsize=1, env=env,
    )
    line = _worker_proc.stdout.readline()
    if not line or json.loads(line).get("status") != "ready":
        raise RuntimeError("hermes worker failed to start")


def llm_reply(user_text: str) -> str:
    """True-DATA reply via the persistent worker, with call dialogue history.

    2026-09-07: history is passed to the worker each turn (append-only) so the
    Hermes agent runs ONE cached conversation instead of rebuilding per turn —
    removed ~9.5s of per-turn scaffolding overhead (measured).
    """
    global _worker_proc
    with _dialogue_lock:
        _dialogue.append({"role": "user", "content": user_text})
    t0 = time.perf_counter()
    with _worker_lock:
        _ensure_worker()
        line = None
        for attempt in (1, 2):  # restart-on-death with single retry (spec #8a)
            _worker_proc.stdin.write(json.dumps({"prompt": user_text}) + "\n")
            _worker_proc.stdin.flush()
            line = _worker_proc.stdout.readline()
            if line:
                break
            log.warning("worker died mid-turn (attempt %d); respawning", attempt)
            _worker_proc = None
            _ensure_worker()
        if not line:
            raise RuntimeError("worker died mid-call")
        resp = json.loads(line)
    elapsed = time.perf_counter() - t0
    if resp.get("error"):
        log.warning("worker error: %s", resp["error"])
    text = (resp.get("content") or "").strip()
    if not text:
        text = "I'm here, Captain, but my response came back empty."
    with _dialogue_lock:
        _dialogue.append({"role": "assistant", "content": text})
    log.info("LLM turn: %.2fs, %d chars", elapsed, len(text))
    return text


# ----------------------------------------------------------------------------
# TTS — Siri Voice 1 via `say` (streamed sentence-by-sentence)
# ----------------------------------------------------------------------------
_SENTENCE_END = (".", "!", "?")
_PAUSE_MARKS = (";", ":")


def split_sentences(text: str) -> list[str]:
    out, buf = [], []
    for ch in text:
        buf.append(ch)
        if ch in _SENTENCE_END:
            out.append("".join(buf).strip())
            buf = []
    tail = "".join(buf).strip()
    if tail:
        out.append(tail)
    return [s for s in out if s]


class TTSCanceled(Exception):
    pass


def tts_sentences(sentences: list[str], emit, cancelled) -> None:
    """Synthesize each sentence and emit it at the bridge's 24 kHz.

    2026-09-07 latency pass #2: synthesis moved to tts_engine, which holds ONE
    persistent NSSpeechSynthesizer instead of spawning `say` per sentence.
    `say` pays ~665ms of speech-engine init on EVERY invocation regardless of
    text length (measured 654-718ms over 10 identical runs; bare process spawn
    is 4ms) and that landed on time-to-first-audio for every sentence. The
    persistent engine does the same work in 25-37ms.

    The VOICE IS UNCHANGED — tts_engine verifies byte-identical output against
    `say` at startup and falls back to `say` permanently if it ever differs.
    """
    for sentence in sentences:
        if cancelled():
            raise TTSCanceled()
        clean = sentence.strip()
        if not clean:
            continue
        audio = tts_engine.synthesize(clean)
        if cancelled():
            raise TTSCanceled()
        emit(audio, is_first=True)


_SENTENCE_BOUNDARY = ".!?…"
_CLAUSE_BOUNDARY = ",;:—–"
# The first chunk of a turn may break at a CLAUSE once it is at least this long.
# Long enough that we never ship a chopped fragment ("Well,"), short enough to
# get audio moving. Later chunks always wait for a full sentence, which reads
# better and costs nothing — by then playback is already ahead of synthesis.
MIN_FIRST_CHUNK_CHARS = int(os.environ.get("DFV_MIN_FIRST_CHUNK", "24"))


def _is_real_boundary(buf: str, idx: int) -> bool:
    """Reject punctuation that is not a speech boundary: 3.5, 1,000, e.g."""
    prev_c = buf[idx - 1] if idx > 0 else " "
    next_c = buf[idx + 1] if idx + 1 < len(buf) else " "
    if prev_c.isdigit() and next_c.isdigit():
        return False          # decimal point or thousands separator
    return True


def _pop_sentences(buf: str, allow_clause: bool = False) -> tuple[list[str], str]:
    """Pop speakable chunks off a growing delta buffer (streaming TTS).

    2026-09-07 latency pass #2: this used to split ONLY on .!?… so a reply that
    opened with a long clause — measured live: "Honestly, Captain, I can't argue
    with the logic — an evening run clears the head..." — held all audio for
    3.8s waiting on the first period. With TTS now at ~40ms/chunk there is no
    reason to wait: `allow_clause` lets the FIRST chunk of a turn break at a
    comma or dash once it has enough words to sound deliberate.
    """
    out: list[str] = []
    start = 0
    for idx, ch in enumerate(buf):
        is_sentence = ch in _SENTENCE_BOUNDARY
        is_clause = (allow_clause and not out and ch in _CLAUSE_BOUNDARY
                     and (idx - start) >= MIN_FIRST_CHUNK_CHARS)
        if (is_sentence or is_clause) and _is_real_boundary(buf, idx):
            out.append(buf[start:idx + 1])
            start = idx + 1
    return out, buf[start:]


# ----------------------------------------------------------------------------
# The call loop
# ----------------------------------------------------------------------------
class CallSession:
    def __init__(self, stub: pb_grpc.FaceTimeMediaStub, call_id: str):
        self.stub = stub
        self.call_id = call_id
        self.writer = queue.Queue()
        self.playing = threading.Event()
        self.barge_in = threading.Event()
        self.capture_f32_24k = queue.Queue(maxsize=500)
        self.capture_lock = threading.Lock()   # guards VAD ring buffer
        self.turn_active = threading.Semaphore(1)  # single-flight turn guard
        # Speculative STT bookkeeping. `spec_epoch` invalidates an in-flight
        # speculation the moment the caller starts speaking again.
        self.spec_lock = threading.Lock()
        self.spec_epoch = 0
        self.spec_started = False
        self.spec_result: tuple[int, int, str] | None = None  # (epoch, samples, text)
        self.silence_run = 0
        self.speech_run = 0
        self.in_speech = False
        self.utterance: list[np.ndarray] = []
        self.utterance_ms = 0
        self.rec_writer_caller = None
        self.rec_writer_agent = None

    # ---------------- daemon I/O ----------------
    def audio_loop(self) -> None:
        """Writer thread: pull (kind, payload) from queue -> daemon stream."""
        # Audio() RPC: we write first packet (START) and read CAPTURE/EVENT.
        # gRPC bidi: requests iterator must be independent; use a generator
        # that yields from self.writer, reading responses in this same thread.
        def gen():
            yield pb.AudioPacket(
                call_id=self.call_id, kind=pb.AUDIO_PACKET_KIND_START,
                sample_rate=SAMPLE_RATE_BRIDGE, channels=1, sequence=0,
            )
            while True:
                kind, payload = self.writer.get()
                if kind == "STOP":
                    yield pb.AudioPacket(
                        call_id=self.call_id, kind=pb.AUDIO_PACKET_KIND_STOP,
                        sample_rate=SAMPLE_RATE_BRIDGE, channels=1,
                        sequence=self._next_seq(),
                    )
                    return
                elif kind == "CLEAR":
                    yield pb.AudioPacket(
                        call_id=self.call_id, kind=pb.AUDIO_PACKET_KIND_CLEAR,
                        sample_rate=SAMPLE_RATE_BRIDGE, channels=1,
                        sequence=self._next_seq(),
                    )
                else:  # PLAYBACK with payload
                    yield pb.AudioPacket(
                        call_id=self.call_id, kind=pb.AUDIO_PACKET_KIND_PLAYBACK,
                        pcm16=payload, sample_rate=SAMPLE_RATE_BRIDGE, channels=1,
                        sequence=self._next_seq(),
                    )

        try:
            for resp in self.stub.Audio(gen()):
                if resp.kind == pb.AUDIO_PACKET_KIND_CAPTURE:
                    self.on_capture(resp)
                elif resp.kind == pb.AUDIO_PACKET_KIND_EVENT:
                    log.info("audio event: %s", resp.event)
        except grpc.RpcError as e:
            if e.code() not in (grpc.StatusCode.CANCELLED,):
                log.warning("audio stream closed: %s", e.code())

    def _next_seq(self) -> int:
        self._seq = getattr(self, "_seq", 0) + 1
        return self._seq

    def on_capture(self, resp) -> None:
        if RECORDINGS_DIR and self.rec_writer_caller is not None:
            try:
                self.rec_writer_caller.writeframes(resp.pcm16)
            except Exception:
                pass
        f32_24k = pcm16_to_f32(resp.pcm16)
        n_samples = len(f32_24k)
        # Packet cadence from the ACTUAL payload (daemon emits ~100ms packets;
        # the old hardcoded CHUNK_MS=20 made end-of-utterance detection 5x
        # slower than configured — a major per-turn latency tax).
        packet_ms = n_samples / SAMPLE_RATE_BRIDGE * 1000.0
        # Capture gain (see CAPTURE_GAIN note): caller audio arrives far below
        # the agent's own level; bring it up BEFORE the VAD sees it.
        f32_24k = np.clip(f32_24k * CAPTURE_GAIN, -1.0, 1.0)
        # Receive-path watchdog: if we've been connected a while and NEVER
        # seen a non-silent packet, the caller->BlackHole16 route is dead
        # (typically FaceTime output reverted to speakers). Say so once.
        rms_now = float(np.sqrt(np.mean(f32_24k ** 2))) if n_samples else 0.0
        self.max_capture_rms = max(getattr(self, "max_capture_rms", 0.0), rms_now)
        self.capture_packets = getattr(self, "capture_packets", 0) + 1
        if (
            not getattr(self, "dead_path_warned", False)
            and self.capture_packets >= 200  # ~20s of packets
            and self.max_capture_rms < 0.012
        ):
            self.dead_path_warned = True
            log.error(
                "RECEIVE PATH DEAD: %.1fs of capture, max RMS %.5f — FaceTime "
                "output is not reaching BlackHole 16ch. Telling Captain.",
                self.capture_packets * packet_ms / 1000.0,
                self.max_capture_rms,
            )
            def _warn_dead():
                try:
                    self.playing.set()
                    tts_sentences(
                        split_sentences(
                            "Captain, I can't hear you — my receive path is dead. "
                            "Hang up and I'll call right back."
                        ),
                        emit=self._emit_speech,
                        cancelled=lambda: self.barge_in.is_set(),
                    )
                except TTSCanceled:
                    pass
                finally:
                    self.playing.clear()
                    self.barge_in.clear()
            threading.Thread(target=_warn_dead, daemon=True).start()
        with self.capture_lock:
            if self.in_speech:
                self.utterance.append(f32_24k)
                self.utterance_ms += packet_ms
            # energy-based VAD (no model during playback; caller mic path is
            # post-loopback so it stays clean) + Silero confirm at end
            rms = rms_now
            if self.in_speech:
                if rms < 0.004:  # post-gain floor: caller noise floor ~0.003
                    self.silence_run += packet_ms
                else:
                    # Speech resumed — any speculation in flight is now stale.
                    if self.silence_run and self.spec_started:
                        self._invalidate_speculation()
                    self.silence_run = 0
                self._last_silence_run_ms = self.silence_run
                if (SPECULATIVE_STT_MS
                        and not self.spec_started
                        and self.silence_run >= SPECULATIVE_STT_MS
                        and self.silence_run < MIN_SILENCE_MS):
                    self._maybe_start_speculative_stt()
                if (self.silence_run >= MIN_SILENCE_MS or
                        self.utterance_ms >= MAX_UTTERANCE_MS):
                    self._finalize_utterance()
            else:
                if self.playing.is_set() and rms > 0.02:
                    # caller speaks over TTS -> barge-in
                    self.barge_in.set()
                    self.playing.clear()
                    self.writer.put(("CLEAR", None))
                    self._start_speech(rms)
                elif not self.playing.is_set() and rms > 0.006:
                    # post-gain trigger: caller speech 0.01-0.05 after 10x gain,
                    # noise floor 0.003 — 0.006 separates them with margin
                    self.speech_run = getattr(self, "speech_run", 0) + packet_ms
                    if self.speech_run >= SPEECH_START_MS:
                        self.speech_run = 0
                        self._start_speech(rms)
                elif not self.playing.is_set():
                    self.speech_run = 0

    def _start_speech(self, rms: float) -> None:
        self.in_speech = True
        self.utterance_ms = 0
        self.silence_run = 0
        self.speech_run = 0
        self.utterance = []
        log.info("speech start (rms %.4f)", rms)

    # ---------------- speculative STT ----------------
    def _prepare_for_stt(self, audio: np.ndarray, trim_ms: float) -> np.ndarray | None:
        """Trim trailing silence, normalize, downsample. None if not worth decoding."""
        trim_samples = int(min(trim_ms, 700) / 1000.0 * SAMPLE_RATE_BRIDGE)
        if trim_samples > 0 and len(audio) > trim_samples:
            audio = audio[:-trim_samples]
        if len(audio) < SAMPLE_RATE_BRIDGE * 0.25:  # <250ms
            return None
        peak = float(np.abs(audio).max()) if audio.size else 0.0
        if 0.0 < peak < 0.5:
            audio = np.clip(audio * (0.7 / peak), -1.0, 1.0)
        f32_16k = Resampler(up=False).process(audio)
        if float(np.sqrt(np.mean(f32_16k ** 2))) < UTTERANCE_RMS_GATE:
            return None
        return f32_16k

    def _invalidate_speculation(self) -> None:
        """Discard any in-flight speculation — the caller resumed speaking."""
        with self.spec_lock:
            self.spec_epoch += 1
            self.spec_started = False
            self.spec_result = None

    def _maybe_start_speculative_stt(self) -> None:
        """Transcribe the utterance-so-far during the endpointing window.

        WHY THIS IS SAFE. The endpointing window is SPECULATIVE_STT_MS..
        MIN_SILENCE_MS of pure silence — by definition no speech arrives in it,
        or `silence_run` would have reset and invalidated this. And the finalized
        utterance gets its trailing silence trimmed anyway, so the audio decoded
        here is the same SPEECH the final path would decode. If anything does
        differ, _finalize_utterance re-checks the sample count and falls back to
        transcribing for real, so a stale speculation can never be spoken to.

        The cost of being wrong is one discarded transcript. The win is that STT
        (~250ms on the dev host, ~500ms on the live host) overlaps the silence
        wait instead of running after it.
        """
        if stt is None or not self.utterance:
            return
        audio = np.concatenate(self.utterance)
        prepared = self._prepare_for_stt(audio, self.silence_run)
        if prepared is None:
            return
        with self.spec_lock:
            self.spec_started = True
            epoch = self.spec_epoch
        n = len(prepared)

        def _run() -> None:
            try:
                text = stt.transcribe(prepared)
            except Exception:
                log.exception("speculative STT failed")
                return
            with self.spec_lock:
                if epoch == self.spec_epoch:
                    self.spec_result = (epoch, n, text)

        threading.Thread(target=_run, name="dfv-spec-stt", daemon=True).start()

    def _claim_speculation(self) -> int | None:
        """Snapshot the epoch to resolve against, or None if nothing is in flight.

        Called from _finalize_utterance, which runs on the gRPC capture thread —
        so it must not block. The actual waiting happens in _process_turn, which
        has its own thread.
        """
        with self.spec_lock:
            if not self.spec_started:
                return None
            self.spec_started = False
            return self.spec_epoch

    def _resolve_speculation(self, epoch: int, n_samples: int,
                             timeout_s: float) -> str | None:
        """Wait briefly for the in-flight speculation, then validate it.

        Waiting beats giving up: the speculation is already partway through the
        decode, so finishing it is strictly cheaper than starting a second one.
        On a host where STT takes longer than the remaining endpointing window
        this is the difference between the feature helping and it costing double.
        """
        deadline = time.perf_counter() + timeout_s
        while True:
            with self.spec_lock:
                result = self.spec_result
                current = self.spec_epoch
            if current != epoch:
                return None            # invalidated by resumed speech
            if result is not None:
                self.spec_result = None
                got_epoch, got_n, text = result
                if got_epoch != epoch:
                    return None
                # Guard against decoding a different span than we finalized on.
                if abs(got_n - n_samples) > SAMPLE_RATE_STT * 0.12:  # >120ms
                    log.info("speculation span mismatch (%d vs %d samples); "
                             "re-transcribing", got_n, n_samples)
                    return None
                return text
            if time.perf_counter() >= deadline:
                log.info("speculation did not land within %.0fms; transcribing",
                         timeout_s * 1000)
                return None
            time.sleep(0.005)

    def _finalize_utterance(self) -> None:
        audio = np.concatenate(self.utterance) if self.utterance else np.zeros(0)
        self.in_speech = False
        self.utterance = []
        self.silence_run = 0
        # Tail trim, normalize, downsample, RMS gate — shared with the
        # speculative path so both decode exactly the same preparation.
        f32_16k = self._prepare_for_stt(
            audio, getattr(self, "_last_silence_run_ms", 0))
        if f32_16k is None:
            log.info("utterance too short or below RMS gate, dropping")
            self._invalidate_speculation()
            return
        spec_epoch = self._claim_speculation()
        threading.Thread(
            target=self._process_turn, args=(f32_16k, spec_epoch), daemon=True
        ).start()

    def _process_turn(self, f32_16k: np.ndarray,
                      spec_epoch: int | None = None) -> None:
        # Single-flight. Turns are spawned per finalized utterance, so two
        # utterances in quick succession used to run concurrently: both blocked
        # on _worker_lock, both queued speech into the same playback stream, and
        # whichever finished first cleared the OTHER turn's barge_in flag. The
        # result was overlapping replies and barge-in that stopped working.
        if not self.turn_active.acquire(blocking=False):
            log.info("turn already in flight; dropping overlapping utterance")
            return
        try:
            t0 = time.perf_counter()
            text = None
            if spec_epoch is not None:
                text = self._resolve_speculation(
                    spec_epoch, len(f32_16k), SPECULATIVE_WAIT_MS / 1000.0)
            if text is not None:
                stt_ms, how = (time.perf_counter() - t0) * 1000, "speculative"
            else:
                text = stt.transcribe(f32_16k)
                stt_ms = (time.perf_counter() - t0) * 1000
                how = "live"
            if not text:
                log.info("empty transcript; saying nothing")
                return
            log.info("Captain: %s (STT %.0fms, %s)", text, stt_ms, how)
            self.playing.set()
            spoke = {"any": False}

            def _speak(sentence: str, is_final: bool = False) -> None:
                spoke["any"] = True
                self._speak_sentence(sentence, is_final=is_final)

            try:
                # Streaming turn: first sentence speaks while the model is
                # still generating the rest (2026-09-07 latency pass).
                llm_reply_streaming(
                    text,
                    speak=_speak,
                    cancelled=lambda: self.barge_in.is_set(),
                )
            except TTSCanceled:
                log.info("TTS canceled by barge-in")
            except Exception:
                # Worker died, stream broke, JSON was malformed — the caller is
                # still on the line and must hear something.
                log.exception("turn failed mid-stream")
                if not spoke["any"] and not self.barge_in.is_set():
                    try:
                        self._speak_sentence(TURN_FAILED_LINE, is_final=True)
                    except Exception:
                        log.exception("fallback line failed to speak")
            finally:
                self.playing.clear()
                self.barge_in.clear()
        except Exception:
            log.exception("turn failed")
        finally:
            self.turn_active.release()

    def _speak_sentence(self, sentence: str, is_final: bool = False) -> None:
        """TTS one sentence and emit it (streaming turn path)."""
        try:
            tts_sentences([sentence], emit=self._emit_speech,
                          cancelled=lambda: self.barge_in.is_set())
        except TTSCanceled:
            raise
        except Exception:
            log.exception("sentence TTS failed")

    def _emit_speech(self, f32_24k: np.ndarray, is_first: bool = False) -> None:
        pcm = f32_to_pcm16(f32_24k)
        if RECORDINGS_DIR and self.rec_writer_agent is not None:
            try:
                self.rec_writer_agent.writeframes(pcm)
            except Exception:
                log.debug("agent recording write failed", exc_info=True)
        self.writer.put(("PLAYBACK", pcm))


stt: STT | None = None


def _init_models() -> None:
    """Load and prewarm STT + TTS. Every first-call cost belongs HERE.

    Both engines have a large one-time cost (MLX model load and graph build,
    NSSpeechSynthesizer's 411ms first synthesis). Paying them at service start
    is the difference between a normal first turn and the 'why was his first
    reply so slow' complaint.
    """
    global stt
    t0 = time.perf_counter()
    log.info("loading STT…")
    stt = STT()
    log.info("STT ready (%s) in %.2fs", stt.engine_name(), time.perf_counter() - t0)

    t0 = time.perf_counter()
    try:
        tts_engine.init()
        tts_engine.preload(PRELOAD_PHRASES)
        log.info("TTS ready (%s) in %.2fs",
                 tts_engine.engine_name(), time.perf_counter() - t0)
    except Exception as e:
        log.warning("tts init failed: %s", e)


def _warm_worker() -> None:
    """Warm the worker WITHOUT polluting the call transcript.

    This used to go through llm_reply(), which appends both the ping and its
    reply to _dialogue. Every call therefore began with a synthetic
    "System ping."/"ready." exchange that was written into the transcript and
    fed to the post-call memory summarizer — and left _dialogue non-empty, so
    persist_call_memory() fired even for calls where nobody said anything.
    """
    try:
        t0 = time.perf_counter()
        with _worker_lock:
            _ensure_worker()
            _worker_proc.stdin.write(json.dumps(
                {"prompt": "System ping. Reply with the single word: ready."}) + "\n")
            _worker_proc.stdin.flush()
            line = _worker_proc.stdout.readline()
        if not line:
            raise RuntimeError("worker gave no response to warm ping")
        log.info("worker warm: %.2fs", time.perf_counter() - t0)
    except Exception as e:
        log.warning("worker warmup failed (will retry on first call): %s", e)


# Optional per-call recording (mirror of the daemon's WAV tee, ours at 24k)
def _open_recorders(call_id: str):
    if not RECORDINGS_DIR:
        return None, None
    d = Path(RECORDINGS_DIR) / time.strftime("%Y%m%d-%H%M%S")
    d.mkdir(parents=True, exist_ok=True)
    caller = wave.open(str(d / "caller_24k.wav"), "wb")
    caller.setnchannels(1); caller.setsampwidth(2); caller.setframerate(SAMPLE_RATE_BRIDGE)
    agent = wave.open(str(d / "agent_24k.wav"), "wb")
    agent.setnchannels(1); agent.setsampwidth(2); agent.setframerate(SAMPLE_RATE_BRIDGE)
    return caller, agent


# ----------------------------------------------------------------------------
# Outbound call triggers (file-based; consumed by the warm voice service)
# ----------------------------------------------------------------------------
def _consume_outbound_trigger() -> bool:
    """Atomically consume ~/.facetime-bridge/outbound.trigger if present.

    The trigger is a zero-byte or JSON file dropped by an authorized local
    client (the Hermes gateway, a CLI, launchd). Rename-first makes the
    consume atomic against a second writer.
    """
    p = Path(TRIGGER_PATH)
    if not p.exists():
        return False
    try:
        consumed = p.with_suffix(".trigger.consumed")
        p.rename(consumed)
        return True
    except FileNotFoundError:
        return False  # raced with another consumer; fine
    except OSError as e:
        log.warning("trigger consume failed: %s", e)
        return False


def _mark_trigger_result(result: dict) -> None:
    """Write the call outcome next to the consumed trigger for the caller."""
    try:
        out = Path(TRIGGER_PATH).with_suffix(".trigger.result")
        out.write_text(json.dumps(result) + "\n")
    except OSError:
        pass


def _await_outbound_connect(stub, timeout_s: float = 180.0) -> "object | None":
    """Poll PROBE until the outbound call connects (Captain picks up) or ends.

    2026-09-06 15:05 flight (post-reboot cold Phone): trigger → cold launch →
    auth gate → dial → callee ring → answer took ~100+s before the in-call
    surface became probe-visible as 'connected'. 'idle'/'unknown' is the
    EXPECTED state the whole time the callee's phone is ringing — it must
    never be treated as terminal. Only 'ended'/'failed' end the wait early;
    otherwise hold the line for the full window (an unanswered ring-out
    eventually leaves idle, and the wait expires cleanly).
    """
    deadline = time.time() + timeout_s
    started = time.time()
    last_heartbeat = started
    while time.time() < deadline:
        try:
            pr = stub.Control(pb.ControlRequest(command=pb.CONTROL_COMMAND_PROBE))
        except grpc.RpcError:
            time.sleep(1.0)
            continue
        if pr.state == "connected":
            log.info("outbound call connected")
            return pr
        if pr.state in ("ended", "failed"):
            log.info("outbound call ended before connect: %s", pr.state)
            return None
        if pr.state in ("dialing", "ringing", "prompt"):
            log.info("outbound surface activity: %s", pr.state)
        if time.time() - last_heartbeat >= 30.0:
            log.info("outbound still waiting (state=%s, %.0fs elapsed)",
                     pr.state, time.time() - started)
            last_heartbeat = time.time()
        time.sleep(1.0)
    log.warning("outbound connect wait timed out after %.0fs", timeout_s)
    return None


def _ax2_env() -> dict:
    """Env for ax2 calls: the authorized-caller contract must ride along."""
    return {**os.environ,
            "FACETIME_BRIDGE_AUTHORIZED_CALLER_E164": AUTHORIZED_E164 or ""}


def _nc_process_name(name: str) -> bool:
    """NC surface-name matcher: 'NotificationCenter' (executable name, what
    ax2 actually reports — verified live 09-08) and 'Notification Center'
    (display spelling) both name the same tray."""
    return (name or "").replace(" ", "").lower() == "notificationcenter"


def _ax2_path() -> str:
    """Path to the bridge's AX helper binary for CLI probes/presses.

    ax3 carries the --ax-press / --frames additions. ax2's PATH is
    provenance-bound (09-08: a different binary at that exact path passes
    --self-check but gets taskgate-killed on any AX-touching call), so the
    NEW code lives at ax3 — while the launchd DAEMON keeps running from ax2
    (its plist is untouched and it never uses the new subcommands).
    """
    for name in ("facetime-bridge-ax3", "facetime-bridge-ax2"):
        p = os.path.expanduser(f"~/.local/bin/{name}")
        if os.path.exists(p):
            return p
    return os.path.expanduser("~/.local/bin/facetime-bridge-ax")


def _ax2_snapshot_frames() -> list:
    """`ax2 --ax-snapshot --frames`: full (unfiltered) AX dump with frames.

    Unlike the default snapshot (keyword-filtered, no frames — the shape
    `_call_timer_running` depends on), this returns every surface with
    per-node frame {x,y,w,h} in points. Raises on any failure; callers
    decide how to degrade.
    """
    import subprocess as _sp
    r = _sp.run(
        [_ax2_path(), "--ax-snapshot", "--frames"],
        capture_output=True, text=True, timeout=20, env=_ax2_env(),
    )
    return json.loads(r.stdout or "[]")


def _newest_call_node(surfaces) -> "dict | None":
    """Newest enabled pressable 'Call' node carrying a frame, else None.

    A failed press leaves a STALE banner behind and the fresh prompt arrives
    after it, so the newest match is the only safe target (2026-09-06
    lesson). The array is walked in dump order and the LAST match wins —
    mirroring both the old AppleScript heuristic (item n of hitList) and
    ax2's own --ax-press newest preference.
    """
    matches = []
    for s in (surfaces or []):
        if not s.get("enabled", True):
            continue
        if not _nc_process_name(s.get("process") or ""):
            continue
        hay = " ".join([str(t) for t in (s.get("texts") or [])]
                       + [str(s.get("label") or "")]).lower()
        if "call" not in hay:
            continue
        if "AXPress" not in (s.get("actions") or []) and s.get("role") != "AXButton":
            continue
        if not s.get("frame"):
            continue
        matches.append(s)
    if not matches:
        return None
    if len(matches) > 1:
        log.info("%d 'Call' nodes visible — targeting the NEWEST (last) one",
                 len(matches))
    return matches[-1]


def _in_call_banner_visible() -> bool:
    """True when a pressable in-call banner control (mute/hang up/end) exists.

    Independent connect proof for the outbound path (issue #6 fix 2): the
    timer text can be unreadable while a live call IS running. Uses the
    DEFAULT keyword-filtered snapshot (same channel and shape
    `_call_timer_running` depends on).
    """
    import subprocess as _sp
    try:
        r = _sp.run(
            [_ax2_path(), "--ax-snapshot"],
            capture_output=True, text=True, timeout=8, env=_ax2_env(),
        )
        surfaces = json.loads(r.stdout or "[]")
    except Exception as e:
        log.warning("in-call banner probe failed (%s): %s", type(e).__name__, e)
        return False
    for s in surfaces:
        if not s.get("enabled", True):
            continue
        if "AXPress" not in (s.get("actions") or []):
            continue
        hay = " ".join([str(t) for t in (s.get("texts") or [])]
                       + [str(s.get("label") or ""),
                          str(s.get("identifier") or "")]).lower()
        # Only the in-call banner offers Mute / Hang Up / End; a dial or
        # prompt surface never does.
        if any(w in hay for w in ("mute", "hang up", "end call")) or "end" in hay.split():
            return True
    return False


def _click_to_call_prompt_visible() -> bool:
    """True while a 'Click to Call' banner is still unanswered in NC.

    Guards the stale-surface grace in _place_call_direct: a visible prompt
    means the call was NEVER confirmed. Probe failure fails SAFE (True) —
    an unreadable screen must never enable the heuristic connect.
    """
    import subprocess as _sp
    try:
        r = _sp.run(
            [_ax2_path(), "--ax-snapshot"],
            capture_output=True, text=True, timeout=8, env=_ax2_env(),
        )
        surfaces = json.loads(r.stdout or "[]")
    except Exception as e:
        log.warning("Click-to-Call prompt probe failed (%s): %s",
                    type(e).__name__, e)
        return True
    for s in surfaces:
        hay = " ".join([str(t) for t in (s.get("texts") or [])]
                       + [str(s.get("label") or "")]).lower()
        if "click to call" in hay:
            return True
    return False


def _place_call_direct(stub) -> "object | None":
    """Place the outbound call via the FaceTime Audio URL and wait for connect.

    2026-09-06 learnings, all verified live:
    - The daemon's Control(CALL) times out (PROMPT_TIMEOUT): no pressable AX
      action matches its authorized-call vocabulary within 25s.
    - macOS sometimes shows a 'Click to Call' confirmation prompt instead of
      dialing (seen after repeated URL opens). When present, press its Call
      button ourselves (single deliberate press, same AX channel as daemon).
    - The daemon probe reads OUTBOUND calls as idle the entire time (the
      outbound surface carries no caller identity, fail-closed classifier).
      So we can't use PROBE for the connect wait. Instead watch the Phone
      process: it spawns when the call UI engages (dialing or connected) and
      dies when the call ends. 'connected' = Phone alive + running call timer
      in the AX snapshot ('FaceTime Audio M:SS' with a nonzero/updating time).
    """
    import subprocess as _sp
    if not AUTHORIZED_E164:
        log.error("FACETIME_BRIDGE_AUTHORIZED_CALLER_E164 is unset — refusing to "
                  "dial. Set it in the launchd plist for ai.data.facetime-voice.")
        return None
    url = f"facetime-audio://{AUTHORIZED_E164}"
    t_dial = time.time()
    try:
        _sp.run(["open", url], check=True, timeout=15)
    except Exception as e:
        log.error("URL open failed: %s", e)
        return None

    # 2026-09-08 (issue #6): the 12s blind connect is GONE. On tonight's
    # live call the Phone surface was up but the prompt was never pressed
    # (System Events cannot see NC banner buttons on macOS 26), and the old
    # fallback returned 'connected' 12s in anyway — the false connect that
    # left the callee ringing while we played a greeting to nobody.
    # Connect now requires PROOF, in this order:
    #   1. running call timer in the AX snapshot (unchanged)
    #   2. pressable in-call banner controls (mute/hang up/end) — the timer
    #      text is not always readable, but a live call always shows them
    #   3. after 45s of Phone-up with the prompt-press budget spent and NO
    #      Click-to-Call prompt visible: the honest stale-surface heuristic
    #      (logged as unproven). A visible prompt NEVER fakes a connect.
    CLICK_TO_CALL_WINDOW_S = 20.0   # prompt appears within ~10s if at all
    CLICK_TO_CALL_TRIES = 3
    TIMER_PROBE_EVERY_S = 1.5
    STALE_SURFACE_GRACE_S = 45.0    # replaces the 12s blind connect
    deadline = time.time() + 90.0

    presses = 0
    next_press_at = t_dial + 3.0
    phone_seen_at = None
    next_timer_probe = 0.0

    while time.time() < deadline:
        now = time.time()

        if (phone_seen_at is None and presses < CLICK_TO_CALL_TRIES
                and now >= next_press_at
                and now - t_dial < CLICK_TO_CALL_WINDOW_S):
            presses += 1
            if _press_click_to_call_if_present():
                presses = CLICK_TO_CALL_TRIES  # pressed; stop traversing AX
            next_press_at = time.time() + 5.0

        phone_up = _sp.run(["pgrep", "-x", "Phone"],
                           capture_output=True).returncode == 0
        if phone_up:
            if phone_seen_at is None:
                phone_seen_at = now
                log.info("Phone call surface appeared (%.1fs after dial)",
                         now - t_dial)
            if now >= next_timer_probe:
                next_timer_probe = now + TIMER_PROBE_EVERY_S
                if _call_timer_running():
                    log.info("outbound call connected (timer running, %.1fs after dial)",
                             time.time() - t_dial)
                    return _ProbeLike("connected")
                # Timer unreadable ≠ not connected. A live call always
                # carries banner controls; accept them as the second proof.
                if _in_call_banner_visible():
                    log.info("outbound call connected (in-call banner present, "
                             "%.1fs after dial)", time.time() - t_dial)
                    return _ProbeLike("connected")
                # Grace: Phone up well past the prompt window with the press
                # budget spent and no unanswered prompt on screen. The prompt
                # check fails SAFE — an unreadable screen counts as visible.
                if (now - phone_seen_at > STALE_SURFACE_GRACE_S
                        and presses >= CLICK_TO_CALL_TRIES
                        and not _click_to_call_prompt_visible()):
                    log.warning("connect unproven, proceeding on "
                                "stale-surface heuristic (Phone up %.0fs, "
                                "no timer, no banner, no prompt)",
                                now - phone_seen_at)
                    return _ProbeLike("connected")
                if _click_to_call_prompt_visible():
                    log.info("prompt still unanswered — connect unproven")
        elif phone_seen_at is not None:
            # Surface appeared then vanished — call ended/failed.
            log.info("Phone surface vanished before connect")
            return None

        time.sleep(0.25)
    log.warning("outbound connect proof never arrived — cleaning up")
    return None


class _ProbeLike:
    """Minimal stand-in for a ControlResponse (state/authorized/ok)."""
    def __init__(self, state: str):
        self.state = state
        self.authorized = True
        self.ok = True


def _press_click_to_call_if_present() -> bool:
    """Press the NEWEST 'Call' button on a 'Click to Call' prompt.

    2026-09-08 (issue #6): System Events cannot see Notification Center
    banner buttons on macOS 26 ('entire contents' blind — re-confirmed on
    two live calls), so the AppleScript press NEVER fired. The bridge's
    ax2 helper CAN see them: try its --ax-press first (it presses the
    NEWEST enabled pressable node matching 'Call' in the named process),
    then fall back to one frames-snapshot + cliclick at the computed
    button center.

    A failed attempt leaves a STALE banner in the NC tray and the fresh
    prompt arrives after it — the newest match is the only safe target.
    ax2 already prefers the newest node; the frames fallback mirrors that
    (last match wins, matching dump order).
    """
    import subprocess as _sp
    ax2 = _ax2_path()
    try:
        r = _sp.run(
            [ax2, "--ax-press", "--process", "Notification Center",
             "--contains", "Call"],
            capture_output=True, text=True, timeout=20, env=_ax2_env(),
        )
        out = json.loads((r.stdout or "{}"))
        if out.get("pressed"):
            log.info("pressed Click-to-Call prompt via ax2 (matched: %s)",
                     out.get("matched", ""))
            return True
        log.info("ax2 --ax-press found no 'Call' node (reason: %s)",
                 out.get("reason", r.stderr or "?"))
    except Exception as e:
        log.warning("ax2 --ax-press failed (%s): %s", type(e).__name__, e)

    # ONE fallback: frames snapshot → newest 'Call' node → cliclick center.
    try:
        surfaces = _ax2_snapshot_frames()
    except Exception as e:
        log.warning("click-to-call frames snapshot failed (%s): %s",
                    type(e).__name__, e)
        return False
    node = _newest_call_node(surfaces)
    if node is None:
        log.info("no pressable 'Call' node with a frame — nothing to press")
        return False
    f = node["frame"]
    x = int(f["x"] + f["w"] / 2)
    y = int(f["y"] + f["h"] / 2)
    log.info("ax-press fallback: cliclick at (%d,%d) on %r",
             x, y, " | ".join([str(t) for t in (node.get("texts") or [])])[:80])
    try:
        click = _sp.run(["cliclick", f"c:{x},{y}"],
                        capture_output=True, text=True, timeout=10)
        if click.returncode == 0:
            return True
        log.warning("cliclick failed (rc=%d): %s",
                    click.returncode, (click.stderr or click.stdout or "").strip())
    except Exception as e:
        log.warning("cliclick click failed: %s", e)
    return False


def _call_timer_running() -> bool:
    """True when the AX snapshot shows a running call timer (live call).

    2026-09-07: was silently always-False. AUTHORIZED_E164 was None (see the
    constant above), and subprocess rejects a None env value with TypeError,
    which the bare `except` swallowed. Now the env value is guaranteed a str,
    and a genuine probe failure is logged instead of being indistinguishable
    from "no timer" — the difference between them cost 60s on every dial.
    """
    import json as _json
    import subprocess as _sp
    ax = os.path.expanduser("~/.local/bin/facetime-bridge-ax2")
    if not os.path.exists(ax):
        ax = os.path.expanduser("~/.local/bin/facetime-bridge-ax")
    try:
        r = _sp.run(
            [ax, "--ax-snapshot"],
            capture_output=True, text=True, timeout=8,
            env={**os.environ,
                 "FACETIME_BRIDGE_AUTHORIZED_CALLER_E164": AUTHORIZED_E164 or ""},
        )
        surfaces = _json.loads(r.stdout or "[]")
        for s in surfaces:
            for t in s.get("texts", []):
                # 'FaceTime Audio 00:17' — a timer; 'FaceTime Audio - , 0:17'
                # also observed. Any digits pattern means live timer.
                if "FaceTime Audio" in t and any(c.isdigit() for c in t):
                    return True
        return False
    except Exception as e:
        log.warning("call-timer probe failed (%s): %s", type(e).__name__, e)
        return False


# In-call banner mic-control vocabulary. 'muted'/'unmute' imply the mic is
# CURRENTLY muted (the control offers to unmute); a bare 'mute' label alone
# does not — a MUTE toggle says nothing about which side it is on.
_MUTED_HINTS = ("muted", "unmute")
_UNMUTED_HINTS = ("unmuted",)
_MUTE_ANY = ("mute",)


def _banner_mute_state(surfaces) -> str:
    """Classify the in-call banner mic control: 'muted'|'unmuted'|'unknown'.

    Logs every matching node's raw texts verbatim at INFO — the wording of
    the control is not assumed, only recorded, and re-classified here if the
    live wording ever drifts.
    """
    found = False
    for s in (surfaces or []):
        if not _nc_process_name(s.get("process") or ""):
            continue
        texts = [str(t) for t in (s.get("texts") or [])]
        label = str(s.get("label") or "")
        ident = str(s.get("identifier") or "")
        hay = " ".join(texts + [label, ident])
        if not any(w in hay.lower() for w in _MUTE_ANY):
            continue
        found = True
        log.info("mute-control node found: texts=%r label=%r identifier=%r "
                 "actions=%r enabled=%r", texts, label, ident,
                 s.get("actions"), s.get("enabled"))
        low = hay.lower()
        # ORDER MATTERS: 'unmuted' contains 'muted' as a substring, so the
        # unmuted check must run first or every unmuted control reads muted.
        if any(h in low for h in _UNMUTED_HINTS):
            return "unmuted"
        if any(h in low for h in _MUTED_HINTS):
            return "muted"
    if not found:
        log.info("no mute-control node in the in-call banner snapshot")
    return "unknown"


def _ensure_outbound_mic_unmuted(session=None) -> str:
    """Prove (or force) the outbound mic unmuted right after connect.

    2026-09-08 live call: the in-call banner mic icon was SLASHED from the
    first second — TTS played into a muted mic for 85s and the callee heard
    nothing. This runs exactly once per outbound call, immediately after
    connect is confirmed, and NEVER raises: a failure here must degrade to a
    log line, not kill the call path.

    Order: ax2 snapshot to classify the banner mic control → if it reads
    muted, ax2 press + re-snapshot to verify → System Events Video>Mute
    menu toggle as the fallback (SE CAN read app menus, unlike NC banners;
    AXMenuItemMarkChar reads 'missing value' before AND after on this build
    — logged as inconclusive).

    Returns the final state: 'unmuted' | 'unknown'.
    """
    method = "none"
    state = "unknown"
    try:
        # (a) Snapshot and classify. Log everything found verbatim — do not
        # assume semantics from partial wording.
        surfaces = _ax2_snapshot_frames()
        state = _banner_mute_state(surfaces)
        log.info("outbound mic state per banner: %s", state)

        if state == "muted":
            import subprocess as _sp
            log.info("banner shows mic MUTED — pressing the mute control")
            try:
                r = _sp.run(
                    [_ax2_path(), "--ax-press", "--process",
                     "Notification Center", "--contains", "Mute"],
                    capture_output=True, text=True, timeout=20, env=_ax2_env(),
                )
                out = json.loads((r.stdout or "{}"))
                log.info("ax2 mute press: %r", out)
                method = "ax2-press"
                # (b) RE-SNAPSHOT to verify the muted indication is gone.
                after = _banner_mute_state(_ax2_snapshot_frames())
                log.info("post-press banner mute state: %s", after)
                if after == "unmuted":
                    state = "unmuted"
                elif after == "muted":
                    log.warning("banner still reads muted after press")
                else:
                    # Pressed but the banner no longer classifies — treat
                    # the press as taken effect rather than press again.
                    state = "unmuted"
            except Exception as e:
                log.warning("ax2 mute press failed: %s", e)

        if state != "unmuted":
            # (c) Fallback: FaceTime's own Video > Mute menu via System
            # Events. SE cannot see NC banners but CAN read app menus.
            method = "menu-toggle"
            try:
                import subprocess as _sp
                before = (
                    'tell application "System Events" to tell process '
                    '"FaceTime" to get value of attribute "AXMenuItemMarkChar" '
                    'of menu item "Mute" of menu 1 of menu bar item "Video" '
                    'of menu bar 1'
                )
                click = (
                    'tell application "System Events" to tell process '
                    '"FaceTime" to click menu item "Mute" of menu 1 of menu '
                    'bar item "Video" of menu bar 1'
                )
                r0 = _sp.run(["osascript", "-e", before],
                             capture_output=True, text=True, timeout=10)
                log.info("menu AXMenuItemMarkChar BEFORE: %r", (r0.stdout or "").strip())
                _sp.run(["osascript", "-e", click],
                        capture_output=True, text=True, timeout=10)
                r1 = _sp.run(["osascript", "-e", before],
                             capture_output=True, text=True, timeout=10)
                log.info("menu AXMenuItemMarkChar AFTER: %r", (r1.stdout or "").strip())
            except Exception as e:
                log.warning("menu-toggle unmute failed: %s", e)
    except Exception as e:
        # NEVER raise: a probe failure must not kill the call path.
        log.warning("mic-unmute check could not run: %s", e)
    finally:
        # (d) Exactly one summary line, always. 'no' = we KNOW the mic is
        # still muted; 'unknown' = state never classified.
        verdict = ("yes" if state == "unmuted"
                   else "no" if state == "muted" else "unknown")
        fn = log.info if state == "unmuted" else log.warning
        fn("outbound mic unmuted: %s (method=%s)", verdict, method)
    return state


def _run_call(session, was_outbound: bool, stub) -> None:
    """Greeting (outbound), audio loop, and end-watcher for one call."""
    # Outbound calls connect with the callee hearing dead air (the loop
    # is otherwise mute until they speak first). Speak immediately so
    # Captain knows the line is live — three 2026-09-06 flights ended
    # with him answering into silence and hanging up.
    if was_outbound:
        try:
            session.playing.set()
            tts_sentences(
                split_sentences(GREETING_LINE),
                emit=session._emit_speech,
                cancelled=lambda: session.barge_in.is_set(),
            )
        except TTSCanceled:
            log.info("greeting canceled by barge-in")
        finally:
            session.playing.clear()
            session.barge_in.clear()

    # run until the call ends
    t = threading.Thread(target=session.audio_loop, daemon=True)
    t.start()

    # The daemon emits no 'ended' event on its WaitIncoming stream; the
    # in-call Phone surface vanishing IS the end signal. Poll Control
    # PROBE (state snapshot) until it reports ended/idle/unknown.
    ended = threading.Event()

    def watch_events():
        try:
            idle_seen = 0
            while not ended.is_set():
                pr = stub.Control(pb.ControlRequest(command=pb.CONTROL_COMMAND_PROBE))
                if pr.state in ("ended", "failed"):
                    log.info("call %s", pr.state)
                    break
                if pr.state in ("idle", "unknown"):
                    idle_seen += 1
                    # Three consecutive idle/unknown probes (~6s) after a
                    # live call means the call surface is gone for good.
                    if idle_seen >= 3:
                        log.info("call surface gone (idle x%d)", idle_seen)
                        break
                else:
                    idle_seen = 0
                time.sleep(2.0)
        except grpc.RpcError as e:
            log.warning("probe watcher rpc error: %s", e.code())
        finally:
            ended.set()
            session.writer.put(("STOP", None))

    threading.Thread(target=watch_events, daemon=True).start()

    ended.wait()


def persist_call_memory(dialogue: list[dict], call_id: str) -> str | None:
    """Auto-persist a finished call: transcript file + DATA memory entry.

    2026-09-07 requirement from Captain: call information must survive the
    hangup — if the iMessage DATA can't recall what happened on a call, the
    call wasn't useful. Writes the full transcript to
    ~/.hermes/call_transcripts/<ts>.md and asks the worker to add a compact
    memory entry pointing at it. Runs even if the worker write fails (the
    transcript file is the durable artifact).
    """
    if not dialogue:
        return None
    try:
        from datetime import datetime, timezone
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%SZ")
        tdir = os.path.expanduser("~/.hermes/call_transcripts")
        os.makedirs(tdir, exist_ok=True)
        path = os.path.join(tdir, f"{ts}-{call_id}.md")
        lines = [f"# Voice call {call_id} — {ts}", ""]
        for m in dialogue:
            who = "Captain" if m["role"] == "user" else "DATA"
            lines.append(f"**{who}:** {m['content']}")
            lines.append("")
        with open(path, "w") as f:
            f.write("\n".join(lines))
        # Best-effort memory entry via the worker (it has the memory tool).
        try:
            with _worker_lock:
                _ensure_worker()
                summary_prompt = (
                    f"System: A voice call just ended. Full transcript saved at {path}. "
                    f"Add ONE memory entry (store: memory) summarizing anything "
                    f"durable from this call: appointments made, facts stated, "
                    f"tasks requested, decisions. Include the transcript path. "
                    f"Keep it under 300 chars. Do not reply with anything else.\n\n"
                    + "\n".join(
                        (m["role"] + ": " + m["content"]) for m in dialogue[-20:]
                    )
                )
                _worker_proc.stdin.write(json.dumps({"prompt": summary_prompt}) + "\n")
                _worker_proc.stdin.flush()
                line = _worker_proc.stdout.readline()
                if line:
                    json.loads(line)  # drain response
        except Exception as e:
            log.warning("memory persist via worker failed (transcript saved): %s", e)
        log.info("call transcript persisted: %s", path)
        return path
    except Exception as e:
        log.warning("call transcript persistence failed: %s", e)
        return None


def llm_reply_streaming(user_text: str, speak, cancelled) -> str:
    """Streaming turn: speak sentences as the model generates them.

    speak(sentence, is_final) is called per sentence; time-to-first-audio
    drops from full-turn to first-sentence (~1-2.5s projected). Falls back to
    llm_reply + batch TTS if the worker returns no deltas (tool turns, older
    worker). Returns the full reply text for the dialogue transcript.
    """
    with _dialogue_lock:
        _dialogue.append({"role": "user", "content": user_text})
    t0 = time.perf_counter()
    collected: list[str] = []
    buf = ""
    # Two distinct facts, and conflating them is a bug: `spoke_reply` gates
    # "has the model's ACTUAL answer been voiced yet" (a filler must not
    # suppress the real answer on a tool turn, where content arrives only at
    # the end), while `spoke_anything` gates "did the caller hear ANY audio"
    # for the never-go-silent guard.
    spoke_reply = False
    spoke_anything = False
    tier = "fast"
    with _worker_lock:
        _ensure_worker()
        _worker_proc.stdin.write(json.dumps({
            "prompt": user_text, "stream": True,
        }) + "\n")
        _worker_proc.stdin.flush()
        while True:
            line = _worker_proc.stdout.readline()
            if not line:
                raise RuntimeError("worker died mid-stream")
            msg = json.loads(line)
            if "filler" in msg:
                # The worker routed this turn to the tool-enabled agent, which
                # streams nothing until its whole loop finishes. Speak now so
                # the caller hears acknowledgement instead of dead air.
                filler = (msg.get("filler") or "").strip()
                if filler:
                    tier = "full"
                    spoke_anything = True
                    speak(filler, is_final=False)
                continue
            if "delta" in msg:
                buf += msg["delta"]
                # Only the turn's first chunk may break early at a clause.
                sentences, buf = _pop_sentences(buf, allow_clause=not spoke_reply)
                for s in sentences:
                    collected.append(s)
                    if s.strip():
                        spoke_reply = True
                        spoke_anything = True
                        speak(s.strip(), is_final=False)
            elif "content" in msg or "error" in msg:
                if msg.get("error"):
                    log.warning("worker stream error: %s", msg["error"])
                tier = msg.get("tier", tier)
                final = (msg.get("content") or "").strip()
                # Any completed sentences never streamed as deltas (tool turns
                # emit only the final content) — speak the whole thing now.
                if not spoke_reply and final:
                    speak(final, is_final=True)
                    spoke_reply = True
                    spoke_anything = True
                if final and not collected:
                    collected.append(final)
                tail = buf.strip()
                if tail:
                    collected.append(tail)
                    speak(tail, is_final=True)
                    spoke_anything = True
                # NEVER GO SILENT. 2026-09-07: on a worker error `final` is ""
                # and every branch above was skipped, so nothing was spoken —
                # but the transcript still recorded the "came back empty" line
                # below, which made the logs look like a completed turn while
                # the Captain heard pure dead air. That is the "sometimes he
                # just never says anything" failure. An audible failure is
                # always better than silence on a live call.
                if not spoke_anything:
                    log.error("turn produced no speakable text (error=%s) — "
                              "speaking audible fallback", msg.get("error"))
                    try:
                        speak(TURN_FAILED_LINE, is_final=True)
                    except TTSCanceled:
                        raise
                    except Exception:
                        log.exception("even the fallback line failed to speak")
                break
    elapsed = time.perf_counter() - t0
    text = " ".join(collected).strip() or "I'm here, Captain, but my response came back empty."
    with _dialogue_lock:
        _dialogue.append({"role": "assistant", "content": text})
    log.info("streaming turn [%s tier]: %.2fs, %d chars", tier, elapsed, len(text))
    return text


def main() -> int:
    _ensure_log_dir()
    os.makedirs(os.path.dirname(SOCKET_PATH), exist_ok=True)
    log.info("DATA FaceTime Voice starting; socket=%s", SOCKET_PATH)

    # Health-check FIRST — if the daemon isn't ready (e.g. BlackHole not yet
    # installed), exit fast and let launchd retry. Don't burn CPU loading
    # whisper for a daemon that can't take a call.
    try:
        channel = grpc.insecure_channel(f"unix:{SOCKET_PATH}")
        grpc.channel_ready_future(channel).result(timeout=10)
    except grpc.FutureTimeoutError:
        log.error("daemon socket not answering — is facetime-bridge-ax running?")
        return 2
    stub = pb_grpc.FaceTimeMediaStub(channel)
    health = stub.Health(pb.HealthRequest())
    if not health.ready:
        log.error("daemon not ready: in=%s out=%s (BlackHole installed?)",
                  health.input_device, health.output_device)
        return 2
    log.info("daemon ready: in=%s out=%s", health.input_device, health.output_device)

    _init_models()
    threading.Thread(target=_warm_worker, daemon=True).start()

    # 2026-09-07 latency lesson: attach_live.py calls _init_models() itself,
    # but the worker warm-up used to live only here — an attach before the
    # main loop's warm thread finished meant Captain's first words hit a cold
    # worker (28s turn). Warm in BOTH paths; the worker caches in-process.
    def _warm_when_loaded():
        while stt is None:
            time.sleep(0.2)
        _warm_worker()

    threading.Thread(target=_warm_when_loaded, daemon=True).start()

    # Event pump: WaitIncoming blocks with no timeout, which would starve the
    # outbound trigger poll. Run the stream in its own thread pushing events
    # into a queue; the main loop services both the queue and the trigger.
    events: "queue.Queue" = queue.Queue()

    def pump_wait_incoming():
        while True:
            # Fresh channel per attempt: a daemon restart replaces the UDS
            # socket and the old channel hangs in CONNECTING forever, silently
            # stranding inbound auto-answer (2026-09-06 16:26 bug — call fell
            # through to Live Voicemail).
            pump_channel = grpc.insecure_channel(f"unix:{SOCKET_PATH}")
            pump_stub = pb_grpc.FaceTimeMediaStub(pump_channel)
            try:
                for ev in pump_stub.WaitIncoming(pb.WaitIncomingRequest()):
                    log.info("call event: %s authorized=%s", ev.state, ev.authorized)
                    events.put(ev)
                    if ev.state in ("connected", "ended", "failed"):
                        break  # stream is done (daemon returns after answering)
            except grpc.RpcError as e:
                log.warning("wait-incoming stream error: %s — reconnecting", e.code())
                time.sleep(2.0)
            finally:
                pump_channel.close()

    threading.Thread(target=pump_wait_incoming, daemon=True).start()

    # Wait for an authorized call, answer it, run the loop.
    in_call = False
    while True:
        # 2026-09-07 latency fix: adopt ALREADY-CONNECTED outbound calls.
        # attach_live.py used to cold-start whisper (~10-17s) + agent (~40s)
        # per call, so Captain waited ~15s after answering. This service is
        # permanently warm — when a call goes live that we didn't place
        # (manual banner dial from the iMessage side), adopt it here.
        if not in_call:
            try:
                pr = stub.Control(pb.ControlRequest(command=pb.CONTROL_COMMAND_PROBE))
                # NOTE: PROBE never sets authorized (that rides WaitIncoming);
                # an outbound call on this socket can only be to the
                # authorized E164 (daemon fails closed), so connected is proof.
                if pr.state == "connected":
                    log.info("adopting live call (warm attach — no cold start)")
                    in_call = True
                    call_id = f"dfv-adopt-{int(time.time())}"
                    caller_rec, agent_rec = _open_recorders(call_id)
                    session = CallSession(stub, call_id)
                    session.rec_writer_caller = caller_rec
                    session.rec_writer_agent = agent_rec
                    saved_output = None
                    try:
                        import audio_default
                        saved_output = audio_default.get_default_output()
                        if saved_output != "BlackHole 16ch":
                            audio_default.set_default_output("BlackHole 16ch")
                            log.info("default output %s -> BlackHole 16ch (adopt)", saved_output)
                    except Exception as e:
                        log.warning("audio_default routing failed: %s", e)
                    try:
                        _run_call(session, True, stub)
                    finally:
                        try:
                            if _dialogue:
                                persist_call_memory(list(_dialogue), call_id)
                                with _dialogue_lock:
                                    _dialogue.clear()
                        except Exception as e:
                            log.warning("post-call persist error: %s", e)
                        if saved_output and saved_output != "BlackHole 16ch":
                            try:
                                import audio_default
                                audio_default.set_default_output(saved_output)
                            except Exception as e:
                                log.warning("default output restore failed: %s", e)
                        for w in (caller_rec, agent_rec):
                            if w:
                                try:
                                    w.close()
                                except Exception:
                                    pass
                        log.info("adopted call complete; re-arming")
                        time.sleep(1.0)
                    continue
            except grpc.RpcError:
                pass  # daemon busy or socket hiccup; normal poll continues
        answer_resp = None
        was_outbound = False
        # Outbound: a trigger file asks the warm service to place the call.
        # The daemon dials ONLY the authorized E.164 (fail-closed by design).
        if _consume_outbound_trigger():
            log.info("outbound trigger accepted — placing call")
            answer_resp = _place_call_direct(stub)
            if answer_resp is None:
                log.error("outbound call failed to connect")
                _mark_trigger_result({"ok": False, "error": "call failed to place or connect"})
                continue
            log.info("outbound call connected")
            # Issue #6 fix 3: exactly once per outbound call, right after
            # connect is confirmed — tonight's call ran 85s mic-muted.
            # Never raises; worst case is a log line.
            _ensure_outbound_mic_unmuted()
            _mark_trigger_result({"ok": True, "state": "connected"})
            was_outbound = True
        else:
            # Inbound path — block on the event queue, but wake often enough
            # to keep servicing the outbound trigger file.
            try:
                ev = events.get(timeout=TRIGGER_POLL_S)
            except queue.Empty:
                continue
            if ev.state == "connected" and ev.authorized:
                # Daemon auto-answered (its WaitIncoming answers and returns).
                answer_resp = ev
            elif ev.state == "ringing" and ev.authorized:
                # Wait for the daemon's own 'connected' event; fall back to a
                # single explicit ANSWER only if the auto-answer never lands.
                deadline = time.time() + 15.0
                while time.time() < deadline:
                    try:
                        ev2 = events.get(timeout=1.0)
                    except queue.Empty:
                        pr = stub.Control(pb.ControlRequest(command=pb.CONTROL_COMMAND_PROBE))
                        if pr.state == "connected" and pr.authorized:
                            answer_resp = pr
                            break
                        continue
                    if ev2.state == "connected" and ev2.authorized:
                        answer_resp = ev2
                        break
                    if ev2.state in ("ended", "failed"):
                        break
                if answer_resp is None:
                    resp = stub.Control(pb.ControlRequest(command=pb.CONTROL_COMMAND_ANSWER))
                    if not resp.ok:
                        log.error("answer failed: %s %s", resp.error_code, resp.message)
                        continue
                    answer_resp = resp
            else:
                continue

        call_id = f"dfv-{int(time.time())}"
        caller_rec, agent_rec = _open_recorders(call_id)
        session = CallSession(stub, call_id)
        session.rec_writer_caller = caller_rec
        session.rec_writer_agent = agent_rec

        # 2026-09-07 (14:50 call): FaceTime follows the SYSTEM default output
        # at call start regardless of its per-app menu — with default output
        # on the MacBook speakers, Captain's voice played out the speakers
        # and BlackHole 16ch captured pure silence (proven by caller.wav).
        # Flip default output to BlackHole 16ch for the call; restore after.
        # (Default input must stay BlackHole 16ch at all times — the daemon's
        # engine clamps its input format to whatever the default input is at
        # first touch. Restoring THAT to speakers would break the next call.)
        saved_output = None
        try:
            import audio_default
            saved_output = audio_default.get_default_output()
            if saved_output != "BlackHole 16ch":
                audio_default.set_default_output("BlackHole 16ch")
                log.info("default output %s -> BlackHole 16ch (call routing)", saved_output)
        except Exception as e:
            log.warning("audio_default routing failed: %s", e)
        try:
            _run_call(session, was_outbound, stub)
        finally:
            # Captain's standing requirement (2026-09-07): call content must
            # survive the hangup — transcript + memory entry, every call.
            try:
                if _dialogue:
                    persist_call_memory(list(_dialogue), call_id)
                    with _dialogue_lock:
                        _dialogue.clear()
            except Exception as e:
                log.warning("post-call persist error: %s", e)
            if saved_output and saved_output != "BlackHole 16ch":
                try:
                    import audio_default
                    audio_default.set_default_output(saved_output)
                    log.info("default output restored to %s", saved_output)
                except Exception as e:
                    log.warning("default output restore failed: %s", e)

        for w in (caller_rec, agent_rec):
            if w:
                try:
                    w.close()
                except Exception:
                    pass
        in_call = False  # re-arm call adoption for the next call
        log.info("call complete; re-arming for next call")
        time.sleep(1.0)

    return 0


if __name__ == "__main__":
    sys.exit(main())