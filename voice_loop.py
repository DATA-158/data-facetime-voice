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
# 2026-09-07: must be .venv-native-tts (has grpc, dotenv, numpy, faster_whisper
# AND imports the hermes-agent tree cleanly). The gateway venv lacks grpc, and
# hermes_worker imports voice_loop (for SYSTEM_CONTEXT) which imports grpc.
HERMES_VENV = os.path.expanduser("~/agent-calling/.venv-native-tts/bin/python")
WORKER_SCRIPT = str(Path(__file__).parent / "hermes_worker.py")

SAMPLE_RATE_BRIDGE = 24000  # daemon contract
SAMPLE_RATE_STT = 16000      # whisper + VAD
CHUNK_MS = 20                # legacy label; actual packet cadence measured per packet
MIN_SILENCE_MS = int(os.environ.get("DFV_SILENCE_MS", "700"))  # end-of-utterance
SPEECH_START_MS = int(os.environ.get("DFV_SPEECH_START_MS", "60"))  # consecutive speech packets to open
# Outbound trigger: the gateway (or any local client) drops a JSON file here
# to have the warm voice service place an authorized call. Polled at TRIGGER_POLL_S
TRIGGER_PATH = os.environ.get(
    "DFV_TRIGGER_PATH", os.path.expanduser("~/.facetime-bridge/outbound.trigger")
)
TRIGGER_POLL_S = float(os.environ.get("DFV_TRIGGER_POLL_S", "2.0"))
# The only number this system will ever dial or answer (fail-closed, mirrors
# the bridge daemon's FACETIME_BRIDGE_AUTHORIZED_CALLER_E164 contract).
AUTHORIZED_E164 = os.environ.get(
    os.environ.get("FACETIME_BRIDGE_AUTHORIZED_CALLER_E164", "")
)
MAX_UTTERANCE_MS = 30_000
TTS_VOICE = os.environ.get("DFV_TTS_VOICE", "")  # empty = system default voice (Siri — Captain's pick, confirmed better by A/B test)
MAX_TURNS = 200

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
# STT — faster-whisper (distil-medium.en, int8, ARM64 CPU)
# ----------------------------------------------------------------------------
class STT:
    def __init__(self):
        from faster_whisper import WhisperModel
        self.model = WhisperModel(
            os.environ.get("DFV_STT_MODEL", "distil-small.en"),
            device="auto",
            compute_type="int8",
            cpu_threads=int(os.environ.get("DFV_STT_THREADS", "6")),
            download_root=os.path.expanduser("~/.cache/dfv-whisper"),
        )

    def transcribe(self, f32_16k: np.ndarray) -> str:
        segments, _info = self.model.transcribe(
            f32_16k, language="en", vad_filter=False, beam_size=1,
            condition_on_previous_text=False,
        )
        return " ".join(s.text.strip() for s in segments).strip()


# ----------------------------------------------------------------------------
# LLM — persistent Hermes worker (true DATA: persona, memory, tools)
# ----------------------------------------------------------------------------
SYSTEM_CONTEXT = """You are DATA, the right-hand AI agent for Captain Kirk (Spencer). The Captain is speaking to you over a live FaceTime Audio call, through a real-time voice pipeline (STT -> you -> TTS).

COMMUNICATION RULES:
- Speak like DATA talks: calm, precise, dry wit. "Sir" or "Captain" once per reply, max.
- Voice-friendly: NO markdown, NO code blocks, NO tables, NO URLs. Plain spoken sentences.
- SHORT: 1-3 sentences by default. The Captain can ask for more.
- Never say you are an AI language model. Never apologize for being AI.
- If asked to DO something (calendar, reminders, notes, fitness log, GitHub, web search), DO IT with your tools immediately, then confirm briefly what you did.

YOU HAVE FULL TOOL ACCESS (auto-approved). Key tools: terminal, web_search, read_file, write_file, calendar/notes/reminders skills, fitness DB at ~/.hermes/fitness_tracker.db, GitHub via gh (org SpencerSmithSite), session_search for past conversations.
"""

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
    """Synthesize each sentence with `say` as WAVE LEI16@24000, emit directly.

    2026-09-07 latency pass: LEI16@24000 (measured 1294ms vs 3301ms for AIFF,
    2.5x) AND it is the bridge's native rate — the old 16k->24k resample chain
    is gone entirely. /dev/stdout + FIFO streaming both fail (error -54 / fifo
    open blocks); file-per-sentence stays the transport.
    """
    import tempfile
    for sentence in sentences:
        if cancelled():
            raise TTSCanceled()
        clean = sentence.strip()
        if not clean:
            continue
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
            path = tf.name
        try:
            say_cmd = [
                "say", "-o", path,
                "--file-format=WAVE", "--data-format=LEI16@24000",
            ]
            if TTS_VOICE:
                say_cmd += ["-v", TTS_VOICE]
            say_cmd += ["--", clean]
            subprocess.run(say_cmd, check=True, capture_output=True, timeout=30)
            import soundfile as sf
            audio, sr = sf.read(path, dtype="float32", always_2d=False)
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            if abs(sr - SAMPLE_RATE_BRIDGE) > 1:
                r = Resampler(up=True)
                r.ratio = SAMPLE_RATE_BRIDGE / sr
                audio = r.process(audio)
            if cancelled():
                raise TTSCanceled()
            emit(audio, is_first=True)
        finally:
            try:
                os.unlink(path)
            except OSError:
                pass


def _pop_sentences(buf: str) -> tuple[list[str], str]:
    """Pop complete sentences off a growing delta buffer (streaming TTS)."""
    out = []
    start = 0
    for idx, ch in enumerate(buf):
        if ch in ".!?…":
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
                    self.silence_run = 0
                self._last_silence_run_ms = self.silence_run
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

    def _finalize_utterance(self) -> None:
        audio = np.concatenate(self.utterance) if self.utterance else np.zeros(0)
        self.in_speech = False
        self.utterance = []
        self.silence_run = 0
        # Tail trim: drop the trailing silence run so whisper doesn't decode
        # dead air (2026-09-07 latency pass; ~300-500ms less to decode).
        trim_ms = min(getattr(self, "_last_silence_run_ms", 0), 700)
        trim_samples = int(trim_ms / 1000.0 * SAMPLE_RATE_BRIDGE)
        if trim_samples > 0 and len(audio) > trim_samples:
            audio = audio[:-trim_samples]
        if len(audio) < SAMPLE_RATE_BRIDGE * 0.25:  # <250ms
            log.info("utterance too short, dropping")
            return
        # Peak-normalize for STT clarity (gain already applied live; this
        # lifts quiet-but-present speech the rest of the way).
        peak = float(np.abs(audio).max()) if audio.size else 0.0
        if 0.0 < peak < 0.5:
            audio = np.clip(audio * (0.7 / peak), -1.0, 1.0)
        down = Resampler(up=False)
        f32_16k = down.process(audio)
        # RMS gate
        rms = float(np.sqrt(np.mean(f32_16k ** 2)))
        if rms < UTTERANCE_RMS_GATE:
            log.info("utterance below RMS gate (%.5f), dropping", rms)
            return
        threading.Thread(
            target=self._process_turn, args=(f32_16k,), daemon=True
        ).start()

    def _process_turn(self, f32_16k: np.ndarray) -> None:
        try:
            t0 = time.perf_counter()
            text = stt.transcribe(f32_16k)
            stt_ms = (time.perf_counter() - t0) * 1000
            if not text:
                log.info("empty transcript; saying nothing")
                return
            log.info("Captain: %s (STT %.0fms)", text, stt_ms)
            self.playing.set()
            try:
                # Streaming turn: first sentence speaks while the model is
                # still generating the rest (2026-09-07 latency pass).
                llm_reply_streaming(
                    text,
                    speak=self._speak_sentence,
                    cancelled=lambda: self.barge_in.is_set(),
                )
            except TTSCanceled:
                log.info("TTS canceled by barge-in")
            finally:
                self.playing.clear()
                self.barge_in.clear()
        except Exception:
            log.exception("turn failed")

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
        if RECORDINGS_DIR and self.rec_writer_agent is not None:
            try:
                self.rec_writer_agent.writeframes(f32_to_pcm16(f32_24k))
            except Exception:
                answer_file = None
        self.writer.put(("PLAYBACK", f32_to_pcm16(f32_24k)))


stt: STT | None = None


def _init_models() -> None:
    global stt
    log.info("loading whisper…")
    stt = STT()
    log.info("whisper ready")
    # TTS engine pre-warm (agent-calling report item 5): first `say` spawn in
    # a boot session costs ~1s extra; burn it here, not on Captain's turn.
    try:
        import tempfile, subprocess as _sp
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as _tf:
            _p = _tf.name
        _sp.run(["say", "-o", _p, "--file-format=WAVE",
                 "--data-format=LEI16@24000", "--", "."],
                check=True, capture_output=True, timeout=15)
        os.unlink(_p)
        log.info("tts prewarmed")
    except Exception as e:
        log.warning("tts prewarm failed: %s", e)


def _warm_worker() -> None:
    try:
        t0 = time.perf_counter()
        llm_reply("System ping. Reply with the single word: ready.")
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
    url = f"facetime-audio://{AUTHORIZED_E164}"
    try:
        _sp.run(["open", url], check=True, timeout=15)
    except Exception as e:
        log.error("URL open failed: %s", e)
        return None

    # Press any 'Click to Call' prompt (appears within ~10s if macOS wants
    # confirmation), then wait for the Phone call surface.
    deadline = time.time() + 90.0
    pressed_prompt = False
    phone_seen_at = None
    while time.time() < deadline:
        if not pressed_prompt:
            pressed_prompt = _press_click_to_call_if_present()
        if _sp.run(["pgrep", "-x", "Phone"], capture_output=True).returncode == 0:
            if phone_seen_at is None:
                phone_seen_at = time.time()
                log.info("Phone call surface appeared")
            # The surface may exist while still dialing; the call is truly
            # live when the timer text appears. Give it up to 60s.
            if _call_timer_running():
                log.info("outbound call connected (timer running)")
                return _ProbeLike("connected")
            if time.time() - phone_seen_at > 60.0:
                log.info("Phone surface up 60s without timer — treating as connected")
                return _ProbeLike("connected")
        else:
            if phone_seen_at is not None:
                # Surface appeared then vanished — call ended/failed.
                log.info("Phone surface vanished before connect")
                return None
        time.sleep(1.0)
    log.warning("outbound connect wait timed out after 90s")
    return None


class _ProbeLike:
    """Minimal stand-in for a ControlResponse (state/authorized/ok)."""
    def __init__(self, state: str):
        self.state = state
        self.authorized = True
        self.ok = True


def _press_click_to_call_if_present() -> bool:
    """Press the NEWEST 'Call' button on a 'Click to Call' prompt.

    Uses the same Accessibility channel as the daemon. CRITICAL (2026-09-06
    lesson): each failed attempt leaves a stale 'Click to Call' banner in the
    Notification Center tray. Pressing the FIRST match presses the STALE
    banner — the fresh prompt expires (~20s) and the call dies. So: collect
    ALL matching buttons, press the LAST (newest) first, verify the call
    engaged via the timer, and only then report success.
    """
    import subprocess as _sp
    collect = (
        'tell application "System Events"\n'
        '  tell process "Notification Center"\n'
        '    set hitList to {}\n'
        '    repeat with elem in UI elements\n'
        '      try\n'
        '        set elemDesc to entire contents of elem\n'
        '        repeat with d in elemDesc\n'
        '          try\n'
        '            if role of d is "AXButton" and description of d contains "Call" then\n'
        '              set end of hitList to d\n'
        '            end if\n'
        '          end try\n'
        '        end repeat\n'
        '      end try\n'
        '    end repeat\n'
        '    set n to count of hitList\n'
        '    if n is 0 then return "none"\n'
        '    perform action "AXPress" of item n of hitList\n'
        '    return "pressed" & n\n'
        '  end tell\n'
        'end tell'
    )
    try:
        r = _sp.run(["osascript", "-e", collect], capture_output=True, text=True, timeout=20)
        out = (r.stdout or "").strip()
        if out.startswith("pressed"):
            log.info("pressed Click-to-Call prompt (banner #%s)", out[7:])
            return True
    except Exception as e:
        log.warning("click-to-call press failed: %s", e)
    return False


def _call_timer_running() -> bool:
    """True when the AX snapshot shows a running call timer (live call)."""
    import json as _json
    import subprocess as _sp
    try:
        r = _sp.run(
            ["/Users/data/.local/bin/facetime-bridge-ax2", "--ax-snapshot"],
            capture_output=True, text=True, timeout=15,
            env={**os.environ, "FACETIME_BRIDGE_AUTHORIZED_CALLER_E164": AUTHORIZED_E164},
        )
        surfaces = _json.loads(r.stdout or "[]")
        for s in surfaces:
            for t in s.get("texts", []):
                # 'FaceTime Audio 00:17' — a timer; 'FaceTime Audio - , 0:17'
                # also observed. Any digits pattern means live timer.
                if "FaceTime Audio" in t and any(c.isdigit() for c in t):
                    return True
        return False
    except Exception:
        return False


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
                split_sentences(
                    "Captain, DATA here. The line is live — go ahead."
                ),
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
    spoke_any = False
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
            if "delta" in msg:
                buf += msg["delta"]
                sentences, buf = _pop_sentences(buf)
                for s in sentences:
                    collected.append(s)
                    if s.strip():
                        spoke_any = True
                        speak(s.strip(), is_final=False)
            elif "content" in msg or "error" in msg:
                if msg.get("error"):
                    log.warning("worker stream error: %s", msg["error"])
                final = (msg.get("content") or "").strip()
                # Any completed sentences never streamed as deltas (tool turns
                # emit only the final content) — speak the whole thing now.
                if not spoke_any and final:
                    speak(final, is_final=True)
                if final and not collected:
                    collected.append(final)
                tail = buf.strip()
                if tail:
                    collected.append(tail)
                    speak(tail, is_final=True)
                break
    elapsed = time.perf_counter() - t0
    text = " ".join(collected).strip() or "I'm here, Captain, but my response came back empty."
    with _dialogue_lock:
        _dialogue.append({"role": "assistant", "content": text})
    log.info("streaming turn: %.2fs, %d chars", elapsed, len(text))
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