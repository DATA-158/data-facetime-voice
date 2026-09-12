#!/usr/bin/env python3
"""DATA FaceTime voice agent — the lean loop.

Built directly on facetime-bridge's contract (docs/INTEGRATION.md): the daemon
owns FaceTime (answer / call / hangup / audio); this process owns the
conversation. Nothing here touches Accessibility, system audio defaults,
FaceTime menus, or the Phone process. If the daemon refuses, we do not work
around it.

    caller ─FaceTime─▶ BlackHole 16ch ─▶ daemon ─CAPTURE─▶ VAD ─▶ STT ─▶ Hermes
    caller ◀─FaceTime─ BlackHole 2ch  ◀─ daemon ◀─PLAYBACK─ TTS ◀─ sentences ◀─┘

Inbound:  daemon's WaitIncoming auto-answers the authorized caller; on
          "connected" we open the Audio stream and talk.
Outbound: touch ~/.facetime-bridge/outbound.trigger (see deploy/dfv-call); we
          send Control(CALL) and open the Audio stream as soon as the daemon
          reports dialing/connected. The first speech we hear is the connect.

Everything local except the LLM call: Silero VAD (onnxruntime), MLX whisper,
AVSpeechSynthesizer with the system default voice.
"""
from __future__ import annotations

import json
import logging
import os
import queue
import re
import subprocess
import sys
import threading
import time
from pathlib import Path

import grpc
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import facetime_media_pb2 as pb  # noqa: E402
import facetime_media_pb2_grpc as pbg  # noqa: E402
# 2026-09-11, measured on DATA's host (8 GB Air, loaded): base.en 370-520 ms vs
# small.en 1300-1800 ms per utterance, identical transcripts. Override with env.
os.environ.setdefault("DFV_STT_MODEL", "mlx-community/whisper-base.en-mlx")
import stt_engine  # noqa: E402

# ---------------------------------------------------------------------------
# Config (env-overridable; no secrets here — the E.164 lives in the daemon)
# ---------------------------------------------------------------------------
BRIDGE_RATE = 24000
STT_RATE = 16000
SOCKET = "unix:" + os.path.expanduser(os.environ.get("FACETIME_BRIDGE_SOCKET", "~/.facetime-bridge/bridge.sock"))
TRIGGER = Path(os.environ.get("DFV_TRIGGER_PATH", "~/.facetime-bridge/outbound.trigger")).expanduser()
LOG_DIR = HERE / "logs"
HERMES_PY = os.path.expanduser(os.environ.get("HERMES_PYTHON", "~/.hermes/hermes-agent/venv/bin/python"))

VAD_THRESHOLD = float(os.environ.get("DFV_VAD_THRESHOLD", "0.5"))
VAD_THRESHOLD_PLAYING = float(os.environ.get("DFV_VAD_THRESHOLD_PLAYING", "0.85"))
SILENCE_MS = int(os.environ.get("DFV_SILENCE_MS", "600"))
MIN_SPEECH_MS = int(os.environ.get("DFV_MIN_SPEECH_MS", "250"))
BARGE_IN_MS = int(os.environ.get("DFV_BARGE_IN_MS", "300"))
MAX_UTTERANCE_S = 30
# Caller audio off BlackHole 16ch is very quiet (live call 2026-09-11 15:46:
# speech peak 0.008, rms 0.002; noise floor 0.00009). Gain before VAD/STT,
# soft-limited so ringback (0.027 rms) can't clip.
INPUT_GAIN = float(os.environ.get("DFV_INPUT_GAIN", "16"))
# Outbound answer detection: FaceTime ringback = ~0.75 s bursts every 3 s
# (rms ~0.025 pre-gain, ~300 Hz). Bursts stop at answer. No burst for
# RINGBACK_GAP_S after at least one burst => answered.
RINGBACK_RMS = 0.012          # pre-gain
RINGBACK_GAP_S = float(os.environ.get("DFV_RINGBACK_GAP_S", "4.5"))
INBOUND_SETTLE_S = float(os.environ.get("DFV_INBOUND_SETTLE_S", "2.5"))
MAX_SENTENCES = int(os.environ.get("DFV_MAX_SENTENCES", "3"))

GREETING = os.environ.get("DFV_GREETING_LINE", "DATA here. Go ahead, Captain.")
TOOL_FILLER = os.environ.get("DFV_TOOL_FILLER", "Let me check that, Captain.")
LOST_LINE = os.environ.get("DFV_TURN_FAILED_LINE", "Captain, I lost that one. Say again?")
CAP_LINE = os.environ.get("DFV_VOICE_CAP_CLOSING_LINE", "Full details on iMessage, Captain.")
STALL_LINE = os.environ.get("DFV_STALL_LINE", "One moment, Captain.")
STALL_AFTER_S = float(os.environ.get("DFV_STALL_AFTER_S", "3.5"))
# "Thinking" sound: a soft blip every THINK_PERIOD_S while a turn has produced
# no speech yet (the LLM's first token is the one latency we can't shorten).
# Pushed in 100 ms slices just-in-time so at most ~150 ms is ever queued, and
# flushed with CLEAR the instant the first sentence is ready — it never delays
# DATA. Pauses under the filler/stall line; stops on speech or barge-in.
THINKING_SOUND = os.environ.get("DFV_THINKING_SOUND", "blip").lower()   # blip | off
THINK_AFTER_S = float(os.environ.get("DFV_THINK_AFTER_S", "1.0"))
THINK_PERIOD_S = float(os.environ.get("DFV_THINK_PERIOD_S", "0.6"))
THINK_GAIN = float(os.environ.get("DFV_THINK_GAIN", "0.07"))             # ~-23 dBFS

log = logging.getLogger("dfv")


# ---------------------------------------------------------------------------
# Resampling — small windowed-sinc FIR, then linear pick. Voice band only.
# ---------------------------------------------------------------------------
def _lowpass_taps(cutoff_hz: float, rate: float, taps: int = 31) -> np.ndarray:
    n = np.arange(taps) - (taps - 1) / 2
    fc = cutoff_hz / rate
    h = 2 * fc * np.sinc(2 * fc * n) * np.hamming(taps)
    return (h / h.sum()).astype(np.float32)


_TAPS_24_TO_16 = _lowpass_taps(7000, 24000)


def resample(x: np.ndarray, src: int, dst: int) -> np.ndarray:
    if src == dst or len(x) == 0:
        return x.astype(np.float32)
    if src > dst:
        x = np.convolve(x, _TAPS_24_TO_16 if src == 24000 else _lowpass_taps(dst * 0.45, src), mode="same")
    n_out = int(round(len(x) * dst / src))
    idx = np.linspace(0, len(x) - 1, n_out)
    return np.interp(idx, np.arange(len(x)), x).astype(np.float32)


# ---------------------------------------------------------------------------
# Silero VAD on onnxruntime (no torch). 512-sample frames at 16 kHz = 32 ms.
# ---------------------------------------------------------------------------
class SileroVAD:
    FRAME = 512
    CONTEXT = 64

    def __init__(self):
        import onnxruntime as ort
        path = os.environ.get("DFV_SILERO_ONNX") or str(
            Path.home() / ".cache/torch/hub/snakers4_silero-vad_master/src/silero_vad/data/silero_vad.onnx")
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        self.sess = ort.InferenceSession(path, sess_options=opts, providers=["CPUExecutionProvider"])
        self.reset()

    def reset(self):
        self.state = np.zeros((2, 1, 128), dtype=np.float32)
        self.context = np.zeros((1, self.CONTEXT), dtype=np.float32)

    def prob(self, frame16k: np.ndarray) -> float:
        x = np.concatenate([self.context, frame16k.reshape(1, -1)], axis=1).astype(np.float32)
        out, self.state = self.sess.run(None, {"input": x, "state": self.state, "sr": np.array(16000, dtype=np.int64)})
        self.context = x[:, -self.CONTEXT:]
        return float(out[0][0])


# ---------------------------------------------------------------------------
# TTS — AVSpeechSynthesizer in tts_helper.py, system default voice, cached.
# ---------------------------------------------------------------------------
_EMOJI = re.compile("[\U0001F000-\U0001FAFF\U00002600-\U000027BF\U0001F900-\U0001F9FF⭐✅❌️]")
_MARKDOWN = re.compile(r"[*_`#>]+|\[([^\]]+)\]\([^)]*\)")


def clean_for_speech(text: str) -> str:
    text = _MARKDOWN.sub(lambda m: m.group(1) or "", text)
    text = _EMOJI.sub("", text)
    return re.sub(r"\s+", " ", text).strip()


class TTS:
    def __init__(self):
        self._lock = threading.Lock()
        self._cache: dict[str, np.ndarray] = {}
        # tts_helper_ns.py = NSSpeechSynthesizer = the SYSTEM voice (Siri).
        # (AVSpeechSynthesizer cannot reach Siri voices at all; it silently
        # renders Samantha — the "robotic woman" of 2026-09-11. Removed.)
        helper = os.environ.get("DFV_TTS_HELPER", "tts_helper_ns.py")
        self._proc = subprocess.Popen(
            [sys.executable, str(HERE / helper)],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=0)
        status = json.loads(self._proc.stdout.readline().decode())
        if status.get("status") != "ready":
            raise RuntimeError(f"tts helper: {status}")
        log.info("TTS helper %s ready: %s", helper, status)

    def render(self, text: str) -> np.ndarray:
        """int16 mono at BRIDGE_RATE."""
        if text in self._cache:
            return self._cache[text]
        with self._lock:
            if self._proc.poll() is not None:
                raise RuntimeError("tts helper died")
            self._proc.stdin.write((json.dumps({"text": text, "voice": ""}) + "\n").encode())
            self._proc.stdin.flush()
            hdr = json.loads(self._proc.stdout.readline().decode())
            if not hdr.get("ok"):
                raise RuntimeError(f"tts: {hdr.get('error')}")
            n = hdr["samples"] * 4
            buf = bytearray()
            while len(buf) < n:
                chunk = self._proc.stdout.read(n - len(buf))
                if not chunk:
                    raise RuntimeError("tts helper closed")
                buf += chunk
        f32 = np.frombuffer(bytes(buf), dtype=np.float32)
        pcm = np.clip(resample(f32, int(hdr["rate"]), BRIDGE_RATE) * 32767, -32768, 32767).astype(np.int16)
        return pcm

    def preload(self, *phrases: str):
        for p in phrases:
            t = time.perf_counter()
            self._cache[p] = self.render(p)
            log.info("preloaded %r (%.2fs audio, %.0fms)", p, len(self._cache[p]) / BRIDGE_RATE, (time.perf_counter() - t) * 1000)


# ---------------------------------------------------------------------------
# LLM — persistent Hermes worker (hermes_worker.py), JSON lines over pipes.
# ---------------------------------------------------------------------------
class Worker:
    def __init__(self):
        self._lock = threading.Lock()
        self._spawn()

    def _spawn(self):
        env = dict(os.environ)
        self._proc = subprocess.Popen(
            [HERMES_PY, str(HERE / "hermes_worker.py")],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=open(LOG_DIR / "hermes_worker.log", "ab"),
            text=True, bufsize=1, env=env)
        t = time.perf_counter()
        line = self._proc.stdout.readline()
        log.info("worker ready in %.1fs: %s", time.perf_counter() - t, line.strip())

    def refresh(self) -> int:
        """Reload DATA's recent cross-channel context (call start)."""
        with self._lock:
            if self._proc.poll() is not None:
                self._spawn()
            self._proc.stdin.write(json.dumps({"refresh": True}) + "\n")
            self._proc.stdin.flush()
            line = self._proc.stdout.readline()
            try:
                return int(json.loads(line).get("refreshed", 0))
            except Exception:
                return 0

    def stream(self, prompt: str, on_delta, on_filler, cancelled) -> str:
        """Run one turn. Calls on_delta(text) per delta; returns final text.
        If cancelled() becomes true, drains the rest of the turn quietly."""
        with self._lock:
            if self._proc.poll() is not None:
                log.warning("worker died; respawning")
                self._spawn()
            self._proc.stdin.write(json.dumps({"prompt": prompt, "stream": True}) + "\n")
            self._proc.stdin.flush()
            while True:
                line = self._proc.stdout.readline()
                if not line:
                    raise RuntimeError("worker closed stdout")
                msg = json.loads(line)
                if "delta" in msg:
                    if not cancelled():
                        on_delta(msg["delta"])
                elif "filler" in msg:
                    if not cancelled():
                        on_filler(msg["filler"])
                elif "content" in msg:
                    return msg["content"], msg.get("tier", "?")
                elif "error" in msg:
                    raise RuntimeError(msg["error"])


# ---------------------------------------------------------------------------
# Sentence splitter for streaming deltas → TTS.
# ---------------------------------------------------------------------------
_BOUNDARY = re.compile(r"(?<=[.!?])\s+|(?<=[.!?])$|\n+")


def looks_hallucinated(text: str, audio_s: float) -> bool:
    """Whisper on near-silence emits loops ('I'm sorry. ' x30 on 1.0s of audio,
    live call 2026-09-11 15:54). Reject repetition and impossible speech rates."""
    words = text.split()
    if len(words) >= 6:
        from collections import Counter
        top = Counter(w.strip(".,!?").lower() for w in words).most_common(1)[0][1]
        if top / len(words) > 0.5:
            return True
    return len(text) / max(audio_s, 0.3) > 45  # > ~45 chars/s is not human


class SentenceBuffer:
    def __init__(self, first_clause_chars: int = 40):
        self.buf = ""
        self.emitted = 0
        self.first_clause_chars = first_clause_chars

    def feed(self, delta: str) -> list[str]:
        self.buf += delta
        out = []
        while True:
            m = _BOUNDARY.search(self.buf)
            if m and m.start() > 0:
                out.append(self.buf[:m.start()])
                self.buf = self.buf[m.end():]
                continue
            # First audio: break at a clause once we have enough, don't wait for a period.
            if self.emitted == 0 and not out and len(self.buf) >= self.first_clause_chars:
                cm = re.search(r"[,;:—-]\s+", self.buf[self.first_clause_chars // 2:])
                if cm:
                    cut = self.first_clause_chars // 2 + cm.end()
                    out.append(self.buf[:cut])
                    self.buf = self.buf[cut:]
                    continue
            break
        self.emitted += len(out)
        return [s for s in (clean_for_speech(s) for s in out) if s]

    def flush(self) -> list[str]:
        s, self.buf = clean_for_speech(self.buf), ""
        return [s] if s else []


# ---------------------------------------------------------------------------
# Bridge client
# ---------------------------------------------------------------------------
class Bridge:
    def __init__(self):
        self.channel = grpc.insecure_channel(SOCKET)
        self.stub = pbg.FaceTimeMediaStub(self.channel)

    def health(self):
        return self.stub.Health(pb.HealthRequest(), timeout=5)

    def control(self, cmd: int, timeout: float = 60.0):
        return self.stub.Control(pb.ControlRequest(command=cmd), timeout=timeout)

    def probe(self):
        try:
            return self.control(pb.CONTROL_COMMAND_PROBE, timeout=10)
        except grpc.RpcError as e:
            log.warning("probe failed: %s", e.code())
            return None


def thinking_pattern() -> np.ndarray:
    """One period of the thinking sound: a 40 ms 520 Hz blip with a raised-
    cosine envelope, then silence to THINK_PERIOD_S. int16 at BRIDGE_RATE."""
    n = int(THINK_PERIOD_S * BRIDGE_RATE)
    out = np.zeros(n, dtype=np.float32)
    blip = int(0.040 * BRIDGE_RATE)
    t = np.arange(blip) / BRIDGE_RATE
    env = 0.5 - 0.5 * np.cos(2 * np.pi * np.arange(blip) / blip)
    out[:blip] = THINK_GAIN * env * np.sin(2 * np.pi * 520 * t)
    return (out * 32767).astype(np.int16)


class AudioSession:
    """One bidi Audio stream: START first, CAPTURE in, PLAYBACK/CLEAR/STOP out."""

    def __init__(self, bridge: Bridge, call_id: str):
        self.call_id = call_id
        self.outq: queue.Queue = queue.Queue()
        self.capture: queue.Queue = queue.Queue()
        self.seq = 0
        self.ready = threading.Event()
        self.closed = threading.Event()
        self.scheduled_until = 0.0  # wall time when queued playback finishes
        self._lock = threading.Lock()
        self.outq.put(self._pkt(pb.AUDIO_PACKET_KIND_START))
        self._resp = bridge.stub.Audio(self._gen())
        threading.Thread(target=self._reader, daemon=True, name="audio-reader").start()

    def _pkt(self, kind, pcm=b""):
        self.seq += 1
        return pb.AudioPacket(call_id=self.call_id, kind=kind, pcm16=pcm,
                              sample_rate=BRIDGE_RATE, channels=1, sequence=self.seq)

    def _gen(self):
        while True:
            p = self.outq.get()
            if p is None:
                return
            yield p

    def _reader(self):
        try:
            for pkt in self._resp:
                if pkt.kind == pb.AUDIO_PACKET_KIND_EVENT:
                    log.info("audio event: %s", pkt.event)
                    if pkt.event == "ready":
                        self.ready.set()
                elif pkt.kind == pb.AUDIO_PACKET_KIND_CAPTURE:
                    self.capture.put(np.frombuffer(pkt.pcm16, dtype=np.int16).astype(np.float32) / 32768.0)
        except grpc.RpcError as e:
            if not self.closed.is_set():
                log.warning("audio stream ended: %s %s", e.code(), e.details())
        finally:
            self.closed.set()
            self.capture.put(None)

    def play(self, pcm: np.ndarray):
        with self._lock:
            now = time.monotonic()
            self.scheduled_until = max(now, self.scheduled_until) + len(pcm) / BRIDGE_RATE
        b = pcm.tobytes()
        step = 2400 * 2  # 100 ms per packet
        for i in range(0, len(b), step):
            self.outq.put(self._pkt(pb.AUDIO_PACKET_KIND_PLAYBACK, b[i:i + step]))

    def play_raw(self, pcm: np.ndarray):
        """Queue audio WITHOUT counting it as speech (thinking blips): VAD
        thresholds and the 'playing' pause logic ignore it."""
        self.outq.put(self._pkt(pb.AUDIO_PACKET_KIND_PLAYBACK, pcm.tobytes()))

    @property
    def playing(self) -> bool:
        return time.monotonic() < self.scheduled_until

    def clear(self):
        with self._lock:
            self.scheduled_until = 0.0
        self.outq.put(self._pkt(pb.AUDIO_PACKET_KIND_CLEAR))

    def stop(self):
        self.closed.set()
        self.outq.put(self._pkt(pb.AUDIO_PACKET_KIND_STOP))
        self.outq.put(None)


# ---------------------------------------------------------------------------
# The call
# ---------------------------------------------------------------------------
class Call:
    def __init__(self, bridge: Bridge, audio: AudioSession, vad: SileroVAD, tts: TTS, stt, worker: Worker, simulate: bool = False):
        self.bridge, self.audio, self.vad, self.tts, self.stt, self.worker = bridge, audio, vad, tts, stt, worker
        self.simulate = simulate
        self.reopen_on_answer = False
        self.answered = threading.Event()
        self.reopening = False
        self.last_ringback = 0.0
        self.greet_lock = threading.Lock()
        self.utterances: queue.Queue = queue.Queue()
        self.barge = threading.Event()
        self.ended = threading.Event()
        self.turn_lock = threading.Lock()
        self.greeted = False
        self.transcript = []

    # ---- capture → VAD → utterances ---------------------------------------
    def listen(self):
        vad = self.vad
        vad.reset()
        frame = SileroVAD.FRAME
        pending16 = np.zeros(0, dtype=np.float32)
        speech: list[np.ndarray] = []
        in_speech = False
        speech_ms = 0
        silence_ms = 0
        frame_ms = frame * 1000 / STT_RATE
        while not self.ended.is_set():
            audio = self.audio
            try:
                chunk = audio.capture.get(timeout=0.5)
            except queue.Empty:
                continue
            if chunk is None:
                if self.ended.is_set():
                    break
                if audio is not self.audio:
                    pass  # old session drained; fall through to reset and continue on the new one
                vad.reset()  # session swapped (reopen_audio); keep listening
                pending16 = np.zeros(0, dtype=np.float32)
                in_speech = False
                continue
            if not self.answered.is_set():
                # Pre-answer: track ringback bursts on the RAW chunk.
                rms = float(np.sqrt(np.mean(chunk ** 2))) if len(chunk) else 0.0
                now = time.monotonic()
                if rms > RINGBACK_RMS:
                    if self.last_ringback == 0.0:
                        log.info("ringback heard (rms %.3f)", rms)
                    self.last_ringback = now
                elif self.last_ringback and now - self.last_ringback > RINGBACK_GAP_S:
                    log.info("ringback stopped %.1fs ago — answered", now - self.last_ringback)
                    self.answered.set()
            chunk = np.tanh(chunk * INPUT_GAIN)
            pending16 = np.concatenate([pending16, resample(chunk, BRIDGE_RATE, STT_RATE)])
            while len(pending16) >= frame:
                f, pending16 = pending16[:frame], pending16[frame:]
                p = vad.prob(f)
                playing = self.audio.playing
                thr = VAD_THRESHOLD_PLAYING if playing else VAD_THRESHOLD
                if p >= thr:
                    if not in_speech:
                        in_speech, speech, speech_ms, silence_ms = True, [], 0, 0
                        log.debug("speech start p=%.2f playing=%s", p, playing)
                    speech.append(f)
                    speech_ms += frame_ms
                    silence_ms = 0
                    if playing and speech_ms >= BARGE_IN_MS and not self.barge.is_set():
                        log.info("barge-in")
                        self.barge.set()
                        self.audio.clear()
                elif in_speech:
                    speech.append(f)
                    silence_ms += frame_ms
                    if silence_ms >= SILENCE_MS or speech_ms > MAX_UTTERANCE_S * 1000:
                        in_speech = False
                        if speech_ms >= MIN_SPEECH_MS:
                            self.utterances.put(np.concatenate(speech))
                        speech = []

    # ---- utterances → STT → LLM → TTS ------------------------------------
    def converse(self):
        while not self.ended.is_set():
            try:
                utt = self.utterances.get(timeout=0.5)
            except queue.Empty:
                continue
            # Latest wins: if several piled up during a long turn, keep the last.
            while not self.utterances.empty():
                utt = self.utterances.get_nowait()
            t0 = time.perf_counter()
            text = self.stt.transcribe(utt)
            t_stt = time.perf_counter() - t0
            if not text or len(text.strip(" .")) < 2:
                log.info("STT empty (%.0fms) — ignored", t_stt * 1000)
                continue
            if looks_hallucinated(text, len(utt) / STT_RATE):
                log.info("STT hallucination dropped (%.1fs audio): %r", len(utt) / STT_RATE, text[:60])
                continue
            log.info("Captain: %s  (STT %.0fms, %.1fs audio)", text, t_stt * 1000, len(utt) / STT_RATE)
            self.transcript.append(("user", text))
            if not self.greeted:
                # Speech before the ringback detector fired: that IS the answer.
                self.answered.set()
                self.on_answered()
                if re.fullmatch(r"[\s\W]*(hello|hi|hey|yo|data|you there|are you there)[\s\W]*", text, re.I):
                    continue
            self.turn(text)

    def on_answered(self):
        """Outbound only. The stream we listened on was opened pre-answer and
        its playback leg is dead once Phone grabs BlackHole 2ch (live calls
        2026-09-11 15:32/15:36/15:42: DATA heard the Captain, the Captain heard
        nothing; 15:46 re-opened post-answer and WAS heard). Re-open, greet."""
        with self.greet_lock:
            if self.greeted:
                return
            self.greeted = True
            if self.reopen_on_answer:
                self.reopen_audio()
            self.say(GREETING)
            rms = bh2_tap(1.5)
            log.log(logging.INFO if rms > 0.01 else logging.ERROR,
                    "BH2 tap during greeting: rms=%.4f (%s)", rms,
                    "playback leg LIVE" if rms > 0.01 else "playback leg DEAD")

    def answer_watch(self):
        """Wait for the ringback detector (listen thread) and greet."""
        while not self.ended.is_set():
            if self.answered.wait(0.25):
                self.on_answered()
                return

    def reopen_audio(self):
        """The daemon is single-session (one AudioBridge at a time): opening a
        second Audio stream while the first is alive kills BOTH (live call
        2026-09-11 15:54). So: STOP the old stream, wait for the daemon to close
        it, then open the new one. watch_end ignores the close while
        self.reopening is set. Measured 0.12s end-to-end (call 15:46)."""
        t = time.perf_counter()
        self.reopening = True
        try:
            old = self.audio
            old.stop()
            if not old.closed.wait(5):
                log.warning("reopen_audio: old stream did not close within 5s; opening anyway")
            new = AudioSession(self.bridge, old.call_id + "-r")
            deadline = time.time() + 20
            while time.time() < deadline and not new.ready.is_set() and not new.closed.is_set():
                time.sleep(0.02)
            if not new.ready.is_set():
                log.error("reopen_audio: new stream not ready (closed=%s) — retrying once", new.closed.is_set())
                new.stop()
                time.sleep(1.0)
                new = AudioSession(self.bridge, old.call_id + "-r2")
                if not new.ready.wait(20):
                    log.error("reopen_audio: retry failed; playback leg is likely dead")
                    self.audio = new
                    return
            self.audio = new
            log.info("audio stream re-opened post-answer in %.2fs", time.perf_counter() - t)
        finally:
            self.reopening = False

    def say(self, text: str):
        pcm = self.tts.render(text)
        self.audio.play(pcm)
        log.info("DATA: %s", text)

    def turn(self, user_text: str):
        self.barge.clear()
        sentences = SentenceBuffer()
        spoken = 0
        t0 = time.perf_counter()
        first_audio = [None]

        thinking_stop = threading.Event()

        def thinking():
            # Soft blips while the turn has produced no speech. Just-in-time
            # pacing keeps the daemon queue shallow so the flush is instant.
            if thinking_stop.wait(THINK_AFTER_S):
                return
            pattern = thinking_pattern()
            step = int(0.1 * BRIDGE_RATE)
            pos = 0
            while not thinking_stop.is_set() and not self.barge.is_set():
                if self.audio.playing:          # filler / stall line on air
                    time.sleep(0.1)
                    continue
                chunk = pattern[pos:pos + step]
                pos = (pos + step) % len(pattern)
                self.audio.play_raw(chunk)
                time.sleep(0.095)

        if THINKING_SOUND == "blip":
            threading.Thread(target=thinking, daemon=True, name="thinking").start()

        def speak(s: str, filler: bool = False) -> bool:
            nonlocal spoken
            if self.barge.is_set() or spoken >= MAX_SENTENCES:
                return False
            t = time.perf_counter()
            pcm = self.tts.render(s)
            if self.barge.is_set():
                return False
            if not filler and not thinking_stop.is_set():
                # First real sentence: end the blips and flush whatever slice is
                # still queued so DATA's voice starts now, not 100 ms later.
                thinking_stop.set()
                if THINKING_SOUND == "blip":
                    self.audio.clear()
            self.audio.play(pcm)
            spoken += 1
            if first_audio[0] is None:
                first_audio[0] = time.perf_counter() - t0
                stall_timer.cancel()
            log.info("DATA: %s  (tts %.0fms)", s, (time.perf_counter() - t) * 1000)
            return True

        def on_delta(d):
            for s in sentences.feed(d):
                if not speak(s):
                    break

        def stall():
            # Provider is thinking (live call 2026-09-11 15:46: one FAST turn
            # took 13.2s to first token). Never leave the line dead that long.
            if first_audio[0] is None and not self.barge.is_set():
                log.warning("no audio %.1fs into the turn — speaking stall line", STALL_AFTER_S)
                self.audio.play(self.tts.render(STALL_LINE))
        stall_timer = threading.Timer(STALL_AFTER_S, stall)
        stall_timer.daemon = True
        stall_timer.start()
        tier = "?"
        try:
            final, tier = self.worker.stream(user_text, on_delta, lambda f: speak(f, filler=True),
                                             lambda: self.barge.is_set() or spoken >= MAX_SENTENCES)
            for s in sentences.flush():
                speak(s)
            final = clean_for_speech(final)
            capped = spoken >= MAX_SENTENCES and sentences.buf.strip()
            if capped and not self.barge.is_set():
                # We cut the reply short on purpose; say so rather than trail off.
                self.audio.play(self.tts.render(CAP_LINE))
            self.transcript.append(("assistant", final))
            log.info("turn done [%s]: first audio %s, total %.2fs, %d sentence(s)%s", tier,
                     f"{first_audio[0]:.2f}s" if first_audio[0] else "none",
                     time.perf_counter() - t0, spoken, " [barged]" if self.barge.is_set() else "")
            if spoken == 0 and not self.barge.is_set():
                self.say(LOST_LINE)
        except Exception as e:
            log.exception("turn failed: %s", e)
            if not self.barge.is_set():
                self.say(LOST_LINE)
        finally:
            stall_timer.cancel()
            thinking_stop.set()

    # ---- lifecycle --------------------------------------------------------
    def watch_end(self):
        """Poll the daemon; two consecutive idle/ended scans end the call."""
        misses = 0
        while not self.ended.is_set():
            time.sleep(2.0)
            if self.audio.closed.is_set() and not self.reopening:
                log.info("audio stream closed by daemon")
                self.ended.set()
                break
            if self.simulate:
                continue  # no FaceTime call to probe; ends with the stream
            r = self.bridge.probe()
            if r is None:
                continue
            if r.state in ("idle", "ended"):
                misses += 1
                if misses >= 2:
                    log.info("call ended (probe: %s)", r.state)
                    self.ended.set()
            else:
                misses = 0

    def run(self):
        workers = [self.listen, self.converse, self.watch_end]
        if self.reopen_on_answer:
            workers.append(self.answer_watch)
        else:
            self.answered.set()
        threads = [threading.Thread(target=f, daemon=True, name=f.__name__) for f in workers]
        for t in threads:
            t.start()
        self.ended.wait()
        self.audio.stop()
        with open(LOG_DIR / "transcripts.jsonl", "a") as fh:
            fh.write(json.dumps({"t": time.time(), "call_id": self.audio.call_id, "turns": self.transcript}) + "\n")


# ---------------------------------------------------------------------------
# Service
# ---------------------------------------------------------------------------
def wait_connected(bridge: Bridge, timeout: float = 75.0) -> bool:
    """Outbound: poll the daemon until the call surface reads 'connected'."""
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        r = bridge.probe()
        if r is not None and r.state != last:
            log.info("probe: %s", r.state)
            last = r.state
        if r is not None and r.state == "connected":
            return True
        if r is not None and last == "connected":
            return False
        time.sleep(1.0)
    return False


_MUTE_ENABLED_SCRIPT = """
tell application "System Events"
  if not (exists process "Phone") then return "no-phone"
  tell process "Phone"
    try
      return (enabled of menu item "Mute" of menu "Video" of menu bar 1) as string
    on error
      return "no-menu"
    end try
  end tell
end tell
"""


def phone_mute_enabled() -> str:
    """'true' when Phone.app's Video>Mute is enabled — which macOS only does for
    a LIVE (answered) call. 'false' while ringing; 'no-phone' when no call UI."""
    try:
        r = subprocess.run(["osascript", "-e", _MUTE_ENABLED_SCRIPT], capture_output=True, text=True, timeout=5)
        return (r.stdout or r.stderr).strip()
    except Exception as e:
        return f"err:{e}"


def wait_answered(timeout: float = 75.0) -> bool:
    """Outbound: block until the far end actually picks up. The daemon's
    'connected' fires as soon as the Phone surface exists (dialing), and the
    audio stream must not open before the answer (see run_call). Polls
    Phone.app's Video>Mute enabled state; logs every transition."""
    deadline = time.time() + timeout
    last = None
    while time.time() < deadline:
        st = phone_mute_enabled()
        if st != last:
            log.info("answer-wait: Phone Video>Mute enabled=%s", st)
            last = st
        if st == "true":
            return True
        if st == "no-phone" and last == "false":
            return False  # call surface went away without answer
        time.sleep(0.5)
    return False


def wait_for_answer_tap(vad: "SileroVAD", timeout: float = 75.0, alive=lambda: True) -> str:
    """Outbound answer detection WITHOUT touching the daemon: tap BlackHole 16ch
    (FaceTime's output) with sounddevice. FaceTime ringback = ~0.75 s bursts
    every 3 s; when they stop for RINGBACK_GAP_S the far end has picked up.
    Fallback: far-end speech (Silero) also means answered. Returns
    'ringback-gap' | 'speech' | 'timeout' | 'ended'.

    Why not the daemon's stream: it is single-session and refuses the next
    stream for a while after a STOP (measured 2026-09-11 15:55: open/stop/open
    -> 2nd open fails with UNKNOWN, 3rd works). Opening it once, post-answer,
    is the only reliable order."""
    import sounddevice as sd
    q: "queue.Queue[np.ndarray]" = queue.Queue()
    with sd.InputStream(device="BlackHole 16ch", channels=2, samplerate=48000, blocksize=4800,
                        callback=lambda d, f, t, st: q.put(d.mean(axis=1).copy())):
        vad.reset()
        last_burst = 0.0
        pending = np.zeros(0, dtype=np.float32)
        speech_ms = 0.0
        t_end = time.time() + timeout
        while time.time() < t_end:
            if not alive():
                return "ended"
            try:
                blk = q.get(timeout=0.5)
            except queue.Empty:
                continue
            now = time.monotonic()
            rms = float(np.sqrt(np.mean(blk ** 2)))
            if rms > RINGBACK_RMS:
                if last_burst == 0.0:
                    log.info("ringback heard (rms %.3f)", rms)
                last_burst = now
                continue
            if last_burst and now - last_burst > RINGBACK_GAP_S:
                log.info("ringback stopped %.1fs ago — answered", now - last_burst)
                return "ringback-gap"
            # speech fallback (quiet far-end voice, ~0.002 rms pre-gain)
            pending = np.concatenate([pending, resample(np.tanh(blk * INPUT_GAIN), 48000, STT_RATE)])
            while len(pending) >= SileroVAD.FRAME:
                f, pending = pending[:SileroVAD.FRAME], pending[SileroVAD.FRAME:]
                if vad.prob(f) >= VAD_THRESHOLD:
                    speech_ms += SileroVAD.FRAME * 1000 / STT_RATE
                    if speech_ms >= 400:
                        log.info("far-end speech heard — answered")
                        return "speech"
                else:
                    speech_ms = 0.0
        return "timeout"


def bh2_tap(seconds: float = 1.5) -> float:
    """Ground truth for the playback leg: RMS on BlackHole 2ch INPUT (what
    FaceTime's mic reads) while the greeting plays. Fail-open diagnostic."""
    try:
        import sounddevice as sd
        buf = []
        with sd.InputStream(device="BlackHole 2ch", channels=2, samplerate=48000,
                            callback=lambda d, f, t, st: buf.append(d[:, 0].copy())):
            time.sleep(seconds)
        a = np.concatenate(buf) if buf else np.zeros(1, dtype=np.float32)
        return float(np.sqrt(np.mean(a ** 2)))
    except Exception as e:
        log.warning("bh2 tap failed: %s", e)
        return -1.0


def run_call(bridge, vad, tts, stt, worker, outbound: bool, simulate: bool = False):
    # ORDER MATTERS (2026-09-11 live call 15:32): the audio stream must open
    # AFTER the call is connected. Opened early, the daemon's playback engine
    # is already running on BlackHole 2ch when Phone.app seizes that device as
    # its mic; CoreAudio reconfigures it and the engine goes silent — DATA
    # hears the Captain, the Captain hears nothing. Upstream's documented
    # lifecycle (WaitIncoming -> connected -> Audio) has the same rule.
    if outbound and not simulate:
        if not wait_connected(bridge):
            log.error("outbound call never reached 'connected' — not opening audio")
            return
        # 'connected' here only means the Phone surface exists (dialing). Wait
        # for the real answer on a device tap, THEN open the daemon stream.
        how = wait_for_answer_tap(vad, alive=lambda: (bridge.probe() or pb.ControlResponse(state="connected")).state not in ("idle", "ended"))
        if how in ("timeout", "ended"):
            log.error("outbound call not answered (%s) — not opening audio", how)
            return
        log.info("answered (%s) — opening audio stream", how)
    elif not simulate:
        # Inbound: the daemon pressed Answer and confirmed 'connected'. Give
        # Phone.app a moment to open its devices before ours start (the audio
        # law: our engines must start AFTER Phone holds BlackHole 2ch).
        time.sleep(INBOUND_SETTLE_S)
    # Context refresh off the critical path: the state.db read took 7.3 s on
    # the 20:36 inbound call and delayed the audio open. The worker lock
    # serializes it against the first turn, so the prompt is fresh by then.
    def _refresh():
        t = time.perf_counter()
        n = worker.refresh()
        log.info("worker context refreshed: %d recent lines (%.0fms)", n, (time.perf_counter() - t) * 1000)
    threading.Thread(target=_refresh, daemon=True, name="ctx-refresh").start()
    call_id = f"dfv-{int(time.time())}"
    audio = AudioSession(bridge, call_id)
    if not audio.ready.wait(30):
        log.error("daemon never sent 'ready' for the audio stream")
        audio.stop()
        return
    log.info("audio stream open (%s)", "outbound" if outbound else "inbound")
    call = Call(bridge, audio, vad, tts, stt, worker, simulate=simulate)
    call.greeted = True
    call.say(GREETING)
    rms = bh2_tap(1.5)
    log.log(logging.INFO if rms > 0.01 else logging.ERROR,
            "BH2 tap during greeting: rms=%.4f (%s)", rms, "playback leg LIVE" if rms > 0.01 else "playback leg DEAD")
    call.run()
    log.info("call finished: %d transcript turns", len(call.transcript))


def warm_daemon_audio(bridge: Bridge):
    """The daemon's FIRST Audio stream after it starts is slow (5 s at 16:44,
    >30 s on the 20:36 inbound call — the Captain hung up before 'ready').
    Open and close one stream now, while idle, so a call never pays it."""
    t = time.perf_counter()
    try:
        s = AudioSession(bridge, f"warm-{int(time.time())}")
        ok = s.ready.wait(60)
        s.stop()
        s.closed.wait(5)
        log.info("daemon audio path warmed: ready=%s in %.1fs", ok, time.perf_counter() - t)
    except Exception as e:
        log.warning("daemon audio warm-up failed: %s", e)


def wait_incoming(bridge: Bridge, events: queue.Queue, stop: threading.Event):
    """Daemon-side auto-answer. Re-subscribes forever."""
    while not stop.is_set():
        try:
            for ev in bridge.stub.WaitIncoming(pb.WaitIncomingRequest()):
                log.info("incoming event: state=%s authorized=%s err=%s", ev.state, ev.authorized, ev.error_code)
                if ev.state == "connected":
                    # Only reachable after the daemon's own identity-verified
                    # Answer press: the ringing card carried the configured
                    # E.164 or contact name. Anyone else rings out.
                    events.put("inbound")
                    break
                if ev.error_code:
                    if ev.error_code == "CALLER_NOT_AUTHORIZED":
                        log.warning("incoming call from an unauthorized caller — not answering")
                    break
        except grpc.RpcError as e:
            log.warning("WaitIncoming: %s %s", e.code(), e.details())
            time.sleep(2)


def main():
    LOG_DIR.mkdir(exist_ok=True)
    logging.basicConfig(level=os.environ.get("DFV_LOG_LEVEL", "INFO"),
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout)])
    for noisy in ("grpc", "urllib3", "httpx"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    t = time.perf_counter()
    bridge = Bridge()
    h = bridge.health()
    log.info("bridge ready=%s in=%s out=%s", h.ready, h.input_device, h.output_device)
    vad = SileroVAD()
    log.info("VAD ready")
    tts = TTS()
    tts.preload(GREETING, TOOL_FILLER, LOST_LINE, CAP_LINE, STALL_LINE)
    stt = stt_engine.STT()
    worker = Worker()
    warm_daemon_audio(bridge)
    log.info("all warm in %.1fs — waiting for a call", time.perf_counter() - t)

    if "--simulate" in sys.argv:
        # No FaceTime: open the audio stream now and converse with whatever is
        # played into BlackHole 16ch (see sim_call.py). Ends on Ctrl-C.
        log.info("SIMULATE: treating the daemon audio stream as a connected inbound call")
        run_call(bridge, vad, tts, stt, worker, outbound=False, simulate=True)
        return

    events: queue.Queue = queue.Queue()
    stop = threading.Event()
    threading.Thread(target=wait_incoming, args=(bridge, events, stop), daemon=True).start()

    while True:
        # Outbound trigger file (dfv-call). Consume atomically.
        if TRIGGER.exists():
            try:
                TRIGGER.rename(TRIGGER.with_suffix(".consumed"))
                events.put("outbound")
            except OSError:
                pass
        try:
            kind = events.get(timeout=1.0)
        except queue.Empty:
            continue
        if kind == "outbound":
            log.info("placing outbound call via daemon")
            try:
                r = bridge.control(pb.CONTROL_COMMAND_CALL, timeout=90)
            except grpc.RpcError as e:
                r = None
                log.error("CALL rpc failed: %s %s", e.code(), e.details())
            result = f"{r.ok} {r.state} {r.error_code} {r.message}" if r else "rpc-failed"
            log.info("CALL -> %s", result)
            TRIGGER.with_suffix(".result").write_text(result + "\n")
            if not (r and r.ok):
                continue
        run_call(bridge, vad, tts, stt, worker, outbound=(kind == "outbound"))


if __name__ == "__main__":
    main()
