# DATA FaceTime Voice

Realtime-ish AI voice calls over FaceTime Audio on macOS. An AI assistant
("DATA") that places and answers FaceTime Audio calls, hears the caller via
Whisper STT, thinks with a persistent warm LLM worker (Hermes Agent), and
speaks with macOS TTS — while both the STT/LLM/TTS stack and the FaceTime
bridge stay permanently warm under launchd.

## Architecture

Three layers:

1. **facetime-bridge** (Swift daemon, separate repo — see below): the native
   bridge that talks to FaceTime via Apple's private media APIs. Places calls,
   captures call audio, plays agent audio into the call. Runs as a signed
   launchd daemon.
2. **data-facetime-voice** (this repo, Python ~1,800 lines): the conversation
   brain. Whisper STT → persistent LLM worker → macOS TTS, with turn-taking
   VAD, barge-in, streaming sentence-by-sentence speech, and call lifecycle
   (inbound answer, outbound dial, warm adoption of live calls).
3. **Glue**: two launchd services (bridge + voice), shell entrypoints in
   `deploy/`, and a pre-dial audio-routing pin script.

```
iPhone ←FaceTime Audio→ FaceTime.app ←AudioHose APIs→ bridge daemon (Swift)
     → BlackHole 16ch (capture) → voice_loop.py (VAD/STT/LLM/TTS)
     → BlackHole 2ch (playback) → FaceTime mic → caller's ear
```

## Files

- `voice_loop.py` — main service: audio event loop, VAD, STT, turn handling,
  streaming LLM→TTS, warm adoption of in-flight calls, call persistence.
- `hermes_worker.py` — persistent Hermes worker holding TWO warm agents: a
  no-tool FAST agent for conversation and the full-toolset agent for turns that
  need to act. Routes per turn and emits a spoken filler on tool turns.
- `tts_engine.py` — persistent macOS synthesizer (same voice as `say`, ~25ms
  instead of ~665ms). Verifies byte-identical output at startup, falls back to
  `say` if it ever differs. Run it directly to self-test: `python tts_engine.py`.
- `stt_engine.py` — MLX GPU whisper with faster-whisper CPU fallback.
  Self-test: `python stt_engine.py`.
- `bench_turn.py` — end-to-end turn latency harness; measures real
  time-to-first-audio without needing a FaceTime call.
- `attach_live.py` — cold-start call attach (superseded by warm adoption;
  kept as fallback).
- `audio_default.py` — CoreAudio default-device probe/flip helpers.
- `facetime_media_pb2*.py` — gRPC stubs for the bridge daemon's media API.
- `config.json` — non-secret runtime config.
- `deploy/` — launchd plists + shell scripts (dfv-call, pin, restart).

## Latency (measured, 2026-09-07 pass #2)

All numbers from an M-series MacBook Air, `bench_turn.py`, three runs.

**Time to first audio after the caller stops speaking: mean 1.4-1.6s, worst 2.2s**
(was 5-8s on a good turn and 30s+ on a bad one).

Per-stage, measured:

| stage | before | after | how |
|---|---|---|---|
| endpointing | 700ms | 700ms | unchanged (VAD silence window) |
| STT | 2112ms | ~250ms | MLX GPU whisper-small.en instead of CPU int8 |
| LLM -> first chunk | 2100-4700ms | 500-1500ms | two-tier tools + reasoning off for chat |
| TTS first sentence | 665ms | 25-70ms | one persistent synthesizer instead of `say` per sentence |

### What actually mattered

- **`say` costs ~665ms of speech-engine init per invocation**, flat, regardless
  of text length (10 identical runs: 654-718ms; a bare process spawn is 4ms).
  It cannot be prewarmed across processes. One long-lived `NSSpeechSynthesizer`
  does the same sentence in 25-37ms. **The voice is unchanged** — `tts_engine`
  proves byte-identical output against `say` at startup (same SHA-256) and
  falls back to `say` permanently if it ever differs.
- **STT was 6x slower than it needed to be.** faster-whisper distil-small.en on
  CPU took 2112ms on a 6.8s utterance; the same audio on the GPU via MLX takes
  333ms and transcribes more accurately. Note `cpu_threads` was a footgun:
  raising 6 -> 10 made the CPU path *slower* (2112ms -> 3167ms).
- **Tool schemas dominated the LLM turn.** The 6 production toolsets are 10
  tools / ~6046 tokens of JSON schema on every request, including "you there?".
  First-token latency with them: 2.08 / 4.69 / 3.59 / 0.68s. Without them:
  0.40 / 0.65 / 0.87 / 1.62s. The worker now runs two agents — a no-tool FAST
  agent for conversation and the full agent for anything that acts — and routes
  per turn with a local heuristic (no extra model call).
- **Tool turns stream nothing** until the whole agent loop finishes, so the
  worker emits a spoken filler ("Let me check that, Captain.") the moment a turn
  routes to the tool tier. Dead air on a tool turn: ~1.0s instead of 7s+.
- **First chunk breaks at a clause, not a sentence.** A reply opening with a
  long clause held all audio for 3.8s waiting on the first period. With TTS at
  ~40ms there is no reason to wait.

### Bugs that were the real "it takes minutes" / "he never speaks"

- `AUTHORIZED_E164` was `os.environ.get(os.environ.get(...))` — a double lookup
  that read the env var *named by* the phone number, so it was always `None`.
  The dial URL became `facetime-audio://None`, and `_call_timer_running()` fed
  that `None` to `subprocess(env=...)`, raising `TypeError` into a bare
  `except` — so connection was **never** detected and every outbound call fell
  through to the blind 60s/90s timeout branches.
- On a worker error, `llm_reply_streaming` spoke nothing at all while still
  writing "my response came back empty" into the transcript — the logs looked
  like a completed turn while the caller heard pure silence. There is now an
  audible fallback on every path.
- `DFV_REASONING_EFFORT` never did anything: the worker mutated a `load_config()`
  dict that was never passed to `AIAgent` (and `load_config()` returns a fresh
  dict each call), so every voice turn ran at the config default `medium`.
  Reasoning tokens are emitted *before* content, so on a call they are pure dead
  air. Now set via `reasoning_config`: off for chat, low for tool turns.
- The dial loop ran an AppleScript `entire contents of` traversal of Notification
  Center **every second** until it succeeded — one of the slowest calls in the AX
  API. It is now bounded and only runs inside the window where the prompt appears.
- Turns had no single-flight guard: two quick utterances ran concurrently, queued
  overlapping speech, and each cleared the other's barge-in flag.

### Still on the table

- Endpointing is a flat 700ms and is now the single largest remaining stage.
  Silero VAD (already a dependency) would allow ~400-500ms safely; the note
  below about 500ms fragmenting speech was measured against the *energy* VAD.
- Speculative STT (start transcribing at the onset of silence rather than after
  it is confirmed) would take STT off the critical path almost entirely.
- Tool turns still take 1-7s behind the filler. Trimming the voice toolset, or a
  second filler when the answer runs long, would tighten that.

## Hard-won gotchas (read before touching)

- **CoreAudio first-touch law**: AVAudioEngine fixes its INPUT format from
  the system default input at first touch. Pin system defaults BEFORE any
  engine touches a device. Mono-clamp if default input is the MacBook mic.
- **FaceTime locks its output at call start**: whatever the SYSTEM default
  output is when the call connects, that's what caller audio routes through.
  Pin before dialing; per-app menus can drift or lie.
- **Device seizure (silent-call root cause)**: FaceTime seizes the playback
  BlackHole as its input at call start; a playback engine that selected the
  same device at startup silently loses the route mid-call. Measured:
  pre-call renders hit the device (0.83 peak), in-call renders read 0.0000
  while the daemon-side recorder shows full-scale audio. Keep the TTS leg
  and FaceTime's mic on separate devices, or re-render post-connect.
- **gRPC + fork = crash**: forking TTS subprocesses while gRPC streams run
  eventually aborts the process. Set `GRPC_ENABLE_FORK_SUPPORT=1`.
- **Silence window**: 700ms endpointing. 500ms fragments speech into
  sub-second clips that each spawn garbage LLM turns.
- **launchd env is not shell env**: a venv that works in your terminal may
  miss modules under launchd. Point HERMES_VENV at a venv that actually has
  grpc/dotenv/numpy/faster_whisper.
- **Agent telemetry lies**: recordings and taps are ground truth; log lines
  like "call connected" are not.

## Secrets

No secrets in the repo. The authorized caller E.164 is injected via launchd
(`FACETIME_BRIDGE_AUTHORIZED_CALLER_E164`). Recordings and logs are
gitignored (they contain call audio).

## Status / roadmap

- Working end-to-end: inbound answer, outbound dial, warm adoption,
  streaming responses, post-call persistence.
- Known open issue: in-call playback route loss (see gotcha above) — fix in
  progress (TTS leg onto spare channels of the capture device).
- Not portable: Apple's private FaceTime media APIs (the Swift bridge does
  the heavy lifting there).

## Sister repo

The native bridge this orchestrates lives at
[kingbootoshi/facetime-bridge](https://github.com/kingbootoshi/facetime-bridge)
(upstream) with local patches for AX traversal and audio cadence.