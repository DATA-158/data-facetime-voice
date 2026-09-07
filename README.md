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
- `hermes_worker.py` — persistent Hermes AIAgent worker process (one warm
  LLM agent reused across turns; ~1s raw API, 4-6s warm turn).
- `attach_live.py` — cold-start call attach (superseded by warm adoption;
  kept as fallback).
- `audio_default.py` — CoreAudio default-device probe/flip helpers.
- `facetime_media_pb2*.py` — gRPC stubs for the bridge daemon's media API.
- `config.json` — non-secret runtime config.
- `deploy/` — launchd plists + shell scripts (dfv-call, pin, restart).

## Latency design notes (measured on an M-series MacBook Air)

- First spoken audio after caller finishes: **~5s** (streaming: sentences
  speak as the LLM generates them; persistent warm worker ~4-6s; TTS
  ~1.3s/sentence, prewarmed).
- Call adoption when dialing out: trigger file → warm service consumes in
  ~2s → Phone dials → connected in ~15-20s total.
- STT: faster-whisper distil-small.en, 6 cpu_threads, tail-trimmed
  utterances. distil-medium.en ~2.3s in-call if you have headroom.
- TTS: `say` writing LEI16@24000 WAV (~1.3s/sentence; AIFF was 3.3s).
  `say` cannot stream to stdout (err -54) — per-chunk WAV files are the way.
- The 15-20s "why is this slow" trap: building a fresh agent per turn.
  Persistent worker + streaming first-sentence is the fix.

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