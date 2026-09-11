# DATA FaceTime Voice

Realtime AI voice calls over FaceTime Audio on macOS. DATA (a Hermes agent)
places and answers FaceTime Audio calls with the Captain, hears him via local
Whisper, thinks with a warm Hermes worker, and speaks with the Mac's own
system voice (Siri). Everything is local except the LLM call.

First fully working two-way call: 2026-09-11 15:58 on DATA's Mac — greeting
~5 s after pickup, turns 1.0–1.3 s.

## Architecture

```
iPhone ←FaceTime Audio→ Phone.app/FaceTime.app
      output → BlackHole 16ch → facetime-bridge daemon → CAPTURE ─┐ gRPC
      mic    ← BlackHole 2ch  ← facetime-bridge daemon ← PLAYBACK ┘ unix socket
                                                      voice_agent.py
                        Silero VAD → MLX whisper base.en → hermes_worker → NSSpeechSynthesizer
```

- **[facetime-bridge](../facetime-bridge)** (Swift daemon, fork of
  kingbootoshi/facetime-bridge with local macOS 26 patches): owns FaceTime —
  probe / call / answer / hangup via Accessibility, fail-closed to ONE
  authorized caller (E.164 + trusted contact name), and the two audio legs.
  We do not work around it. If it refuses, the call does not happen.
- **`voice_agent.py`** (this repo): the conversation. One file. Built on
  `docs/INTEGRATION.md` of the bridge and nothing else.
- **`hermes_worker.py`**: persistent Hermes agents (FAST no-tool tier for
  chat, FULL tier for actions), JSON lines over pipes.
- **`tts_helper_ns.py`**: NSSpeechSynthesizer = the system voice, warm.
- **`stt_engine.py`**: MLX whisper on the GPU.

## Files

| file | role |
|---|---|
| `voice_agent.py` | service: call lifecycle, VAD, STT, turns, barge-in, `--simulate` |
| `hermes_worker.py` | warm Hermes agents, tier routing, filler on tool turns |
| `tts_helper_ns.py` | system-voice TTS process (Siri); `tts_helper.py` = AVSpeech fallback (cannot do Siri) |
| `stt_engine.py` | MLX whisper (`DFV_STT_MODEL`, default base.en) |
| `voice_persona.py` | DATA's voice-call system prompt (no audio deps; the worker imports only this) |
| `sim_call.py` | play a recorded caller into BlackHole 16ch, record BlackHole 2ch — full pipeline test, **no FaceTime call** |
| `probe_audio_path.py` | tone test of both daemon audio legs, no call |
| `audio_procs.swift` | which process holds which CoreAudio device (`swift audio_procs.swift Phone`) |
| `deploy/` | launchd plists, `dfv-call` (outbound trigger) |
| `voice_loop.py`, `attach_live.py`, `audio_default.py`, `deploy/ft_enforce_routes.scpt`, `data-facetime-voice-pin` | **legacy, superseded** — kept one release for reference |

## Running

```bash
# service (launchd): ~/Library/LaunchAgents/ai.data.facetime-voice.plist -> voice_agent.py
launchctl kickstart -k gui/$(id -u)/ai.data.facetime-voice
tail -f ~/Library/Logs/data-facetime-voice/voice_agent.log

# outbound call (DATA/Hermes side)
touch ~/.facetime-bridge/outbound.trigger        # or deploy/dfv-call

# full pipeline without a call
python voice_agent.py --simulate &
python sim_call.py recordings/<call>/caller.wav --seconds 60
```

Inbound: the daemon's `WaitIncoming` answers a ringing call only when the
card carries the configured caller; `voice_agent` then opens audio and talks.
Anyone else rings out.

## The audio laws (measured live, 2026-09-11 — read before touching)

1. **Open the daemon's Audio stream only AFTER the far end has answered.**
   Opened earlier, its playback engine dies when Phone.app takes BlackHole
   2ch at answer: DATA hears the Captain, the Captain hears nothing. Four
   silent calls today, every silent call in the history. `recordings/*/
   caller.wav|agent.wav` are ground truth; a BlackHole 2ch tap during the
   greeting reads ~0.14 rms when live, 0.0000 when dead.
2. **The daemon is single-session** and refuses the next Audio stream for a
   while after a STOP (open/stop/open: 2nd fails, 3rd works). Never re-open
   mid-call; never run two streams.
3. **Answer detection (macOS 26):** the daemon's `connected` = "Phone window
   exists" (1 s after dialing); Phone's Video▸Mute becomes enabled mid-ring.
   Neither is an answer. FaceTime ringback = 0.75 s bursts (~300 Hz) every
   3 s on BlackHole 16ch; 4.5 s without a burst = answered. Pre-answer
   listening is a sounddevice tap, not the daemon (`wait_for_answer_tap`).
4. **System default input must be BlackHole 16ch** (output too). With
   BlackHole 2ch as default input the daemon's capture engine dies within
   2–6 s. Per-app: FaceTime.app and Phone.app Video menu → mic BlackHole 2ch,
   output BlackHole 16ch (already set; honored).
5. **The Click-to-Call card on macOS 26.6 shows the contact NAME, not the
   number.** The daemon authorizes it via `FACETIME_BRIDGE_AUTHORIZED_CALLER_NAME`
   (`promptNameIdentity`, same trust model as the incoming-ring patch).
   Until that build is installed and granted Accessibility, `voice_agent`
   presses the button itself (`DFV_AX_PRESS_SHIM=1`, needs AX trust for
   `~/.local/bin/facetime-bridge-ax3` in the service's context).
6. **The system voice is a Siri voice** (`com.apple.siri.natural.Aaron`).
   AVSpeechSynthesizer cannot load Siri voices at all — it silently renders
   Samantha. NSSpeechSynthesizer with voice=nil renders the system voice,
   sample-identical to `say`, ~0.2 s per second of audio, warm. `say` is the
   same voice plus 0.6 s of init per call.
7. Caller audio off BlackHole 16ch is −42 dBFS. 24 dB gain + limiter before
   VAD/STT. Whisper loops ("I'm sorry" ×30) on ~1 s of noise → dropped.
8. **This is an 8 GB MacBook Air.** With Docker Desktop, Brave and the Hermes
   Electron app running it sat at load avg 20 and 3.7 GB swap; `say` took
   8 s, warm-up 35 s. Quit them. Nothing measured elsewhere transfers here.
9. launchd on macOS 26 cannot open a job's stdout under `~/Documents`
   (EX_CONFIG, no log). Service logs go to `~/Library/Logs/data-facetime-voice/`.
10. glm-5.3-flash via ollama-cloud with reasoning `none` leaks its thinking
    as untagged content — DATA speaks it. `low` is separated cleanly.

## Latency (live call 2026-09-11 15:58, Siri voice not yet in)

utterance end → first audio: **1.0–1.3 s** (STT 130–210 ms, LLM ~0.8 s,
TTS ~100 ms). With the Siri voice, TTS is 0.3–0.9 s per sentence
(simulator: first audio ~2.0 s). Pickup → greeting: ~5 s. A stall filler
("One moment, Captain.") covers provider hiccups past 3.5 s.

## Secrets

None in the repo. `FACETIME_BRIDGE_AUTHORIZED_CALLER_E164` and `_NAME` live in
the bridge daemon's launchd plist. Recordings, logs and transcripts are
gitignored.
