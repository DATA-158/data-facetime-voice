# Latency pass #2 — engineering handover

Turn latency on FaceTime Audio calls went from **30 s+ of dead air** to **~1.4 s**.

Most of that was not tuning. Four separate bugs were breaking calls outright, and one
of them meant a documented optimisation had **never once executed**.

This document records what was measured, what was wrong, what changed and why, and
what is still unproven. Read [Verified, and not](#verified-and-not) carefully — the
FaceTime-dependent paths have not been exercised against a live call.

Every number here was measured on an M-series MacBook Air, not estimated. Where a
figure is reconstructed from separately-measured stages rather than observed as a
single run, it says so.

| | Before | After |
|---|---|---|
| Time to first audio (mean) | ~7.1 s best case | **1.34 s** |
| Worst turn | 30 s+ observed | **1.72 s** |
| Speech synthesis, per sentence | 665 ms | **25–37 ms** |
| Transcription | 2112 ms | **333 ms** |

---

## Contents

- [The latency budget](#the-latency-budget)
- [How this was measured](#how-this-was-measured)
- [Four bugs](#four-bugs)
- [The performance work](#the-performance-work)
- [The new structure](#the-new-structure)
- [Verified, and not](#verified-and-not)
- [Deploying and operating](#deploying-and-operating)
- [Still to do](#still-to-do)

---

## The latency budget

The pipeline is strictly serial — nothing starts until the previous stage finishes —
so time-to-first-audio is the sum of four stages, and the only question worth asking
is which one owns the milliseconds.

```
                 0s        2s        4s        6s
                 |---------|---------|---------|
BEFORE  7067 ms  ####  ############  ####################  ####
                 endpt      STT            LLM              TTS
                 700       2112           3590              665

AFTER   1718 ms  ####  #  ####  .
                 endpt STT  LLM  TTS
                 700   225  751   42
```

| Stage | Before | After | What changed |
|---|---:|---:|---|
| Endpointing | 700 ms | 700 ms | Untouched — now the largest remaining stage |
| STT | 2112 ms | ~250 ms | MLX GPU `whisper-small.en` replaces CPU int8 |
| LLM → first chunk | 2080–4690 ms | 265–988 ms | Two-tier tools; reasoning off for chat; clause chunking |
| TTS | 665 ms | 25–70 ms | One persistent synthesizer instead of `say` per sentence |
| **Total** | **7067 ms** | **1718 ms** | Mean across three turns: **1339 ms** |

`BEFORE` is reconstructed from per-stage measurements. `AFTER` is measured end to end
(`bench_turn.py`), worst of three turns.

### On the 30-second turns

The 7 s budget above is the *healthy* path. The 30 s+ turns came from two things on
top of it:

- A tool turn streams nothing until its whole agent loop finishes — measured at 7.4 s
  to first token in one case.
- A worker error produced total silence rather than a slow reply.

Both are addressed below.

---

## How this was measured

Three harnesses, all committed so they can be re-run on your host rather than taken on
faith:

| Harness | What it does |
|---|---|
| `bench_turn.py` | Drives a full turn — utterance audio → STT → the real persistent worker → chunking → TTS — and reports true time-to-first-audio. Needs no FaceTime call. |
| `python tts_engine.py` | Self-tests the synthesizer: proves voice fidelity against `say`, then benchmarks synthesis across sentence lengths. |
| `python stt_engine.py` | Reports which backend was selected and times three transcriptions of a fixed utterance. |

The LLM stage was additionally profiled directly against the Hermes agent and against
the raw Ollama endpoint, which is how the tool-schema and reasoning costs were
separated from the model's own speed.

---

## Four bugs

Numbered because they are a closed set of four, not because they happen in sequence.
Each independently broke calls.

### 01 — The authorised number was always `None`

> Symptom: *"telling DATA to call me takes minutes"*

A double lookup. It read the environment variable **named by** the phone number:

```python
AUTHORIZED_E164 = os.environ.get(
    os.environ.get("FACETIME_BRIDGE_AUTHORIZED_CALLER_E164", ""))
# -> None, always
```

Two consequences, both confirmed by running the expression:

1. The dial URL became the literal string `facetime-audio://None`.
2. `_call_timer_running()` passed that `None` into `subprocess(env=…)`, which raises
   `TypeError` — swallowed by a bare `except`. It therefore returned `False` **every
   single time**, connection was never detected, and every outbound call fell through
   to the blind 60 s and 90 s timeout branches.

The voice plist also never set the variable at all, so **the code fix alone is not
sufficient** — see [Deploying](#deploying-and-operating).

### 02 — A failed turn produced silence, and logs that looked fine

> Symptom: *"…IF he starts speaking"*

On a worker error the final content was empty, so every speak branch was skipped — yet
the transcript still recorded *"I'm here, Captain, but my response came back empty."*
The log read like a completed turn while the caller heard nothing at all.

There is now an audible fallback on every path, including when the worker dies
mid-stream. An audible failure beats dead air on a live call.

Note the two distinct flags in `llm_reply_streaming`: `spoke_reply` gates whether the
model's *actual answer* has been voiced, `spoke_anything` gates the never-go-silent
guarantee. Conflating them would let a filler suppress the real answer on a tool turn.

### 03 — `DFV_REASONING_EFFORT` had never once taken effect

> Symptom: unexplained multi-second first-token delays

The worker mutated a config dict that was never passed to the agent:

```python
cfg = load_config()
cfg["agent"]["reasoning_effort"] = "low"   # discarded
_agent = AIAgent(model=…, provider=…)      # cfg never passed
```

`load_config()` returns a fresh dict on every call (verified), so the mutation went
nowhere and every voice turn ran at the config default, `medium`.

This matters more than it sounds: reasoning tokens are emitted **before** any content,
so on a phone call they are pure dead air.

| `reasoning_effort` | Thinking emitted | First content |
|---|---:|---:|
| unset | 418 chars | 0.74 s |
| `low` | 237 chars | 0.55 s |
| `none` | 0 chars | **0.25 s** |

Now applied through the real `reasoning_config` constructor argument, and set **per
tier**: off for conversation, `low` for tool turns where deliberation genuinely helps
tool choice.

### 04 — DATA was running on a degraded persona

> Symptom: subtly off-character replies

`hermes_worker` imported `voice_loop` purely to read the `SYSTEM_CONTEXT` string —
which drags in grpc, numpy, soundfile and pyobjc as a side effect. The worker runs
under a **different interpreter** to the voice loop, and that venv has none of them:

```
grpc MISSING · numpy MISSING · soundfile MISSING · AppKit MISSING
  -> voice_loop import FAILED -> degraded fallback persona
```

The import raised, the exception was caught, and DATA silently ran a stub persona.

Nothing about a system prompt needs an audio stack, so it now lives in
`voice_persona.py` with zero dependencies. `HERMES_VENV` is env-configurable too,
instead of a hardcoded path that breaks on any other machine.

---

## The performance work

### Speech synthesis: 665 ms → 25 ms, same voice

`say` pays a fixed speech-engine initialisation on **every** invocation, independent of
text length. Ten consecutive identical runs:

```
687  663  718  677  668  660  664  677  662  654   ms
bare subprocess spawn, for comparison:              4 ms
```

So it is not fork cost, and it cannot be prewarmed across processes — it is engine
startup inside `say`, and it landed on the first sentence of every single turn. One
long-lived `NSSpeechSynthesizer` does the same sentence in 25–37 ms after a one-off
411 ms first call.

**The voice does not change, and that is enforced rather than assumed.** At startup
`tts_engine` renders a probe phrase through both paths and compares SHA-256:

```
say        : 159220 bytes  sha256=f0ad702a7dfdf1af…
persistent : 159220 bytes  sha256=f0ad702a7dfdf1af…
IDENTICAL AUDIO
```

If they ever differ — or the voice cannot be loaded at all — it falls back to `say`
permanently and logs why.

> **Siri voices are restricted from `NSSpeechSynthesizer`.** If your default is one,
> expect the fidelity check to fail: you keep your voice exactly as-is and simply forgo
> the speedup on that host. Confirm which branch you land on from the log.

One implementation note worth keeping: completion is detected by polling
`isSpeaking()`, **not** the delegate callback. `NSSpeechSynthesizer` schedules its
delegate on the main runloop, so a delegate never fires on a worker thread and every
synthesis times out. This cost a debugging cycle.

### Transcription: 6.3× faster on the GPU

All timings against the same 6.79 s utterance, best of three:

| Backend | Model | Time | Note |
|---|---|---:|---|
| faster-whisper, CPU | `distil-small.en` | 2112 ms | the previous default, 6 threads |
| faster-whisper, CPU | `distil-small.en` | 3167 ms | 10 threads — *slower*, oversubscribed |
| faster-whisper, CPU | `distil-medium.en` | 5849 ms | README suggested 2.3 s; it is not |
| MLX, GPU | `whisper-base.en` | 125 ms | available if you want more headroom |
| **MLX, GPU** | **`whisper-small.en`** | **333 ms** | **now the default** — and more accurate |
| MLX, GPU | `distil-large-v3` | 1019 ms | still beats the old CPU path |

Two things worth carrying forward. `cpu_threads` was a footgun — the obvious "make it
faster" knob made the old path 50 % slower. And moving STT to the GPU means it stops
competing with TTS and the rest of the pipeline for CPU.

Inference remains entirely local. The only network access is a one-time model download,
exactly as faster-whisper already did.

### Two-tier tools

The six production toolsets are 10 tools and roughly **6046 tokens of JSON schema**
attached to every request — including *"you there?"*, which is most of them.
First-token latency across four consecutive turns, same model, same prompt:

| Configuration | Turn 1 | 2 | 3 | 4 |
|---|---:|---:|---:|---:|
| With all 6 toolsets | 2.08 s | 4.69 s | 3.59 s | 0.68 s |
| With no toolsets | 0.40 s | 0.65 s | 0.87 s | 1.62 s |

The worker now holds **two warm agents**:

- **FAST** — no toolsets. Handles conversation, streams from the first token.
- **FULL** — every production toolset. Handles anything that acts.

Routing is a local regex heuristic in `needs_tools()`; an extra classifier model call
would cost more latency than the tool schema it avoids. It scores 19/19 on the
fixtures, including the trap that *"heading out for a run this evening"* is
conversation while *"run the test suite"* is not — bare `run` was originally in the
keyword list and sent small talk down a 6.5 s path.

Both agents are mirrored the other's turns (`_mirror_turn`), so DATA keeps one
continuous thread when a turn switches tier.

**Tool turns now speak immediately.** A tool turn emits no content deltas until its
whole loop finishes, so it used to be pure silence for as long as that took. The worker
emits a filler line the instant a turn routes to FULL, and the voice loop speaks it.
Dead air on a tool turn: **~1.0 s instead of 7 s+**.

### Three smaller ones

- **First chunk breaks at a clause, not a sentence.** A reply opening *"Honestly,
  Captain, I can't argue with the logic — an evening run clears the head."* held all
  audio for 3.8 s waiting on the first period. With TTS at ~40 ms there is no reason to
  wait. Only the first chunk of a turn does this; later chunks wait for real sentence
  boundaries, which reads better and costs nothing because playback is already ahead of
  synthesis. Guarded against splitting `3.5` and `1,000`.

- **The dial loop's AX traversal is bounded.** It ran an AppleScript
  `entire contents of` walk over every Notification Center element **every second**
  until it succeeded — one of the slowest calls in the accessibility API, hammering AX
  exactly while the call was trying to come up. Now 3 attempts inside the window where
  the prompt actually appears, with `pgrep` (4 ms) as the fast poll.

- **A single-flight turn guard.** Turns were spawned per finalised utterance with no
  mutual exclusion, so two quick utterances ran concurrently, queued overlapping speech
  into the same stream, and each cleared the *other's* barge-in flag.

---

## The new structure

The shape of the change is one idea: **things that were tangled by import are now
separated by responsibility.** Synthesis, transcription and persona each own a module,
each degrades independently, and each self-tests.

### New files

| File | Responsibility |
|---|---|
| `tts_engine.py` | All synthesis. Persistent `NSSpeechSynthesizer` on a dedicated worker thread, the voice-fidelity proof, and permanent `say` fallback. |
| `stt_engine.py` | All transcription. Selects MLX GPU, falls back to faster-whisper CPU, prewarms at service start. API-compatible with the old inline `STT` class. |
| `voice_persona.py` | Just `SYSTEM_CONTEXT`, and deliberately nothing else. Zero imports, so both interpreters can read it. This is the fix for bug 04. |
| `bench_turn.py` | The end-to-end latency harness. Uses the real splitter from `voice_loop` so it measures the same chunking production performs. |

### Changed files

| File | Change |
|---|---|
| `voice_loop.py` | Still the service and call lifecycle, but no longer implements synthesis or transcription. Gains the never-go-silent guarantee, the single-flight guard, clause-aware chunking, fixed E.164 handling, and the rewritten dial loop. |
| `hermes_worker.py` | Two warm agents instead of one, per-tier reasoning config, the routing heuristic, and filler emission on tool turns. |
| `deploy/ai.data.facetime-voice.plist` | Now actually declares `FACETIME_BRIDGE_AUTHORIZED_CALLER_E164`, which it never did. |

### One turn, end to end

```
1  VAD          endpoint utterance            700 ms
2  stt_engine   transcribe (GPU)             ~250 ms
3  worker       route tier (local regex)       <1 ms
4  agent        stream deltas             265–988 ms
5  voice_loop   pop chunk (clause or sentence)
6  tts_engine   synthesize                  25–70 ms
7  bridge       playback (24 kHz PCM)
```

On a tool turn, step 3 emits the filler and step 6 speaks it immediately, while step 4
continues in the background. Steps 5–7 then repeat per chunk; because synthesis
(~40 ms) is far shorter than the audio it produces (1.7–10 s), everything after the
first chunk is fully hidden behind playback.

### Worker protocol

```
stdin :  {"prompt": "...", "stream": true}
stdout:  {"status": "ready"}                        once, at startup
         {"filler":  "..."}                         tool turns only, emitted first
         {"delta":   "..."}                         streamed content
         {"content": "...", "tier": "fast|full"}    terminal message
         {"error":   "...", "tier": "..."}          terminal message
```

The terminal message always arrives. `llm_reply_streaming` reads to it every time —
stopping early leaves lines in the pipe that the *next* turn reads as its own.

---

## Verified, and not

> The test host shares one Apple ID and one phone number with the callee, and
> **FaceTime cannot call itself** — one account is one IDS identity, so there is no
> device-to-device call within it. Everything downstream of FaceTime was exercised;
> FaceTime itself was not.

| Area | Status | Evidence |
|---|---|---|
| STT / LLM / TTS / chunking | Verified | Component benchmarks + `bench_turn.py`, reproducible |
| Voice fidelity vs `say` | Verified | Byte-identical SHA-256; fallback path also exercised |
| Tier routing | Verified | 19/19 on fixtures |
| Bridge daemon under launchd | Verified | `HEALTH ready=True`, `PROBE ok=True state=idle` |
| Full stack warm start | Verified | Both services ran; STT, TTS and worker all reported ready |
| VAD endpointing on real call audio | **Unverified** | Needs a live call |
| Barge-in | **Unverified** | Needs a live call |
| Rewritten `_place_call_direct` | **Unverified** | Never run against a live dial — **watch this first** |

### A false alarm you will hit

`facetime-bridge doctor` reports `accessibility: unavailable` even when the grant is
correct. The helper is ad-hoc/linker-signed with no Team ID, so a shell-launched
invocation is attributed by TCC to the parent terminal, not the binary. Under launchd
it is its own responsible process and works — confirmed by querying the daemon over
gRPC. Do not chase this.

---

## Deploying and operating

### 1 — Apply

```bash
cd ~/Documents/Programming/Datas_Projects/data-facetime-voice
git fetch origin pull/1/head:latency-pass-2 && git checkout latency-pass-2
```

### 2 — Two required config changes

In `~/Library/LaunchAgents/ai.data.facetime-voice.plist`, inside
`EnvironmentVariables`. The first is the actual fix for the dialling delay — the code
fix alone is not sufficient, because the plist never declared it:

```xml
<key>FACETIME_BRIDGE_AUTHORIZED_CALLER_E164</key>
<string>+1XXXXXXXXXX</string>
<key>HERMES_VENV</key>
<string>/Users/YOURUSER/agent-calling/.venv-native-tts/bin/python</string>
```

### 3 — Two new dependencies

```bash
~/agent-calling/.venv-native-tts/bin/pip install mlx-whisper pyobjc-framework-Cocoa
```

Skipping this breaks nothing — both engines fall back safely to the old
implementations. You simply do not get the speedups.

### 4 — Restart and watch

```bash
launchctl kickstart -k gui/$(id -u)/ai.data.facetime-voice
tail -f logs/voice_loop.log
```

| Log line | Means |
|---|---|
| `STT engine: MLX GPU model=…` | Good. If it says `faster-whisper CPU`, mlx-whisper did not install. |
| `voice fidelity verified vs 'say': identical` | Good — you have the 665 ms → 25 ms win. |
| `TTS voice fidelity check FAILED … keeping 'say'` | **Expected if your default is a Siri voice.** Voice preserved, no TTS speedup. |
| `streaming turn [fast tier]` | Conversation took the no-tool path — this should be the common case. |
| `streaming turn [full tier]` | Tool path. A filler was spoken first. |
| `turn produced no speakable text …` | Bug 02's guard fired. The caller heard a fallback; investigate the cause. |

### 5 — Check without calling

```bash
~/agent-calling/.venv-native-tts/bin/python bench_turn.py
~/agent-calling/.venv-native-tts/bin/python tts_engine.py
~/agent-calling/.venv-native-tts/bin/python stt_engine.py
```

### Environment variables

All optional except the two in step 2.

| Variable | Default | Effect |
|---|---|---|
| `DFV_TTS_ENGINE` | `auto` | `say` forces the old path; `native` skips the fidelity self-test |
| `DFV_TTS_VOICE` | *(empty)* | Empty = system default voice. Leave it alone unless you mean it. |
| `DFV_STT_ENGINE` | `auto` | `mlx` or `faster` to pin a backend |
| `DFV_STT_MODEL` | `whisper-small.en-mlx` | Any MLX repo or faster-whisper name |
| `DFV_STT_THREADS` | `6` | faster-whisper CPU threads. Do not raise — see above. |
| `DFV_TOOL_TIER` | `auto` | `always` or `never` to override routing |
| `DFV_TOOL_FILLER` | `"Let me check that, Captain."` | Spoken when a turn routes to the tool tier |
| `DFV_TURN_FAILED_LINE` | `"Captain, I lost that one. Say again?"` | Spoken when a turn yields nothing sayable |
| `DFV_REASONING_EFFORT_FAST` | `none` | Reasoning for conversational turns |
| `DFV_REASONING_EFFORT` | `low` | Reasoning for tool turns |
| `DFV_MIN_FIRST_CHUNK` | `24` | Chars before the first chunk may break at a clause |
| `DFV_SILENCE_MS` | `700` | Endpointing window |
| `DFV_TRIGGER_POLL_S` | `0.25` | Outbound trigger poll (was 2.0) |

---

## Still to do

1. **Endpointing is now the largest stage** at a flat 700 ms — 52 % of a good turn.
   Silero VAD is already a dependency and would support ~400–500 ms safely. The
   existing note that 500 ms fragments speech was measured against the *energy* VAD, so
   it does not rule this out.

2. **Speculative STT.** Begin transcribing at the onset of silence rather than after it
   is confirmed, and STT leaves the critical path almost entirely.

3. **Tool turns still run 1–7 s behind the filler.** Trimming the voice toolset, or a
   second filler when the answer runs long, would close the gap.

4. **A live call has not happened.** The three unverified rows above, in order of risk.

5. **`config.json` is read by nothing.** It advertises `distil-medium.en` and
   `Siri Voice 1`, neither of which is in effect. Either wire it up or delete it — as it
   stands it is actively misleading.
