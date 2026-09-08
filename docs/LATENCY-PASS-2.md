# Latency passes #2 and #3 — engineering handover

Turn latency on FaceTime Audio calls went from **30 s+ of dead air** to a measured
**3.1–4.0 s on the live host** (1.34 s on the development host — see the note below on
why the two differ). Pass #3 targets what remained, and should take the live host to
roughly **2.0–2.9 s**.

> **Pass #3 is PR #3 (TTS) and PR #4 (speculative STT).** This document covers both.
> If you are reading it before those merge, the sections marked *pass #3* describe code
> that is not on `main` yet.

Most of that was not tuning. Four separate bugs were breaking calls outright, and one
of them meant a documented optimisation had **never once executed**. The 30 s+ turns
were eliminated by those fixes, not by the speedups.

This document records what was measured, what was wrong, what changed and why, and
what is still unproven. Read [Verified, and not](#verified-and-not) carefully — the
FaceTime-dependent paths have not been exercised against a live call.

Every number here was measured, not estimated. Where a figure is reconstructed from
separately-measured stages rather than observed as a single run, it says so.

> ### Two hosts, two sets of numbers — read this first
>
> The development benchmarks were taken on an M-series MacBook Air (the "bench host").
> DATA then deployed and re-measured on the **live host**, and the results differ in two
> ways that matter:
>
> - **The TTS speedup was not available on the live host after pass #2.** Its default
>   voice is a Siri voice, which `NSSpeechSynthesizer` cannot render, so the fidelity
>   check failed and the engine correctly stayed on `say`. **Pass #3 revisits exactly
>   this**: `AVSpeechSynthesizer` is a different engine that CAN reach Siri voices, and
>   it renders sample-identical audio. Whether it works on the live host is the single
>   open question — the startup log answers it.
> - **The live host runs `glm-5.3-flash` on both tiers**, per Captain directive. The LLM
>   figures below were taken against `minimax-m3:cloud`, so treat them as showing the
>   *shape* of the tool-schema and reasoning costs, not as predictions for that model.
>
> Where the two disagree, **the live host is the number that counts.**

| | Before | Bench host | Live host (DATA, verified) |
|---|---|---|---|
| Time to first audio (mean) | ~7.1 s best case | 1.34 s | **3.1–4.0 s** |
| Worst turn | 30 s+ observed | 1.72 s | — |
| Speech synthesis, per sentence | 665 ms | 21–37 ms | 665 ms after pass #2; **~70 ms if pass #3's AV backend loads the Siri voice** |
| Transcription | 2112 ms | 333 ms | **~500 ms** (was 1.6–2.1 s there) |

The honest headline for the live host is therefore **5–8 s → 3.1–4.0 s**, with the
worst-case 30 s+ turns eliminated by the bug fixes rather than by the speedups. DATA
attributes the remaining time to provider first-token variance on tool turns.

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
- [Verifying pass #3 on your host](#verifying-pass-3-on-your-host)

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
(`bench_turn.py`), worst of three turns, **on the bench host**.

On the live host the same budget resolves differently: STT lands at ~500 ms rather
than 250 ms, TTS stays at 665 ms because of the `say` fallback, and the LLM stage is
a different model — giving the measured 3.1–4.0 s mean. Endpointing is 700 ms on both.

**After pass #3** that budget is expected to become, on the live host:

| Stage | After pass #2 | After pass #3 |
|---|---:|---:|
| Endpointing | 700 ms | 700 ms — now the largest stage you control |
| STT | ~500 ms | **~0** — hidden inside the endpointing window |
| LLM | ~1200–2100 ms | unchanged |
| TTS | 665 ms | **~70 ms** if the AV backend loads the Siri voice, else 665 ms |

Roughly **3.1–4.0 s → 2.0–2.9 s**. These are projections: bench-host measurements
applied to DATA's verified live-host figures, not yet measured on the live host.

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

#### Three backends, tried in order

Pass #2 shipped only `NSSpeechSynthesizer`, and when its fidelity check failed on
the live host I concluded that Siri voices were simply unreachable and the speedup
was unavailable there. **That conclusion was wrong**, and pass #3 corrects it:
`NSSpeechSynthesizer` is the *legacy* API. `AVSpeechSynthesizer` is a different
engine, and it reaches premium and Siri voices.

| Backend | Per sentence | Siri voices | Notes |
|---|---:|---|---|
| `NSSpeechSynthesizer` | 21–37 ms | ✗ cannot load | fastest; tried first |
| `AVSpeechSynthesizer` | 62–77 ms | ✓ can | also streams (303 buffers) |
| `say` | 632–675 ms | ✓ | the floor |

Order is NS → AV → say. NS is fastest so it leads; AV catches exactly the hosts
where NS fails, which is the live host's case. Either way a host lands 10–30×
faster than `say`.

#### The fidelity gate is sample-level, not a file hash

**The voice does not change, and that is enforced rather than assumed.** At startup
each candidate backend renders a probe phrase, `say` renders the same phrase, both
are decoded to raw samples, and the backend is adopted ONLY if the audio matches:

```
77562 samples each · max sample error 0.000000 · correlation 1.000000
```

Pass #2 compared SHA-256 of the output *files*. That cannot work across backends:
`say` writes AIFF and `AVSpeechSynthesizer` emits raw float32 buffers, so a hash
reports a false mismatch on audio that is in fact identical — and it would equally
miss a silent voice *substitution* whenever the container happened to match.
Comparing decoded samples is correct in both directions.

If a backend differs in any way, or the voice cannot be loaded at all, it falls back
and logs why. There is no path in which the voice changes.

#### Two main-runloop traps, and why there is a helper process

Both fast APIs signal completion only on the MAIN runloop, and the voice loop's main
thread is busy running the service:

- `NSSpeechSynthesizer` schedules its **delegate** there, so on a worker thread the
  delegate never fires and every synthesis times out. Worked around by polling
  `isSpeaking()` instead, which needs no runloop.
- `AVSpeechSynthesizer`'s `writeUtterance:toBufferCallback:` has no polling
  equivalent — on a worker thread it produces **0 buffers**, every time.

Restructuring `voice_loop`'s threading around a runloop would be invasive in a
process that also owns gRPC streams and call lifecycle. So `AVSpeechSynthesizer`
lives in `tts_helper.py`: a small process whose main thread does nothing but pump a
runloop and synthesize, with stdin read on a background thread. Engine init is paid
once for the life of that process.

#### Fixed lines are pre-synthesized

The filler, failure and greeting lines are fixed strings spoken verbatim over and
over — and on a tool turn the filler **is** the first audio the caller hears.
Pre-synthesizing them at startup drops them to **0.00 ms**, measured. This helps on
every host regardless of which backend was adopted.

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

On the **live host** DATA measured MLX at **~500 ms**, down from 1.6–2.1 s on CPU there. Slower than the bench host's 333 ms, but the same 3–4× improvement.

Two things worth carrying forward. `cpu_threads` was a footgun — the obvious "make it
faster" knob made the old path 50 % slower. And moving STT to the GPU means it stops
competing with TTS and the rest of the pipeline for CPU.

Inference remains entirely local. The only network access is a one-time model download,
exactly as faster-whisper already did.

### Speculative STT — transcribing during the silence window *(pass #3)*

Endpointing waits `MIN_SILENCE_MS` (700 ms) of silence to decide the caller is done,
and only THEN starts transcribing. Those costs were serial for no reason: the silence
window is, by definition, time in which no new speech arrives.

Transcription now starts once the caller has been quiet for `DFV_SPECULATIVE_STT_MS`
(default 300 ms). By the time endpointing confirms at 700 ms the transcript is usually
already in hand.

```
before:  [speech] ---- 700ms silence ---- [STT 500ms] -> LLM
after:   [speech] ---- 700ms silence ---- -> LLM
                        └─ STT runs here ─┘
```

Worth ~250 ms on the bench host and ~500 ms on the live host, on every turn.

**Why it is safe, which matters more than the speedup here.** This is the most
race-prone code in the voice loop — a transcript is produced on one thread,
invalidated from the audio callback on another, and consumed on a third:

- The speculative window is pure silence. If speech resumes, `silence_run` resets and
  the speculation is invalidated immediately (epoch bump), so a stale half-utterance
  can never be spoken to.
- The finalized utterance has its trailing silence trimmed anyway, so the audio
  decoded speculatively is the same SPEECH the ordinary path decodes.
- The result is validated on sample count before use; a span mismatch over 120 ms
  re-transcribes for real.
- Worst case is one discarded transcript. There is no path where a wrong transcript
  reaches the agent.

One design detail worth understanding before changing it: `_finalize_utterance` runs
on the gRPC **capture thread** and must never block, so it only claims an epoch. The
waiting happens in `_process_turn`, which already has its own thread. That wait
(`DFV_SPECULATIVE_WAIT_MS`, 700 ms) collects an in-flight speculation rather than
abandoning it — the decode is already partway done, so finishing it is strictly
cheaper than starting a second one. On a host where STT outruns the remaining
endpointing window, that wait is the difference between this feature helping and it
costing double. **The live host's ~500 ms STT is exactly that case.**

`test_speculative_stt.py` drives the real `on_capture` path with synthetic packets and
asserts all three properties. Run it after any change to the VAD or endpointing.

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
| `tts_helper.py` | *(pass #3)* Hosts `AVSpeechSynthesizer` in its own process, because its buffer callback only fires on the main runloop. Self-tests with `--self-test`. |
| `test_speculative_stt.py` | *(pass #3)* Regression test for speculative STT; drives the real capture path. Exit 0 = pass. |

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
| Voice fidelity vs `say` | Verified | Sample-identical (max error 0.000000) for both fast backends; fallback path also exercised |
| Tier routing | Verified | 19/19 on fixtures |
| Bridge daemon under launchd | Verified | `HEALTH ready=True`, `PROBE ok=True state=idle` |
| Full stack warm start | Verified | Both services ran; STT, TTS and worker all reported ready |
| VAD endpointing on real call audio | **Unverified** | Needs a live call |
| Barge-in | **Unverified** | Needs a live call |
| Rewritten `_place_call_direct` | **Unverified** | Never run against a live dial — **watch this first** |

### Verified again on the live host

DATA re-ran verification after deploying (merge `193c808`) and confirmed independently:

- Syntax clean across all changed files.
- `AUTHORIZED_E164` double-getenv fixed, and **fail-closed confirmed** — an empty value
  refuses to dial rather than dialling `None`.
- Both FAST and FULL tiers resolve to `glm-5.3-flash` per the config default.
- TTS fidelity check **failed and correctly kept `say`** (Siri default voice).
- MLX STT installed and measured at ~500 ms.
- `bench_turn.py` on that host: **mean 3.1–4.0 s** time-to-first-audio.
- Service restarted and warm; STT `mlx` ready, TTS `say` ready, both workers warm.

The FaceTime-dependent rows above remain unverified on either host.

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
| `DFV_TTS_ENGINE` | `auto` | Force a backend: `ns`, `av` or `say`. Default tries NS → AV → say |
| `DFV_TTS_CACHE_MAX` | `64` | Max pre-synthesized phrases held in memory |
| `DFV_SPECULATIVE_STT_MS` | `300` | Silence before speculative transcription starts. **0 disables** |
| `DFV_SPECULATIVE_WAIT_MS` | `700` | How long a turn waits to collect an in-flight speculation |
| `DFV_GREETING_LINE` | *(DATA's greeting)* | Spoken when an outbound call connects |
| `DFV_TOOL_FILLER` | `"Let me check that, Captain."` | Must match the worker's value; the voice loop preloads it |
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

## Verifying pass #3 on your host

Three commands, none of which need a call:

```bash
python tts_engine.py            # which backend was adopted, and the fidelity proof
python tts_helper.py --self-test
python test_speculative_stt.py  # exit 0 = speculative STT is behaving
```

The line that matters most is at service start:

```
TTS engine: AVSpeechSynthesizer — voice fidelity verified vs `say`: sample-identical
```

- `AVSpeechSynthesizer` → the Siri voice IS reachable; TTS drops to ~70 ms.
- `NSSpeechSynthesizer` → also fine, and faster still (~25 ms).
- `say (the fallback floor)` → neither fast backend matched your voice. Nothing has
  changed for you, the voice is intact, and TTS stays at 665 ms. Not a failure —
  that is the gate doing its job.

Then per turn, `Captain: "…" (STT 0ms, speculative)` means the speculation is landing;
`(STT 500ms, live)` means it is not, and the threshold needs tuning on real audio.

---

## Still to do

1. **Make a live call before optimising further.** Nothing downstream of FaceTime's
   dial and answer has ever run — the rewritten `_place_call_direct`, barge-in, and VAD
   endpointing on real audio are all unproven. Two questions the logs will answer that
   no synthetic benchmark can: does `rms` on real FaceTime audio land anywhere near the
   0.006 trigger with `CAPTURE_GAIN=10`, and does the STT line read `speculative` or
   `live` when the Captain speaks at a natural pace? If it reads `live` every turn, the
   300 ms threshold needs real-call tuning and pass #3's STT win is not being realised.

2. **Endpointing, at a flat 700 ms, is now the largest stage anyone controls** — and
   the next big win is *semantic* endpointing: pass #3 already produces a transcript at
   300 ms, so it can decide whether the caller actually finished ("…around six?" is
   complete; "I was thinking about" is not) and finalize early. Worth ~300 ms on most
   turns, and it costs nothing extra because the transcript is already there.

   **Do not build this before a live call.** Getting it wrong means cutting the Captain
   off mid-sentence, and every VAD number in this project so far came from synthetic
   `say` audio through a fake capture path. Tune it on real-call telemetry.

3. **Tool turns still run 1–7 s behind the filler.** Trimming the voice toolset, or a
   second filler when the answer runs long, would close the gap.

4. **TTS on the live host is still 665 ms/sentence** — the largest single remaining
   stage there, and untouchable without changing the default voice. If that constraint
   ever relaxes, this is the biggest available win on that machine.

5. **A live call has not happened.** The three unverified rows above, in order of risk.

6. **`config.json` is read by nothing.** It advertises `distil-medium.en` and
   `Siri Voice 1`, neither of which is in effect. Either wire it up or delete it — as it
   stands it is actively misleading.
