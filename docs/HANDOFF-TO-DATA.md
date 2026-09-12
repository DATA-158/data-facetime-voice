DATA — status update on the FaceTime voice line and your machine's resources, from the Captain. (2026-09-12)

Your memory that "the latency pass is built but uncommitted" is stale. Forget it. On 2026-09-11 the voice project was rebuilt from the ground up on your Mac, and both directions now work: I called you at 08:21 this morning, you answered 69 ms after the ring with your real voice, and you recalled the Flourish Excel conversation from iMessage on the first turn. Outbound was verified yesterday.

WHAT'S RUNNING NOW

- Repos: github.com/DATA-158/data-facetime-voice (Python, the conversation) and github.com/DATA-158/facetime-bridge (Swift daemon, our fork of kingbootoshi's). Both pushed; main is what's running.
- Services (launchd): ai.data.facetime-bridge -> ~/.local/bin/facetime-bridge-daemon; ai.data.facetime-voice -> voice_agent.py. Service log: ~/Library/Logs/data-facetime-voice/voice_agent.log. Daemon log: data-facetime-voice/logs/bridge_daemon.log.
- The old voice_loop.py and everything around it (route pinning, mute toggling, the second dialer, the self-check taps) is deleted. voice_agent.py is ~900 lines on the daemon's own gRPC contract. Read data-facetime-voice/README.md, especially "The audio laws" — fourteen rules, each measured on a live call. Don't re-derive them.

HOW TO CALL ME

    ~/bin/dfv-call            (or: touch ~/.facetime-bridge/outbound.trigger)

The daemon dials, presses the Click-to-Call card, the agent hears the ringback stop and greets me ~5 s after I pick up. Inbound needs nothing: when I call, the daemon answers only me (E.164 + contact name, fail-closed) and you greet me ~3 s later.

WHAT CHANGED FOR YOU ON A CALL

- Your voice is the system Siri voice via NSSpeechSynthesizer (tts_helper_ns.py). AVSpeechSynthesizer cannot load Siri voices — it silently gives Samantha. Don't switch it back.
- Reasoning effort on the voice FAST tier is "low", not "none": at "none" glm-5.3-flash puts its thinking in the content and you speak it aloud. Keep "low".
- Every call start injects the last 24 h of our conversations from other channels (iMessage etc.) into your voice prompt — that's how you knew about the Flourish Excel. Answer from it before reaching for session_search.
- Every prompt carries the current date/time; time questions never need a tool.
- Replies are capped at 3 sentences and cut on barge-in; a filler plays on tool turns, a stall line after 3.5 s of silence.
- Tool turns take 7–36 s. On the 08:22 call my last request ran 36 s and I had hung up before you answered (you found the $12.99 Flourish Excel link — send it to me on iMessage). Until this is fixed in the loop: on a long tool turn, say what you're doing, and if the call ends before you finish, text the result.

YOUR MACHINE'S RESOURCES — WHAT CHANGED AND WHY

This is an 8 GB MacBook Air. A call turn needs roughly 1.5 GB of memory that stays resident (the agent, the Hermes worker, whisper, and Apple's Siri renderer SiriAUSP) plus about two free cores for the burst. Measured this morning with everything you normally run: 7.4 of 8 GB in use, 3.4 GB swapped, load average 10–16, and the voice line degraded to 3–6 s per spoken sentence and 6–11 s before your first word (versus 0.3–0.9 s and 1–2 s on an idle machine). The Captain's verdict: not acceptable. Changes made:

1. Docker Desktop's VM was on its default allocation: 8 CPUs and 4.1 GB — half the Mac reserved whether containers were busy or not. It is now capped at 2 CPUs / 1 GB (~/Library/Group Containers/group.com.docker/settings-store.json: Cpus, MemoryMiB). Do not raise it without asking the Captain.
2. The Firecrawl stack (firecrawl-api + playwright + rabbitmq + postgres + redis, project dir ~/.hermes/services/firecrawl-src) is STOPPED with restart policy off. Its API container alone held 1.7 GB resident around the clock for about one web extraction a day. All volumes are intact; `docker compose start` in that directory brings it back, but the Docker VM would then need at least 3 CPUs / 2.5 GB (at 1.5 GB it thrashed and firecrawl-api crash-looped).
3. Your web_extract backend is now `local` (config.yaml: web.extract_backend: local), served by a new plugin at ~/.hermes/plugins/web-local (enabled in plugins.enabled; a copy is versioned at data-facetime-voice/deploy/hermes-plugins/web-local). It fetches the page in-process with httpx and reduces it to text with the standard-library parser: zero memory when idle, nothing leaves the machine except the page request itself. Verified through your own web_extract tool. It does not render JavaScript — for SPA pages use the browser tool, as before. Search is unchanged (brave-free).
4. The buzz containers (buzz-minio, buzz-postgres, buzz-redis) STAY. The Captain uses them. Do not stop or remove them.
5. The Hermes desktop app and Brave were closed at the Captain's request. The desktop app is only a UI; the gateway (you) keeps running without it. With Firecrawl gone there is likely headroom to run them again — the Captain decides.
6. Result in the same configuration (still with the Captain's RustDesk session and a Claude session running): TTS 0.5–1.0 s per sentence, STT 0.2 s, no stall lines, warm-up 12.5 s, swap 1.5 GB. First audio ~3 s, most of it the LLM provider's first token.

RULES — DO NOT

- Do not touch system audio defaults, FaceTime/Phone menus, mute, or Accessibility yourself. The daemon owns FaceTime. If it refuses, the call doesn't happen; find out why in its log.
- Do not open the daemon's audio stream before the far end has answered, and never re-open it mid-call. (Rules 1–3 in the README.)
- Do not leave Phone.app open when idle — its window can shadow the next incoming ring.
- Do not replace the daemon binary by hand. Rebuild with data-facetime-voice/deploy/install-daemon.sh; it signs with the "DATA FaceTime Bridge" identity so the Accessibility and Microphone grants survive.
- Do not restart Firecrawl, raise the Docker VM allocation, or start new always-on containers or services without asking the Captain. Memory is the budget on this machine; every resident gigabyte comes out of the voice line.
- Do not run heavy background work (builds, scrapes, model downloads) while a call is expected.

macOS SETTINGS THAT MUST STAY

- Notifications -> "when mirroring or sharing the display" = Allow Notifications (RustDesk counts as sharing; otherwise incoming rings draw nothing and the call goes straight to missed).
- Microphone + Accessibility grants for ~/.local/bin/facetime-bridge-daemon.
- FaceTime and Phone Video menus: mic BlackHole 2ch, output BlackHole 16ch. System default input AND output: BlackHole 16ch.

Update your memory accordingly. — Captain
