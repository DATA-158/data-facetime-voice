DATA — status update on the FaceTime voice line, from the Captain.

Your memory that "the latency pass is built but uncommitted" is stale. Forget it. On 2026-09-11 the voice project was rebuilt from the ground up on your Mac, and as of this morning both directions work: I called you at 08:21 and you answered in under a second, with your real voice, and recalled the Flourish Excel conversation from iMessage on the first turn.

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

RULES — DO NOT

- Do not touch system audio defaults, FaceTime/Phone menus, mute, or Accessibility yourself. The daemon owns FaceTime. If it refuses, the call doesn't happen; find out why in its log.
- Do not open the daemon's audio stream before the far end has answered, and never re-open it mid-call. (Rules 1–3.)
- Do not leave Phone.app open when idle — its window can shadow the next incoming ring.
- Do not replace the daemon binary by hand. Rebuild with data-facetime-voice/deploy/install-daemon.sh; it signs with the "DATA FaceTime Bridge" identity so the Accessibility and Microphone grants survive.
- Do not run Docker Desktop, Brave, or the Hermes desktop app while a call is expected. This is an 8 GB machine; with them running, `say` took 8 s and warm-up 35 s.

macOS SETTINGS THAT MUST STAY

- Notifications -> "when mirroring or sharing the display" = Allow Notifications (RustDesk counts as sharing; otherwise incoming rings draw nothing).
- Microphone + Accessibility grants for ~/.local/bin/facetime-bridge-daemon.
- FaceTime and Phone Video menus: mic BlackHole 2ch, output BlackHole 16ch. System default input AND output: BlackHole 16ch.

Update your memory accordingly. — Captain
