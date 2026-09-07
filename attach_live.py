#!/usr/bin/env python3
"""Attach the DATA voice loop to the CURRENTLY LIVE outbound call."""
import sys
import time
import threading
import subprocess
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import grpc
import facetime_media_pb2 as pb
import facetime_media_pb2_grpc as pb_grpc
import voice_loop as vl


def phone_alive() -> bool:
    return subprocess.run(["pgrep", "-x", "Phone"], capture_output=True).returncode == 0


def main() -> int:
    if not phone_alive():
        print("no live call surface (Phone process absent)")
        return 1
    channel = grpc.insecure_channel(f"unix:{vl.SOCKET_PATH}")
    grpc.channel_ready_future(channel).result(timeout=10)
    stub = pb_grpc.FaceTimeMediaStub(channel)
    # 2026-09-07: warm the LLM worker in PARALLEL with whisper loading — the
    # original serial order meant Captain's first words hit a 28s cold worker.
    import threading as _t
    warm_done = {}
    def _warm_async():
        try:
            vl._warm_worker()
            warm_done["ok"] = True
        except Exception as e:
            print("worker warm failed:", e)
    _t.Thread(target=_warm_async, daemon=True).start()
    vl._init_models()
    call_id = f"dfv-attach-{int(time.time())}"
    session = vl.CallSession(stub, call_id)

    # 2026-09-07 (15:00 call): FaceTime locks its output route at CALL START.
    # The attach flips the default AFTER dialing — too late; caller audio went
    # to the speakers again (watchdog: 20s, max RMS 0.004). Routing must be
    # pinned BEFORE the green button. Here we're already mid-call: flip
    # immediately (may or may not re-route live) and warn loudly if capture
    # stays dead — the pre-dial pin is the real fix (data-facetime-voice-pin).
    saved_output = None
    try:
        import audio_default
        saved_output = audio_default.get_default_output()
        if saved_output != "BlackHole 16ch":
            audio_default.set_default_output("BlackHole 16ch")
            print(f"default output {saved_output!r} -> BlackHole 16ch (mid-call — may be late)")
    except Exception as e:
        print("audio_default routing failed:", e)

    try:
        t = threading.Thread(target=session.audio_loop, daemon=True)
        t.start()
        print("attached; greeting")

        def greet():
            time.sleep(0.8)
            try:
                session.playing.set()
                vl.tts_sentences(
                    vl.split_sentences(
                        "Captain, DATA here. The line is live — go ahead."
                    ),
                    emit=session._emit_speech,
                    cancelled=lambda: session.barge_in.is_set(),
                )
            except Exception as e:
                print("greet failed:", e)
            finally:
                session.playing.clear()
                session.barge_in.clear()

        threading.Thread(target=greet, daemon=True).start()

        gone = 0
        while True:
            time.sleep(2.0)
            if phone_alive():
                gone = 0
            else:
                gone += 1
                if gone >= 3:
                    print("call ended (Phone surface gone)")
                    break
    finally:
        # Captain's standing requirement (2026-09-07): call content must
        # survive the hangup — transcript + memory entry, every call.
        try:
            if vl._dialogue:
                vl.persist_call_memory(list(vl._dialogue), call_id)
                with vl._dialogue_lock:
                    vl._dialogue.clear()
        except Exception as e:
            print("post-call persist error:", e)
        if saved_output and saved_output != "BlackHole 16ch":
            try:
                import audio_default
                audio_default.set_default_output(saved_output)
                print(f"default output restored to {saved_output!r}")
            except Exception as e:
                print("default output restore failed:", e)
    session.writer.put(("STOP", None))
    time.sleep(1.0)
    print("attach session done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
