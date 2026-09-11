#!/usr/bin/env python3
"""Prove the bridge daemon's two audio legs with tones — no FaceTime call needed.

Leg 1 (PLAYBACK): we send a 1 kHz tone to the daemon as PLAYBACK packets; the
daemon renders it into BlackHole 2ch (FaceTime's mic). We tap BlackHole 2ch as
an INPUT with sounddevice and expect to see the tone there.

Leg 2 (CAPTURE): we play a 440 Hz tone into BlackHole 16ch as an OUTPUT with
sounddevice (standing in for FaceTime's speaker output); the daemon captures
BlackHole 16ch and streams it to us as CAPTURE packets. We expect the tone there.

Both legs are what a real call exercises; only the far ends (FaceTime) are
replaced by sounddevice. Exit 0 only when both legs carry signal.
"""
import os
import queue
import sys
import threading
import time

import grpc
import numpy as np
import sounddevice as sd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import facetime_media_pb2 as pb  # noqa: E402
import facetime_media_pb2_grpc as pbg  # noqa: E402

RATE = 24000
SOCK = "unix:" + os.path.expanduser(os.environ.get("FACETIME_BRIDGE_SOCKET", "~/.facetime-bridge/bridge.sock"))
TONE_S = 2.0


def tone(freq, seconds, rate, amp=0.5):
    t = np.arange(int(seconds * rate)) / rate
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


def dominant_freq(x, rate):
    if len(x) < 1024:
        return 0.0, 0.0
    spec = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    f = np.fft.rfftfreq(len(x), 1 / rate)
    return float(f[int(np.argmax(spec))]), float(np.sqrt(np.mean(x ** 2)))


def main():
    ch = grpc.insecure_channel(SOCK)
    stub = pbg.FaceTimeMediaStub(ch)
    h = stub.Health(pb.HealthRequest(), timeout=5)
    print(f"health ready={h.ready} in={h.input_device} out={h.output_device}")

    call_id = f"probe-{int(time.time())}"
    outq: "queue.Queue[pb.AudioPacket | None]" = queue.Queue()

    def request_iter():
        while True:
            pkt = outq.get()
            if pkt is None:
                return
            yield pkt

    def mk(kind, pcm=b"", seq=0):
        return pb.AudioPacket(call_id=call_id, kind=kind, pcm16=pcm,
                              sample_rate=RATE, channels=1, sequence=seq)

    outq.put(mk(pb.AUDIO_PACKET_KIND_START))
    responses = stub.Audio(request_iter())

    captured = []
    ready = threading.Event()
    done = threading.Event()

    def reader():
        try:
            for pkt in responses:
                if pkt.kind == pb.AUDIO_PACKET_KIND_EVENT:
                    print(f"  daemon event: {pkt.event!r}")
                    if pkt.event == "ready":
                        ready.set()
                elif pkt.kind == pb.AUDIO_PACKET_KIND_CAPTURE:
                    captured.append(np.frombuffer(pkt.pcm16, dtype=np.int16).astype(np.float32) / 32768.0)
        except grpc.RpcError as e:
            if not done.is_set():
                print(f"  stream error: {e.code()} {e.details()}")
        finally:
            done.set()

    threading.Thread(target=reader, daemon=True).start()
    if not ready.wait(10):
        print("FAIL: daemon never sent 'ready'")
        return 2
    print("audio stream open")

    # ---- Leg 1: PLAYBACK -> BlackHole 2ch ----------------------------------
    bh2_in = sd.query_devices("BlackHole 2ch")["index"] if isinstance(sd.query_devices("BlackHole 2ch"), dict) else None
    tap = []
    def on_bh2(indata, frames, t, status):
        tap.append(indata[:, 0].copy())
    with sd.InputStream(device="BlackHole 2ch", channels=2, samplerate=48000, callback=on_bh2):
        time.sleep(0.3)
        pcm = (tone(1000, TONE_S, RATE) * 32767).astype(np.int16)
        seq = 1
        for i in range(0, len(pcm), 480):  # 20 ms packets
            outq.put(mk(pb.AUDIO_PACKET_KIND_PLAYBACK, pcm[i:i + 480].tobytes(), seq))
            seq += 1
        time.sleep(TONE_S + 0.7)
    bh2 = np.concatenate(tap) if tap else np.zeros(0, dtype=np.float32)
    f1, rms1 = dominant_freq(bh2, 48000)
    leg1 = rms1 > 0.01 and abs(f1 - 1000) < 30
    print(f"LEG 1 PLAYBACK -> BlackHole 2ch: {'PASS' if leg1 else 'FAIL'}  "
          f"({len(bh2)/48000:.2f}s tapped, rms={rms1:.4f}, peak freq={f1:.0f} Hz, expected 1000)")

    # ---- Leg 2: BlackHole 16ch -> CAPTURE ----------------------------------
    captured.clear()
    stereo = np.zeros((int(TONE_S * 48000), 16), dtype=np.float32)
    t440 = tone(440, TONE_S, 48000)
    stereo[:, 0] = t440
    stereo[:, 1] = t440
    sd.play(stereo, samplerate=48000, device="BlackHole 16ch")
    sd.wait()
    time.sleep(0.5)
    cap = np.concatenate(captured) if captured else np.zeros(0, dtype=np.float32)
    f2, rms2 = dominant_freq(cap, RATE)
    leg2 = rms2 > 0.01 and abs(f2 - 440) < 30
    print(f"LEG 2 BlackHole 16ch -> CAPTURE: {'PASS' if leg2 else 'FAIL'}  "
          f"({len(captured)} packets, {len(cap)/RATE:.2f}s, rms={rms2:.4f}, peak freq={f2:.0f} Hz, expected 440)")

    done.set()
    outq.put(mk(pb.AUDIO_PACKET_KIND_STOP))
    outq.put(None)
    time.sleep(0.5)
    return 0 if (leg1 and leg2) else 1


if __name__ == "__main__":
    sys.exit(main())
