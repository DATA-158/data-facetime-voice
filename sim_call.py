#!/usr/bin/env python3
"""Simulated caller: play a recorded caller.wav into BlackHole 16ch (what FaceTime
would output) while recording BlackHole 2ch (what FaceTime's mic would carry
back to the caller). Run `voice_agent.py --simulate` first.

    python sim_call.py recordings/<call>/caller.wav [--seconds 60] [--out logs/sim_bh2.wav]

Prints when DATA's audio appears on BlackHole 2ch (energy > floor) with a
timestamp relative to playback start, so turn latency can be read straight off.
"""
import argparse
import sys
import threading
import time

import numpy as np
import sounddevice as sd
import soundfile as sf

ap = argparse.ArgumentParser()
ap.add_argument("wav")
ap.add_argument("--seconds", type=float, default=0, help="truncate caller audio")
ap.add_argument("--out", default="logs/sim_bh2.wav")
ap.add_argument("--tail", type=float, default=8.0, help="keep recording after playback ends")
args = ap.parse_args()

caller, sr = sf.read(args.wav, dtype="float32", always_2d=False)
if caller.ndim > 1:
    caller = caller.mean(axis=1)
if args.seconds:
    caller = caller[: int(args.seconds * sr)]
# up to 48 kHz for the device
n48 = int(len(caller) * 48000 / sr)
c48 = np.interp(np.linspace(0, len(caller) - 1, n48), np.arange(len(caller)), caller).astype(np.float32)
out = np.zeros((n48, 16), dtype=np.float32)
out[:, 0] = c48
out[:, 1] = c48

rec = []
t0 = time.monotonic()
last_state = False
def on_in(indata, frames, t, status):
    global last_state
    mono = indata[:, 0].copy()
    rec.append(mono)
    rms = float(np.sqrt(np.mean(mono ** 2)))
    loud = rms > 0.01
    if loud != last_state:
        print(f"{time.monotonic()-t0:7.2f}s  BH2 {'DATA speaking' if loud else 'silent'} (rms={rms:.3f})", flush=True)
        last_state = loud

with sd.InputStream(device="BlackHole 2ch", channels=2, samplerate=48000, blocksize=4800, callback=on_in):
    print(f"playing {len(caller)/sr:.1f}s of caller audio into BlackHole 16ch", flush=True)
    sd.play(out, samplerate=48000, device="BlackHole 16ch")
    sd.wait()
    print(f"{time.monotonic()-t0:7.2f}s  playback done; recording tail {args.tail}s", flush=True)
    time.sleep(args.tail)

audio = np.concatenate(rec)
sf.write(args.out, audio, 48000)
print(f"wrote {args.out} ({len(audio)/48000:.1f}s)")
