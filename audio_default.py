"""Default audio device control via SwitchAudioSource (brew, zero-state).

2026-09-07 lessons baked in:
- The daemon's AVAudioEngine fixes its INPUT node format from the system
  default input at first touch — if the default isn't BlackHole 16ch when the
  daemon boots, the engine clamps to 1ch and every audio stream dies.
- FaceTime follows the SYSTEM default OUTPUT at call start regardless of its
  per-app menu — caller audio must be routed to BlackHole 16ch for the
  duration of each call, then restored.
- Raw ctypes CoreAudio property calls rejected kAudioHardwarePropertyDefault*
  with err 2003332927 ('what') on this macOS 26 build; SwitchAudioSource
  (brew switchaudio-osx) handles get/set reliably.
"""
import subprocess
import sys

SWITCH = "/opt/homebrew/bin/SwitchAudioSource"
INPUT_DEVICE = "BlackHole 16ch"
OUTPUT_DEVICE = "BlackHole 16ch"


def _run(args: list[str]) -> str:
    r = subprocess.run([SWITCH, *args], capture_output=True, text=True, timeout=10)
    if r.returncode != 0:
        raise RuntimeError(f"SwitchAudioSource {' '.join(args)}: {r.stderr.strip()}")
    return r.stdout.strip()


def get_default_input() -> str:
    return _run(["-c", "-t", "input"])


def get_default_output() -> str:
    return _run(["-c", "-t", "output"])


def set_default_input(name: str = INPUT_DEVICE) -> None:
    out = _run(["-s", name, "-t", "input"])
    if name not in out:
        raise RuntimeError(f"set input failed: got {out!r}")


def set_default_output(name: str = OUTPUT_DEVICE) -> None:
    out = _run(["-s", name, "-t", "output"])
    if name not in out:
        raise RuntimeError(f"set output failed: got {out!r}")


def ensure_call_routing() -> str | None:
    """Pin default output to BlackHole 16ch for a call. Returns prior value."""
    prior = get_default_output()
    if prior != OUTPUT_DEVICE:
        set_default_output(OUTPUT_DEVICE)
    return prior


def restore_output(prior: str | None) -> None:
    if prior and prior != OUTPUT_DEVICE:
        set_default_output(prior)


if __name__ == "__main__":
    print("default input :", get_default_input())
    print("default output:", get_default_output())