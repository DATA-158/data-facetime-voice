#!/usr/bin/env python3
"""Regression tests for the issue #6 live-call placement fixes.

Three failures on the 2026-09-08 live call, all fixed here:
  1. the Click-to-Call prompt was never pressed (System Events cannot see
     NC banner buttons on macOS 26) — now pressed via ax2 --ax-press,
     with a frames-snapshot + cliclick fallback;
  2. BLIND_CONNECT_AFTER_S returned 'connected' 12s in while the prompt
     was still unanswered — removed; connect now requires proof;
  3. the call ran 85s mic-muted — now classified and force-unmuted
     immediately after connect.

Everything subprocess-facing is stubbed via unittest.mock.patch, so this
test never touches AX, cliclick, or the bridge binary. Importing
voice_loop is MLX-free (verified: whisper loads lazily inside STT).

Run:  python test_connect_proof.py     (exit 0 = pass)
"""
import json
import os
import sys
import unittest
from unittest import mock

os.environ.setdefault("DFV_CAPTURE_GAIN", "1.0")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import voice_loop as vl  # noqa: E402  (must not load MLX — asserted below)

NC = "Notification Center"
AX2 = "/Users/fake/.local/bin/facetime-bridge-ax2"


def ax_env_ok(kwargs):
    """The authorized-caller env contract must ride on every ax2 call."""
    return "FACETIME_BRIDGE_AUTHORIZED_CALLER_E164" in (kwargs.get("env") or {})


def run_result(stdout="", returncode=0, stderr=""):
    r = mock.Mock()
    r.stdout = stdout
    r.stderr = stderr
    r.returncode = returncode
    return r


# ---- fixtures -------------------------------------------------------------
# A stale banner (older, disabled) plus the fresh 'Call' prompt.
PROMPT_FRAMES = [
    {"bundleID": "com.apple.nc", "process": NC, "role": "AXGroup",
     "identifier": "banner-stale", "enabled": False,
     "texts": ["FaceTime Audio", "Call"], "actions": ["AXPress"],
     "frame": {"x": 100, "y": 100, "w": 200, "h": 40}},
    {"bundleID": "com.apple.nc", "process": NC, "role": "AXButton",
     "identifier": "banner-fresh", "enabled": True,
     "texts": ["Call"], "actions": ["AXPress"],
     "frame": {"x": 1000, "y": 200, "w": 120, "h": 30}},
]
MUTED_FRAMES = [
    {"bundleID": "com.apple.nc", "process": NC, "role": "AXButton",
     "identifier": "mic", "enabled": True,
     "texts": ["microphone muted"], "actions": ["AXPress"],
     "frame": {"x": 10, "y": 10, "w": 20, "h": 20}},
]
UNMUTED_FRAMES = [
    {"bundleID": "com.apple.nc", "process": NC, "role": "AXButton",
     "identifier": "mic", "enabled": True,
     "texts": ["microphone unmuted"], "actions": ["AXPress"],
     "frame": {"x": 10, "y": 10, "w": 20, "h": 20}},
]
NOMUTE_FRAMES = [
    {"bundleID": "com.apple.nc", "process": NC, "role": "AXButton",
     "identifier": "mic", "enabled": True,
     "texts": ["FaceTime Audio 00:03"], "actions": ["AXPress"],
     "frame": {"x": 10, "y": 10, "w": 20, "h": 20}},
]

ok = True


def run(name, fn):
    global ok
    print(f"\n--- {name} ---")
    try:
        fn()
    except AssertionError as e:
        print(f"  FAIL: {e}")
        ok = False


def t1_prompt_press():
    """ax2 --ax-press pressed=true -> True, correct argv."""
    calls = []

    def fake_run(argv, **kwargs):
        calls.append((argv, kwargs))
        assert "--ax-press" in argv, f"expected ax-press, got {argv}"
        assert "--process" in argv and NC in argv
        assert "--contains" in argv and "Call" in argv
        assert ax_env_ok(kwargs), "authorized-caller env missing"
        return run_result(stdout=json.dumps({"pressed": True, "matched": "Call button"}))

    with mock.patch("os.path.exists", return_value=True), \
         mock.patch("subprocess.run", side_effect=fake_run):
        assert vl._press_click_to_call_if_present() is True
    assert len(calls) == 1, "should not fall through after a successful press"
    print("  pressed=true -> True, argv correct, no fallback run")
run("1. prompt-press via ax2 --ax-press", t1_prompt_press)


def t2_prompt_fallback():
    """ax-press pressed=false -> frames snapshot -> cliclick at center."""
    calls = []

    def fake_run(argv, **kwargs):
        calls.append((list(argv), dict(kwargs)))
        if "--ax-press" in argv:
            return run_result(stdout=json.dumps({"pressed": False, "reason": "no match"}))
        if "--ax-snapshot" in argv and "--frames" in argv:
            return run_result(stdout=json.dumps(PROMPT_FRAMES))
        if argv and argv[0] == "cliclick":
            return run_result(stdout="", returncode=0)
        raise AssertionError(f"unexpected subprocess call: {argv}")

    with mock.patch("os.path.exists", return_value=True), \
         mock.patch("subprocess.run", side_effect=fake_run):
        assert vl._press_click_to_call_if_present() is True
    cliclick = [c for c in calls if c[0][0] == "cliclick"]
    assert cliclick, "cliclick fallback never ran"
    # newest = last matching node: center of {x:1000, w:120, y:200, h:30}
    assert cliclick[0][0] == ["cliclick", "c:1060,215"], \
        f"wrong coords: {cliclick[0][0]}"
    press = [c for c in calls if "--ax-press" in c[0]]
    assert len(press) == 1 and press[0][0][1:6] == ["--ax-press", "--process",
                                                    NC, "--contains", "Call"]
    print("  pressed=false -> snapshot parsed, cliclick c:1060,215 issued")
run("2. prompt-press falls back to frames + cliclick", t2_prompt_fallback)


def t3_blind_connect_gone():
    """No timer + no banner + Phone-up at 12s must NOT read 'connected'."""
    src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "voice_loop.py")).read()
    assert "BLIND_CONNECT_AFTER_S" not in src, \
        "blind-connect constant still present"
    assert "treating as connected" not in src, \
        "blind-connect fallback message still present"

    # Behavioral: Phone up 12s, timer unreadable, banner probe False,
    # press budget unspent-ish. With the blind return gone, the function
    # must keep waiting (deadline eventually logs 'never arrived').
    with mock.patch("os.path.exists", return_value=True), \
         mock.patch("subprocess.run") as run_mock, \
         mock.patch.object(vl, "_call_timer_running", return_value=False), \
         mock.patch.object(vl, "_in_call_banner_visible", return_value=False), \
         mock.patch.object(vl, "_click_to_call_prompt_visible", return_value=False), \
         mock.patch.object(vl, "_press_click_to_call_if_present", return_value=False), \
         mock.patch.object(vl.time, "sleep", lambda s: None):
        run_mock.return_value = run_result(returncode=1)  # pgrep miss initially
        # Simulate: pgrep says Phone IS up (returncode 0).
        run_mock.return_value = run_result(returncode=0)
        vl._place_call_direct.__globals__["time"].time
        # Drive the loop: phone seen at t0; probes fail; at +12s the OLD
        # code returned connected — assert it has not, and that the loop
        # is still waiting at +20s.
        clock = {"t": 1000.0}
        real_time = vl.time.time

        def fake_time():
            return clock["t"]

        def advance_press_budget():
            # press window already spent (t_dial +20s ago)
            clock["t"] += 0
            return None

        with mock.patch.object(vl.time, "time", fake_time):
            clock["t"] -= 30.0  # pretend dial happened 30s ago: press window over

            # Re-implement the loop's early steps manually is fragile; instead
            # drive _place_call_direct with sleep as a no-op and a deadline
            # that arrives quickly: patch deadline via clock advance.
            # The loop polls every 0.25s of real time — sleep is patched to
            # a no-op, so we burn iterations by advancing the fake clock.
            result = []

            def run_once():
                # each loop turn advances the clock 0.3s via sleep patch
                pass

            # Simply run: with sleep a no-op and the clock advancing only
            # when we say so, we advance 45s+ over many turns by making
            # sleep bump the clock.
            def bump(s):
                clock["t"] += 0.3

            # NOTE: vl.time.sleep was already patched above; re-patch to bump.
            # vl.time.time is mocked within this with-block only.
            vl_time = vl.time
            orig_sleep = vl_time.sleep
            vl_time.sleep = bump
            try:
                # Run the loop in a thread with a hard cap, because with
                # sleep bumped the deadline (90s) arrives after 300 turns.
                import threading
                done = threading.Event()

                def target():
                    try:
                        result.append(vl._place_call_direct(stub=None))
                    except Exception as e:  # pragma: no cover
                        result.append(e)
                    finally:
                        done.set()

                th = threading.Thread(target=target, daemon=True)
                th.start()
                assert done.wait(20), "loop did not terminate"
            finally:
                vl_time.sleep = orig_sleep

        r = result[0]
        assert isinstance(r, vl._ProbeLike) is False or r is None, \
            f"got a connect result without proof: {r!r}"
        assert r is None, f"expected None (cleanup), got {r!r}"
    print("  12s blind connect is gone: no-timer/no-banner now returns None "
          "at deadline (and the constant no longer exists)")
run("3. blind-connect false positive removed", t3_blind_connect_gone)


def t4a_unmute_muted():
    """muted -> ax2-press 'Mute' -> re-snapshot -> unmuted:yes (ax2-press)."""
    snaps = {"n": 0}

    def fake_run(argv, **kwargs):
        if "--ax-press" in argv:
            assert "--contains" in argv and "Mute" in argv
            assert "--process" in argv and NC in argv
            return run_result(stdout=json.dumps({"pressed": True, "matched": "Mute"}))
        if "--ax-snapshot" in argv:
            snaps["n"] += 1
            return run_result(stdout=json.dumps(MUTED_FRAMES if snaps["n"] == 1
                                                else UNMUTED_FRAMES))
        raise AssertionError(f"unexpected subprocess call: {argv}")

    with mock.patch("os.path.exists", return_value=True), \
         mock.patch("subprocess.run", side_effect=fake_run):
        state = vl._ensure_outbound_mic_unmuted()
    assert state == "unmuted", state
    assert snaps["n"] >= 2, "re-snapshot after press never ran"
    print("  muted detected -> pressed 'Mute' -> re-snapshot verified -> "
          "unmuted (ax2-press)")
run("4a. unmute: muted -> press -> verify", t4a_unmute_muted)


def t4b_unmute_unmuted():
    """Already unmuted -> NO press issued."""
    presses = []

    def fake_run(argv, **kwargs):
        if "--ax-press" in argv:
            presses.append(list(argv))
            return run_result(stdout=json.dumps({"pressed": True}))
        if "--ax-snapshot" in argv:
            return run_result(stdout=json.dumps(UNMUTED_FRAMES))
        raise AssertionError(f"unexpected subprocess call: {argv}")

    with mock.patch("os.path.exists", return_value=True), \
         mock.patch("subprocess.run", side_effect=fake_run):
        state = vl._ensure_outbound_mic_unmuted()
    assert state == "unmuted", state
    assert not presses, f"press issued on an unmuted mic: {presses}"
    print("  unmuted detected -> no press issued")
run("4b. unmute: unmuted -> no press", t4b_unmute_unmuted)


def t4c_unmute_unknown_menu_fallback():
    """unknown -> menu fallback attempted -> 'unknown (none)' — never raises."""
    osa = []

    def fake_run(argv, **kwargs):
        if argv and argv[0] == "osascript":
            osa.append(list(argv))
            return run_result(stdout="missing value", returncode=0)
        if "--ax-snapshot" in argv:
            return run_result(stdout=json.dumps(NOMUTE_FRAMES))
        raise AssertionError(f"unexpected subprocess call: {argv}")

    with mock.patch("os.path.exists", return_value=True), \
         mock.patch("subprocess.run", side_effect=fake_run):
        state = vl._ensure_outbound_mic_unmuted()  # must not raise
    assert state == "unknown", state
    assert osa, "menu fallback not attempted"
    scripts = [c[2] for c in osa]  # argv = ['osascript', '-e', <script>]
    assert any("AXMenuItemMarkChar" in s for s in scripts), \
        "mark-char read (BEFORE/AFTER) not attempted"
    assert any("click menu item" in s for s in scripts), "menu click not attempted"
    assert len(osa) == 3, f"expected BEFORE+click+AFTER (3 calls), got {len(osa)}"
    print(f"  unknown -> menu fallback ran ({len(osa)} osascript calls), "
          "state stays unknown, no exception")
run("4c. unmute: unknown -> menu fallback, never raises", t4c_unmute_unknown_menu_fallback)


def t5_never_raises():
    """ax2 snapshot explodes -> single warning line, no exception."""
    with mock.patch("os.path.exists", return_value=True), \
         mock.patch("subprocess.run", side_effect=OSError("boom")):
        state = vl._ensure_outbound_mic_unmuted()
    assert state == "unknown", state
    print("  subprocess hard-fail -> swallowed, state unknown")
run("5. unmute helper never raises on hard failure", t5_never_raises)


def t6_no_mlx_import():
    """Importing voice_loop must not load MLX/whisper (lazy STT)."""
    assert "mlx" not in sys.modules, "MLX loaded at import time"
    assert not any("whisper" in m for m in sys.modules), "whisper loaded at import"
    print("  voice_loop import is dependency-light (no MLX/whisper)")
run("6. import stays MLX-free", t6_no_mlx_import)


print("\n" + ("ALL CHECKS PASSED" if ok else "FAILURES ABOVE"))
sys.exit(0 if ok else 1)