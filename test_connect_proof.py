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
import time
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


def t4c_unmute_unknown_no_toggle():
    """unknown -> NO toggle of any kind (call-3 fix): state stays unknown,
    zero osascript calls, never raises. A blind toggle muted call 3's live
    mic; unclassifiable state must mean DO NOTHING."""
    osa = []

    def fake_run(argv, **kwargs):
        if argv and argv[0] == "osascript":
            osa.append(list(argv))
            return run_result(stdout="missing value")
        if "--ax-snapshot" in argv:
            return run_result(stdout=json.dumps(NOMUTE_FRAMES))
        raise AssertionError(f"unexpected subprocess call: {argv}")

    with mock.patch("os.path.exists", return_value=True), \
         mock.patch("subprocess.run", side_effect=fake_run):
        state = vl._ensure_outbound_mic_unmuted()  # must not raise
    assert state == "unknown", state
    assert not osa, \
        f"BLIND menu toggle ran on unknown state: {osa}"
    print("  unknown -> no blind toggle (0 osascript calls), state unknown")
run("4c. unmute: unknown -> NO toggle (call-3 fix)", t4c_unmute_unknown_no_toggle)


def t4d_unmute_muted_press_unconfirmed_menu_gated():
    """muted -> ax2 press not confirmed -> menu path GATED on Video>Mute
    enabled=true (live call) -> toggle runs -> post-toggle banner verify."""
    snaps = {"n": 0}
    osa = []

    def fake_run(argv, **kwargs):
        if "--ax-press" in argv:
            return run_result(stdout=json.dumps({"pressed": False,
                                                 "reason": "no match"}))
        if "--ax-snapshot" in argv:
            snaps["n"] += 1
            return run_result(stdout=json.dumps(MUTED_FRAMES if snaps["n"] == 1
                                                else UNMUTED_FRAMES))
        if argv and argv[0] == "osascript":
            osa.append(list(argv))
            if "enabled of menu item" in argv[2]:
                return run_result(stdout="true")   # live call: Mute enabled
            return run_result(stdout="missing value")  # mark char, as live
        raise AssertionError(f"unexpected subprocess call: {argv}")

    with mock.patch("os.path.exists", return_value=True), \
         mock.patch("subprocess.run", side_effect=fake_run):
        state = vl._ensure_outbound_mic_unmuted()
    assert state == "unmuted", state
    clicks = [c for c in osa if "click menu item" in c[2]]
    enabled_q = [c for c in osa if "enabled of menu item" in c[2]]
    assert len(enabled_q) == 1, "Mute enabled probe not read before toggling"
    assert len(clicks) == 1, f"expected the gated menu click, got {osa}"
    assert enabled_q[0] == osa[0], "enabled probe must precede the click"
    assert snaps["n"] == 2, "post-toggle banner re-verify never ran"
    print("  muted + press unconfirmed -> enabled=true -> menu toggle -> "
          "banner re-verified unmuted")
run("4d. muted: gated menu toggle only when Video>Mute is enabled",
    t4d_unmute_muted_press_unconfirmed_menu_gated)


def t4e_unmute_menu_disabled_never_clicks():
    """Mute menu DISABLED (no active call) -> click refused: only the enabled
    probe runs, state stays 'muted' (honest 'no'), no toggle, never raises."""
    osa = []

    def fake_run(argv, **kwargs):
        if "--ax-press" in argv:
            return run_result(stdout=json.dumps({"pressed": False}))
        if "--ax-snapshot" in argv:
            return run_result(stdout=json.dumps(MUTED_FRAMES))
        if argv and argv[0] == "osascript":
            osa.append(list(argv))
            if "enabled of menu item" in argv[2]:
                return run_result(stdout="false")  # no call active
            return run_result(stdout="missing value")
        raise AssertionError(f"unexpected subprocess call: {argv}")

    with mock.patch("os.path.exists", return_value=True), \
         mock.patch("subprocess.run", side_effect=fake_run):
        state = vl._ensure_outbound_mic_unmuted()  # must not raise
    assert state == "muted", state
    clicks = [c for c in osa if "click menu item" in c[2]]
    assert not clicks, f"clicked Mute while DISABLED (would arm mute): {osa}"
    assert len(osa) == 1 and "enabled of menu item" in osa[0][2], osa
    print("  Mute disabled (no call) -> toggle refused, state stays muted")
run("4e. disabled Mute menu -> never clicks (pre-call arm guard)",
    t4e_unmute_menu_disabled_never_clicks)


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


# ---------------------------------------------------------------------------
# 2026-09-09 silent-call fixes (bugs 1-3)
# ---------------------------------------------------------------------------

def t7_press_gate_runs_without_surface():
    """Bug 1: presses fire from t_dial+3s even with Phone already up.

    The morning call (2026-09-09 06:54): Phone's surface appeared 0.4s after
    dial, so the old `phone_seen_at is None` gate was False forever and zero
    press attempts ran during the whole 90s window.
    """
    import threading
    clock = {"t": 1000.0}
    t0 = clock["t"]
    press_at = []

    def fake_press():
        press_at.append(clock["t"] - t0)
        return False

    def fake_time():
        return clock["t"]

    def bump(s):
        clock["t"] += 0.5

    with mock.patch.object(vl, "AUTHORIZED_E164", "+15550000000"), \
         mock.patch("os.path.exists", return_value=True), \
         mock.patch("subprocess.run", return_value=run_result(returncode=0)), \
         mock.patch.object(vl, "_press_click_to_call_if_present",
                           side_effect=fake_press), \
         mock.patch.object(vl, "_call_timer_running", return_value=False), \
         mock.patch.object(vl, "_in_call_banner_visible", return_value=False), \
         mock.patch.object(vl, "_click_to_call_prompt_visible",
                           return_value=True), \
         mock.patch.object(vl.time, "time", fake_time), \
         mock.patch.object(vl.time, "sleep", bump):
        result = []
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
        assert done.wait(30), "dial loop did not terminate"

    r = result[0]
    assert r is None, f"expected None at deadline (no proof), got {r!r}"
    assert len(press_at) == 3, f"expected exactly 3 press attempts, got {press_at}"
    assert all(dt < 20.0 for dt in press_at), f"presses outside window: {press_at}"
    assert press_at[0] < 4.0, f"first press not at ~t_dial+3s: {press_at}"
    print(f"  Phone up at 0s did NOT block presses: attempts at "
          f"{[round(x, 1) for x in press_at]}s after dial")


def t8_adopt_proof_helper():
    """Bug 2: adopt requires timer/banner proof; probe failures never raise."""
    import threading
    # (a) timer proof short-circuits -> True, banner not consulted.
    with mock.patch.object(vl, "_call_timer_running", return_value=True) as mt, \
         mock.patch.object(vl, "_in_call_banner_visible",
                           return_value=False) as mb:
        assert vl._adopt_connect_proof(stub=None, timeout_s=1.0) is True
        assert mt.called, "timer proof never probed"
        assert not mb.called, "banner probed although the timer already proved"
    print("  timer proof -> adopt allowed (banner not needed)")

    # (b) no proof at all -> keeps polling until timeout, then False.
    clock = {"t": 0.0}
    calls = {"n": 0}

    def no_proof():
        calls["n"] += 1
        return False

    with mock.patch.object(vl, "_call_timer_running", side_effect=no_proof), \
         mock.patch.object(vl, "_in_call_banner_visible", return_value=False), \
         mock.patch.object(vl.time, "time", lambda: clock["t"]), \
         mock.patch.object(vl.time, "sleep", lambda s: clock.__setitem__(
             "t", clock["t"] + 1.1)):
        assert vl._adopt_connect_proof(stub=None, timeout_s=2.2) is False
    assert calls["n"] >= 2, "proof helper stopped polling before timeout"
    print(f"  no proof -> polled {calls['n']}x then failed safe (False)")

    # (c) probes RAISING -> swallowed, still fails safe, never raises.
    clock2 = {"t": 0.0}
    with mock.patch.object(vl, "_call_timer_running",
                           side_effect=RuntimeError("boom")), \
         mock.patch.object(vl.time, "time", lambda: clock2["t"]), \
         mock.patch.object(vl.time, "sleep", lambda s: clock2.__setitem__(
             "t", clock2["t"] + 1.1)):
        assert vl._adopt_connect_proof(stub=None, timeout_s=1.1) is False
    print("  probe exceptions -> swallowed, fails safe")
run("7. prompt-press gate no longer blocked by Phone surface", t7_press_gate_runs_without_surface)
run("8. adopt requires connect proof, never raises", t8_adopt_proof_helper)


def t9_bh2_tap_fail_open():
    """Bug 3 helper: broken tap returns (0.0, 0.0); working tap returns peak."""
    import types
    import numpy as np
    saved = sys.modules.get("sounddevice")

    class FakeStreamOK:
        def __init__(self, **kwargs):
            assert kwargs["device"] == "BlackHole 2ch"
            assert kwargs["samplerate"] == 48000 and kwargs["channels"] == 2
            assert kwargs["dtype"] == "float32" and kwargs["blocksize"] == 960
            self._cb = kwargs["callback"]

        def __enter__(self):
            blk = np.zeros((960, 2), dtype="float32")
            blk[:, :] = 0.25
            self._cb(blk, 960, 0.0, None)
            return self

        def __exit__(self, *a):
            return False

    def _install(cls):
        m = types.ModuleType("sounddevice")
        m.InputStream = cls
        sys.modules["sounddevice"] = m

    def _restore():
        if saved is not None:
            sys.modules["sounddevice"] = saved
        else:
            sys.modules.pop("sounddevice", None)

    try:
        _install(FakeStreamOK)
        with mock.patch.object(vl.time, "sleep", lambda s: None):
            peak, rms = vl._bh2_tap_peak(3.0)
        assert (round(peak, 6), round(rms, 6)) == (0.25, 0.25), (peak, rms)

        class FakeStreamBoom:
            def __init__(self, **kwargs):
                raise RuntimeError("boom")

        _install(FakeStreamBoom)
        with mock.patch.object(vl.time, "sleep", lambda s: None):
            peak, rms = vl._bh2_tap_peak(3.0)
        assert (peak, rms) == (0.0, 0.0), "hard failure must fail-open"
    finally:
        _restore()
    print("  BH2 tap: correct device/format, (0.25,0.25) on signal, "
          "fail-open on error")
run("9. BH2 tap helper works and never raises", t9_bh2_tap_fail_open)


def t10_run_call_bh2_selfcheck():
    """Bug 3 wiring + live-call-2 regression: dead leg (after audio flowed)
    -> ERROR + audible warning; healthy -> pass line; no audio emitted ->
    distinct 'skipped' WARNING (NEVER the false 'playback leg dead' ERROR);
    flag off -> tap never runs (trigger path unchanged)."""
    import threading
    import grpc

    class FakeRpcError(grpc.RpcError):
        def code(self):
            return "UNAVAILABLE"

    class FakeWriter:
        def __init__(self):
            self.items = []

        def put(self, item):
            self.items.append(item)

    class FakeSession:
        def __init__(self):
            self.playing = threading.Event()
            self.barge_in = threading.Event()
            self.writer = FakeWriter()
            self.emitted = []
            self.first_emit = threading.Event()

        def _emit_speech(self, pcm):
            self.emitted.append(len(pcm))
            # Mirror the real hook: the FIRST emission arms the tap.
            if not self.first_emit.is_set():
                self.first_emit.set()

        def audio_loop(self):
            time.sleep(60)  # daemon thread; never joined in tests

    stub = mock.Mock()
    stub.Control.side_effect = FakeRpcError()  # ends the call watcher at once
    spoken = []

    def fake_tts(sentences, emit=None, cancelled=None):
        spoken.append(" ".join(sentences))
        if emit is not None:
            emit(b"\x00\x01")  # greeting flows -> first_emit set, as in-call

    # (a) dead leg AFTER audio emitted -> greeting + warning, ERROR logged.
    with mock.patch.object(vl, "tts_sentences", fake_tts), \
         mock.patch.object(vl, "_bh2_tap_peak", lambda seconds=3.0: (0.0, 0.0)), \
         mock.patch.object(vl.log, "error") as err:
        vl._run_call(FakeSession(), True, stub, bh2_selfcheck=True)
    assert len(spoken) == 2, f"expected greeting + warning, got {spoken}"
    assert "audio leg" in spoken[1], spoken[1]
    assert err.called and "playback leg dead" in str(err.call_args), \
        "ERROR line for the dead playback leg missing"
    print("  dead BH2 leg (audio was flowing) -> ERROR + audible warning")

    # (b) healthy leg -> greeting only, no ERROR.
    spoken.clear()
    with mock.patch.object(vl, "tts_sentences", fake_tts), \
         mock.patch.object(vl, "_bh2_tap_peak",
                           lambda seconds=3.0: (0.5, 0.2)), \
         mock.patch.object(vl.log, "error") as err:
        vl._run_call(FakeSession(), True, stub, bh2_selfcheck=True)
    assert len(spoken) == 1, f"healthy leg must not warn, got {spoken}"
    assert not err.called, "false ERROR on a healthy leg"
    print("  healthy BH2 leg (peak 0.5) -> no warning, no ERROR")

    # (b2) LIVE CALL 2 REGRESSION: no audio ever emitted (stream never
    # flushed) -> tap must NOT log 'playback leg dead'; it logs the distinct
    # 'no audio emitted yet' WARNING instead.
    spoken.clear()
    with mock.patch.object(vl, "tts_sentences",
                           lambda s, emit=None, cancelled=None: None), \
         mock.patch.object(vl, "_bh2_tap_peak",
                           side_effect=AssertionError("tap ran with no audio")), \
         mock.patch.object(vl, "DFV_FIRST_EMIT_WAIT_S", 0.05), \
         mock.patch.object(vl.log, "error") as err, \
         mock.patch.object(vl.log, "warning") as warn:
        vl._run_call(FakeSession(), True, stub, bh2_selfcheck=True)
    assert len(spoken) == 0, spoken
    assert not err.called, \
        "false 'playback leg dead' ERROR when no audio was emitted yet"
    msgs = [str(c) for c in warn.call_args_list]
    assert any("no audio emitted yet" in m for m in msgs), msgs
    print("  no audio emitted -> 'self-check skipped: no audio emitted yet' "
          "WARNING, no false ERROR")

    # (c) flag off -> tap never runs (outbound-trigger path stays unchanged).
    spoken.clear()
    with mock.patch.object(vl, "tts_sentences", fake_tts), \
         mock.patch.object(vl, "_bh2_tap_peak",
                           side_effect=AssertionError("tap ran with flag off")):
        vl._run_call(FakeSession(), True, stub, bh2_selfcheck=False)
    assert len(spoken) == 1, spoken
    print("  bh2_selfcheck=False -> tap never invoked")
run("10. _run_call BH2 self-check wiring", t10_run_call_bh2_selfcheck)


def t12_selfcheck_starts_on_first_emit():
    """Bug A wiring: audio_loop starts BEFORE the greeting (START packet must
    precede TTS), _emit_speech sets first_emit on the FIRST packet, and the
    tap thread only samples after that event (no tap-before-audio race)."""
    import threading
    import numpy as np

    class FakeWriter:
        def __init__(self):
            self.items = []

        def put(self, item):
            self.items.append(item)

    session = vl.CallSession(stub=None, call_id="t12")
    session.writer = FakeWriter()

    # _emit_speech arms first_emit on the first packet only.
    assert not session.first_emit.is_set()
    session._emit_speech(np.zeros(4, dtype="float32"))
    assert session.first_emit.is_set(), "first emit must arm the tap"
    session._emit_speech(np.zeros(4, dtype="float32"))
    print("  _emit_speech sets first_emit on first packet")

    # _run_call must start audio_loop BEFORE the greeting: patch audio_loop
    # to record order AND queue the START packet exactly like the real one.
    import grpc

    class FakeRpcError(grpc.RpcError):
        def code(self):
            return "UNAVAILABLE"

    stub = mock.Mock()
    stub.Control.side_effect = FakeRpcError()

    order = []

    def fake_audio_loop(self):
        order.append("audio_loop")
        self.writer.put(("START", None))

    def fake_tts(sentences, emit=None, cancelled=None):
        order.append("greeting")

    with mock.patch.object(vl.CallSession, "audio_loop", fake_audio_loop), \
         mock.patch.object(vl, "tts_sentences", fake_tts):
        vl._run_call(session, True, stub, bh2_selfcheck=False)
    assert order[:2] == ["audio_loop", "greeting"], \
        f"audio stream must attach before the greeting, got {order}"
    assert any(item[0] == "START" for item in session.writer.items), \
        "audio_loop must have queued the START packet before the greeting"
    print("  audio_loop (START packet) precedes the greeting")


def t13_bargein_releases_guard_promptly():
    """Bug B: TTSCanceled must end the turn quickly — llm_reply_streaming
    stops reading deltas, drains to the terminator, raises; the worker pipe
    ends clean (no cross-turn poisoning) and is not killed."""
    import json as _json

    class FakeStdout:
        def __init__(self, lines):
            self._lines = list(lines)

        def readline(self):
            return self._lines.pop(0) if self._lines else ""

    class FakeWorkerProc:
        """Worker stand-in: 1000 buffered delta lines then the terminator."""

        def __init__(self):
            msgs = [{"delta": f"chunk {i} "} for i in range(1000)]
            msgs.append({"content": "final", "tier": "fast"})
            self.stdout = FakeStdout(_json.dumps(m) + "\n" for m in msgs)
            self.stdin = mock.Mock()
            self.killed = False

        def poll(self):
            return None

        def kill(self):
            self.killed = True

    proc = FakeWorkerProc()
    saved_proc = vl._worker_proc
    vl._worker_proc = proc
    try:
        with mock.patch.object(vl, "_ensure_worker", lambda: None):
            t0 = time.perf_counter()
            try:
                vl.llm_reply_streaming("hello", speak=lambda s, is_final=False: None,
                                       cancelled=lambda: True)
                raise AssertionError("expected TTSCanceled")
            except vl.TTSCanceled:
                pass
            dt = time.perf_counter() - t0
        # The turn abandoned speech at the first delta and drained the
        # remaining buffered lines to the terminator — fast, and the
        # terminator was read (not left to poison the next turn).
        assert dt < 5.0, f"canceled turn took {dt:.2f}s — guard held too long"
        assert proc.killed is False, \
            "healthy pipe must not be killed by the drain"
        assert vl._worker_proc is proc, "worker must survive a clean drain"
        with vl._dialogue_lock:
            last = vl._dialogue[-1] if vl._dialogue else None
            vl._dialogue.clear()
        assert last and last["content"] == "[turn canceled by barge-in]", last
    finally:
        vl._worker_proc = saved_proc
    print(f"  canceled turn released in {dt * 1000:.0f}ms, drained to "
          f"terminator, worker kept")

    # (b) THE EXACT CALL-2 SIGNATURE: speak() itself raises TTSCanceled
    # mid-stream (_speak_sentence re-raises after swallowing synthesis) while
    # the barge_in flag races — the terminator is still unconsumed, so the
    # pipe MUST be drained before the lock releases.
    proc2 = FakeWorkerProc()
    vl._worker_proc = proc2
    try:
        with mock.patch.object(vl, "_ensure_worker", lambda: None):
            def _boom(sentence, is_final=False):
                raise vl.TTSCanceled()
            try:
                vl.llm_reply_streaming("hello", speak=_boom,
                                       cancelled=lambda: False)
                raise AssertionError("expected TTSCanceled")
            except vl.TTSCanceled:
                pass
        assert proc2.killed is False
        assert not proc2.stdout._lines, \
            "orphaned deltas+terminator must be drained, not left for the " \
            "next turn"
        assert vl._worker_proc is proc2
    finally:
        vl._worker_proc = saved_proc
    print("  speak-raised TTSCanceled -> pipe drained to terminator too")


def t14_overlapping_utterance_queued_not_dropped():
    """Bug B: an utterance finalized while a turn holds the guard is QUEUED
    (latest wins), and runs as a fresh turn once the dead turn releases."""
    import numpy as np

    s = vl.CallSession(stub=None, call_id="t14")
    assert s.turn_active.acquire(blocking=False)
    runs = []

    def fake_run_turn(audio, spec_epoch=None):
        runs.append((audio, spec_epoch))

    fake_audio = np.zeros(16, dtype="float32")
    with mock.patch.object(s, "_run_turn", side_effect=fake_run_turn):
        # Overlapping utterance while the guard is held -> queued, not run.
        s._process_turn(fake_audio, 7)
        assert not runs, "queued utterance must not run while guard is held"
        assert s.pending_turn.qsize() == 1, "utterance must be queued"
        # A second overlap REPLACES the queued one (latest wins).
        s._process_turn(np.zeros(8, dtype="float32"), 9)
        assert s.pending_turn.qsize() == 1
        s.turn_active.release()
        # Release path: the kick starts the queued utterance as a fresh turn.
        s._kick_pending_turn()
        deadline = time.time() + 2.0
        while len(runs) < 1 and time.time() < deadline:
            time.sleep(0.01)
    assert len(runs) == 1, f"queued utterance must run once, got {runs}"
    assert runs[0][1] == 9, f"the LATEST queued utterance must run, got {runs}"
    assert s.pending_turn.qsize() == 0, "kick must consume the queue"
    print("  overlapping utterance queued (latest wins), kicked after release")


def t11_adopt_block_wiring():
    """Bug 2 wiring: proof gate precedes adoption; unmute + selfcheck present."""
    src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            "voice_loop.py")).read()
    i_proof = src.find("if not _adopt_connect_proof(stub):")
    i_adopt = src.find('"adopting live call (warm attach — no cold start)"')
    i_flag = src.find("bh2_selfcheck=True")
    assert i_proof != -1, "adopt connect-proof gate missing"
    assert i_adopt != -1, "adopt log line missing"
    assert i_flag != -1, "adopt path does not request the BH2 self-check"
    assert i_proof < i_adopt < i_flag, "adopt wiring order wrong (proof->adopt->flag)"
    n_unmute = src.count("_ensure_outbound_mic_unmuted()")
    assert n_unmute == 2, \
        f"unmute must run exactly on trigger + adopt paths, found {n_unmute}"
    print("  adopt path wiring: proof gate -> unmute -> bh2_selfcheck=True")
run("11. adopt block wiring (proof, unmute, selfcheck)", t11_adopt_block_wiring)
run("12. BH2 self-check starts on first emit", t12_selfcheck_starts_on_first_emit)
run("13. barged-in turn releases guard promptly + drains pipe",
    t13_bargein_releases_guard_promptly)
run("14. overlapping utterance queued (not dropped), runs after release",
    t14_overlapping_utterance_queued_not_dropped)


# pytest-compat: module-level checks above run at import; the unittest bridge
# below lets pytest collect a real test that reflects the ok flag, while the
# direct script run keeps its documented `python test_connect_proof.py` exit.
class SilentCallRegression(unittest.TestCase):
    def test_all_checks_passed(self):
        self.assertTrue(ok, "one or more checks failed — see output above")


if __name__ == "__main__":
    print("\n" + ("ALL CHECKS PASSED" if ok else "FAILURES ABOVE"))
    sys.exit(0 if ok else 1)