"""Hardware-facing capture sources (issue #172) driven through fakes.

``catap`` is replaced by a fake module in ``sys.modules`` and ``Popen`` by a
fake process, so no Core Audio or avfoundation call happens here. Real soxr
runs (it is pure). The opt-in ``hardware`` test at the bottom is the only
test that touches real devices and is excluded from CI.
"""

from __future__ import annotations

import logging
import queue
import signal
import subprocess
import sys
import threading
import time
import tomllib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf

from vemoizer import capture_sources
from vemoizer.capture import Chunk, TrackWriter
from vemoizer.capture_sources import CatapSystemSource, FfmpegMicSource

SR = 16_000
REPO = Path(__file__).resolve().parent.parent


class Clock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t


def wait_until(predicate, timeout: float = 5.0) -> None:
    end = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > end:
            raise AssertionError("timed out waiting for condition")
        time.sleep(0.01)


def make_buf(samples: np.ndarray, rate: float, input_sample_time=None):
    return SimpleNamespace(
        data=np.asarray(samples, dtype="<f4").tobytes(),
        frame_count=len(samples),
        format=SimpleNamespace(sample_rate=float(rate), num_channels=1, is_float=True),
        input_sample_time=input_sample_time,
    )


def device(name: str, *, default_in=False, default_out=False, alert_out=False):
    """``default_out`` is the playback device (``is_default_output``); ``alert_out``
    is the system alert/sound-effects device (``is_default_system_output``)."""
    return SimpleNamespace(
        name=name,
        uid=name,
        is_default_input=default_in,
        is_default_output=default_out,
        is_default_system_output=alert_out,
        streams=(),
    )


class FakeSession:
    def __init__(self, on_buffer) -> None:
        self.on_buffer = on_buffer
        self.started = 0
        self.stopped = 0
        self.capture_failed = False

    def start(self) -> None:
        self.started += 1

    def stop(self) -> None:
        self.stopped += 1


class FakeCatap:
    """Installs a fake ``catap`` (and ``catap.events``) into ``sys.modules``."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, devices=()) -> None:
        self.sessions: list[FakeSession] = []
        self.watchers: list = []
        self.devices = list(devices)
        events = SimpleNamespace(watch_default_output_device=self._watch)
        module = SimpleNamespace(
            record_system_audio=self._record,
            list_audio_devices=lambda: self.devices,
            events=events,
        )
        monkeypatch.setitem(sys.modules, "catap", module)
        monkeypatch.setitem(sys.modules, "catap.events", events)

    def _record(self, *, on_buffer=None, mono=False, **_kw) -> FakeSession:
        session = FakeSession(on_buffer)
        self.sessions.append(session)
        return session

    def _watch(self, callback):
        self.watchers.append(callback)
        return SimpleNamespace(close=lambda: None)


def spy_soxr(monkeypatch: pytest.MonkeyPatch) -> dict[str, list]:
    import soxr

    real = soxr.ResampleStream
    calls: dict[str, list] = {"rates": [], "threads": []}

    class Spy:
        def __init__(self, in_rate, out_rate, num_channels, dtype="float32", **kw):
            calls["rates"].append(in_rate)
            self._stream = real(in_rate, out_rate, num_channels, dtype=dtype, **kw)

        def resample_chunk(self, x, last=False):
            calls["threads"].append(threading.get_ident())
            return self._stream.resample_chunk(x, last=last)

    monkeypatch.setattr(soxr, "ResampleStream", Spy)
    return calls


def feed_tap(src, session, signal_48k: np.ndarray, frame: int, clk: Clock) -> None:
    for i in range(0, len(signal_48k), frame):
        block = signal_48k[i : i + frame]
        clk.t = (i + len(block)) / 48_000
        session.on_buffer(make_buf(block, 48_000, input_sample_time=i))


def drain_after_stop(src) -> list[Chunk]:
    src.stop()
    return list(src.chunks())


# --- CatapSystemSource -------------------------------------------------------


def test_tap_48k_to_16k_length_within_one_sample_per_second(monkeypatch) -> None:
    FakeCatap(monkeypatch)
    clk = Clock()
    src = CatapSystemSource(clock=clk, poll_s=0.01)
    src.start()
    sig = np.sin(np.linspace(0, 2000, 4 * 48_000)).astype(np.float32) * 0.3
    feed_tap(src, src._session, sig, 512, clk)
    chunks = drain_after_stop(src)
    total = sum(len(c.samples) for c in chunks)
    assert abs(total - 4 * SR) <= 4  # four seconds of audio, a few samples of slack


def test_chunked_equals_one_shot(monkeypatch) -> None:
    sig = np.sin(np.linspace(0, 500, 2 * 48_000)).astype(np.float32) * 0.2
    outputs = []
    for frame in (480, 960):
        FakeCatap(monkeypatch)
        clk = Clock()
        src = CatapSystemSource(clock=clk, poll_s=0.01)
        src.start()
        feed_tap(src, src._session, sig, frame, clk)
        outputs.append(np.concatenate([c.samples for c in drain_after_stop(src)]))
    np.testing.assert_allclose(outputs[0], outputs[1], atol=1e-6)


def test_format_change_flushes_old_stream_and_marks_discontinuity(monkeypatch) -> None:
    FakeCatap(monkeypatch)
    calls = spy_soxr(monkeypatch)
    clk = Clock()
    src = CatapSystemSource(clock=clk, poll_s=0.01)
    src.start()
    session = src._session
    sig_a = np.zeros(int(0.3 * 48_000), dtype=np.float32)
    feed_tap(src, session, sig_a, 512, clk)
    # Format change at t = 0.3 s: the tap now delivers 24 kHz.
    for i in range(20):
        clk.t = 0.3 + (i + 1) * 512 / 24_000
        session.on_buffer(make_buf(np.zeros(512, np.float32), 24_000))
    chunks = drain_after_stop(src)
    assert calls["rates"] == [48_000, 24_000]
    assert any(len(c.samples) == 0 for c in chunks), "expected a discontinuity marker"


def test_input_sample_time_jump_becomes_discontinuity_frames(monkeypatch) -> None:
    FakeCatap(monkeypatch)
    clk = Clock()
    src = CatapSystemSource(clock=clk, poll_s=0.01)
    src.start()
    session = src._session
    jump = 4800  # native (48 kHz) frames
    for i in range(200):
        base = i * 512 + (jump if i >= 100 else 0)
        clk.t = (i + 1) * 512 / 48_000
        session.on_buffer(make_buf(np.zeros(512, np.float32), 48_000, base))
    chunks = drain_after_stop(src)
    assert sum(c.discontinuity_frames for c in chunks) == round(jump * SR / 48_000)


@pytest.mark.parametrize("input_time_restarts", [True, False])
def test_format_switch_keeps_the_wall_clock_gap(
    monkeypatch, input_time_restarts
) -> None:
    # AirPods HFP: 0.3 s at 48 kHz, 0.5 s with no buffers, then 2.5 s at 24 kHz.
    # The stem must span all 3.3 s, whether the new stream's input_sample_time
    # restarts at 0 or continues in the new rate's frames.
    FakeCatap(monkeypatch)
    clk = Clock()
    src = CatapSystemSource(clock=clk, poll_s=0.01)
    src.start()
    session = src._session
    for i in range(30):
        clk.t = (i + 1) * 480 / 48_000
        session.on_buffer(make_buf(np.zeros(480, np.float32), 48_000, i * 480))
    switch_s = 0.3 + 0.5
    base = 0 if input_time_restarts else round(switch_s * 24_000)
    for j in range(250):
        clk.t = switch_s + (j + 1) * 240 / 24_000
        session.on_buffer(make_buf(np.zeros(240, np.float32), 24_000, base + j * 240))
    chunks = drain_after_stop(src)
    stem = sum(len(c.samples) + c.discontinuity_frames for c in chunks)
    assert abs(stem - round(3.3 * SR)) <= 8


def _feed_run(session, clk, *, rate, block, start_wall, start_in, seconds):
    """Real-time buffers: wall clock and input_sample_time both advance per block."""
    t, i = start_wall, start_in
    for _ in range(round(seconds * rate / block)):
        t += block / rate
        clk.t = t
        session.on_buffer(make_buf(np.zeros(block, np.float32), rate, i))
        i += block
    return t


@pytest.mark.parametrize("trigger", ["format_switch", "reopen"])
def test_jump_then_new_run_does_not_double_count_the_jump(monkeypatch, trigger) -> None:
    # 2.5 s (the 2 s rate check closes inside it), a 0.5 s stall (input_sample_time
    # jumps 0.5 s while the wall clock advances 0.5 s), 1 s more, then a new run
    # with no wall-clock gap: the stem must span 5.0 s. Counting the jump again in
    # the new run's gap would add another 0.5 s.
    fake = FakeCatap(monkeypatch)
    clk = Clock()
    src = CatapSystemSource(clock=clk, poll_s=0.01)
    src.start()
    session = src._session
    t = _feed_run(
        session, clk, rate=48_000, block=480, start_wall=0.0, start_in=0, seconds=2.5
    )
    stall = 24_000  # 0.5 s of 48 kHz frames
    t = _feed_run(
        session,
        clk,
        rate=48_000,
        block=480,
        start_wall=t + 0.5,
        start_in=stall + 120_000,
        seconds=1.0,
    )
    if trigger == "format_switch":
        _feed_run(
            session, clk, rate=24_000, block=240, start_wall=t, start_in=0, seconds=1.0
        )
    else:
        first = fake.sessions[0]
        first.capture_failed = True
        wait_until(lambda: len(fake.sessions) == 2)
        _feed_run(
            fake.sessions[1],
            clk,
            rate=48_000,
            block=480,
            start_wall=t,
            start_in=0,
            seconds=1.0,
        )
    chunks = drain_after_stop(src)
    stem = sum(len(c.samples) + c.discontinuity_frames for c in chunks)
    assert abs(stem - round(5.0 * SR)) <= 8


def test_capture_failed_reopens_and_keeps_yielding(monkeypatch) -> None:
    fake = FakeCatap(monkeypatch)
    clk = Clock()
    src = CatapSystemSource(clock=clk, poll_s=0.01)
    src.start()
    first = fake.sessions[0]
    first.capture_failed = True
    wait_until(lambda: len(fake.sessions) == 2)
    second = fake.sessions[1]
    assert first.stopped == 1 and second.started == 1
    sig = np.full(48_000, 0.1, np.float32)
    feed_tap(src, second, sig, 512, clk)
    chunks = drain_after_stop(src)
    assert sum(len(c.samples) for c in chunks) > 0


def test_callback_only_enqueues_and_soxr_runs_on_worker(monkeypatch) -> None:
    FakeCatap(monkeypatch)
    calls = spy_soxr(monkeypatch)
    clk = Clock()
    src = CatapSystemSource(clock=clk, poll_s=0.01)
    src.start()
    feed_tap(src, src._session, np.zeros(48_000, np.float32), 512, clk)
    drain_after_stop(src)
    assert calls["threads"], "soxr was never called"
    assert threading.get_ident() not in calls["threads"]


def test_default_output_change_warns_once(monkeypatch, caplog) -> None:
    fake = FakeCatap(monkeypatch, devices=[device("Speakers", default_out=True)])
    src = CatapSystemSource(clock=Clock(), poll_s=0.01)
    src.start()
    with caplog.at_level(logging.WARNING, logger="vemoizer.capture_sources"):
        fake.watchers[0](None)
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert warnings == ["default output changed to Speakers"]
    drain_after_stop(src)


def test_output_device_is_playback_default_not_alert_device(
    monkeypatch, caplog
) -> None:
    # The alert device must not be picked: audio plays through is_default_output.
    fake = FakeCatap(
        monkeypatch,
        devices=[
            device("Alerts", alert_out=True),
            device("Speakers", default_out=True),
        ],
    )
    src = CatapSystemSource(clock=Clock(), poll_s=0.01)
    src.start()
    assert capture_sources._default_output(sys.modules["catap"]).name == "Speakers"
    with caplog.at_level(logging.WARNING, logger="vemoizer.capture_sources"):
        fake.watchers[0](None)
    warnings = [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING]
    assert warnings == ["default output changed to Speakers"]
    drain_after_stop(src)


# --- FfmpegMicSource ---------------------------------------------------------


class FakeStdout:
    """A pipe the test feeds: ``push`` bytes, ``eof`` to simulate exit."""

    def __init__(self) -> None:
        self._q: queue.SimpleQueue[bytes] = queue.SimpleQueue()

    def push(self, data: bytes) -> None:
        self._q.put(data)

    def eof(self) -> None:
        self._q.put(b"")

    def read(self, _n: int = -1) -> bytes:
        return self._q.get()


class FakeProc:
    def __init__(self, hang_once: bool = False) -> None:
        self.stdout = FakeStdout()
        self.stderr = FakeStdout()
        self.pid = 4242
        self.returncode: int | None = None
        self.signals: list[int] = []
        self.killed = False
        self.waits: list[float | None] = []
        self._hang_once = hang_once

    def poll(self) -> int | None:
        return self.returncode

    def send_signal(self, sig: int) -> None:
        self.signals.append(sig)
        self.stdout.eof()  # a dying ffmpeg closes its pipe
        self.stderr.eof()

    def wait(self, timeout: float | None = None) -> int:
        self.waits.append(timeout)
        if self._hang_once and timeout is not None:
            self._hang_once = False
            raise subprocess.TimeoutExpired("ffmpeg", timeout)
        self.returncode = 0 if self.returncode is None else self.returncode
        return self.returncode

    def kill(self) -> None:
        self.killed = True
        self.returncode = -9


def _noise_bytes(seconds: float) -> bytes:
    rng = np.random.default_rng(2)
    return rng.uniform(-0.1, 0.1, round(seconds * SR)).astype("<f4").tobytes()


def _zero_bytes(seconds: float) -> bytes:
    return np.zeros(round(seconds * SR), dtype="<f4").tobytes()


def test_ffmpeg_argv_is_pinned() -> None:
    procs: list[tuple[list[str], dict]] = []

    def popen(argv, **kw):
        procs.append((argv, kw))
        return FakeProc()

    src = FfmpegMicSource("Built-in Microphone", popen=popen, clock=Clock())
    src.start()
    try:
        argv, kw = procs[0]
        assert argv == [
            "ffmpeg", "-hide_banner", "-nostdin", "-fflags", "nobuffer",
            "-flush_packets", "1", "-f", "avfoundation",
            "-i", ":Built-in Microphone", "-ac", "1", "-ar", "16000",
            "-af", "aresample=async=1", "-f", "f32le", "pipe:1",
        ]  # fmt: skip
        assert kw["start_new_session"] is True
    finally:
        drain_after_stop(src)


def test_ffmpeg_argv_resamples_on_device_timestamps() -> None:
    # Without aresample=async=1 avfoundation's short 48 kHz delivery (~89 %)
    # is never corrected and the mic stem drifts (#199).
    procs: list[tuple[list[str], dict]] = []

    def popen(argv, **kw):
        procs.append((argv, kw))
        return FakeProc()

    src = FfmpegMicSource("Mic", popen=popen, clock=Clock())
    src.start()
    try:
        argv, _kw = procs[0]
        assert "-af" in argv
        assert argv[argv.index("-af") + 1] == "aresample=async=1"
        assert argv[argv.index("-ar") + 1] == "16000"
    finally:
        drain_after_stop(src)


def test_ffmpeg_stderr_is_piped_and_logged_at_debug(caplog) -> None:
    procs: list[tuple[FakeProc, dict]] = []

    def popen(_argv, **kw):
        proc = FakeProc()
        procs.append((proc, kw))
        return proc

    src = FfmpegMicSource("Mic", popen=popen, clock=Clock())
    with caplog.at_level(logging.DEBUG, logger="vemoizer.capture_sources"):
        src.start()
        proc, kw = procs[0]
        assert kw["stderr"] is subprocess.PIPE
        proc.stderr.push(b"Input #0, avfoundation\n[mic] progress\r")
        wait_until(lambda: any("progress" in r.getMessage() for r in caplog.records))
        # stdout keeps flowing while stderr is still open (never blocks the pump).
        proc.stdout.push(_noise_bytes(0.1))
        drain_after_stop(src)
    debug = [r for r in caplog.records if r.levelno == logging.DEBUG]
    assert any("Input #0, avfoundation" in r.getMessage() for r in debug)


def test_restart_re_resolves_default_input(monkeypatch) -> None:
    names = iter(["Mic A", "Mic B"])
    procs: list[tuple[list[str], FakeProc]] = []

    def popen(argv, **_kw):
        proc = FakeProc()
        procs.append((argv, proc))
        return proc

    src = FfmpegMicSource(None, popen=popen, resolve=lambda: next(names), clock=Clock())
    src.start()
    procs[0][1].stdout.push(_noise_bytes(0.5))
    procs[0][1].stdout.eof()  # unexpected exit
    wait_until(lambda: len(procs) == 2)
    assert ":Mic B" in procs[1][0]
    drain_after_stop(src)


def test_restart_gap_is_wall_clock_discontinuity() -> None:
    clk = Clock()
    procs: list[FakeProc] = []

    def popen(_argv, **_kw):
        proc = FakeProc()
        procs.append(proc)
        return proc

    src = FfmpegMicSource("Mic", popen=popen, clock=clk)
    src.start()
    clk.t = 10.0
    procs[0].stdout.push(_noise_bytes(1.0))  # anchor = 10.0 - 1.0 = 9.0
    procs[0].stdout.eof()
    wait_until(lambda: len(procs) == 2)
    clk.t = 15.0
    procs[1].stdout.push(_noise_bytes(0.5))  # anchor = 15.0 - 0.5 = 14.5
    src.stop()
    chunks = list(src.chunks())
    assert chunks[0].first_sample_t == pytest.approx(9.0)
    assert chunks[1].first_sample_t == pytest.approx(14.5)
    # previous chunk ended at 10.0; the restart lands at 14.5: a 4.5 s gap.
    assert chunks[1].discontinuity_frames == round(4.5 * SR)


def test_arrival_jitter_in_source_clock_inserts_no_zeros(tmp_path) -> None:
    # Twenty 0.5 s blocks, each arriving up to 600 ms late. Chunk times follow
    # the sample count, so the stem must come out contiguous with no padding.
    rng = np.random.default_rng(3)
    arrivals = iter(np.arange(20) * 0.5 + rng.uniform(0.0, 0.6, 20))
    procs: list[FakeProc] = []

    def popen(_argv, **_kw):
        proc = FakeProc()
        procs.append(proc)
        return proc

    src = FfmpegMicSource("Mic", popen=popen, clock=lambda: next(arrivals))
    src.start()
    block = _noise_bytes(0.5)
    for _ in range(20):
        procs[0].stdout.push(block)
    src.stop()
    chunks = list(src.chunks())
    assert len(chunks) == 20
    assert all(c.discontinuity_frames == 0 for c in chunks)
    np.testing.assert_allclose(np.diff([c.first_sample_t for c in chunks]), 0.5)
    writer = TrackWriter(tmp_path / "mic.flac", t0=chunks[0].first_sample_t)
    for chunk in chunks:
        writer.write(chunk)
    writer.close()
    y, _sr = sf.read(tmp_path / "mic.flac", dtype="float32")
    assert len(y) == 20 * round(0.5 * SR)


def test_exact_zeros_after_restart_warn(caplog) -> None:
    clk = Clock()
    procs: list[FakeProc] = []

    def popen(_argv, **_kw):
        proc = FakeProc()
        procs.append(proc)
        return proc

    src = FfmpegMicSource("Mic", popen=popen, clock=clk)
    src.start()
    procs[0].stdout.eof()
    wait_until(lambda: len(procs) == 2)
    clk.t = 1.0
    with caplog.at_level(logging.WARNING, logger="vemoizer.capture_sources"):
        procs[1].stdout.push(_zero_bytes(0.5))
        wait_until(lambda: any("exact zeros" in r.getMessage() for r in caplog.records))
    drain_after_stop(src)


def test_stop_sends_sigint_then_waits_then_kills() -> None:
    proc = FakeProc(hang_once=True)
    src = FfmpegMicSource("Mic", popen=lambda *_a, **_k: proc, clock=Clock())
    src.start()
    src.stop()
    assert proc.signals == [signal.SIGINT]
    assert proc.waits[0] is not None  # bounded wait first
    assert proc.killed
    assert proc.returncode is not None  # reaped


def test_gives_up_after_repeated_spawn_failures() -> None:
    first = FakeProc()
    spawns = []

    def popen(_argv, **_kw):
        spawns.append(_argv)
        if len(spawns) == 1:
            return first
        raise OSError(2, "ffmpeg: not found")

    src = FfmpegMicSource("Mic", popen=popen, clock=Clock())
    src.start()
    first.stdout.eof()  # unexpected exit; every restart then fails to spawn
    wait_until(lambda: src.failed)
    src.stop()


# --- factory stub, marker and import-hygiene checks --------------------------


def test_real_capture_factory_is_stubbed_by_conftest(system_effects) -> None:
    with pytest.raises(RuntimeError, match="stubbed"):
        capture_sources._open_real_sources("MacBook Pro Microphone")
    assert system_effects["capture"] == [["MacBook Pro Microphone"]]


def test_hardware_marker_excluded_by_default() -> None:
    config = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
    addopts = config["tool"]["pytest"]["ini_options"]["addopts"]
    assert "not hardware" in addopts


def test_cli_help_does_not_import_catap() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys, vemoizer.cli; print('catap' in sys.modules)",
        ],
        capture_output=True,
        text=True,
        check=True,
        cwd=REPO,
    )
    assert result.stdout.strip() == "False"


@pytest.mark.hardware
def test_real_mic_delivers_16k_samples_per_second() -> None:
    # Arrival times come from our clock, so the rate is measured in wall time
    # (not from sample-count timestamps, which would hide the drift).
    arrivals: list[float] = []

    def clock() -> float:
        arrivals.append(time.monotonic())
        return arrivals[-1]

    src = FfmpegMicSource(clock=clock)
    src.start()
    time.sleep(13.0)
    chunks = drain_after_stop(src)
    assert len(chunks) == len(arrivals)
    start = arrivals[-1] - 10.0
    first = next(i for i, t in enumerate(arrivals) if t >= start)
    samples = sum(len(c.samples) for c in chunks[first + 1 :])
    rate = samples / (arrivals[-1] - arrivals[first])
    assert rate == pytest.approx(SR, rel=0.01)


@pytest.mark.hardware
def test_real_system_tap_produces_nonzero_while_afplay_plays() -> None:
    src = CatapSystemSource()
    src.start()
    player = subprocess.Popen(["afplay", "/System/Library/Sounds/Ping.aiff"])
    time.sleep(1.5)
    player.wait()
    chunks = drain_after_stop(src)
    peak = max(
        (float(np.max(np.abs(c.samples))) for c in chunks if len(c.samples)),
        default=0.0,
    )
    assert peak > 0.0
