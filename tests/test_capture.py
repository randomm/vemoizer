"""Pure capture layer (issue #172): TrackWriter placement, silence math, watchdog, skew.

No hardware, no network: the writer's file object is a fake except for the
real-file tests (flush readability, disk full), which write FLAC into ``tmp_path``.
"""

from __future__ import annotations

import logging
import resource
import shutil
import signal
import subprocess

import numpy as np
import pytest
import soundfile as sf

from vemoizer.capture import (
    Chunk,
    SilenceWatchdog,
    SkewLogger,
    TrackWriter,
    is_digital_silence,
)

SR = 16_000


class FakeFile:
    """Stands in for ``soundfile.SoundFile``; optionally raises on write N or close."""

    def __init__(
        self,
        fail_on_write: int | None = None,
        error: BaseException | None = None,
        close_error: BaseException | None = None,
    ) -> None:
        self.writes: list[np.ndarray] = []
        self.flushes = 0
        self.closed = False
        self._calls = 0
        self._fail_on_write = fail_on_write
        self._error = error
        self._close_error = close_error

    def write(self, data: np.ndarray) -> None:
        self._calls += 1
        if self._fail_on_write is not None and self._calls == self._fail_on_write:
            raise self._error or OSError(28, "No space left on device")
        self.writes.append(np.array(data, copy=True))

    def flush(self) -> None:
        self.flushes += 1

    def close(self) -> None:
        self.closed = True
        if self._close_error is not None:
            raise self._close_error

    @property
    def total(self) -> int:
        return sum(len(w) for w in self.writes)


def _writer(fake: FakeFile, t0: float = 0.0, **kwargs) -> TrackWriter:
    return TrackWriter("unused.flac", t0, open_file=lambda _path: fake, **kwargs)


def test_three_chunks_round_trip_within_one_24bit_lsb(tmp_path) -> None:
    rng = np.random.default_rng(0)
    x = rng.uniform(-0.5, 0.5, SR * 3).astype(np.float32)
    path = tmp_path / "me.flac"
    writer = TrackWriter(path, t0=100.0)
    for i in range(3):
        writer.write(Chunk(x[i * SR : (i + 1) * SR], 100.0 + i))
    writer.close()

    y, sr = sf.read(path, dtype="float32")
    assert sr == SR
    np.testing.assert_allclose(y, x, atol=1 / 2**23)


def test_contiguous_chunks_insert_no_zeros() -> None:
    fake = FakeFile()
    writer = _writer(fake)
    for i in range(6):
        writer.write(Chunk(np.full(8000, 0.25, np.float32), first_sample_t=i * 0.5))
    assert fake.total == 6 * 8000
    assert not any(np.all(w == 0.0) for w in fake.writes)


def test_discontinuity_inserts_exactly_that_many_zeros() -> None:
    fake = FakeFile()
    writer = _writer(fake)
    writer.write(Chunk(np.full(1000, 0.5, np.float32), 0.0))
    writer.write(
        Chunk(np.full(1000, 0.5, np.float32), 0.0625, discontinuity_frames=1234)
    )
    zeros = np.concatenate(fake.writes[1:2])
    assert len(zeros) == 1234 and not np.any(zeros)
    assert fake.total == 1000 + 1234 + 1000


def test_late_track_is_head_padded_to_shared_index_zero() -> None:
    fake = FakeFile()
    writer = _writer(fake, t0=10.0)
    writer.write(Chunk(np.full(SR, 0.1, np.float32), first_sample_t=10.4))
    assert fake.total == SR + round(0.4 * SR)
    assert not np.any(fake.writes[0][: round(0.4 * SR)])


def test_restart_chunk_lands_at_its_first_sample_time() -> None:
    # The source turns a wall-clock restart gap into discontinuity_frames;
    # the writer then places the restarted audio at the wall-clock position.
    fake = FakeFile()
    writer = _writer(fake)
    writer.write(Chunk(np.full(SR, 0.1, np.float32), 0.0))
    gap = 2 * SR
    writer.write(Chunk(np.full(SR, 0.2, np.float32), 3.0, discontinuity_frames=gap))
    assert fake.total == SR + gap + SR
    assert writer.position_s == pytest.approx(4.0)


def test_skew_is_logged_both_ways_and_never_corrected(caplog) -> None:
    logger = SkewLogger(every_s=60.0)
    with caplog.at_level(logging.INFO, logger="vemoizer.capture"):
        assert logger.observe(me_t=30.0, them_t=30.2) is None  # before 60 s: silent
        ahead = logger.observe(me_t=60.0, them_t=60.2)
        behind = logger.observe(me_t=120.0, them_t=119.9)
    assert ahead == pytest.approx(0.2)
    assert behind == pytest.approx(-0.1)
    messages = [r.getMessage() for r in caplog.records]
    assert any("them ahead" in m for m in messages)
    assert any("them behind" in m for m in messages)


def test_flush_is_called_every_interval() -> None:
    fake = FakeFile()
    writer = _writer(fake, flush_frames=SR)
    for i in range(3):
        writer.write(Chunk(np.full(SR // 2, 0.1, np.float32), i * 0.5))
    assert fake.flushes == 1  # 1.5 s written: one flush at the 1 s mark
    writer.close()


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not on PATH")
def test_flushed_flac_is_readable_before_close(tmp_path) -> None:
    # The ingest decoder is ffmpeg. soundfile cannot read an unclosed FLAC (its
    # header frame count is unset), but ffmpeg decodes everything the scheduled
    # flushes wrote; only the last partial FLAC block stays buffered until close().
    path = tmp_path / "me.flac"
    writer = TrackWriter(path, t0=0.0)
    for i in range(20):
        writer.write(Chunk(np.full(SR, 0.1, np.float32), float(i)))
    decoded = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(path), "-f", "f32le", "-"],
        capture_output=True,
        check=False,
    ).stdout
    assert len(decoded) // 4 >= 19 * SR
    writer.close()
    y, _sr = sf.read(path, dtype="float32")
    assert len(y) == 20 * SR


@pytest.mark.parametrize(
    "error",
    [
        pytest.param(OSError(28, "No space left on device"), id="oserror"),
        pytest.param(sf.LibsndfileError(28, "System error."), id="libsndfile"),
        pytest.param(AssertionError("short write"), id="soundfile-short-write"),
    ],
)
def test_write_failure_stops_recording_and_keeps_earlier_data(caplog, error) -> None:
    fake = FakeFile(fail_on_write=3, error=error)
    writer = _writer(fake)
    with caplog.at_level(logging.ERROR, logger="vemoizer.capture"):
        results = [
            writer.write(Chunk(np.full(100, 0.1, np.float32), i * 100 / SR))
            for i in range(4)
        ]
    assert results == [True, True, False, False]
    assert fake.total == 200  # the first two writes are intact
    assert fake.closed
    assert not writer.usable
    errors = [r for r in caplog.records if r.levelno == logging.ERROR]
    assert len(errors) == 1


def test_close_failure_is_logged_not_raised(caplog) -> None:
    fake = FakeFile(close_error=sf.LibsndfileError(5, "System error."))
    writer = _writer(fake)
    writer.write(Chunk(np.full(100, 0.1, np.float32), 0.0))
    with caplog.at_level(logging.ERROR, logger="vemoizer.capture"):
        writer.close()
    assert [r.levelno for r in caplog.records] == [logging.ERROR]


def test_real_libsndfile_disk_full_stops_recording(tmp_path, caplog) -> None:
    # RLIMIT_FSIZE makes the real libsndfile write fail the way a full disk does
    # (LibsndfileError, not OSError). The limit is restored before the test ends.
    path = tmp_path / "me.flac"
    writer = TrackWriter(path, t0=0.0, flush_frames=SR)
    writer.write(Chunk(np.zeros(SR, np.float32), 0.0))
    soft, hard = resource.getrlimit(resource.RLIMIT_FSIZE)
    previous = signal.signal(signal.SIGXFSZ, signal.SIG_IGN)
    resource.setrlimit(resource.RLIMIT_FSIZE, (path.stat().st_size + 512, hard))
    rng = np.random.default_rng(0)
    try:
        with caplog.at_level(logging.ERROR, logger="vemoizer.capture"):
            results = [
                writer.write(Chunk(rng.uniform(-0.9, 0.9, SR).astype(np.float32), 1.0))
                for _ in range(10)
            ]
    finally:
        resource.setrlimit(resource.RLIMIT_FSIZE, (soft, hard))
        signal.signal(signal.SIGXFSZ, previous)
    assert False in results
    assert not writer.usable
    assert len([r for r in caplog.records if r.levelno == logging.ERROR]) == 1
    writer.close()


def test_out_of_range_samples_are_counted_and_warned_once(caplog) -> None:
    x = np.array([0.5, 1.2, -1.5, 0.1], dtype=np.float32)
    writer = _writer(FakeFile())
    with caplog.at_level(logging.WARNING, logger="vemoizer.capture"):
        writer.write(Chunk(x, 0.0))
        writer.write(Chunk(x, 4 / SR))
    assert writer.clipped == 4
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1


def test_is_digital_silence_is_exact_zero_only() -> None:
    assert is_digital_silence(np.zeros(1000, np.float32))
    rng = np.random.default_rng(1)
    minus_80_dbfs = rng.normal(0.0, 1e-4, 1000).astype(np.float32)
    assert not is_digital_silence(minus_80_dbfs)


def test_watchdog_fires_once_and_rearms_on_signal() -> None:
    now = [0.0]
    watchdog = SilenceWatchdog(seconds=2.0, clock=lambda: now[0])
    now[0] = 1.9
    assert not watchdog.check()
    now[0] = 2.0
    assert watchdog.check()
    now[0] = 5.0
    assert not watchdog.check()  # already fired: once only
    watchdog.signal()
    now[0] = 6.9
    assert not watchdog.check()
    now[0] = 7.0
    assert watchdog.check()
