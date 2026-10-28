"""TrackWriter subscribers (issue #180): the live path sees exactly what was written.

No hardware and no files: the writer's file object is a recording fake.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest

from vemoizer.capture import Chunk, TrackWriter

SR = 16_000


class _RecordingFile:
    def __init__(self) -> None:
        self.writes: list[np.ndarray] = []

    def write(self, data: np.ndarray) -> None:
        self.writes.append(np.array(data, copy=True))

    def flush(self) -> None:
        return None

    def close(self) -> None:
        return None


def _writer(t0: float = 0.0) -> tuple[TrackWriter, _RecordingFile]:
    file = _RecordingFile()
    return TrackWriter("unused.flac", t0, open_file=lambda _p: file), file


def test_subscriber_receives_the_same_arrays_the_writer_wrote() -> None:
    writer, file = _writer(t0=0.0)
    seen: list[np.ndarray] = []
    writer.subscribe(seen.append)
    # First chunk is head-padded (0.5 s), then a discontinuity pad of 100 frames.
    writer.write(Chunk(np.full(320, 0.25, dtype=np.float32), 0.5))
    writer.write(
        Chunk(np.full(160, -0.5, dtype=np.float32), 0.52, discontinuity_frames=100)
    )

    assert len(seen) == len(file.writes) == 4
    for got, wrote in zip(seen, file.writes, strict=True):
        np.testing.assert_array_equal(got, wrote)
    assert np.concatenate(seen).size == writer.position_s * SR


def test_subscriber_exception_is_logged_and_never_stops_the_writer(
    caplog: pytest.LogCaptureFixture,
) -> None:
    writer, file = _writer()
    other: list[np.ndarray] = []

    def broken(_block: np.ndarray) -> None:
        raise RuntimeError("vad exploded")

    writer.subscribe(broken)
    writer.subscribe(other.append)
    with caplog.at_level(logging.WARNING, logger="vemoizer.capture"):
        results = [
            writer.write(Chunk(np.full(160, 0.1, dtype=np.float32), i * 0.01))
            for i in range(5)
        ]

    assert results == [True] * 5
    assert writer.usable
    assert sum(len(w) for w in file.writes) == 5 * 160
    assert len(other) == 5  # the healthy subscriber keeps receiving
    assert "vad exploded" in caplog.text


def test_failing_subscriber_is_detached_after_its_first_error(
    caplog: pytest.LogCaptureFixture,
) -> None:
    writer, _file = _writer()
    calls: list[int] = []

    def broken(_block: np.ndarray) -> None:
        calls.append(1)
        raise ValueError("bad block")

    writer.subscribe(broken)
    with caplog.at_level(logging.WARNING, logger="vemoizer.capture"):
        for i in range(3):
            writer.write(Chunk(np.full(160, 0.1, dtype=np.float32), i * 0.01))

    assert calls == [1]
    assert caplog.text.count("bad block") == 1


def test_no_subscriber_call_for_a_block_the_file_refused() -> None:
    class _FailingFile(_RecordingFile):
        def write(self, data: np.ndarray) -> None:
            raise OSError(28, "No space left on device")

    file = _FailingFile()
    writer = TrackWriter("unused.flac", 0.0, open_file=lambda _p: file)
    seen: list[np.ndarray] = []
    writer.subscribe(seen.append)

    assert writer.write(Chunk(np.full(160, 0.1, dtype=np.float32), 0.0)) is False
    assert seen == []
