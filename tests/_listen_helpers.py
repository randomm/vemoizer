"""Fakes for the ``listen`` tests (issue #173): a clock, sources, hardware.

No real source is ever built here; ``fake_hardware`` returns started fakes
in place of ``capture_sources._open_real_sources``.
"""

from __future__ import annotations

import queue
import threading
from collections.abc import Callable, Sequence

import numpy as np

from vemoizer.capture import Chunk, drain

SR = 16_000


class FakeClock:
    """Time moves only through ``sleep``; the real sleep yields to the pump threads."""

    def __init__(self) -> None:
        self.t = 0.0
        self.sleeps = 0

    def __call__(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> None:
        self.sleeps += 1
        self.t += seconds
        threading.Event().wait(0.001)


def tone(seconds: float, value: float = 0.2) -> np.ndarray:
    return np.full(round(seconds * SR), value, dtype=np.float32)


def zeros(seconds: float) -> np.ndarray:
    return np.zeros(round(seconds * SR), dtype=np.float32)


class FakeSource:
    """A started source that delivers its queued chunks until ``stop()``."""

    failed = False

    def __init__(self, chunks: Sequence[Chunk] = ()) -> None:
        self._chunks = list(chunks)
        self._q: queue.SimpleQueue[Chunk | None] = queue.SimpleQueue()
        self.started = False
        self.stop_calls = 0

    def start(self) -> None:
        self.started = True
        for chunk in self._chunks:
            self._q.put(chunk)

    def chunks(self):
        return drain(self._q)

    def stop(self) -> None:
        self.stop_calls += 1
        self._q.put(None)


def fake_hardware(
    mic_chunks: Sequence[Chunk],
    tap_chunks: Sequence[Chunk],
    opened: list[str | None] | None = None,
) -> Callable[[str | None], tuple[FakeSource, FakeSource]]:
    """A ``_open_real_sources`` stand-in; records the ``--mic`` name it got."""

    def open_sources(mic_name: str | None) -> tuple[FakeSource, FakeSource]:
        if opened is not None:
            opened.append(mic_name)
        mic, tap = FakeSource(mic_chunks), FakeSource(tap_chunks)
        mic.start()
        tap.start()
        return mic, tap

    return open_sources
