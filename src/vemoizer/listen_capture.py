"""Per-source pumps and the start-up preflight for ``listen`` (issue #173).

A pump thread moves one source's chunks into its track writer and into a
:class:`TrackState` the main thread reads. Nothing here opens hardware; the
sources arrive already started.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from vemoizer.audio_contract import SAMPLE_RATE
from vemoizer.capture import Chunk, SilenceWatchdog, is_digital_silence

#: How long a source may take to deliver its first audio (ffmpeg avfoundation
#: can need over a second to open the device).
START_WAIT_S = 4.0

#: Exact zeros over this many seconds of audio count as a dead input.
ZERO_WINDOW_S = 2.0

#: Extra wall time allowed for the zero window to fill before judging it.
PREFLIGHT_SLACK_S = 2.0

_POLL_S = 0.05
_ZERO_WINDOW_SAMPLES = round(ZERO_WINDOW_S * SAMPLE_RATE)


class _Writer(Protocol):
    def write(self, chunk: Chunk) -> bool: ...


class _Source(Protocol):
    def chunks(self) -> Any: ...


class TrackState:
    """What the main thread needs to know about one source's audio.

    Written by the pump thread, read by the main thread; every access holds
    the lock. Empty chunks are markers, not audio, and never count.
    """

    def __init__(self, name: str, watchdog: SilenceWatchdog | None = None) -> None:
        self.name = name
        self._watchdog = watchdog
        self._lock = threading.Lock()
        self._seen = 0
        self._nonzero_in_window = False
        self._end_t: float | None = None

    @property
    def started(self) -> bool:
        with self._lock:
            return self._seen > 0

    @property
    def window_done(self) -> bool:
        with self._lock:
            return self._seen >= _ZERO_WINDOW_SAMPLES

    @property
    def end_t(self) -> float | None:
        """Session-clock time just after the latest audio; None before any."""
        with self._lock:
            return self._end_t

    @property
    def silent(self) -> bool:
        """The first two seconds of audio were all exact zeros."""
        with self._lock:
            return self._seen >= _ZERO_WINDOW_SAMPLES and not self._nonzero_in_window

    def observe(self, chunk: Chunk) -> None:
        samples = chunk.samples
        if samples.size == 0:
            return
        with self._lock:
            room = max(0, _ZERO_WINDOW_SAMPLES - self._seen)
            if room and np.any(samples[:room]):
                self._nonzero_in_window = True
            self._seen += samples.size
            self._end_t = chunk.first_sample_t + samples.size / SAMPLE_RATE
            if self._watchdog is not None and not is_digital_silence(samples):
                self._watchdog.signal()

    def check_watchdog(self) -> bool:
        """True once when the source has been silent for the watchdog's span."""
        with self._lock:
            return self._watchdog is not None and self._watchdog.check()


@dataclass(frozen=True)
class Problem:
    """A preflight finding: ``kind`` is ``not_started`` or ``silent``."""

    track: str
    kind: str


def _wait(
    done: Callable[[], bool],
    clock: Callable[[], float],
    sleep: Callable[[float], None],
    deadline: float,
) -> None:
    while not done() and clock() < deadline:
        sleep(_POLL_S)


def preflight(
    states: Sequence[TrackState],
    clock: Callable[[], float],
    sleep: Callable[[float], None],
    t0: float,
) -> list[Problem]:
    """Judge the sources before any recording is trusted.

    Waits up to :data:`START_WAIT_S` for every source's first audio; a source
    that never starts is reported (and the zero window is not waited for).
    Otherwise waits for each source's zero window and reports the silent ones.
    """
    _wait(
        lambda: all(s.started for s in states),
        clock,
        sleep,
        t0 + START_WAIT_S,
    )
    missing = [Problem(s.name, "not_started") for s in states if not s.started]
    if missing:
        return missing
    _wait(
        lambda: all(s.window_done for s in states),
        clock,
        sleep,
        t0 + START_WAIT_S + ZERO_WINDOW_S + PREFLIGHT_SLACK_S,
    )
    return [Problem(s.name, "silent") for s in states if s.silent]


def pump(source: _Source, writer: _Writer, state: TrackState) -> threading.Thread:
    """Move *source*'s chunks into *writer* and *state* until the source ends."""

    def run() -> None:
        for chunk in source.chunks():
            state.observe(chunk)
            writer.write(chunk)

    thread = threading.Thread(target=run, name=f"listen-{state.name}", daemon=True)
    thread.start()
    return thread
