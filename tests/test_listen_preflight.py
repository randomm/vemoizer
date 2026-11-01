"""Preflight, track state and the source pump for ``listen`` (issue #173).

Fake clock and fake sources only: no Core Audio, no avfoundation, no ffmpeg.
"""

from __future__ import annotations

import queue

import numpy as np
import pytest
from _listen_helpers import FakeClock, tone, zeros

from vemoizer.capture import Chunk, SilenceWatchdog
from vemoizer.listen_capture import TrackState, preflight, pump


def _settle(state: TrackState, chunks: list[Chunk]) -> TrackState:
    for chunk in chunks:
        state.observe(chunk)
    return state


def test_preflight_reports_source_that_never_delivers_after_four_seconds() -> None:
    clock = FakeClock()
    mic = TrackState("mic")
    tap = _settle(TrackState("tap"), [Chunk(tone(2.5), 0.0)])
    problems = preflight([mic, tap], clock, clock.sleep, t0=0.0)
    assert [(p.track, p.kind) for p in problems] == [("mic", "not_started")]
    # The mic keeps the 4 s deadline; it must not inherit the tap's 10 s.
    assert 4.0 <= clock.t < 10.0


def test_preflight_gives_the_system_tap_ten_seconds_to_start() -> None:
    """issue #205: a cold Core Audio tap can be slower than 4 s."""
    clock = FakeClock()
    mic = _settle(TrackState("mic"), [Chunk(tone(2.5), 0.0)])
    problems = preflight([mic, TrackState("tap")], clock, clock.sleep, t0=0.0)
    assert [(p.track, p.kind) for p in problems] == [("tap", "not_started")]
    assert clock.t >= 10.0


def test_preflight_accepts_a_tap_that_starts_after_the_mic_deadline() -> None:
    """A tap delivering its first audio at 9 s is not judged not_started."""
    clock = FakeClock()
    mic = _settle(TrackState("mic"), [Chunk(tone(2.5), 0.0)])
    assert preflight([mic, _SlowTap(9.0, clock)], clock, clock.sleep, t0=0.0) == []


class _SlowTap(TrackState):
    """A tap that starts delivering only once the fake clock reaches *at_s*."""

    def __init__(self, at_s: float, clock: FakeClock) -> None:
        super().__init__("tap")
        self._at_s = at_s
        self._clock = clock
        self._fed = False

    @property
    def started(self) -> bool:
        if not self._fed and self._clock() >= self._at_s:
            self._fed = True
            self.observe(Chunk(tone(2.5), self._clock()))
        return super().started


def test_preflight_ignores_empty_chunks_when_judging_started() -> None:
    state = _settle(TrackState("mic"), [Chunk(np.zeros(0, np.float32), 0.0)])
    assert state.started is False


def test_two_seconds_of_exact_zeros_is_silent() -> None:
    clock = FakeClock()
    mic = _settle(TrackState("mic"), [Chunk(zeros(2.5), 0.0)])
    tap = _settle(TrackState("tap"), [Chunk(tone(2.5), 0.0)])
    problems = preflight([mic, tap], clock, clock.sleep, t0=0.0)
    assert [(p.track, p.kind) for p in problems] == [("mic", "silent")]


def test_one_second_of_zeros_then_signal_is_not_silent() -> None:
    clock = FakeClock()
    samples = np.concatenate([zeros(1.0), tone(2.0)])
    mic = _settle(TrackState("mic"), [Chunk(samples, 0.0)])
    tap = _settle(TrackState("tap"), [Chunk(tone(2.5), 0.0)])
    assert preflight([mic, tap], clock, clock.sleep, t0=0.0) == []


def test_zeros_after_the_two_second_window_do_not_count() -> None:
    clock = FakeClock()
    samples = np.concatenate([tone(2.0), zeros(3.0)])
    mic = _settle(TrackState("mic"), [Chunk(samples, 0.0)])
    tap = _settle(TrackState("tap"), [Chunk(tone(2.5), 0.0)])
    assert preflight([mic, tap], clock, clock.sleep, t0=0.0) == []


def test_watchdog_fires_once_after_silence_and_rearms_on_signal() -> None:
    clock = FakeClock()
    watchdog = SilenceWatchdog(180.0, clock)
    state = TrackState("tap", watchdog=watchdog)
    clock.t = 179.0
    assert state.check_watchdog() is False
    clock.t = 181.0
    assert state.check_watchdog() is True
    assert state.check_watchdog() is False
    state.observe(Chunk(tone(0.5), 181.0))
    clock.t = 400.0
    assert state.check_watchdog() is True


def test_digital_silence_chunks_do_not_reset_the_watchdog() -> None:
    clock = FakeClock()
    state = TrackState("tap", watchdog=SilenceWatchdog(180.0, clock))
    state.observe(Chunk(zeros(0.5), 0.0))
    clock.t = 200.0
    assert state.check_watchdog() is True


class _Writer:
    def __init__(self) -> None:
        self.chunks: list[Chunk] = []

    def write(self, chunk: Chunk) -> bool:
        self.chunks.append(chunk)
        return True


class _Source:
    def __init__(self) -> None:
        self._q: queue.SimpleQueue[Chunk | None] = queue.SimpleQueue()

    def feed(self, chunk: Chunk) -> None:
        self._q.put(chunk)

    def chunks(self):
        while (item := self._q.get()) is not None:
            yield item

    def end(self) -> None:
        self._q.put(None)


def test_pump_writes_every_chunk_and_observes_it_until_the_source_ends() -> None:
    source = _Source()
    writer = _Writer()
    state = TrackState("mic")
    thread = pump(source, writer, state)
    source.feed(Chunk(tone(0.5), 0.0))
    source.feed(Chunk(tone(0.5), 0.5))
    source.end()
    thread.join(timeout=5)
    assert not thread.is_alive()
    assert len(writer.chunks) == 2
    assert state.started is True


def test_track_state_end_is_the_end_of_the_latest_audio_on_the_session_clock() -> None:
    state = _settle(TrackState("tap"), [Chunk(tone(1.0), 2.0), Chunk(tone(0.5), 3.0)])
    assert state.end_t == pytest.approx(3.5)


def test_track_state_end_is_none_until_real_audio_arrives() -> None:
    state = _settle(TrackState("mic"), [Chunk(np.zeros(0, np.float32), 5.0)])
    assert state.end_t is None
