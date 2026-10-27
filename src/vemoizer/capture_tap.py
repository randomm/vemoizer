"""Tap-side timeline and resampling for live capture (issue #172).

Native catap buffers in, 16 kHz mono chunks out, on one continuous timeline.
The soxr module is injected, so this file never imports catap or Core Audio.
Runs on capture_sources' worker thread only.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from vemoizer.audio_contract import SAMPLE_RATE
from vemoizer.capture import Chunk

log = logging.getLogger(__name__)

RATE_WINDOW_S = 2.0
RATE_TOLERANCE = 0.05
STANDARD_RATES = (8000, 11025, 16000, 22050, 24000, 32000, 44100, 48000, 88200, 96000)


class UnsupportedTapFormat(ValueError):
    """The tap delivered something other than mono float32."""


class TapConverter:
    """Native tap buffers in, 16 kHz chunks out, on one continuous timeline.

    Buffers are held for ``RATE_WINDOW_S`` so the real delivery rate can be
    checked against the declared one before the soxr stream is built.
    """

    def __init__(self, soxr: Any) -> None:
        self._soxr = soxr
        self._declared: float | None = None
        self._stream: Any = None
        self._pending: list[tuple[Any, float]] = []
        self._anchor: float | None = None
        self._frames_out = 0
        self._prev_end: float | None = None
        self._gap = 0
        self._jump = 0
        self._prev_in: float | None = None

    def feed(self, buf: Any, arrival: float) -> list[Chunk]:
        if not buf.format.is_float or buf.format.num_channels != 1:
            raise UnsupportedTapFormat("tap must be mono float32")
        rate = float(buf.format.sample_rate)
        out: list[Chunk] = []
        if rate != self._declared:
            out += self._end_run()
            if self._declared is not None:
                out.append(self._marker(rate))
            self._declared = rate
            self._new_run()
        if self._stream is None:
            self._pending.append((buf, arrival))
            if arrival - self._pending[0][1] >= RATE_WINDOW_S:
                out += self._validate_and_convert()
            return out
        out += self._convert(buf, arrival)
        return out

    def flush(self) -> list[Chunk]:
        return self._end_run()

    def reset(self) -> list[Chunk]:
        """Drop unvalidated audio after a failure; keep the gap for the reopen."""
        out: list[Chunk] = []
        if self._stream is not None:
            out = self._emit(
                self._stream.resample_chunk(np.zeros(0, np.float32), last=True)
            )
        self._stream = None
        self._pending = []
        self._declared = None
        self._new_run()
        return out

    def _new_run(self) -> None:
        """Start a new timeline run; its gap comes from wall-clock arrival times.

        input_sample_time is only comparable within one run: after a format
        change the new stream's counter may restart or continue in other units.
        """
        self._anchor = None
        self._frames_out = 0
        self._prev_in = None
        self._jump = 0

    def _end_run(self) -> list[Chunk]:
        out: list[Chunk] = []
        if self._pending:
            out += self._validate_and_convert()
        if self._stream is not None:
            tail = self._stream.resample_chunk(np.zeros(0, np.float32), last=True)
            out += self._emit(tail)
            self._stream = None
        return out

    def _marker(self, rate: float) -> Chunk:
        first_t = (self._anchor or 0.0) + self._frames_out / SAMPLE_RATE
        log.info("tap format changed to %.0f Hz; stream rebuilt", rate)
        return Chunk(np.zeros(0, np.float32), first_t, 0)

    def _validate_and_convert(self) -> list[Chunk]:
        rate = self._measured_rate()
        self._stream = self._soxr.ResampleStream(rate, SAMPLE_RATE, 1, dtype="float32")
        pending, self._pending = self._pending, []
        if self._anchor is None:
            first, first_arrival = pending[0]
            self._anchor = first_arrival - first.frame_count / first.format.sample_rate
            self._gap = 0
            if self._prev_end is not None:
                self._gap = max(0, round((self._anchor - self._prev_end) * SAMPLE_RATE))
            # The gap zeros lie before the anchor, so the first chunk's samples
            # start exactly at the anchor once _emit adds them back.
            self._frames_out = -self._gap
        out: list[Chunk] = []
        for buf, arrival in pending:
            out += self._convert(buf, arrival)
        return out

    def _measured_rate(self) -> int:
        declared = self._declared or float(self._pending[0][0].format.sample_rate)
        span = self._pending[-1][1] - self._pending[0][1]
        if len(self._pending) < 2 or span <= 0:
            return int(declared)
        frames = sum(buf.frame_count for buf, _ in self._pending[1:])
        measured = frames / span
        if abs(measured / declared - 1) <= RATE_TOLERANCE:
            return int(declared)
        actual = min(STANDARD_RATES, key=lambda r: abs(r - measured))
        log.warning(
            "system tap declared %.0f Hz but delivers ~%.0f Hz; resampling from %d Hz",
            declared,
            measured,
            actual,
        )
        return actual

    def _convert(self, buf: Any, arrival: float) -> list[Chunk]:
        rate = float(buf.format.sample_rate)
        if buf.input_sample_time is not None:
            if self._prev_in is not None and buf.input_sample_time > self._prev_in:
                jump = buf.input_sample_time - self._prev_in
                self._jump += round(jump * SAMPLE_RATE / rate)
            self._prev_in = buf.input_sample_time + buf.frame_count
        return self._emit(
            self._stream.resample_chunk(np.frombuffer(buf.data, dtype="<f4"))
        )

    def _emit(self, y: np.ndarray) -> list[Chunk]:
        disc = self._gap + self._jump
        self._gap = 0
        self._jump = 0
        if not len(y) and not disc:
            return []
        assert self._anchor is not None  # set before any conversion runs
        # The writer puts the disc zeros before y, so y starts after them. The
        # timeline advances by the zeros too, or the next run's gap counts them twice.
        first_t = self._anchor + (self._frames_out + disc) / SAMPLE_RATE
        self._frames_out += disc + len(y)
        self._prev_end = first_t + len(y) / SAMPLE_RATE
        return [Chunk(y, first_t, disc)]
