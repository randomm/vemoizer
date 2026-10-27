"""Live lag accounting: lag = emit − (capture_t0 + utterance end) (issue #178, P9).

Pure functions only: nearest-rank percentiles and the lag log. The worker
integration (virtual clock, queue wait) lives in ``test_live_decode.py``.
"""

from __future__ import annotations

import pytest

from vemoizer.live_decode import LagLog, LagSummary, nearest_rank


@pytest.mark.parametrize(
    ("values", "pct", "expected"),
    [
        ([1.0, 2.0, 3.0, 4.0], 50, 2.0),
        ([1.0, 2.0, 3.0, 4.0], 90, 4.0),
        (list(map(float, range(1, 11))), 90, 9.0),
        (list(map(float, range(1, 12))), 90, 10.0),
        ([7.0], 90, 7.0),
    ],
)
def test_nearest_rank_percentile(
    values: list[float], pct: float, expected: float
) -> None:
    assert nearest_rank(values, pct) == expected


def test_nearest_rank_sorts_its_input() -> None:
    assert nearest_rank([9.0, 1.0, 5.0], 50) == 5.0


def test_utterance_lag_is_emit_minus_capture_end() -> None:
    log = LagLog()
    # capture starts at 100.0; utterance ends 5.0 s in; emitted at 106.5.
    log.record(utt_end_s=5.0, capture_t0=100.0, emit_t=106.5, word_end_s=[])

    summary = log.utterance_summary()
    assert summary == LagSummary(count=1, p50=1.5, p90=1.5, max=1.5)


def test_word_lag_is_kept_apart_from_utterance_lag() -> None:
    log = LagLog()
    log.record(utt_end_s=5.0, capture_t0=0.0, emit_t=6.0, word_end_s=[1.0, 4.0])

    assert log.utterance_summary().count == 1
    assert log.word_summary().count == 2
    assert log.word_summary().max == pytest.approx(5.0)  # 6.0 - 1.0
    assert log.utterance_summary().max == pytest.approx(1.0)  # 6.0 - 5.0


def test_p90_over_many_utterances() -> None:
    log = LagLog()
    for lag in range(1, 11):
        log.record(utt_end_s=0.0, capture_t0=0.0, emit_t=float(lag), word_end_s=[])

    summary = log.utterance_summary()
    assert summary.p90 == 9.0
    assert summary.p50 == 5.0
    assert summary.max == 10.0


def test_empty_log_summarizes_to_zero_count() -> None:
    summary = LagLog().utterance_summary()
    assert summary.count == 0
    assert summary.max == 0.0
