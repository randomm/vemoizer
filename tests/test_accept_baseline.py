"""Tests for the accept baseline's handling of unreviewed-text keys (issue #243).

Rows from an unreviewed reference carry ``text_drift`` instead of ``wer`` and
no flag keys; the aggregate, the gate and the tolerances must cope without
crashing and without gating on a key the run did not measure.
"""

from __future__ import annotations

from typing import Any

import pytest

from vemoizer import accept_baseline


def _baseline(**rows: dict[str, Any]) -> dict[str, Any]:
    return {
        "fingerprint": "abc",
        "tolerance": {},
        "excerpts": {k: v for k, v in rows.items() if k != "aggregate"},
        "aggregate": rows.get("aggregate", {}),
    }


_REVIEWED = {
    "speaker_name_acc": 0.9,
    "speaker_mapped_acc": 0.95,
    "split_count": 0,
    "merge_count": 0,
    "glossary_insertions": 0,
    "loop_seconds": 0.0,
    "wer": 0.2,
    "unflagged_error_s": 0.0,
    "flag_precision": 1.0,
    "flag_recall": 1.0,
    "title_must_contain_missing": 0,
    "title_forbid_hits": 0,
}
_UNREVIEWED = {
    key: value
    for key, value in _REVIEWED.items()
    if key not in ("wer", "unflagged_error_s", "flag_precision", "flag_recall")
} | {"text_drift": 0.4}


def test_aggregate_averages_text_drift_and_tolerates_missing_keys() -> None:
    agg = accept_baseline.aggregate([_REVIEWED, _UNREVIEWED])
    assert agg["text_drift"] == pytest.approx(0.4)
    assert agg["wer"] == pytest.approx(0.2)  # only the reviewed row has it
    assert agg["flag_precision"] == pytest.approx(1.0)
    assert agg["speaker_name_acc"] == pytest.approx(0.9)


def test_aggregate_of_only_unreviewed_rows_has_no_wer_key() -> None:
    agg = accept_baseline.aggregate([_UNREVIEWED])
    assert "wer" not in agg
    assert "flag_precision" not in agg
    assert agg["text_drift"] == pytest.approx(0.4)


def test_text_drift_is_reported_never_gated() -> None:
    base = _baseline(E1=dict(_REVIEWED), aggregate=dict(_REVIEWED))
    measured = {"E1": dict(_UNREVIEWED, text_drift=0.99)}
    measured["aggregate"] = accept_baseline.aggregate([measured["E1"]])
    regressions = accept_baseline.compare(measured, base)
    assert all(r.key != "text_drift" for r in regressions)


def test_absent_text_keys_are_not_gated_and_do_not_drift() -> None:
    # The baseline holds reviewed keys; the unreviewed row lacks wer/flags.
    base = _baseline(E1=dict(_REVIEWED), aggregate=dict(_REVIEWED))
    measured = {"E1": dict(_UNREVIEWED)}
    measured["aggregate"] = accept_baseline.aggregate([measured["E1"]])
    assert accept_baseline.compare(measured, base) == []


def test_text_drift_absent_from_baseline_is_not_bench_drift() -> None:
    # A baseline written before the key existed must still gate speaker keys.
    base = _baseline(E1=dict(_REVIEWED), aggregate=dict(_REVIEWED))
    measured = {"E1": dict(_REVIEWED, text_drift=0.3)}
    measured["aggregate"] = accept_baseline.aggregate([measured["E1"]])
    assert accept_baseline.compare(measured, base) == []


def test_gated_key_missing_from_baseline_is_still_drift() -> None:
    base = _baseline(E1=dict(_REVIEWED), aggregate=dict(_REVIEWED))
    base["excerpts"]["E1"] = {k: v for k, v in _REVIEWED.items() if k != "split_count"}
    measured = {"E1": dict(_REVIEWED)}
    measured["aggregate"] = accept_baseline.aggregate([measured["E1"]])
    with pytest.raises(accept_baseline.BaselineDrift, match="split_count"):
        accept_baseline.compare(measured, base)


def test_speaker_regression_is_still_gated() -> None:
    base = _baseline(E1=dict(_REVIEWED), aggregate=dict(_REVIEWED))
    measured = {"E1": dict(_UNREVIEWED, speaker_name_acc=0.5)}
    measured["aggregate"] = accept_baseline.aggregate([measured["E1"]])
    found = {(r.scope, r.key) for r in accept_baseline.compare(measured, base)}
    assert ("E1", "speaker_name_acc") in found
    assert all(key != "text_drift" for _scope, key in found)


def test_text_drift_has_a_default_tolerance() -> None:
    assert accept_baseline.default_tolerance("text_drift") == pytest.approx(
        accept_baseline.RATE_TOLERANCE
    )
