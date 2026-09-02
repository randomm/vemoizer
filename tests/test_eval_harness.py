"""Tests for the WER eval harness (issue #11).

Covers the :func:`wer` metric directly (pure function) and the
:func:`run_eval` corpus-walking machinery against a synthetic temp
corpus — no model imports, no network.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from vemoizer.eval_harness import AGGREGATE_KEY, run_eval, wer


# --- wer -------------------------------------------------------------------


def test_wer_identical_strings_is_zero() -> None:
    assert wer("moro aami", "moro aami") == 0.0


def test_wer_identical_up_to_case_punct_whitespace() -> None:
    assert wer("Moro, aami!", "moro  aami") == 0.0


def test_wer_one_substitution_of_three_words() -> None:
    # "a b c" vs "a x c": 1 edit / 3 reference words
    assert wer("a b c", "a x c") == pytest.approx(1 / 3)


def test_wer_all_words_inserted_into_empty_reference() -> None:
    assert wer("", "some words") == 1.0


def test_wer_both_empty_is_zero() -> None:
    assert wer("", "") == 0.0


def test_wer_hypothesis_longer_than_reference_can_exceed_one() -> None:
    # 3 insertions over 2 reference words
    assert wer("a b", "a b c d e") == pytest.approx(1.5)


def test_wer_deletions_count_as_edits() -> None:
    # "a b c" vs "a b": 1 deletion / 3 reference words
    assert wer("a b c", "a b") == pytest.approx(1 / 3)


def test_wer_finnish_special_characters() -> None:
    # ä in both survives normalization; one typo counts
    assert wer("äliti äiti", "äiti äiti") == pytest.approx(0.5)


# --- run_eval --------------------------------------------------------------


def test_run_eval_walks_corpus_and_computes_zero_wer(
    tmp_path: Path,
) -> None:
    (tmp_path / "one.wav").write_bytes(b"RIFF")
    (tmp_path / "one.txt").write_text("moro aami", encoding="utf-8")
    (tmp_path / "two.wav").write_bytes(b"RIFF")
    (tmp_path / "two.txt").write_text("toista tallaista", encoding="utf-8")

    results = run_eval(tmp_path)

    assert results["one"] == 0.0
    assert results["two"] == 0.0
    assert results[AGGREGATE_KEY] == 0.0


def test_run_eval_ignores_unpaired_stems(tmp_path: Path) -> None:
    # .wav without .txt is skipped; .txt without .wav is skipped
    (tmp_path / "orphan.txt").write_text("hello", encoding="utf-8")
    (tmp_path / "lone.wav").write_bytes(b"RIFF")
    (tmp_path / "paired.wav").write_bytes(b"RIFF")
    (tmp_path / "paired.txt").write_text("hello world", encoding="utf-8")

    results = run_eval(tmp_path)

    assert "orphan" not in results
    assert "lone" not in results
    assert "paired" in results
    assert results[AGGREGATE_KEY] == 0.0


def test_run_eval_missing_directory_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        run_eval(tmp_path / "no-such-corpus")


def test_run_eval_empty_corpus_gives_zero_aggregate(tmp_path: Path) -> None:
    assert run_eval(tmp_path) == {AGGREGATE_KEY: 0.0}
