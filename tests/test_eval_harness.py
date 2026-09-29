"""Tests for the WER eval harness (issue #11).

Covers the :func:`wer` metric directly (pure function) and the
:func:`run_eval` corpus-walking machinery against a synthetic temp
corpus — no model imports, no network.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from vemoizer.eval_harness import (
    AGGREGATE_KEY,
    compare_to_baseline,
    corpus_fingerprint,
    run_eval,
    wer,
)

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
#
# run_eval takes the transcription callable as an argument: the harness owns
# corpus walking and scoring, the caller owns model loading. The old stub
# compared each reference against itself (always 0.0) and never invoked a
# model -- these tests pin the seam that replaced it.


def _corpus(tmp_path: Path, samples: dict[str, str]) -> Path:
    for stem, ref in samples.items():
        (tmp_path / f"{stem}.wav").write_bytes(b"RIFF0000WAVE")
        (tmp_path / f"{stem}.txt").write_text(ref, encoding="utf-8")
    return tmp_path


def test_run_eval_scores_the_injected_transcriber(tmp_path: Path) -> None:
    corpus = _corpus(tmp_path, {"one": "moro aami", "two": "toista tallaista"})

    def perfect(wav: Path) -> str:
        return (wav.with_suffix(".txt")).read_text(encoding="utf-8")

    results = run_eval(corpus, perfect)
    assert results == {"one": 0.0, "two": 0.0, AGGREGATE_KEY: 0.0}


def test_run_eval_reports_real_errors(tmp_path: Path) -> None:
    corpus = _corpus(tmp_path, {"one": "a b c"})
    results = run_eval(corpus, lambda wav: "a x c")
    assert results["one"] == pytest.approx(1 / 3)
    assert results[AGGREGATE_KEY] == pytest.approx(1 / 3)


def test_run_eval_transcriber_failure_scores_one_not_crash(tmp_path: Path) -> None:
    """A backend crash on one sample must not abort the whole eval run."""
    corpus = _corpus(tmp_path, {"bad": "a b", "good": "c d"})

    def flaky(wav: Path) -> str:
        if wav.stem == "bad":
            raise RuntimeError("model exploded")
        return wav.with_suffix(".txt").read_text(encoding="utf-8")

    results = run_eval(corpus, flaky)
    assert results["bad"] == 1.0  # empty hypothesis against a real reference
    assert results["good"] == 0.0


def test_run_eval_ignores_unpaired_stems(tmp_path: Path) -> None:
    (tmp_path / "orphan.txt").write_text("hello", encoding="utf-8")
    (tmp_path / "lone.wav").write_bytes(b"RIFF")
    _corpus(tmp_path, {"paired": "hello world"})

    results = run_eval(tmp_path, lambda wav: "hello world")

    assert "orphan" not in results
    assert "lone" not in results
    assert results["paired"] == 0.0


def test_run_eval_missing_directory_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        run_eval(tmp_path / "no-such-corpus", lambda wav: "")


def test_run_eval_empty_corpus_gives_zero_aggregate(tmp_path: Path) -> None:
    assert run_eval(tmp_path, lambda wav: "") == {AGGREGATE_KEY: 0.0}


# --- corpus_fingerprint ----------------------------------------------------


def test_fingerprint_is_stable_for_identical_corpora(tmp_path: Path) -> None:
    (tmp_path / "a").mkdir()
    (tmp_path / "b").mkdir()
    a = _corpus(tmp_path / "a", {"one": "moro"})
    b = _corpus(tmp_path / "b", {"one": "moro"})
    assert corpus_fingerprint(a) == corpus_fingerprint(b)


def test_fingerprint_changes_when_audio_changes(tmp_path: Path) -> None:
    corpus = _corpus(tmp_path, {"one": "moro"})
    before = corpus_fingerprint(corpus)
    (corpus / "one.wav").write_bytes(b"RIFF1111WAVE")
    assert corpus_fingerprint(corpus) != before


def test_fingerprint_changes_when_reference_changes(tmp_path: Path) -> None:
    corpus = _corpus(tmp_path, {"one": "moro"})
    before = corpus_fingerprint(corpus)
    (corpus / "one.txt").write_text("muutettu", encoding="utf-8")
    assert corpus_fingerprint(corpus) != before


# --- compare_to_baseline ---------------------------------------------------


def test_compare_passes_within_tolerance() -> None:
    measured = {"one": 0.11, AGGREGATE_KEY: 0.11}
    baseline = {"one": 0.10, AGGREGATE_KEY: 0.10}
    assert compare_to_baseline(measured, baseline, tolerance=0.02) == []


def test_compare_flags_regressions_beyond_tolerance() -> None:
    measured = {"one": 0.20, AGGREGATE_KEY: 0.20}
    baseline = {"one": 0.10, AGGREGATE_KEY: 0.10}
    regressions = compare_to_baseline(measured, baseline, tolerance=0.02)
    names = [r.name for r in regressions]
    assert "one" in names
    assert AGGREGATE_KEY in names
    reg = regressions[0]
    assert reg.baseline == 0.10
    assert reg.measured == 0.20


def test_compare_improvements_never_flag() -> None:
    measured = {"one": 0.05, AGGREGATE_KEY: 0.05}
    baseline = {"one": 0.10, AGGREGATE_KEY: 0.10}
    assert compare_to_baseline(measured, baseline, tolerance=0.0) == []


def test_compare_new_sample_missing_from_baseline_flags() -> None:
    """A sample the baseline has never seen means the corpus drifted."""
    measured = {"new": 0.0, AGGREGATE_KEY: 0.0}
    regressions = compare_to_baseline(measured, {AGGREGATE_KEY: 0.0}, tolerance=0.02)
    assert [r.name for r in regressions] == ["new"]
