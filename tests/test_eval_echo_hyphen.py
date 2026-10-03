"""Hyphenated glossary terms and the prompt-echo classifier (issue #109).

Regression coverage for the echo classifier's handling of hyphenated
glossary terms. ``textnorm`` (the single normalizer both sides of the
classifier run through) replaces the hyphen — a non-word, non-space
character — with a space on *both* the glossary-token side and the
hypothesis word side, so a term like ``NG-TOPI`` and an echo that
transcribes it as ``"NG TOPI"`` or ``"ng-topi"`` both land on the same
``ng topi`` fragment tokens. The classifier must therefore treat the hyphenated term
symmetrically: a verbatim echo scores 0.0 (echo), the hyphen-transcribed-
as-space variant scores 0.0 (echo), and a real sentence keeps its term
hits.
"""

from __future__ import annotations

import pytest

from vemoizer.eval_harness import glossary_term_hit_rate

#: A reference in which both terms occur (the echo check short-circuits
#: when no term is in the reference, so the reference must carry them).
_REF = "ng-topi ja ibc oli agendalla"
_TERMS = ["NG-TOPI", "IBC"]


def test_verbatim_hyphenated_echo_is_zero() -> None:
    # The canonical echo shape with the hyphen surviving transcription:
    # every word is glossary content, so the term hits must not count.
    assert glossary_term_hit_rate(_REF, "NG-TOPI, IBC.", _TERMS) == 0.0


def test_hyphen_transcribed_as_space_echo_is_zero() -> None:
    # Whisper may transcribe the hyphen as a space; the symmetric
    # tokenization must still classify this as an echo.
    assert glossary_term_hit_rate(_REF, "NG TOPI IBC", _TERMS) == 0.0


def test_lower_case_hyphenated_echo_is_zero() -> None:
    assert glossary_term_hit_rate(_REF, "ng-topi ibc", _TERMS) == 0.0


def test_real_sentence_with_hyphenated_term_is_kept() -> None:
    # A real sentence containing the term must not be zeroed: 1 of 2 terms
    # in the hypothesis → 0.5, not 0.0.
    ref = "we need NG-TOPI and IBC for the audit"
    hyp = "we need NG-TOPI for the audit today"
    assert glossary_term_hit_rate(ref, hyp, _TERMS) == pytest.approx(0.5)


def test_real_sentence_with_both_hyphenated_terms_is_kept() -> None:
    ref = "we need NG-TOPI and IBC for the audit"
    hyp = "we need NG-TOPI and IBC for the audit today"
    # Both terms present → 1.0 (not classified as an echo: 4/9 words
    # outside the glossary, which is above the 0.3 threshold).
    assert glossary_term_hit_rate(ref, hyp, _TERMS) == 1.0


def test_term_run_hyphenated_still_counts_as_hit() -> None:
    # The hyphenated term "NG-TOPI" normalizes to the "ng topi" token run
    # and must still be found as a contiguous run in the hypothesis.
    ref = "ng-topi on agendalla"
    hyp = "ng-topi on agendalla"
    assert glossary_term_hit_rate(ref, hyp, ["NG-TOPI"]) == 1.0
