"""Suspect-region flagging from whisper confidence (issue #71 round 2).

Whisper's per-segment avg_logprob is its own signal about garble and
hallucination; instead of discarding it (old behaviour) or letting an LLM
guess, low-confidence regions are FLAGGED deterministically and rendered
with a warning — never silently rewritten.
"""

from __future__ import annotations

from vemoizer.confidence import flag_suspect_segments
from vemoizer.readability import paragraphs


def _seg(text, logprob=None, start=0.0, end=1.0):
    d = {"start": start, "end": end, "text": text}
    if logprob is not None:
        d["avg_logprob"] = logprob
    return d


def test_low_logprob_flags_garble() -> None:
    out = flag_suspect_segments([_seg("epäselvä kohta", logprob=-0.9)])
    assert out[0]["suspect"] == "garble"


def test_digits_in_shaky_segment_flag_number() -> None:
    out = flag_suspect_segments([_seg("vuonna 2013 aloitin", logprob=-0.6)])
    assert out[0]["suspect"] == "number"


def test_confident_segments_are_untouched() -> None:
    out = flag_suspect_segments([_seg("selvä lause", logprob=-0.2)])
    assert "suspect" not in out[0]


def test_digits_in_confident_segment_do_not_flag() -> None:
    out = flag_suspect_segments([_seg("kello 14 sovittu", logprob=-0.2)])
    assert "suspect" not in out[0]


def test_missing_confidence_never_flags() -> None:
    """Backends without logprobs (dictation path) must be unaffected."""
    out = flag_suspect_segments([_seg("ei tietoa luottamuksesta")])
    assert "suspect" not in out[0]


def test_paragraphs_inherit_worst_suspect() -> None:
    segs = [
        _seg("alku hyvin", logprob=-0.2, start=0.0, end=1.0),
        _seg("sitten sotkua", logprob=-0.9, start=1.1, end=2.0),
    ]
    paras = paragraphs(flag_suspect_segments(segs))
    assert paras[0]["suspect"] == "garble"
