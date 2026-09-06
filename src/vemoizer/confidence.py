"""Suspect-region flagging from whisper segment confidence (issue #71).

Whisper reports per-segment ``avg_logprob``; regions it decoded shakily
are exactly where garble, hallucination and wrong numbers live. Those
regions are FLAGGED deterministically for the reader ("tarkista") and for
the repair prompt — never silently rewritten: a flag is honest, a guess
is not. Thresholds follow the community practice around openai-whisper's
own ``logprob_threshold`` (-1.0 = discard-grade; flagging warns earlier).
"""

from __future__ import annotations

import re
from typing import Any

#: Below this the segment text is likely garbled.
GARBLE_LOGPROB = -0.7

#: Below this, digits in the segment are unreliable (numbers garble at
#: higher confidence than words — a wrong year reads as fluent Finnish).
NUMBER_LOGPROB = -0.5

_DIGIT = re.compile(r"\d")

#: The relative rule needs a real distribution; short recordings skip it.
MIN_SEGMENTS_FOR_RELATIVE = 20

#: Flag when a segment sits this many MADs below the file median.
RELATIVE_MADS = 1.5

#: Fraction of immediately repeated bigrams that marks recognizer output
#: as loop-degenerate even when confidence is not reported.
BIGRAM_REPEAT_RATE = 0.15


def _repeated_bigram_rate(text: str) -> float:
    """Fraction of adjacent bigrams that repeat their predecessor bigram."""
    tokens = text.split()
    if len(tokens) < 4:
        return 0.0
    bigrams = [(tokens[i], tokens[i + 1]) for i in range(len(tokens) - 1)]
    # Loops come in periods 1-3 ("sana sana", "a b a b", "a b c a b c"):
    # compare each bigram against its recent predecessors.
    repeats = sum(
        1
        for i in range(2, len(bigrams))
        if any(bigrams[i] == bigrams[i - k] for k in (1, 2, 3) if i - k >= 0)
    )
    return repeats / max(len(bigrams) - 2, 1)


def flag_suspect_segments(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Mark low-confidence segments with ``suspect: garble|number``.

    Pure; segments without ``avg_logprob`` (backends that report no
    confidence) pass through untouched, so the dictation path is
    unaffected.
    """
    logprobs = [
        float(s["avg_logprob"]) for s in segments if s.get("avg_logprob") is not None
    ]
    relative_cut: float | None = None
    if len(logprobs) >= MIN_SEGMENTS_FOR_RELATIVE:
        ordered = sorted(logprobs)
        median = ordered[len(ordered) // 2]
        mad = sorted(abs(v - median) for v in logprobs)[len(logprobs) // 2]
        # Floor the MAD: a file where nearly every segment shares one value
        # has mad=0, which would exempt exactly the outliers we hunt.
        mad = max(mad, 0.05)
        # Relative rule: whisper's logprob scale varies by model and
        # language, so a fixed cut lets whole files ship flagless while
        # unreadable; an outlier vs the file's own distribution is
        # suspicious regardless of its absolute value.
        relative_cut = median - RELATIVE_MADS * mad

    out: list[dict[str, Any]] = []
    for seg in segments:
        logprob = seg.get("avg_logprob")
        text = str(seg.get("text", ""))
        entry = dict(seg)
        garble = _repeated_bigram_rate(text) >= BIGRAM_REPEAT_RATE
        if logprob is not None:
            value = float(logprob)
            if (
                value < GARBLE_LOGPROB
                or relative_cut is not None
                and value < relative_cut
            ):
                garble = True
            elif value < NUMBER_LOGPROB and _DIGIT.search(text):
                entry["suspect"] = "number"
        if garble:
            entry["suspect"] = "garble"
        out.append(entry)
    return out
