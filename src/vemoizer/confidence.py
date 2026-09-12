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


def flag_suspect_segments(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Mark low-confidence segments with ``suspect: garble|number``.

    Pure; segments without ``avg_logprob`` (backends that report no
    confidence) pass through untouched, so the dictation path is
    unaffected.
    """
    out: list[dict[str, Any]] = []
    for seg in segments:
        logprob = seg.get("avg_logprob")
        entry = dict(seg)
        if logprob is not None:
            if float(logprob) < GARBLE_LOGPROB:
                entry["suspect"] = "garble"
            elif float(logprob) < NUMBER_LOGPROB and _DIGIT.search(
                str(seg.get("text", ""))
            ):
                entry["suspect"] = "number"
        out.append(entry)
    return out
