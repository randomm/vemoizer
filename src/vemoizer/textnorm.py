"""Text normalization for WER computation (issue #11).

Word error rate is only comparable when both sides go through the same
normalization: case, punctuation, and whitespace differences must not
count as edits. Finnish and English alike: casefold (not lower —
covers ß → ss and the rest of Unicode), strip punctuation, collapse
whitespace.
"""

from __future__ import annotations

import re

#: Any run of non-word, non-space characters → single space. This strips
#: punctuation (.,!?;:'"()[]{}…—) and then a later collapse handles the
#: resulting whitespace. Underscores survive because \w includes them —
#: acceptable, they are rare in speech transcripts.
_PUNCT_RE = re.compile(r"[^\w\s]")
_WS_RE = re.compile(r"\s+")


def textnorm(s: str) -> str:
    """Normalize *s* for WER comparison.

    Returns the casefolded string with all punctuation replaced by
    spaces and whitespace runs collapsed to single spaces, stripped.
    Empty strings and pure-punctuation input normalize to ``""``.
    """
    s = s.casefold()
    s = _PUNCT_RE.sub(" ", s)
    s = _WS_RE.sub(" ", s)
    return s.strip()
