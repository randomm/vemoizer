"""Post-decode echo filter: drop prompt-shaped segments (issue #109).

On unclear or quiet audio — and especially in English meetings — whisper
can continue the glossary ``initial_prompt`` instead of transcribing:
``Sanasto, Pia, NG-TOPI, …`` (issue #109). The filter drops segments that
are the prompt being echoed while keeping real sentences that contain one
or more glossary terms, and is fail-open on any error (it never loses a
real segment to a filter bug).
"""

from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

#: The label word the prompt once carried. Even after the neutral form
#: dropped it (issue #109, option 1), the recorded echo evidence shows it as
#: the most-repeated token, so it stays on the echo vocabulary.
_PROMPT_LABEL = "Sanasto"


def echo_vocabulary(prompt: str | None) -> list[str] | None:
    """The prompt terms that can echo back, or ``None`` when there is none.

    ``None`` means no glossary was configured, in which case the filter is a
    no-op (nothing to echo). The terms come straight from the configured
    ``initial_prompt`` — the same string whisper was seeded with — so the
    filter only ever looks for what it was actually primed to repeat.
    """
    if prompt is None or not prompt.strip():
        return None
    terms = [t.strip() for t in prompt.strip().rstrip(".").split(",")]
    kept = [t for t in terms if t]
    if _PROMPT_LABEL.lower() not in {t.lower() for t in kept}:
        kept.insert(0, _PROMPT_LABEL)
    return kept


def _is_echo(text: str, echo_terms: list[str]) -> bool:
    """True when *text* is the prompt being echoed, not real speech.

    A segment is an echo when every word is a glossary term or the (former)
    prompt label, in prompt order. A real sentence that contains one or more
    glossary terms has at least one non-glossary word and is always kept.
    Punctuation-only tokens (commas, periods) are ignored by the token
    pattern; the case of a word is irrelevant.
    """
    vocab = {t.lower() for t in echo_terms}
    matched = 0
    for token in re.findall(r"[\w'][-\w']*", text, re.UNICODE):
        # The token (commas/periods excluded by the pattern) must be a whole
        # glossary term or the former label for the segment to be an echo;
        # case-insensitive.
        norm = token.lower()
        if norm in vocab:
            matched += 1
        else:
            # Any non-glossary word (a real sentence contains at least one)
            # breaks the echo: this is what keeps a genuine sentence that
            # happens to mention a glossary term intact.
            return False
    return matched > 0


def _words_on_timeline(
    segments: list[dict[str, Any]], offset_s: float
) -> list[dict[str, Any]]:
    """The segments' words, shifted onto the recording timeline."""
    words: list[dict[str, Any]] = []
    for seg in segments:
        for w in seg.get("words") or []:
            word = str(w.get("word", "")).strip()
            if word:
                words.append(
                    {
                        "word": word,
                        "start": float(w.get("start", 0.0)) + offset_s,
                        "end": float(w.get("end", 0.0)) + offset_s,
                    }
                )
    return words


def filter_echo_segments(
    segments: list[dict[str, Any]],
    offset_s: float,
    echo_terms: list[str] | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Drop echo segments (and their words) from one window's raw decode.

    Returns ``(kept_segments, kept_words)`` with words on the full-recording
    timeline (shifted by *offset_s*). When *echo_terms* is ``None`` (no
    glossary) nothing is dropped.

    Fail-open: on any unexpected error the original (unfiltered) segments
    and words are returned unchanged, so a filter bug can never lose real
    speech. A dropped echo is logged at INFO with its window time and a
    count of echoed terms, so it is auditable without leaking transcript
    text beyond the term list.
    """
    if echo_terms is None:
        return list(segments), _words_on_timeline(segments, offset_s)
    try:
        kept_segments: list[dict[str, Any]] = []
        for seg in segments:
            text = str(seg.get("text", "")).strip()
            if text and _is_echo(text, echo_terms):
                logger.info(
                    "dropping prompt echo at %.0fs (echoed %d glossary term(s))",
                    float(seg.get("start", 0.0)) + offset_s,
                    len([t for t in echo_terms if t.lower() in text.lower()]),
                )
                continue
            kept_segments.append(seg)
        return kept_segments, _words_on_timeline(kept_segments, offset_s)
    except Exception as e:  # noqa: BLE001 - fail-open: keep unfiltered
        logger.warning("echo filter error, returning unfiltered: %s", e)
        return list(segments), _words_on_timeline(segments, offset_s)
