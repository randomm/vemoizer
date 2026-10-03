"""Post-decode echo filter: drop prompt-shaped segments (issue #109).

On unclear or quiet audio — and especially in English meetings — whisper
can continue the glossary ``initial_prompt`` instead of transcribing:
``Sanasto, Pia, NG-TOPI, …`` (issue #109). The filter drops segments that
are the prompt being echoed while keeping real sentences that contain one
or more glossary terms, and is fail-open on any error (it never loses a
real segment to a filter bug).

The same phenomenon (prompt echo) is classified independently by two
callers: the transcriber uses the strict form :func:`_is_echo` (an echo is
a *run* of prompt terms — or a single segment carrying the label — the
only thing it is safe to drop is something we know to be the prompt), and
the eval harness uses its own proportional form ``_is_prompt_echo`` (a
metric-shape check — a hypothesis that is *almost* pure prompt must not
count its term hits, so a bit of surrounding filler is tolerated). They
classify the same phenomenon independently with different strictness.

The strict form deliberately keeps a single-word segment that is one
glossary term (``"Jira."``, ``"Kubernetes"``): a bare term with no other
words is more often a real one-word answer than a prompt continuation,
and dropping it would lose real speech. A prompt echo is a *run* of
prompt terms (``"term, term, …"``) or a segment that carries the
former label (``"Sanasto"``), which whisper only ever repeats as part of
the prompt.
"""

from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

#: The label word the prompt once carried. Even after the neutral form
#: dropped it (issue #109, option 1), the recorded echo evidence shows it
#: as the most-repeated token, so the echo vocabulary keeps it for
#: backward compatibility with pre-#109 echo recordings — even though the
#: neutral prompt itself never echoes it.
_PROMPT_LABEL = "Sanasto"


def echo_vocabulary(prompt: str | None) -> list[str] | None:
    """The prompt terms that can echo back, or ``None`` when there is none.

    ``None`` means no glossary was configured, in which case the filter is a
    no-op (nothing to echo). The terms come straight from the configured
    ``initial_prompt`` — the same string whisper was seeded with — so the
    filter only ever looks for what it was actually primed to repeat, plus
    the former label word for backward compatibility (see
    :data:`_PROMPT_LABEL`).

    Each term is kept as-is (case preserved, hyphens intact); :func:`_is_echo`
    compares case-insensitively. A hyphenated term matches only when the
    hyphen survives transcription — a ``"NG-TOPI"`` in the vocabulary does
    *not* match a segment echoing ``"NG TOPI"`` (the hyphen tokenized as a
    space), which is the eval harness's proportional form's territory.
    """
    if prompt is None or not prompt.strip():
        return None
    terms = [t.strip() for t in prompt.strip().rstrip(".").split(",")]
    kept = [t for t in terms if t]
    if _PROMPT_LABEL.lower() not in {t.lower() for t in kept}:
        kept.insert(0, _PROMPT_LABEL)
    return kept


def _is_echo(text: str, vocab: set[str]) -> bool:
    """True when *text* is the prompt being echoed, not real speech.

    *vocab* is the pre-computed lower-cased vocabulary set (see
    :func:`vocabulary_set`) — callers that check many segments against the
    same vocabulary should build it once and pass it in, rather than
    rebuilding the set on every call.

    A segment is an echo when every word is a prompt term (or the former
    label, which :func:`echo_vocabulary` always includes) **and** it is
    either a *run* of at least two tokens or carries the label token
    :data:`_PROMPT_LABEL` (case-insensitive). A single-token segment that
    is one glossary term (``"Jira."``, ``"Kubernetes"``) is **not** an echo
    and is kept: an echo is a run of prompt terms or the label, not a lone
    term — a real one-word answer would be lost otherwise. No real
    sentence containing one or more glossary terms can be an echo, because
    it has at least one non-glossary word. The comparison is
    case-insensitive. Hyphenated terms match only when the hyphen survives
    transcription — if whisper transcribes the hyphen as a space, the
    strict form keeps the segment (a documented limitation; the eval
    harness's proportional form is the looser companion). Punctuation-only
    input (commas/periods normalize away) is not an echo — there is nothing
    for the model to have repeated.

    The ``matched > 0`` guard at the end prevents ``_is_echo("", vocab)``
    from returning True when the regex finds no matches (empty input is
    filtered by the caller, but the guard makes the contract
    self-evident).
    """
    if not text:
        return False
    # Tokenize on word boundaries (commas/periods separate tokens; hyphens
    # are word chars, so "NG-TOPI" stays one token and matches the
    # vocabulary's "NG-TOPI" directly).
    tokens = re.findall(r"\w+[-\w]*", text, re.UNICODE)
    matched = 0
    for token in tokens:
        norm = token.lower()
        if norm in vocab:
            matched += 1
        else:
            # Any non-glossary word (a real sentence contains at least one)
            # breaks the echo: this is what keeps a genuine sentence that
            # happens to mention a glossary term intact.
            return False
    if matched == 0:
        # Punctuation-only or empty: nothing for the model to have
        # repeated — not an echo.
        return False
    if matched >= 2:
        # A run of prompt terms (the canonical echo shape: ``"term, term,
        # …"``).
        return True
    # Exactly one token: an echo only when it carries the former label
    # (whisper never echoes a lone term without the label).
    return _PROMPT_LABEL.lower() in tokens[0].lower()


def is_echo(text: str, echo_terms: list[str]) -> bool:
    """Public single-segment echo check (see :func:`_is_echo`).

    Accepts the raw (case-preserved) echo term list and builds the lower
    vocabulary set on the spot — fine for one-off checks; callers checking
    many segments against the same vocabulary should use
    :func:`filter_echo_segments` (which pre-computes the set via
    :func:`vocabulary_set` and reuses it for the whole window).
    """
    if not echo_terms:
        return False
    return _is_echo(text, vocabulary_set(echo_terms))


def vocabulary_set(echo_terms: list[str] | None) -> set[str]:
    """Lower-cased vocabulary set for :func:`_is_echo`.

    Pre-compute once per vocabulary and pass the result to :func:`_is_echo`
    for every segment in the window, instead of rebuilding the set (and
    re-lower-casing every term) on each call. Returns an empty set when
    *echo_terms* is ``None`` (no glossary): with an empty vocabulary every
    token is non-glossary, so :func:`_is_echo` returns ``False`` for any
    non-empty input — the correct no-op behaviour for a missing glossary.
    """
    if echo_terms is None:
        return set()
    return {t.lower() for t in echo_terms}


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

    The lower-cased vocabulary set is computed once up front (via
    :func:`vocabulary_set`) and reused for every segment in the window,
    rather than rebuilt on each :func:`_is_echo` call.

    Fail-open: on any unexpected error the original (unfiltered) segments
    and words are returned unchanged, so a filter bug can never lose real
    speech. A dropped echo is logged at INFO with its window time and a
    count of echoed terms, so it is auditable without leaking transcript
    text beyond the term list.
    """
    if echo_terms is None:
        return list(segments), _words_on_timeline(segments, offset_s)
    vocab = vocabulary_set(echo_terms)
    try:
        kept_segments: list[dict[str, Any]] = []
        for seg in segments:
            text = str(seg.get("text", "")).strip()
            if text and _is_echo(text, vocab):
                lower = text.lower()
                logger.info(
                    "dropping prompt echo at %.0fs (echoed %d glossary term(s))",
                    float(seg.get("start", 0.0)) + offset_s,
                    len([t for t in echo_terms if t.lower() in lower]),
                )
                continue
            kept_segments.append(seg)
        return kept_segments, _words_on_timeline(kept_segments, offset_s)
    except Exception as e:  # noqa: BLE001 - fail-open: keep unfiltered
        # Log the exception type and message only (no traceback): the
        # caught Exception originates inside the segment/word iteration
        # below, and a malformed payload can embed transcript fragments
        # in the exception message — the log line must stay free of
        # transcript text.
        logger.warning(
            "echo filter error, returning unfiltered: %s: %s",
            type(e).__name__,
            e,
        )
        try:
            fallback_words = _words_on_timeline(segments, offset_s)
        except Exception as e2:
            # The fallback itself can raise when the exception above came
            # from inside _words_on_timeline (e.g. a non-numeric word
            # timestamp); the fail-open contract still requires the
            # segments to survive, so degrade to no words rather than
            # propagate and lose the whole window. Log the degradation
            # so it is distinguishable from a fully unfiltered return.
            logger.warning(
                "echo filter fallback: words extraction failed, returning "
                "empty words: %s: %s",
                type(e2).__name__,
                e2,
            )
            fallback_words = []
        return list(segments), fallback_words
