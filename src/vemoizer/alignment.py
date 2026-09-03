"""DTW alignment between decode A and decode B word onsets.

Consensus pipeline stage (project invariant #2): given two word lists with
start times from two different ASR decodes of the *same* audio, DTW over
word onsets produces the monotonic word pairing. Each paired position
carries its word-similarity cost, which the disputed-span stage (ticket 6)
turns into "re-decode this slice" decisions.

This module is deliberately pure: no model imports, no network, no audio
files. It operates on the word-list contract documented on
``TranscriptionResult`` (``words=[{word, start, end}]`` in time order), so
it consumes the output of *any* backend reached through the ``Transcriber``
Protocol — including Canary decode B, whose word-timestamp shape is
normalised upstream before reaching this stage.

Two flavours of the same DTW share one cost function: :func:`dtw_align` pairs
word *strings* for the eval harness, and :func:`align_decodes` pairs the word
*dicts* the consensus pipeline needs, so a disputed span can be anchored to
real timestamps.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

from .progress import format_duration

#: Cost of an insertion or deletion (a word on one side that has no
#: counterpart on the other). Tunable by the disputed-span stage if needed.
GAP_PENALTY: float = 1.0


@dataclass(frozen=True)
class AlignedWord:
    """One DTW alignment position.

    Either side may be ``None`` — an insertion (word only in A, paired
    against the gap on B) or deletion (word only in B, paired against the
    gap on A). The cost on a gap row is ``GAP_PENALTY``; on a real pair it
    is the cost function in :func:`dtw_align`.
    """

    word_a: str | None
    start_a: float | None
    word_b: str | None
    start_b: float | None
    cost: float


def _similarity(a: str, b: str) -> float:
    """Similarity in [0, 1] between two tokens; 1.0 is identical.

    The comparison is case-insensitive and punctuation-insensitive: the two
    decodes routinely differ only in capitalisation or a surrounding comma
    ("Mikko," vs "mikko"), which is not a real dispute. Digits and spaces
    are significant and compared as-is.

    Implementation: Levenshtein distance normalised by the longer token
    length, inverted to a similarity. Tokens are short so this is bounded
    by tens of operations.
    """
    a = a.lower().strip(".,;:!?\"'()[]{}-—–")
    b = b.lower().strip(".,;:!?\"'()[]{}-—–")
    if a == b:
        return 1.0
    if not a or not b:
        return 0.0
    return _levenshtein_similarity(a, b)


def _levenshtein_similarity(a: str, b: str) -> float:
    """1 - Levenshtein(a, b) / max(len(a), len(b)); 2-row rolling table.

    Both inputs are guaranteed non-empty by the caller (:func:`_similarity`
    checks for empty after stripping before calling this). The ``m == 0``
    and ``n == 0`` guards are unreachable in the normal call path; they are
    kept as a defensive measure but are not covered by the test suite.
    """
    m, n = len(a), len(b)
    if m == 0 or n == 0:
        return 1.0 if m == 0 and n == 0 else 0.0
    # prev[j] = distance between a[:i-1] and b[:j]; initialised for i=1:
    # distance between a[:0] and b[:j] == j (j insertions).
    prev = list(range(n + 1))
    for i in range(1, m + 1):
        # curr[j] = distance between a[:i] and b[:j]; invariant: curr[0] == i.
        curr = [i]
        ai = a[i - 1]
        for j in range(1, n + 1):
            match = 0 if ai == b[j - 1] else 1
            curr.append(min(curr[j - 1] + 1, prev[j] + 1, prev[j - 1] + match))
        prev = curr
    return 1.0 - prev[n] / max(m, n)


def _pair_cost(
    words_a: list[dict[str, float | str]],
    words_b: list[dict[str, float | str]],
    i: int,
    j: int,
) -> float:
    """Cost of pairing ``words_a[i]`` with ``words_b[j]`` (see dtw_align)."""
    aw = words_a[i]
    bw = words_b[j]
    time_delta = float(aw["start"]) - float(bw["start"])
    text_cost = 1.0 - _similarity(str(aw["word"]), str(bw["word"]))
    return time_delta * time_delta * text_cost


def dtw_align(
    words_a: list[dict[str, float | str]],
    words_b: list[dict[str, float | str]],
) -> list[AlignedWord]:
    """DTW-align two word lists on word onset times.

    Each input entry is ``{"word": str, "start": float}`` (``end`` is
    accepted but unused — the cost is onset-only by design).

    Cost function: a real pair (i, j) costs

        (|start_a[i] - start_b[j]|)^2 * (1 - similarity(word_a[i], word_b[j]))

    Squaring keeps the time penalty dominated by large onset drift (a word
    whose onset is off by 1.0 s costs 4x what a 0.5 s drift costs). The
    similarity factor is multiplicative so a perfect-word, perfect-time pair
    is zero cost; a pair with either a large time gap or a large text gap is
    punished. An insertion or deletion (a word on one side with no
    counterpart on the other) costs ``GAP_PENALTY``.

    The alignment is monotonic: the sequence of pairings preserves the
    temporal order of both inputs, and DTW with non-negative costs always
    yields a well-defined (unique) path for a given input pair.

    Boundary semantics: if ``words_a`` or ``words_b`` is empty, the result
    is empty. A word that has no counterpart on the other side is emitted
    as a gap row with cost ``GAP_PENALTY``.

    Complexity: O(len(a) * len(b)) time and space in the DP table. For a
    60-minute memo at ~2 words/second the table is ~1.4M cells — a
    budgeted cost, consistent with the VAD stage's wall-clock note.
    """
    if not words_a or not words_b:
        return []

    n, m = len(words_a), len(words_b)
    inf = float("inf")

    # D[i][j] = minimum cumulative cost of an alignment whose last step
    # involves words_a[:i] and words_b[:j] (i and j are 1-indexed counts of
    # words consumed from each list). Row 0 is the all-insertion row
    # (no words from A, j insertions from B); column 0 is the
    # all-deletion row (i deletions from A, no words from B).
    D: list[list[float]] = [[inf] * (m + 1) for _ in range(n + 1)]
    D[0][0] = 0.0
    for j in range(1, m + 1):
        D[0][j] = D[0][j - 1] + GAP_PENALTY
    for i in range(1, n + 1):
        D[i][0] = D[i - 1][0] + GAP_PENALTY
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            pair = _pair_cost(words_a, words_b, i - 1, j - 1)
            diag = D[i - 1][j - 1] + pair
            up = D[i - 1][j] + GAP_PENALTY
            left = D[i][j - 1] + GAP_PENALTY
            best = diag
            if up < best:
                best = up
            if left < best:
                best = left
            D[i][j] = best

    return _backtrace(words_a, words_b, D, n, m)


def _backtrace(
    words_a: list[dict[str, float | str]],
    words_b: list[dict[str, float | str]],
    D: list[list[float]],
    i: int,
    j: int,
) -> list[AlignedWord]:
    """Walk the DP table from (n, m) back to (0, 0) and emit the alignment.

    ``i`` and ``j`` are 1-indexed counts of words consumed from each list
    (starting at n and m). Each step picks the move that achieves the DP
    value at (i, j) — diagonal (pair), up (insert words_b[j-1]), or left
    (delete words_a[i-1]). Ties break diagonal-first, then up-first, then
    left, which is a stable, deterministic order that prefers the
    "straight" path. The result is reversed to time order before returning.
    """
    out: list[AlignedWord] = []
    while i > 0 or j > 0:
        pair = 0.0
        diag = 0.0
        up = 0.0
        left = 0.0
        if i > 0 and j > 0:
            pair = _pair_cost(words_a, words_b, i - 1, j - 1)
            diag = D[i - 1][j - 1] + pair
            up = D[i - 1][j] + GAP_PENALTY
            left = D[i][j - 1] + GAP_PENALTY
        if i == 0:
            # Only B words remain: each is an insertion (consumes a B word).
            bw = words_b[j - 1]
            out.append(
                AlignedWord(
                    word_a=None,
                    start_a=None,
                    word_b=str(bw["word"]),
                    start_b=float(bw["start"]),
                    cost=GAP_PENALTY,
                )
            )
            j -= 1
            continue
        if j == 0:
            # Only A words remain: each is a deletion (consumes an A word).
            aw = words_a[i - 1]
            out.append(
                AlignedWord(
                    word_a=str(aw["word"]),
                    start_a=float(aw["start"]),
                    word_b=None,
                    start_b=None,
                    cost=GAP_PENALTY,
                )
            )
            i -= 1
            continue
        if diag <= up and diag <= left:
            # Diagonal move: pair words_a[i-1] with words_b[j-1].
            aw = words_a[i - 1]
            bw = words_b[j - 1]
            out.append(
                AlignedWord(
                    word_a=str(aw["word"]),
                    start_a=float(aw["start"]),
                    word_b=str(bw["word"]),
                    start_b=float(bw["start"]),
                    cost=pair,
                )
            )
            i -= 1
            j -= 1
        elif up < left:
            # Up move: consume words_a[i-1] as a deletion (B-side gap).
            # words_b[j-1] is NOT consumed at this step.
            aw = words_a[i - 1]
            out.append(
                AlignedWord(
                    word_a=str(aw["word"]),
                    start_a=float(aw["start"]),
                    word_b=None,
                    start_b=None,
                    cost=GAP_PENALTY,
                )
            )
            i -= 1
        else:
            # Left move: consume words_b[j-1] as an insertion (A-side gap).
            # words_a[i-1] is NOT consumed at this step.
            bw = words_b[j - 1]
            out.append(
                AlignedWord(
                    word_a=None,
                    start_a=None,
                    word_b=str(bw["word"]),
                    start_b=float(bw["start"]),
                    cost=GAP_PENALTY,
                )
            )
            j -= 1
    out.reverse()
    return out


logger = logging.getLogger(__name__)

WordPairs = list[tuple[dict[str, Any] | None, dict[str, Any] | None]]


def align_decodes(
    result_a: dict[str, Any], result_b: dict[str, Any]
) -> WordPairs | None:
    """DTW-align the two word streams on word onsets.

    Mirrors :func:`vemoizer.alignment.dtw_align` (same cost function and
    diagonal-first tie-breaking) but emits the original word *dicts* (with
    word and timestamps) instead of the word strings, so the disputed-span
    stage can anchor each span to the actual word times. ``None`` when either
    side produced no words or the alignment itself fails (fail-open).
    """
    words_a = result_a.get("words") or []
    words_b = result_b.get("words") or []
    if not words_a or not words_b:
        # Worth a log line rather than a silent ``None``: a decode that
        # produces text but no word timestamps disables the whole consensus
        # path (no alignment -> no disputed spans -> no re-decode), and the
        # run still completes, so the only symptom is a missing stage.
        logger.warning(
            "alignment skipped: decode A has %d words, decode B has %d "
            "(both sides need word timestamps)",
            len(words_a),
            len(words_b),
        )
        return None
    start = time.monotonic()
    logger.info("alignment: DTW over %d x %d words", len(words_a), len(words_b))
    try:
        pairs = _dtw_word_dicts(words_a, words_b)
    except Exception as e:  # noqa: BLE001 - fail-open stage boundary
        logger.warning("alignment failed: %s", e)
        return None
    logger.info(
        "alignment: %d pairs in %s",
        len(pairs),
        format_duration(time.monotonic() - start),
    )
    return pairs


def _dtw_word_dicts(
    words_a: list[dict[str, Any]], words_b: list[dict[str, Any]]
) -> WordPairs:
    """DTW word-dict alignment (dict-flavoured ``alignment.dtw_align``)."""
    n, m = len(words_a), len(words_b)
    inf = float("inf")
    D: list[list[float]] = [[inf] * (m + 1) for _ in range(n + 1)]
    D[0][0] = 0.0
    for j in range(1, m + 1):
        D[0][j] = D[0][j - 1] + GAP_PENALTY
    for i in range(1, n + 1):
        D[i][0] = D[i - 1][0] + GAP_PENALTY
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            diag = D[i - 1][j - 1] + _pair_cost(words_a, words_b, i - 1, j - 1)
            up = D[i - 1][j] + GAP_PENALTY
            left = D[i][j - 1] + GAP_PENALTY
            best = diag
            if up < best:
                best = up
            if left < best:
                best = left
            D[i][j] = best

    pairs: WordPairs = []
    i, j = n, m
    while i > 0 or j > 0:
        if i == 0:
            pairs.append((None, words_b[j - 1]))
            j -= 1
            continue
        if j == 0:
            pairs.append((words_a[i - 1], None))
            i -= 1
            continue
        diag = D[i - 1][j - 1] + _pair_cost(words_a, words_b, i - 1, j - 1)
        up = D[i - 1][j] + GAP_PENALTY
        left = D[i][j - 1] + GAP_PENALTY
        if diag <= up and diag <= left:
            pairs.append((words_a[i - 1], words_b[j - 1]))
            i -= 1
            j -= 1
        elif up < left:
            pairs.append((words_a[i - 1], None))
            i -= 1
        else:
            pairs.append((None, words_b[j - 1]))
            j -= 1
    pairs.reverse()
    return pairs


def align_pairs_safe(
    result_a: dict[str, Any] | None, result_b: dict[str, Any] | None
) -> WordPairs | None:
    """Alignment wrapper that fails open to ``None``."""
    if result_a is None or result_b is None:
        return None
    try:
        return align_decodes(result_a, result_b)
    except Exception as e:  # noqa: BLE001 - fail-open stage boundary
        logger.warning("alignment failed: %s", e)
        return None
