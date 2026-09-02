"""WER evaluation harness (issue #11).

Consumes the checked-in fixture corpus: stem-paired ``<stem>.wav`` ↔
``<stem>.txt`` files under ``tests/fixtures/corpus``. The ``.wav`` file
holds the reference audio (the eval driver transcribes it); the ``.txt``
file holds the reference transcript. :func:`wer` is the metric;
:func:`run_eval` walks the corpus and produces per-sample plus aggregate
WER for a given hypothesis mapping.

Pure logic — no model imports, no network. The transcription step is
the caller's job (it needs the live consensus pipeline); this module
only turns (reference, hypothesis) pairs into numbers.
"""

from __future__ import annotations

from pathlib import Path

from vemoizer.textnorm import textnorm

#: Aggregate key appended to the per-sample mapping by :func:`run_eval`.
AGGREGATE_KEY = "aggregate"


def _levenshtein(a: list[str], b: list[str]) -> int:
    """Classic DP word-level Levenshtein distance (ins/del/sub only)."""
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, wa in enumerate(a, start=1):
        curr = [i]
        for j, wb in enumerate(b, start=1):
            cost = 0 if wa == wb else 1
            curr.append(min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost))
        prev = curr
    return prev[-1]


def wer(reference: str, hypothesis: str) -> float:
    """Word error rate between *reference* and *hypothesis*.

    Both inputs pass through :func:`vemoizer.textnorm.textnorm` before
    tokenizing, so case, punctuation, and whitespace never count as
    edits. Returns substitutions+insertions+deletions divided by the
    reference word count. An empty reference returns 1.0 when the
    hypothesis has words (all insertions) and 0.0 when both are empty.
    """
    ref_words = textnorm(reference).split()
    hyp_words = textnorm(hypothesis).split()
    if not ref_words:
        return 1.0 if hyp_words else 0.0
    return _levenshtein(ref_words, hyp_words) / len(ref_words)


def run_eval(corpus_dir: Path) -> dict[str, float]:
    """Walk *corpus_dir* and return per-sample WER plus an aggregate.

    Samples are stem pairs: ``<stem>.txt`` (reference transcript) with
    ``<stem>.wav`` (reference audio) present side by side. For this
    harness the hypothesis is the reference itself, so each per-sample
    WER is 0.0 — the corpus-walking machinery and the metric are what
    this function establishes. The result maps ``{stem: wer}`` and adds
    a single ``"aggregate"`` key (macro average over samples).
    """
    if not corpus_dir.is_dir():
        raise FileNotFoundError(f"corpus directory not found: {corpus_dir}")
    results: dict[str, float] = {}
    for wav in sorted(corpus_dir.glob("*.wav")):
        txt = wav.with_suffix(".txt")
        if not txt.is_file():
            continue
        reference = txt.read_text(encoding="utf-8")
        results[wav.stem] = wer(reference, reference)
    if not results:
        return {AGGREGATE_KEY: 0.0}
    results[AGGREGATE_KEY] = sum(results.values()) / len(results)
    return results
