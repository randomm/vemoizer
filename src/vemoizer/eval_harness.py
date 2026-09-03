"""WER evaluation harness (issue #11).

Consumes the checked-in fixture corpus: stem-paired ``<stem>.wav`` ↔
``<stem>.txt`` files under ``tests/fixtures/corpus``. The ``.wav`` file
holds the reference audio (the eval driver transcribes it); the ``.txt``
file holds the reference transcript. :func:`wer` is the metric;
:func:`run_eval` walks the corpus and produces per-sample plus aggregate
WER for a given hypothesis mapping.

Pure logic — no model imports, no network. The transcription step is
the caller's job (it needs the live models); this module walks the
corpus, scores hypotheses, fingerprints the corpus, and compares runs
against a committed baseline.
"""

from __future__ import annotations

import hashlib
import logging
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from vemoizer.textnorm import textnorm

logger = logging.getLogger(__name__)

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


def run_eval(corpus_dir: Path, transcribe: Callable[[Path], str]) -> dict[str, float]:
    """Walk *corpus_dir* and score *transcribe* against the references.

    Samples are stem pairs: ``<stem>.txt`` (reference transcript) with
    ``<stem>.wav`` (reference audio) side by side. *transcribe* maps a WAV
    path to a hypothesis string — the harness owns corpus walking and
    scoring, the caller owns model loading (the Transcriber seam, so this
    module stays free of model imports). A crashing backend scores that
    sample as an empty hypothesis (WER 1.0 against a non-empty reference)
    instead of aborting the run: one bad sample must not hide the other
    numbers. The result maps ``{stem: wer}`` plus one ``"aggregate"`` key
    (macro average over samples).
    """
    if not corpus_dir.is_dir():
        raise FileNotFoundError(f"corpus directory not found: {corpus_dir}")
    results: dict[str, float] = {}
    for wav in sorted(corpus_dir.glob("*.wav")):
        txt = wav.with_suffix(".txt")
        if not txt.is_file():
            continue
        reference = txt.read_text(encoding="utf-8")
        try:
            hypothesis = transcribe(wav)
        except Exception:  # noqa: BLE001 - one sample must not abort the run
            logger.warning("transcription failed for %s; scoring as empty", wav.name)
            hypothesis = ""
        results[wav.stem] = wer(reference, hypothesis)
    if not results:
        return {AGGREGATE_KEY: 0.0}
    results[AGGREGATE_KEY] = sum(results.values()) / len(results)
    return results


def corpus_fingerprint(corpus_dir: Path) -> str:
    """SHA-256 over the corpus contents (paired ``.wav`` + ``.txt`` bytes).

    A committed WER baseline is only meaningful against the exact corpus it
    was measured on; the fingerprint lets the gate refuse to compare numbers
    across a silently changed corpus. Only file *contents* and names are
    hashed — never paths — so two checkouts agree.
    """
    digest = hashlib.sha256()
    for wav in sorted(corpus_dir.glob("*.wav")):
        txt = wav.with_suffix(".txt")
        if not txt.is_file():
            continue
        for path in (wav, txt):
            digest.update(path.name.encode("utf-8"))
            digest.update(path.read_bytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class Regression:
    """One sample whose measured WER exceeds the baseline beyond tolerance."""

    name: str
    baseline: float | None
    measured: float


def compare_to_baseline(
    measured: dict[str, float],
    baseline: dict[str, float],
    *,
    tolerance: float,
) -> list[Regression]:
    """Regressions of *measured* against *baseline* (empty list = gate passes).

    A sample regresses when its measured WER exceeds the baseline by more
    than *tolerance* (small decode nondeterminism must not flake the gate).
    A measured sample missing from the baseline is also flagged — it means
    the corpus drifted and the baseline needs a deliberate update, not a
    silent pass. Improvements never flag; they are recorded by updating the
    baseline in a dedicated commit.
    """
    regressions: list[Regression] = []
    for name, value in sorted(measured.items()):
        if name not in baseline:
            regressions.append(Regression(name, None, value))
            continue
        if value > baseline[name] + tolerance:
            regressions.append(Regression(name, baseline[name], value))
    return regressions
