"""WER evaluation harness (issue #11).

Consumes the checked-in fixture corpus: stem-paired ``<stem>.wav`` ↔
``<stem>.txt`` files under ``tests/fixtures/corpus``. The ``.wav`` file
holds the reference audio (the eval driver transcribes it); the ``.txt``
file holds the reference transcript. :func:`wer` is the metric;
:func:`run_eval` walks the corpus and produces per-sample plus aggregate
WER for a given hypothesis mapping.

The same harness also scores the meeting eval: the multi-speaker
``meeting_sample`` fixture pairs a glossary file (``meeting_sample.terms``,
issue #76) with the reference transcript, and
:func:`glossary_term_hit_rate` measures how often the glossary terms
survive the decode (the metric the glossary prompt work is judged by).

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


def glossary_term_hit_rate(reference: str, hypothesis: str, terms: list[str]) -> float:
    """Glossary term-hit rate (issue #76): terms present in the hypothesis
    divided by terms present in the reference.

    A term "appears" when it stands alone as a whole-word token run,
    case-insensitively: every word of the term (after
    :func:`vemoizer.textnorm.textnorm` normalizes case/punctuation/whitespace)
    must appear as a contiguous run of whole tokens in the text. Words that
    merely contain the term (``"backlogi"`` contains ``"backlog"``) do not
    count; repeated occurrences count once. Term lines that normalize to the
    same form (case/punctuation variants) are de-duplicated, so a glossary
    that lists both ``backlog`` and ``Backlog`` counts the term once.

    Returns ``terms_in_hyp / terms_in_ref``. Returns 1.0 when no term occurs
    in the reference (the hypothesis is vacuously complete for this
    glossary, and the sample contributes nothing to the metric either way);
    returns 0.0 when the reference has terms but the hypothesis captured
    none.
    """
    ref_words = textnorm(reference).split()
    hyp_words = textnorm(hypothesis).split()
    in_ref = _terms_present(ref_words, terms)
    if not in_ref:
        return 1.0
    in_hyp = _terms_present(hyp_words, terms)
    return len(in_ref & in_hyp) / len(in_ref)


def _terms_present(words: list[str], terms: list[str]) -> set[str]:
    """The terms that occur as whole-word token runs in *words*.

    The returned set holds the terms' *normalized* forms, so case or
    punctuation variants of the same term (``backlog`` / ``Backlog`` /
    ``backlog,``) share one slot and can never double-count.
    """
    present: set[str] = set()
    for raw in terms:
        t = textnorm(raw)
        if not t:
            continue
        t_words = t.split()
        n = len(t_words)
        for i in range(len(words) - n + 1):
            if words[i : i + n] == t_words:
                present.add(t)
                break
    return present


def run_eval(
    corpus_dir: Path, transcribe: Callable[[Path], str]
) -> tuple[dict[str, float], dict[str, str]]:
    """Walk *corpus_dir* and score *transcribe* against the references.

    Samples are stem pairs: ``<stem>.txt`` (reference transcript) with
    ``<stem>.wav`` (reference audio) side by side. *transcribe* maps a WAV
    path to a hypothesis string — the harness owns corpus walking and
    scoring, the caller owns model loading (the Transcriber seam, so this
    module stays free of model imports). A crashing backend scores that
    sample as an empty hypothesis (WER 1.0 against a non-empty reference)
    instead of aborting the run: one bad sample must not hide the other
    numbers.

    Returns the per-sample WER mapping (``{stem: wer}`` plus one
    ``"aggregate"`` key, macro average over samples) alongside the
    per-sample hypotheses (``{stem: hypothesis}``); the hypotheses are
    what :func:`run_meeting_eval` reuses for the term-hit metric so the
    meeting walk does not re-decode a sample the WER walk just scored.
    """
    if not corpus_dir.is_dir():
        raise FileNotFoundError(f"corpus directory not found: {corpus_dir}")
    results: dict[str, float] = {}
    hyps: dict[str, str] = {}
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
        hyps[wav.stem] = hypothesis
    if not results:
        return {AGGREGATE_KEY: 0.0}, {}
    results[AGGREGATE_KEY] = sum(results.values()) / len(results)
    return results, hyps


def run_meeting_eval(
    corpus_dir: Path,
    transcribe: Callable[[Path], str],
    reuse: dict[str, str] | None = None,
) -> dict[str, dict[str, float]]:
    """Score the meeting eval fixture (issue #76).

    The meeting fixture is a single multi-speaker sample — a ``.wav`` with
    its reference ``.txt`` and a ``.terms`` glossary side by side. *transcribe*
    maps a WAV path to a hypothesis string. The result maps the sample stem
    to ``{"wer": w, "term_hit": t}`` plus an ``"aggregate"`` key (macro
    average per metric — one sample per metric here, so the aggregate equals
    the sample value; the shape mirrors :func:`run_eval` so the harness
    scales to a future multi-sample meeting set). A crashing backend scores
    the sample as an empty hypothesis (WER 1.0 against a non-empty
    reference, term-hit 0.0).

    *reuse* maps stems to hypotheses already computed by :func:`run_eval`
    for the same corpus (its second return value). Samples present in
    *reuse* skip the decode and score the reused hypothesis, so the WER
    walk and the term-hit walk share the decode and a growing meeting set
    does not compound into one extra pass per sample. Samples absent from
    *reuse* (or not scored by the WER walk) are decoded through
    *transcribe* as before, keeping the WER gate independent of the
    informational term-hit metric: a corpus without a ``.terms`` pair
    scores WER as before, and the meeting metric can be dropped or changed
    without touching the gate.
    """
    if not corpus_dir.is_dir():
        raise FileNotFoundError(f"corpus directory not found: {corpus_dir}")
    results: dict[str, dict[str, float]] = {}
    for wav in sorted(corpus_dir.glob("*.wav")):
        txt = wav.with_suffix(".txt")
        terms_path = wav.with_suffix(".terms")
        if not txt.is_file() or not terms_path.is_file():
            continue
        reference = txt.read_text(encoding="utf-8")
        terms = [
            line.strip()
            for line in terms_path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.strip().startswith("#")
        ]
        reused = reuse.get(wav.stem) if reuse else None
        if reused is not None:
            hypothesis = reused
        else:
            try:
                hypothesis = transcribe(wav)
            except Exception:  # noqa: BLE001 - one sample must not abort the run
                logger.warning(
                    "transcription failed for %s; scoring as empty", wav.name
                )
                hypothesis = ""
        results[wav.stem] = {
            "wer": wer(reference, hypothesis),
            "term_hit": glossary_term_hit_rate(reference, hypothesis, terms),
        }
    if not results:
        return {AGGREGATE_KEY: {"wer": 0.0, "term_hit": 0.0}}
    results[AGGREGATE_KEY] = {
        "wer": sum(r["wer"] for r in results.values()) / len(results),
        "term_hit": sum(r["term_hit"] for r in results.values()) / len(results),
    }
    return results


def corpus_fingerprint(corpus_dir: Path) -> str:
    """SHA-256 over the corpus contents (paired ``.wav`` + ``.txt`` bytes,
    plus any optional ``.terms`` glossary bytes present for a stem).

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
        for path in (wav, txt, wav.with_suffix(".terms")):
            if path.is_file():
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
