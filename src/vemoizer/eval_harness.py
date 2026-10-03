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

from vemoizer.slice_align import slice_similarity
from vemoizer.textnorm import textnorm

logger = logging.getLogger(__name__)

#: A hypothesis whose words fall outside the glossary by less than this
#: fraction is classified as a prompt echo (issue #109). Tuned against
#: the recorded echo shapes (0%–25% outside) vs. the shortest real
#: sentences (50%+ outside); a stricter 0.2 would let a two-word echo
#: slip through, a looser 0.5 would eat real one-term sentences.
_PROMPT_ECHO_OUTSIDE_FRACTION = 0.3

#: Two decodes are "in agreement" when their char-level normalized similarity
#: is at least this fraction (issue #62). Same textnorm + SequenceMatcher
#: ratio as :func:`vemoizer.slice_align.slice_similarity`; 0.8 is tuned
#: against the TTS-corpus agreement band (0.70–0.88, issue #62) so a sample
#: whose decoders merely tell a similar story (0.70–0.88) is not counted
#: as agreeing, while a genuinely matching pair scores well above it.
#: This constant is an informational threshold for the agreement metric
#: only — it does not touch the pipeline's dispute detection (that uses
#: :data:`vemoizer.slice_align.SLICE_DISPUTE_THRESHOLD` for span selection)
#: and it must not be compared with :data:`vemoizer.spans.DISPUTE_THRESHOLD`
#: (the word-pair LCS gate).
AGREEMENT_THRESHOLD = 0.8

#: A sample's shared-decoder hypothesis is "wrong" when its WER against the
#: reference exceeds this (issue #62). Informational threshold — never a
#: gate (invariant #2, the WER gate stays the only regression gate).
_WRONG_WER = 0.3

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

    Prompt echoes (issue #109) are excluded: a hypothesis that is *almost*
    pure glossary — fewer than 30% of its words fall outside the glossary —
    is not a real transcript; it is a continuation of the ``initial_prompt``
    that whisper repeated on unclear or quiet audio. In such a hypothesis the
    term hits are the prompt's, not the decoder's, and counting them would
    inflate the metric exactly when the glossary prompt is failing. A real
    sentence with one or a few glossary terms (``"backlog on täynnä"``)
    keeps every occurrence; a prompt-echo-shaped hypothesis (``"sanasto pia
    ng-topi"``) counts zero regardless of how many terms it lists.

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
    # Compute the glossary token set once for the two calls below rather
    # than rebuilding it inside each (the set is the same for both).
    glossary = _glossary_token_set(terms)
    if _is_prompt_echo(hyp_words, glossary):
        return 0.0
    return len(in_ref & in_hyp) / len(in_ref)


def _is_prompt_echo(words: list[str], glossary: set[str]) -> bool:
    """True when *words* is a prompt echo, not a real transcript (issue #109).

    Proportional (non-strict) form of the echo check. Independent of the
    transcriber's strict drop filter (``echo_filter._is_echo``) — the two
    classify the same phenomenon with different tokenization and
    strictness (see the module docstring in ``echo_filter``): this form
    runs every word through :func:`vemoizer.textnorm.textnorm` (the same
    normalizer as the hypothesis words), which casefolds, replaces
    punctuation — hyphens included — with spaces, and collapses
    whitespace. Because the glossary token set is built from the same
    normalization, a hyphenated term and an echo that transcribes it with
    or without the hyphen land on the same fragment tokens (``NG-TOPI`` →
    ``ng topi`` on both sides), so the hyphen never breaks the match; it
    tolerates up to
    :data:`_PROMPT_ECHO_OUTSIDE_FRACTION` of the words falling outside the
    glossary, because it only gates a metric and must not lose real term
    hits on a real sentence that happens to contain a few glossary words.

    *glossary* is the pre-computed lower-cased token set (see
    :func:`_glossary_token_set`); the caller builds it once per hypothesis
    rather than per word.

    Boundary cases: a bare term run with no other words (0% outside) is an
    echo; a real sentence (``"backlog on täynnä"``, 1/3 outside) is not.
    """
    if not glossary:
        return False
    if not words:
        return False
    in_glossary = sum(1 for w in words if w in glossary)
    if in_glossary < 1:
        return False
    outside = len(words) - in_glossary
    return outside / len(words) < _PROMPT_ECHO_OUTSIDE_FRACTION


def _glossary_token_set(terms: list[str]) -> set[str]:
    """All single tokens that make up any glossary term (normalized).

    Multi-word terms are split: ``"sprint planning"`` contributes both
    ``"sprint"`` and ``"planning"``. Used by :func:`_is_prompt_echo` to
    decide how much of a hypothesis is glossary content versus real speech.
    """
    tokens: set[str] = set()
    for raw in terms:
        t = textnorm(raw)
        if t:
            tokens.update(t.split())
    return tokens


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


def similarity(text_a: str, text_b: str) -> float:
    """Char-level similarity of two texts in ``[0, 1]``.

    Thin alias over :func:`vemoizer.slice_align.slice_similarity` — the
    same textnorm + SequenceMatcher ratio the slice-level dispute detector
    (issue #55) uses. Exposed here so the informational eval metrics
    (e.g. :func:`agreement_on_wrong_sample`) share one definition of
    "similar" with the pipeline's disagreement detector; the two cannot
    quietly drift apart if the threshold is re-tuned.

    Two empty texts are identical (1.0); one empty side is a total
    disagreement (0.0).
    """
    return slice_similarity(text_a, text_b)


def agreement_on_wrong_sample(
    reference: str, hypothesis_a: str, hypothesis_b: str
) -> bool:
    """True when decoders A and B agree with each other AND the shared text
    is wrong (issue #62).

    A sample is "in agreement" when :func:`similarity` of the two
    normalized texts is at least :data:`AGREEMENT_THRESHOLD` (the boundary
    sample, where similarity equals the threshold exactly, **counts** as
    in-agreement). The sample is "wrong" when the WER of *hypothesis_a*
    against *reference* strictly exceeds :data:`_WRONG_WER` (the boundary
    sample, where WER equals the threshold exactly, does **not** count as
    wrong). A sample that is both is the case a two-way comparison cannot
    detect on its own: the pipeline ships decode A, the LLM never
    adjudicates (no dispute), and the WER aggregate is dragged down by the
    shared miss. This is what the ``--backend all`` run's informational
    ``agreement_on_wrong`` number is meant to expose.
    """
    if similarity(hypothesis_a, hypothesis_b) < AGREEMENT_THRESHOLD:
        return False
    return wer(reference, hypothesis_a) > _WRONG_WER


def agreement_on_wrong(
    references: dict[str, str],
    hypothesis_a: dict[str, str],
    hypothesis_b: dict[str, str],
) -> float:
    """Fraction of samples where decoders A and B agree with each other
    (similarity ≥ :data:`AGREEMENT_THRESHOLD`) AND the shared hypothesis is
    wrong against *references* (issue #62).

    *references* maps sample stem -> reference transcript. *hypothesis_a*
    and *hypothesis_b* are the per-sample hypothesis mappings from two
    different decoders — the same shape :func:`run_eval`'s *hypotheses*
    parameter produces. The function does not walk the corpus (the caller
    already walked it twice); it pairs each stem in *hypothesis_a* against
    the same stem in *hypothesis_b* and counts the samples where
    :func:`agreement_on_wrong_sample` is true. A stem present in one
    mapping but missing from the other, or missing from *references*, is
    not counted (the caller is responsible for all three mappings covering
    the same sample set, as the ``--backend all`` run's per-backend walks
    guarantee).

    Returns a fraction in ``[0, 1]``; 0.0 when *hypothesis_a* is empty.
    Informational — the WER gate (:func:`compare_to_baseline`) is the only
    regression gate; this number is reported, never gated (invariant #2).
    """
    if not hypothesis_a:
        return 0.0
    agree_and_wrong = sum(
        1
        for stem, hyp_a in hypothesis_a.items()
        if stem in hypothesis_b
        and stem in references
        and agreement_on_wrong_sample(references[stem], hyp_a, hypothesis_b[stem])
    )
    return agree_and_wrong / len(hypothesis_a)


def run_eval(
    corpus_dir: Path,
    transcribe: Callable[[Path], str],
    hypotheses: dict[str, str] | None = None,
) -> dict[str, float]:
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
    ``"aggregate"`` key, macro average over samples). If a writable
    *hypotheses* dict is passed, it is filled with the per-sample
    hypotheses (``{stem: hypothesis}``) as the walk progresses — that is
    what :func:`run_meeting_eval` reuses via its *reuse* parameter so the
    meeting walk does not re-decode a sample the WER walk just scored,
    and the seam the multi-backend ``agreement_on_wrong`` run reuses to
    pair two decoders' per-sample outputs (issue #62).
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
        if hypotheses is not None:
            hypotheses[wav.stem] = hypothesis
        results[wav.stem] = wer(reference, hypothesis)
    if not results:
        return {AGGREGATE_KEY: 0.0}
    results[AGGREGATE_KEY] = sum(results.values()) / len(results)
    return results


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

    *reuse* is a performance seam owned by the CLI, not the harness: it
    maps stems to hypotheses the WER walk (:func:`run_eval`) already
    decoded (passed out through its *hypotheses* parameter), and the
    meeting walk scores those without re-decoding, so a growing meeting
    set does not compound into one extra decode pass per sample. A direct
    caller that passes *reuse* is responsible for that mapping covering the
    meeting samples — a partial dict silently mixes reused and freshly
    decoded hypotheses in the same result, which confounds any comparison
    the numbers are used for. The meeting walk itself never chooses to
    skip a decode: the WER gate stays independent of the informational
    term-hit metric because a corpus without a ``.terms`` pair scores WER
    as before, and the meeting metric can be dropped or changed without
    touching the gate.
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
