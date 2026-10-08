"""``vemoizer eval`` Typer command (issues #11, #51).

Scores one or more decode backends over the stem-paired fixture corpus and
gates against a committed WER baseline:

    vemoizer eval --backend all --check

Backends: ``parakeet`` and ``canary`` are single decodes through
:func:`vemoizer.pipeline.transcribe_decode_only`; ``consensus`` is the full
pipeline with the LLM forced off (``config_path=os.devnull``) so eval runs
are deterministic and local — pass ``--llm`` to adjudicate with the user's
configured endpoint instead.

The baseline file records the corpus fingerprint alongside the numbers, so
``--check`` refuses to compare against a silently changed corpus (exit 2,
same as a regression). Accuracy claims in PRs come from this command's
output (AGENTS.md invariant #7).
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from pathlib import Path

import typer

from vemoizer.eval_harness import (
    AGGREGATE_KEY,
    agreement_on_wrong,
    compare_to_baseline,
    corpus_fingerprint,
    run_eval,
    run_meeting_eval,
)

#: Gate tolerance: greedy decodes are deterministic in principle, but Metal
#: reductions are not bit-stable across MLX versions; a small slack keeps
#: the gate meaningful without flaking.
DEFAULT_TOLERANCE = 0.02

DEFAULT_BASELINE = Path("tests/fixtures/wer_baseline.json")


def transcribe_decode_only(path: Path | str, *, backend: str) -> dict:
    """Run ingest -> VAD -> one single decode; no consensus, no LLM.

    Lives here because eval is its only caller: the harness scores each
    decode backend on its own so the consensus gain is a measured number
    (invariant #7). The stage chain mirrors ``transcribe_file``'s decode
    stage exactly — same VAD slicing, same merge.
    """
    from contextlib import suppress
    from pathlib import Path as _Path
    from typing import Any

    import vemoizer.pipeline as pipeline_module
    from vemoizer.decode_stage import decode_all
    from vemoizer.ingest import IngestError

    logger = pipeline_module.logger
    # Resolved through the pipeline module so the same seams (and test
    # monkeypatching) govern eval decodes and pipeline decodes alike.
    backends = {
        "parakeet": pipeline_module.ParakeetTranscriber,
        "canary": pipeline_module.CanaryTranscriber,
    }
    if backend not in backends:
        known = ", ".join(sorted(backends))
        raise ValueError(f"unknown backend {backend!r} (known: {known})")
    try:
        audio = pipeline_module.ingest_audio(_Path(path))
    except IngestError as e:
        logger.error("ingest failed for %s: %s", path, e)
        return {"text": "", "segments": [], "error": str(e)}
    if len(audio) == 0:
        return {"text": "", "segments": []}
    slices, _vad_found_speech = pipeline_module._speech_slices(audio)
    transcriber: Any = None
    result: dict[str, Any] | None = None
    try:
        transcriber = backends[backend]()
        result = decode_all(transcriber, slices, f"decode ({backend})")
    except Exception as e:  # noqa: BLE001 - fail-open stage boundary
        logger.warning("decode (%s) failed: %s", backend, e)
    finally:
        if transcriber is not None:
            with suppress(Exception):  # cleanup is best-effort (fail-open)
                transcriber.cleanup()
    if result is None:
        return {"text": "", "segments": []}
    return {
        "text": str(result.get("text", "")).strip(),
        "segments": list(result.get("segments") or []),
    }


def _decode_only(backend: str) -> Callable[[Path], str]:
    def _transcribe(wav: Path) -> str:
        return transcribe_decode_only(wav, backend=backend)["text"]

    return _transcribe


def _consensus(wav: Path) -> str:
    from vemoizer.pipeline import transcribe_file

    # os.devnull parses as empty TOML -> no [llm] -> adjudication skipped:
    # the eval number reflects the local consensus, not a network endpoint.
    return transcribe_file(wav, config_path=os.devnull)["text"]


def _consensus_llm(wav: Path) -> str:
    from vemoizer.pipeline import transcribe_file

    return transcribe_file(wav)["text"]


#: name -> (wav path -> hypothesis). Tests monkeypatch this registry.
BACKENDS: dict[str, Callable[[Path], str]] = {
    "parakeet": _decode_only("parakeet"),
    "canary": _decode_only("canary"),
    "consensus": _consensus,
}


def register_eval(app) -> None:
    """Attach the ``eval`` command to *app* (the main Typer instance)."""

    @app.command("eval", hidden=True)
    def eval(  # noqa: A001, A002 - mirrors vemoizer CLI subcommand name
        corpus: Path = typer.Option(  # noqa: B008
            Path("tests/fixtures/corpus"),
            "--corpus",
            help="Corpus directory of stem-paired .wav/.txt fixtures.",
        ),
        backend: str = typer.Option(
            "all",
            "--backend",
            help="Backend to score: parakeet, canary, consensus, or all.",
        ),
        baseline: Path = typer.Option(  # noqa: B008
            DEFAULT_BASELINE,
            "--baseline",
            help="Committed WER baseline file for --check/--update-baseline.",
        ),
        check: bool = typer.Option(
            False,
            "--check",
            help="Gate against the baseline; exit 2 on any regression.",
        ),
        update_baseline: bool = typer.Option(
            False,
            "--update-baseline",
            help="Write the measured numbers as the new baseline.",
        ),
        llm: bool = typer.Option(
            False,
            "--llm",
            help="Let the consensus backend adjudicate with the configured LLM "
            "(default: LLM off so eval stays local and deterministic).",
        ),
        agreement: bool = typer.Option(
            False,
            "--agreement",
            help="Emit the informational agreement-on-wrong metric "
            "(two decoders agree AND the shared text is wrong; issue #62).",
        ),
    ) -> None:
        """Score decode backends over the fixture corpus (WER)."""
        if not corpus.is_dir():
            typer.echo(f"error: corpus directory not found: {corpus}", err=True)
            raise typer.Exit(code=1)

        names = list(BACKENDS) if backend == "all" else [backend]
        unknown = [n for n in names if n not in BACKENDS]
        if unknown:
            known = ", ".join([*BACKENDS, "all"])
            typer.echo(
                f"error: unknown backend(s): {', '.join(unknown)} (known: {known})",
                err=True,
            )
            raise typer.Exit(code=2)

        measured: dict[str, dict[str, float]] = {}
        if agreement:
            references: dict[str, str] = {}
            per_backend_hyps: dict[str, dict[str, str]] = {}
            _collect_references(corpus, references)
        else:
            references = {}
            per_backend_hyps = {}
        for name in names:
            transcribe = BACKENDS[name]
            if name == "consensus" and llm:
                transcribe = _consensus_llm
            hyps: dict[str, str] = {}
            results = run_eval(corpus, transcribe, hyps)
            measured[name] = results
            if agreement:
                per_backend_hyps[name] = hyps
            typer.echo(f"[{name}]")
            for sample, value in results.items():
                if sample == AGGREGATE_KEY:
                    continue
                typer.echo(f"{sample}\t{value:.4f}")
            typer.echo(f"{AGGREGATE_KEY}\t{results[AGGREGATE_KEY]:.4f}")
            _emit_meeting_term_hits(corpus, transcribe, hyps)

        if agreement:
            _emit_agreement_on_wrong(references, per_backend_hyps)

        fingerprint = corpus_fingerprint(corpus)
        if update_baseline:
            _write_baseline(baseline, fingerprint, measured)
            typer.echo(f"baseline updated: {baseline}")
        if check:
            _check_baseline(baseline, fingerprint, measured)


def _emit_meeting_term_hits(
    corpus: Path, transcribe: Callable[[Path], str], reuse: dict[str, str]
) -> None:
    """Emit the meeting-fixture glossary term-hit rate (issue #76).

    The meeting eval reuses the per-sample hypotheses already computed by
    the WER walk (the *reuse* dict from :func:`run_eval`) for samples that
    both walks score, so the WER walk and the term-hit walk share the
    decode and a growing meeting set does not compound into one extra pass
    per sample. Samples not covered by *reuse* (e.g. a meeting sample the
    WER walk did not score) fall back to a fresh decode through *transcribe*.

    The metric is informational here — it is the number the glossary prompt
    work is judged by (kept only if term hits rise and WER does not
    regress). Absent meeting fixtures (a corpus without a ``.terms`` pair)
    nothing is emitted: the WER gate above is unaffected.

    A 0.0 term-hit on a reused hypothesis reflects the WER-walk decode
    (not a fresh decode), and the harness catches any exception from the
    fallback path and scores the sample as empty (the WER number above,
    computed on the first pass, already says the decode succeeded).
    """
    meeting = run_meeting_eval(corpus, transcribe, reuse=reuse)
    samples = [s for s in meeting if s != AGGREGATE_KEY]
    if not samples:
        return
    for sample in samples:
        typer.echo(f"[term-hit/{sample}]\t{meeting[sample]['term_hit']:.4f}")
    typer.echo(f"[term-hit/{AGGREGATE_KEY}]\t{meeting[AGGREGATE_KEY]['term_hit']:.4f}")


def _collect_references(corpus: Path, references: dict[str, str]) -> None:
    """Fill *references* with the per-sample reference transcripts (stem -> text).

    Reads each ``<stem>.txt`` in *corpus*; a missing file is skipped (the
    sample simply won't be scored by the agreement metric).
    """
    for wav in sorted(corpus.glob("*.wav")):
        txt = wav.with_suffix(".txt")
        if txt.is_file():
            references[wav.stem] = txt.read_text(encoding="utf-8")


def _emit_agreement_on_wrong(
    references: dict[str, str],
    per_backend_hyps: dict[str, dict[str, str]],
) -> None:
    """Emit the informational ``agreement_on_wrong`` metric (issue #62).

    A sample is "agreement on a wrong answer" when two independent decoders
    produce similar text for the same audio AND both hypotheses are wrong
    against the reference — the limit of a two-way consensus pipeline
    (the pipeline ships decode A, the LLM never adjudicates, and the WER
    aggregate inherits the shared error). The number is informational: it
    is reported, never gated (invariant #2, the WER gate stays the only
    regression gate), and it is only meaningful when at least two
    *independent* backends were scored (a single-backend run has no second
    decoder to compare against, so nothing is emitted).

    The pair is pinned by name — ``parakeet`` and ``canary`` — rather than
    "the first two independent backends in dict order", so a future
    re-ordering of :data:`BACKENDS` or the addition of a new independent
    backend cannot silently change which pair is scored. If either
    ``parakeet`` or ``canary`` is missing from *per_backend_hyps* the
    metric is skipped with a clear note.

    The pairing and counting logic lives in the harness's
    :func:`vemoizer.eval_harness.agreement_on_wrong`; this function only
    selects the independent backends and prints the result.
    """
    a_name, b_name = "parakeet", "canary"
    if a_name not in per_backend_hyps or b_name not in per_backend_hyps:
        missing = [n for n in (a_name, b_name) if n not in per_backend_hyps]
        typer.echo(
            f"note: skipping agreement_on_wrong — missing backend(s): "
            f"{', '.join(missing)}",
            err=True,
        )
        return
    value = agreement_on_wrong(
        references, per_backend_hyps[a_name], per_backend_hyps[b_name]
    )
    typer.echo(f"[agreement_on_wrong]\t{value:.4f}")


def _write_baseline(
    path: Path, fingerprint: str, measured: dict[str, dict[str, float]]
) -> None:
    existing = _read_baseline(path)
    backends = dict(existing.get("backends", {})) if existing else {}
    backends.update(measured)  # keep other backends' numbers when re-measuring one
    payload = {
        "corpus_fingerprint": fingerprint,
        "tolerance": (existing or {}).get("tolerance", DEFAULT_TOLERANCE),
        "note": (existing or {}).get(
            "note",
            "WER over tests/fixtures/corpus; update only in a dedicated commit.",
        ),
        "backends": backends,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1, sort_keys=True), encoding="utf-8")


def _read_baseline(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _check_baseline(
    path: Path, fingerprint: str, measured: dict[str, dict[str, float]]
) -> None:
    data = _read_baseline(path)
    if data is None:
        typer.echo(
            f"error: no baseline at {path}; run with --update-baseline first",
            err=True,
        )
        raise typer.Exit(code=2)
    if data.get("corpus_fingerprint") != fingerprint:
        typer.echo(
            "error: corpus fingerprint does not match the baseline — the corpus "
            "changed; re-measure with --update-baseline in a dedicated commit",
            err=True,
        )
        raise typer.Exit(code=2)
    tolerance = float(data.get("tolerance", DEFAULT_TOLERANCE))
    failed = False
    for name, results in measured.items():
        base = data.get("backends", {}).get(name)
        if base is None:
            typer.echo(f"error: baseline has no entry for backend {name}", err=True)
            failed = True
            continue
        for reg in compare_to_baseline(results, base, tolerance=tolerance):
            failed = True
            was = "missing" if reg.baseline is None else f"{reg.baseline:.4f}"
            typer.echo(
                f"regression [{name}] {reg.name}: baseline {was} -> "
                f"measured {reg.measured:.4f}",
                err=True,
            )
    if failed:
        raise typer.Exit(code=2)
    typer.echo("baseline check passed")
