"""``vemoizer eval`` Typer command (issues #11, #51).

Scores one or more decode backends over the stem-paired fixture corpus and
gates against a committed WER baseline:

    vemoizer eval --backend all --check

Backends: ``parakeet`` and ``canary`` are single decodes through
:func:`vemoizer.eval_backends.transcribe_decode_only`; ``whisper-turbo`` is
the meeting profile's decode A alone; ``consensus`` is the full
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
import shutil
import signal
import tempfile
from collections.abc import Callable
from pathlib import Path
from types import FrameType

import typer

from vemoizer import eval_backends
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
            help="Backend to score: parakeet, canary, consensus, whisper-turbo, "
            "or all.",
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
        degrade: str | None = typer.Option(
            None,
            "--degrade",
            help="Informational far-field benchmark (issue #154): profile "
            "far, noisy, or far-noisy; prints plain vs loudnorm WER. "
            "Never gated; skips the regular gate output.",
        ),
    ) -> None:
        """Score decode backends over the fixture corpus (WER)."""
        if not corpus.is_dir():
            typer.echo(f"error: corpus directory not found: {corpus}", err=True)
            raise typer.Exit(code=1)

        names = list(eval_backends.BACKENDS) if backend == "all" else [backend]
        unknown = [n for n in names if n not in eval_backends.BACKENDS]
        if unknown:
            known = ", ".join([*eval_backends.BACKENDS, "all"])
            typer.echo(
                f"error: unknown backend(s): {', '.join(unknown)} (known: {known})",
                err=True,
            )
            raise typer.Exit(code=2)

        if degrade is not None:
            # Informational and exclusive: the gate, agreement and LLM flags
            # are refused (exit 2) rather than silently ignored.
            if check or update_baseline or agreement or llm:
                typer.echo(
                    "error: --degrade is informational; it cannot be combined "
                    "with --check, --update-baseline, --agreement or --llm",
                    err=True,
                )
                raise typer.Exit(code=2)
            _emit_degrade(corpus, names, degrade)
            return

        measured: dict[str, dict[str, float]] = {}
        if agreement:
            references: dict[str, str] = {}
            per_backend_hyps: dict[str, dict[str, str]] = {}
            _collect_references(corpus, references)
        else:
            references = {}
            per_backend_hyps = {}
        for name in names:
            transcribe = eval_backends.BACKENDS[name]
            if name == "consensus" and llm:
                transcribe = eval_backends.consensus_llm
            transcribe = eval_backends.on_large_stack(transcribe)
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


def _exit_on_term(signum: int, frame: object) -> None:
    """Turn SIGTERM into a normal exit so the scratch dir's ``finally`` runs."""
    raise SystemExit(128 + signum)


_Handler = Callable[[int, FrameType | None], object] | int


def _emit_degrade(corpus: Path, names: list[str], profile: str) -> None:
    """Print the informational far-field table (issue #154).

    Variants are written to a temp dir that is removed afterwards; nothing
    is committed or written under the corpus. Audio decodes once at ingest
    (the ingest contract); loudnorm is applied only in the loudnorm arm.
    Every failure exits 2 with a message, never a traceback: benchmark
    refusals (DegradeError), ingest/loudnorm failures, and I/O or decode
    errors while writing or reading the variants.
    """
    from vemoizer.degrade import (
        DegradeError,
        format_table,
        run_degrade_eval,
        run_header,
    )
    from vemoizer.ingest import IngestError, ingest_audio

    work: Path | None = None
    previous: _Handler | None = signal.signal(signal.SIGTERM, _exit_on_term)
    try:
        try:
            # Inside the try: an OSError creating the scratch dir is exit 2 too.
            work = Path(tempfile.mkdtemp(prefix="vemoizer-degrade-"))
            rows = run_degrade_eval(
                corpus,
                profile,
                {n: eval_backends.BACKENDS[n] for n in names},
                work,
                ingest_audio,
            )
        except (DegradeError, IngestError, OSError) as exc:
            typer.echo(f"error: {exc}", err=True)
            raise typer.Exit(code=2) from None
        finally:
            if work is not None:
                shutil.rmtree(work, ignore_errors=True)
    finally:
        # Restore only after the scratch dir is gone, so a SIGTERM in between
        # cannot leak it. signal.signal() returns None for a handler that was
        # not installed from Python, and None is not a valid argument back.
        restore = previous if previous is not None else signal.SIG_DFL
        signal.signal(signal.SIGTERM, restore)
    for line in [*run_header(profile), *format_table(rows)]:
        typer.echo(line)


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
    re-ordering of :data:`eval_backends.BACKENDS` or the addition of a new independent
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
