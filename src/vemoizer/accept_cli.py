"""``vemoizer accept`` Typer command (issue #229, epic #220).

Runs the CURRENT meeting pipeline on every frozen bench excerpt, scores the
result against the owner's reviewed reference
(:mod:`vemoizer.accept_run`), and gates it against the numbers-only baseline
in :mod:`vemoizer.accept_baseline`:

    vemoizer accept --check

The hypothesis is ALWAYS this fresh decode. The corrected sidecar an
excerpt's reference was drafted from is where the answer came from, so this
command never reads it — a bench that scored the sidecar would measure the
sidecar writer, not the pipeline.

The printed rows are anonymised (``E1``..``Ek``, no excerpt ids, speaker
names, paths, titles or transcript text) so the whole run is safe to paste
into a PR body. The bench itself lives in a project layer's
``.vemoizer/bench`` and is never committed.
"""

from __future__ import annotations

import contextlib
import logging
import os
import sys
from pathlib import Path
from typing import Any

import typer

from vemoizer import (
    accept_answers,
    accept_baseline,
    accept_bundle,
    accept_run,
    batch_preset,
)
from vemoizer.owner_config import load_owner_inputs
from vemoizer.presets import replace
from vemoizer.tracks import configured_owner

DEFAULT_BASELINE = Path("tests/fixtures/accept_baseline.json")

#: The project-layer bench directory, next to the config/glossary layers.
BENCH_DIR_NAME = ".vemoizer/bench"

logger = logging.getLogger(__name__)


def register_accept(app: typer.Typer) -> None:
    """Attach the ``accept`` command to *app* (the main Typer instance)."""

    @app.command("accept", hidden=True)
    def accept(
        bench: Path | None = typer.Option(  # noqa: B008 - typer reads option defaults at definition time
            None,
            "--bench",
            help=(
                "Bench directory (default: the nearest .vemoizer/bench found "
                "by walking up from the current directory)."
            ),
        ),
        baseline: Path = typer.Option(  # noqa: B008 - typer reads option defaults at definition time
            DEFAULT_BASELINE,
            "--baseline",
            help="Numbers-only accept baseline for --check/--update-baseline.",
        ),
        check: bool = typer.Option(  # noqa: B008 - typer reads option defaults at definition time
            False,
            "--check",
            help="Gate against the baseline; exit 2 on any regression.",
        ),
        update_baseline: bool = typer.Option(  # noqa: B008 - typer reads option defaults at definition time
            False,
            "--update-baseline",
            help="Write the measured numbers as the new baseline.",
        ),
        answers: Path | None = typer.Option(  # noqa: B008 - typer reads option defaults at definition time
            None,
            "--import-answers",
            help=(
                "Import the owner's quiz answers (bench-answers.json) as probes "
                "into the bench. Runs nothing else."
            ),
        ),
        by: str | None = typer.Option(  # noqa: B008 - typer reads option defaults at definition time
            None,
            "--by",
            help="Owner named as the quiz reviewer (default: the configured owner).",
        ),
    ) -> None:
        """Score the private acceptance bench against the reviewed references."""
        if answers is not None:
            _import_answers(bench, answers, by, gated=check or update_baseline)
            return
        # Deferred: vemoizer.batch pulls the batch/notify stack, and the
        # temp-glossary seam is patched there by name (like run_preset's).
        from vemoizer.batch import _write_temp_glossary

        # The pipeline's INFO logging is the only feedback during a run of
        # minutes (model loads are slow); it goes to stderr, the report to
        # stdout, like every other vemoizer command.
        logging.basicConfig(level=logging.INFO, stream=sys.stderr)

        # absolute(), not resolve(): these paths are used from inside the
        # project's cwd (and after it), so they must not depend on it. The
        # printed baseline path stays the one the owner passed — the report is
        # pasted into PR bodies and never carries an absolute path.
        bench_dir = _resolve_bench(bench).absolute()
        baseline_shown = str(baseline)
        baseline = baseline.absolute()
        # The bench lives in a project layer (<project>/.vemoizer/bench), so
        # the run must happen FROM that project root: every layered lookup
        # (glossary, LLM config, people, run logs) walks up from the cwd, and
        # ``vemoizer accept`` is usually invoked from the vemoizer checkout.
        project_root = _project_root(bench_dir)
        try:
            excerpts, references = accept_bundle.load_bundle(bench_dir)
        except (accept_bundle.BundleError, ValueError) as exc:
            typer.echo(f"error: {exc}", err=True)
            raise typer.Exit(code=2) from None
        labels = {e.id: f"E{index}" for index, e in enumerate(excerpts, 1)}
        drafts = set(accept_bundle.unreviewed(references))
        _refuse_drafts(drafts, labels, gated=check or update_baseline)

        bench_fingerprint = accept_baseline.fingerprint(
            bench_dir, [excerpt.id for excerpt in excerpts]
        )
        # Read the baseline BEFORE decoding: a drifted bench or a missing
        # baseline must fail in milliseconds, not after minutes of MLX.
        base = (
            _baseline_for_check(baseline, bench_fingerprint, baseline_shown)
            if check
            else None
        )

        measured: dict[str, dict[str, Any] | None] = {}
        temp_path: Path | None = None
        with contextlib.chdir(project_root):
            try:
                # Composed HERE, with the project layer under the cwd: the
                # same layered glossary a vemoizer meeting run would merge
                # (batch_preset is the one code path), and the same terms the
                # scorer watches. The temp file goes in the finally below.
                options = batch_preset.resolve_preset_glossary_options("meeting", {})
                terms = accept_run.scoring_terms(options)
                lines = batch_preset.meeting_glossary_lines(options)
                temp_path = Path(_write_temp_glossary(lines)) if lines else None
                # The temp file is part of the options a meeting run hands the
                # pipeline, so it travels on RunOptions and not as an extra
                # argument of the bench's own.
                options = replace(
                    options, glossary_path=str(temp_path) if temp_path else None
                )
                for excerpt in excerpts:
                    label = labels[excerpt.id]
                    logger.info(
                        "accept: %s (model loads are slow on the first run)", label
                    )
                    result = accept_run.run_excerpt(bench_dir, excerpt, options)
                    measured[label] = accept_run.score_run(
                        result, references[excerpt.id], terms
                    )
                    _echo_row(label, measured[label], draft=excerpt.id in drafts)
            finally:
                if temp_path is not None:
                    temp_path.unlink(missing_ok=True)

        failed = sorted(
            (label for label, row in measured.items() if row is None),
            key=lambda label: int(label[1:]),
        )
        aggregate = accept_baseline.aggregate(
            [row for row in measured.values() if row is not None]
        )
        _echo_row(accept_baseline.AGGREGATE_SCOPE, aggregate, draft=False)
        typer.echo(f"fingerprint\t{bench_fingerprint}")

        if update_baseline:
            if failed:
                # A baseline written over failed excerpts silently drops them
                # from the gate, so nothing is written at all — an existing
                # baseline keeps its numbers.
                typer.echo(
                    f"error: refusing to write a baseline over failed excerpts: "
                    f"{', '.join(failed)}",
                    err=True,
                )
                raise typer.Exit(code=2)
            accept_baseline.write_baseline(
                baseline,
                bench_fingerprint=bench_fingerprint,
                excerpts={k: v for k, v in measured.items() if v is not None},
                aggregate_row=aggregate,
            )
            typer.echo(f"baseline updated: {baseline_shown}")
        if base is not None:
            _gate(base, measured, aggregate)


# ---------------------------------------------------------------------------
# Bench discovery
# ---------------------------------------------------------------------------


def _resolve_bench(explicit: Path | None) -> Path:
    """*explicit*, else the nearest project layer's ``.vemoizer/bench``.

    A missing bench is exit 2 with the cwd named: the owner needs to know
    where we looked.
    """
    if explicit is not None:
        return explicit
    found = accept_bundle.find_bench()
    if found is not None:
        return found
    typer.echo(
        f"error: no {BENCH_DIR_NAME} found above {os.getcwd()}; "
        "freeze an excerpt into it (vemoizer.accept_bundle.freeze_excerpt) "
        "or pass --bench",
        err=True,
    )
    raise typer.Exit(code=2)


def _project_root(bench_dir: Path) -> Path:
    """The project that owns *bench_dir*, or exit 2 explaining the layout.

    ``bench_dir.parent.parent`` is only the project whose glossary, config,
    people and run logs drive the run when the bench sits at
    ``<project>/.vemoizer/bench``. Anywhere else the run would measure some
    other layer's configuration, so it is refused before any decode.
    """
    if bench_dir.name != "bench" or bench_dir.parent.name != ".vemoizer":
        typer.echo(
            f"error: the bench must live at <project>/{BENCH_DIR_NAME} — its "
            "project layer drives the glossary and the config this run measures",
            err=True,
        )
        raise typer.Exit(code=2)
    return bench_dir.parent.parent


def _refuse_drafts(drafts: set[str], labels: dict[str, str], *, gated: bool) -> None:
    """Exit 2 on unreviewed references when gating; warn (and continue) otherwise.

    A draft reference is the owner's own draft, not ground truth: measuring
    against it and writing it into a baseline would enshrine an unreviewed
    answer. Without a gate flag the run still happens — the owner uses it to
    preview the numbers before spending the listening time.
    """
    if not drafts:
        return
    names = ", ".join(sorted((labels[i] for i in drafts), key=lambda s: int(s[1:])))
    if not gated:
        logger.warning("unreviewed (draft) references: %s", names)
        return
    typer.echo(
        f"error: refusing to gate on unreviewed references: {names} — "
        "review them in the bench first",
        err=True,
    )
    raise typer.Exit(code=2)


# ---------------------------------------------------------------------------
# Quiz answers (issue #243)
# ---------------------------------------------------------------------------


def _import_answers(
    bench: Path | None, answers: Path, by: str | None, *, gated: bool
) -> None:
    """``--import-answers``: write the quiz probes into the bench, then stop.

    Exit 2 on a bad answers file, an unknown excerpt or no owner; nothing is
    written in any of those cases. The owner is read from the bench's project
    layer, the same config a run of that project would use.
    """
    if gated:
        typer.echo(
            "error: --import-answers runs alone; drop --check and --update-baseline",
            err=True,
        )
        raise typer.Exit(code=2)
    bench_dir = _resolve_bench(bench).absolute()
    with contextlib.chdir(_project_root(bench_dir)):
        owner = configured_owner(by, *load_owner_inputs())
    if owner is None:
        typer.echo(
            "error: no owner: pass --by NAME or set [meeting] owner in the config",
            err=True,
        )
        raise typer.Exit(code=2)
    try:
        excerpts, _refs = accept_bundle.load_bundle(bench_dir)
        counts = accept_answers.import_answers(bench_dir, answers, owner)
    except (accept_bundle.BundleError, ValueError) as exc:  # AnswersError is one
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=2) from None
    labels = {e.id: f"E{index}" for index, e in enumerate(excerpts, 1)}
    for excerpt_id in sorted(counts, key=lambda i: int(labels[i][1:])):
        speaker, term = counts[excerpt_id]
        typer.echo(f"{labels[excerpt_id]}\tprobes={speaker}\tterm_probes={term}")


# ---------------------------------------------------------------------------
# Output and gate (anonymised: safe to paste into a PR body)
# ---------------------------------------------------------------------------


def _echo_row(label: str, values: dict[str, Any] | None, *, draft: bool) -> None:
    """One tab-separated row: label, ``(draft)`` when unreviewed, all metrics.

    ``FAILED`` (and no metrics) for an excerpt whose pipeline run errored:
    its numbers would be an empty-transcript score, i.e. a lie.
    """
    if values is None:
        typer.echo(f"{label}\tFAILED")
        return
    cells = [label, "(draft)"] if draft else [label]
    cells += [f"{key}={accept_baseline.fmt(value)}" for key, value in values.items()]
    typer.echo("\t".join(cells))


def _baseline_for_check(
    path: Path, bench_fingerprint: str, shown: str
) -> dict[str, Any]:
    """The baseline dict for ``--check``, or exit 2 with the reason.

    *shown* is the path as the owner passed it: the error lines quote that,
    never the absolute path used for IO.
    """
    data = accept_baseline.read_baseline(path)
    if data is None:
        typer.echo(
            f"error: no accept baseline at {shown}; run `vemoizer accept "
            "--update-baseline` in a dedicated commit first",
            err=True,
        )
        raise typer.Exit(code=2)
    if data.get("fingerprint") != bench_fingerprint:
        typer.echo(
            "error: bench fingerprint does not match the baseline — the bench "
            "changed; re-measure with --update-baseline in a dedicated commit",
            err=True,
        )
        raise typer.Exit(code=2)
    return data


def _gate(
    base: dict[str, Any],
    measured: dict[str, dict[str, Any] | None],
    aggregate: dict[str, Any],
) -> None:
    """Exit 2 naming every regressed key; print the pass line otherwise."""
    failed = False
    scored: dict[str, dict[str, Any]] = {}
    for label, row in measured.items():
        if row is None:
            # A failed excerpt is the worst possible regression: no transcript
            # at all, and no numbers to compare.
            typer.echo(f"regression [{label}] excerpt: failed pipeline run", err=True)
            failed = True
            continue
        scored[label] = row
    scored[accept_baseline.AGGREGATE_SCOPE] = aggregate
    try:
        regressions = accept_baseline.compare(scored, base)
    except accept_baseline.BaselineDrift as exc:
        # A baseline that cannot describe this run is not a passing gate.
        typer.echo(f"error: {exc}", err=True)
        raise typer.Exit(code=2) from None
    for regression in regressions:
        failed = True
        typer.echo(regression.line(), err=True)
    if failed:
        raise typer.Exit(code=2)
    typer.echo("accept check passed")
