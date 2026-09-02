"""``vemoizer eval`` Typer command (issue #11).

Registers the ``eval`` subcommand on the main app:

    vemoizer eval --corpus tests/fixtures/corpus

Prints per-sample WER plus the aggregate. Exits 1 if the corpus
directory does not exist.
"""

from __future__ import annotations

from pathlib import Path

import typer

from vemoizer.eval_harness import AGGREGATE_KEY, run_eval


def register_eval(app) -> None:
    """Attach the ``eval`` command to *app* (the main Typer instance)."""

    @app.command("eval")
    def eval(  # noqa: A001, A002 - mirrors vemoizer CLI subcommand name
        corpus: Path = typer.Option(  # noqa: B008
            Path("tests/fixtures/corpus"),
            "--corpus",
            exists=False,
            help="Corpus directory of stem-paired .wav/.txt fixtures.",
        ),
    ) -> None:
        """Evaluate the consensus pipeline WER over a fixture corpus."""
        if not corpus.is_dir():
            typer.echo(f"error: corpus directory not found: {corpus}", err=True)
            raise typer.Exit(code=1)
        results = run_eval(corpus)
        for name, value in results.items():
            if name == AGGREGATE_KEY:
                continue
            typer.echo(f"{name}\t{value:.4f}")
        typer.echo(f"{AGGREGATE_KEY}\t{results[AGGREGATE_KEY]:.4f}")
