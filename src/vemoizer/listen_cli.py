"""The ``listen`` Typer command (issue #173): record a call, then transcribe it.

Mirrors the ``meeting`` command's wiring (battery warning, run log, Ctrl-C
tracking); the recording itself lives in :mod:`vemoizer.listen`, imported
lazily so ``--help`` never loads capture code.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import typer

from vemoizer.cli_support import run_log_configure, warn_on_battery
from vemoizer.preset_cli import preset_interrupt_exit
from vemoizer.preset_interrupt import begin_interrupt_tracking


def register_listen(app: typer.Typer) -> None:
    """Attach the ``listen`` command to *app*."""

    @app.command()
    def listen(
        quiet: bool = typer.Option(  # noqa: B008
            False, "--quiet", "-q", help="Suppress the summary output."
        ),
        verbose: bool = typer.Option(  # noqa: B008
            False, "--verbose", "-v", help="Emit per-stage progress logging to stderr."
        ),
        config: Path | None = typer.Option(  # noqa: B008
            None,
            "--config",
            help="LLM config file (default: layered .vemoizer/config.toml search).",
        ),
        glossary: Path | None = typer.Option(  # noqa: B008
            None,
            "--glossary",
            help="Explicit glossary file (replaces both .vemoizer layers).",
        ),
        repair: bool = typer.Option(  # noqa: B008
            True,
            "--repair/--no-repair",
            help="LLM repair pass over the final paragraphs (on by default).",
        ),
        speakers: str | None = typer.Option(  # noqa: B008
            None,
            "--speakers",
            help="People in the recording: N or MIN-MAX (default 2-6).",
        ),
        no_diarize: bool = typer.Option(  # noqa: B008
            False,
            "--no-diarize",
            help="Skip speaker diarization of the remote side.",
        ),
        yes: bool = typer.Option(  # noqa: B008
            False,
            "--yes",
            help="Never prompt: use the fallback owner name if none is configured.",
        ),
        language: str = typer.Option(  # noqa: B008
            "auto",
            "--language",
            help="Recognition language: auto (the default), fi, or en.",
        ),
        mic: str | None = typer.Option(  # noqa: B008
            None,
            "--mic",
            help="Microphone device name (default: the Mac's default input).",
        ),
        me: str | None = typer.Option(  # noqa: B008
            None,
            "--me",
            help="Your name for the transcript (default: asked once, before recording)",
        ),
        duration: float | None = typer.Option(  # noqa: B008
            None,
            "--duration",
            help="Stop recording after SECONDS (default: until Ctrl-C).",
        ),
        live: bool = typer.Option(  # noqa: B008
            False,
            "--live",
            help="Stream a single-decode draft while recording (not the deliverable).",
        ),
        keep_live: bool = typer.Option(  # noqa: B008
            False,
            "--keep-live",
            help="Keep the live draft after the final transcript is written.",
        ),
    ) -> None:
        """Record a call (system audio + mic), then transcribe it as a meeting."""
        warn_on_battery()

        if verbose:
            logging.basicConfig(level=logging.INFO, stream=sys.stderr)
        run_log_configure(
            verbose=verbose,
            quiet=quiet,
            config_path=str(config) if config is not None else None,
        )

        from vemoizer.listen import ListenRun, run_listen

        tracker = begin_interrupt_tracking()
        run = ListenRun(
            me=me,
            mic=mic,
            duration=duration,
            yes=yes,
            quiet=quiet,
            config_path=str(config) if config is not None else None,
            glossary_path=str(glossary) if glossary is not None else None,
            repair=repair,
            no_diarize=no_diarize,
            speakers=speakers,
            language=language,
            live=live,
            keep_live=keep_live,
        )
        try:
            exit_code = run_listen(run, tracker=tracker)
        except KeyboardInterrupt:
            preset_interrupt_exit(tracker, None)
        if exit_code:
            raise typer.Exit(code=exit_code)
