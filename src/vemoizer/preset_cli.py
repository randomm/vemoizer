"""``meeting`` and ``memo`` Typer commands (issue #82, extracted from
``cli.py`` for the 500-line source cap, issue #135).

The commands share the preset wiring (``run_preset``, the layered
config/glossary resolution, the display lifecycle) and the
``--language`` / ``--preprocess`` options. The command order in
``--help`` is preserved: ``register_presets`` is called from ``cli.py``
in the same position the ``@app.command`` decorators used to occupy.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import typer

from vemoizer.cli_support import (
    parse_speakers as _parse_speakers,
)
from vemoizer.cli_support import (
    run_log_configure as _run_log_configure,
)
from vemoizer.cli_support import (
    warn_on_battery as _warn_on_battery,
)


def register_presets(app: typer.Typer) -> None:
    """Attach the ``meeting`` and ``memo`` commands to *app*."""

    @app.command()
    def meeting(
        # B008: typer.Argument/Option in defaults are Typer's documented pattern
        files: list[Path] = typer.Argument(  # noqa: B008
            ...,
            help="One or more audio files (.m4a etc.) from a meeting.",
        ),
        quiet: bool = typer.Option(  # noqa: B008
            False,
            "--quiet",
            "-q",
            help="Suppress the summary output.",
        ),
        verbose: bool = typer.Option(  # noqa: B008
            False,
            "--verbose",
            "-v",
            help="Emit per-stage progress logging to stderr.",
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
            "--repair",
            "--no-repair",
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
            help="Skip speaker diarization (on by default for meetings).",
        ),
        yes: bool = typer.Option(  # noqa: B008
            False,
            "--yes",
            help=(
                "Group mode for 2+ files: run the boundary decodes and accept "
                "every continuation proposal without a prompt (mutually "
                "exclusive with --no-group)."
            ),
        ),
        no_group: bool = typer.Option(  # noqa: B008
            False,
            "--no-group",
            help=(
                "Skip split-recording grouping entirely (each file is "
                "transcribed standalone; no boundary decode, no concat, no "
                "part markers). Mutually exclusive with --yes."
            ),
        ),
        language: str = typer.Option(  # noqa: B008
            "auto",
            "--language",
            help=(
                "Recognition language for the whisper decode: auto (detect "
                "per window, the default), fi, or en (issue #108). "
                'A [meeting] language = "fi"|"en" key in the config file '
                "pins the same choice for meeting (and memo) runs."
            ),
        ),
        preprocess: str | None = typer.Option(  # noqa: B008
            None,
            "--preprocess",
            help=(
                "Audio preprocessing (opt-in): loudnorm (two-pass loudnorm "
                "normalization for far-field recordings; default: off, the "
                "plain decode). Case-insensitive, like --language. Adds a "
                "measurement pass (about a minute per hour of audio)."
            ),
        ),
    ) -> None:
        """Transcribe a meeting: whisper decode, diarization, repair, .md+.json."""
        _warn_on_battery()

        if verbose:
            logging.basicConfig(level=logging.INFO, stream=sys.stderr)
        _run_log_configure(
            verbose=verbose,
            quiet=quiet,
            config_path=str(config) if config is not None else None,
        )

        from vemoizer.batch import run_preset
        from vemoizer.progress_wiring import make_batch_display

        # M4b (issue #105): one display per CLI invocation, closed in a finally
        # below; --quiet suppresses the live progress line too.
        display = make_batch_display(quiet=quiet)
        speaker_count = _parse_speakers(speakers)
        lowered = language.strip().lower()
        lowered_preprocess = (
            preprocess.strip().lower() if preprocess is not None else None
        )
        if lowered_preprocess is not None and lowered_preprocess != "loudnorm":
            typer.echo(
                f"error: unknown preprocess {lowered_preprocess!r} (known: loudnorm)",
                err=True,
            )
            raise typer.Exit(code=2)
        try:
            exit_code = run_preset(
                files,
                command="meeting",
                config_path=str(config) if config is not None else None,
                glossary_path=str(glossary) if glossary is not None else None,
                repair=repair,
                diarize=False if no_diarize else None,
                speakers=speaker_count,
                language=lowered,
                quiet=quiet,
                yes=yes,
                no_group=no_group,
                display=display,
                preprocess=lowered_preprocess,
            )
        except ValueError as e:
            # Unknown --language value (resolve_options validates against
            # LANGUAGE_VALUES, issue #108): clean exit 2, never a traceback.
            typer.echo(f"error: {e}", err=True)
            raise typer.Exit(code=2) from None
        finally:
            if display is not None:
                display.close()
        if exit_code:
            raise typer.Exit(code=exit_code)

    @app.command()
    def memo(
        # B008: typer.Argument/Option in defaults are Typer's documented pattern
        files: list[Path] = typer.Argument(  # noqa: B008
            ...,
            help="One or more audio files (.m4a etc.) to transcribe as a memo.",
        ),
        quiet: bool = typer.Option(  # noqa: B008
            False,
            "--quiet",
            "-q",
            help="Suppress the summary output.",
        ),
        verbose: bool = typer.Option(  # noqa: B008
            False,
            "--verbose",
            "-v",
            help="Emit per-stage progress logging to stderr.",
        ),
        config: Path | None = typer.Option(  # noqa: B008
            None,
            "--config",
            help="LLM config file (default: layered .vemoizer/config.toml search).",
        ),
        glossary: Path | None = typer.Option(  # noqa: B008
            None,
            "--glossary",
            help="Explicit glossary file (correction pairs only for memo).",
        ),
        repair: bool = typer.Option(  # noqa: B008
            True,
            "--repair",
            "--no-repair",
            help="LLM repair pass over the final paragraphs (on by default).",
        ),
        preprocess: str | None = typer.Option(  # noqa: B008
            None,
            "--preprocess",
            help=(
                "Audio preprocessing (opt-in): loudnorm (two-pass loudnorm "
                "normalization for far-field recordings; default: off, the "
                "plain decode). Case-insensitive, like --language. Adds a "
                "measurement pass (about a minute per hour of audio)."
            ),
        ),
    ) -> None:
        """Transcribe a memo: whisper decode, no diarization, repair, .md+.json."""
        _warn_on_battery()

        if verbose:
            logging.basicConfig(level=logging.INFO, stream=sys.stderr)
        _run_log_configure(
            verbose=verbose,
            quiet=quiet,
            config_path=str(config) if config is not None else None,
        )

        from vemoizer.batch import run_preset
        from vemoizer.progress_wiring import make_batch_display

        # M4b (issue #105): one display per CLI invocation, closed in a finally
        # below; --quiet suppresses the live progress line too.
        display = make_batch_display(quiet=quiet)
        lowered_preprocess = (
            preprocess.strip().lower() if preprocess is not None else None
        )
        if lowered_preprocess is not None and lowered_preprocess != "loudnorm":
            typer.echo(
                f"error: unknown preprocess {lowered_preprocess!r} (known: loudnorm)",
                err=True,
            )
            raise typer.Exit(code=2)
        try:
            exit_code = run_preset(
                files,
                command="memo",
                config_path=str(config) if config is not None else None,
                glossary_path=str(glossary) if glossary is not None else None,
                repair=repair,
                quiet=quiet,
                display=display,
                preprocess=lowered_preprocess,
            )
        except ValueError as e:
            # Unknown [meeting] language value (resolve_options validates
            # against LANGUAGE_VALUES, issue #108): clean exit 2, never a
            # traceback — the same contract the meeting command enforces.
            typer.echo(f"error: {e}", err=True)
            raise typer.Exit(code=2) from None
        finally:
            if display is not None:
                display.close()
        if exit_code:
            raise typer.Exit(code=exit_code)
