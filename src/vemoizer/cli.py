"""Typer CLI entry point for vemoizer.

Multi-file batch interface: one or more voice-memo paths as positional
arguments, format selection (default: all formats), and
``--quiet`` / ``--verbose`` verbosity flags.

macOS UX polish (issue #14):
- ``--copy`` — copy transcript text to the clipboard via pbcopy
- Battery warning — warn before long transcription on battery power
- Caffeinate — hold a wake assertion during transcription

Model management (issue #3):
- ``models pull`` — pre-download the three revision-pinned consensus models
  and report per-model + total cache sizes

Preset commands (issue #82):
- ``meeting FILES`` — profile=meeting, diarize, repair, .md+.json to CWD
- ``memo FILES`` — whisper meeting decode without diarization, .md+.json

Progress bars render to stderr via rich; transcripts render to stdout
(rich auto-detects TTY and disables progress on non-TTY stderr).
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
from vemoizer.eval_cli import register_eval
from vemoizer.glossary_check import register_glossary
from vemoizer.names_cli import register_names
from vemoizer.output.formatters import OUTPUT_FORMATS
from vemoizer.render_cli import register_render

app = typer.Typer(
    name="vemoizer",
    help="Local-first voice memo transcription (Finnish/English consensus).",
    no_args_is_help=True,
)
models_app = typer.Typer(
    name="models",
    help="Manage the revision-pinned consensus models.",
    no_args_is_help=True,
)
app.add_typer(models_app, name="models")
register_eval(app)
register_render(app)
register_names(app)


register_glossary(app)


@app.command()
def doctor(
    preprocess: str | None = typer.Option(  # noqa: B008
        None,
        "--preprocess",
        help=(
            "Preprocessing to check: loudnorm (checks the ffmpeg loudnorm "
            "filter; informational only — the preflight gate is the "
            "enforcement point). Omitted: no extra checks (the no-flag "
            "path does not require the loudnorm filter)."
        ),
    ),
) -> None:
    """Run local health checks; exit non-zero on any red check."""
    from .doctor import run_doctor

    lowered = preprocess.strip().lower() if preprocess is not None else None
    if lowered is not None and lowered != "loudnorm":
        typer.echo(f"error: unknown preprocess {lowered!r} (known: loudnorm)", err=True)
        raise typer.Exit(code=2)
    report = run_doctor(echo=lambda line: typer.echo(line), preprocess=lowered or None)
    if not report.ok:
        raise typer.Exit(code=1)


@models_app.command("pull")
def models_pull() -> None:
    """Pre-download and revision-pin all models, then report cache sizes."""
    from vemoizer.models import MODELS, cache_size, pull_models, render_pull_report

    results = pull_models(MODELS)
    sizes = cache_size(MODELS)
    typer.echo(render_pull_report(results, sizes))
    if any(r.error is not None for r in results):
        raise typer.Exit(code=1)


@app.command()
def transcribe(
    # B008: typer.Argument/Option in defaults are Typer's documented pattern
    files: list[Path] = typer.Argument(  # noqa: B008
        ...,
        help="One or more audio files (.m4a etc.) to transcribe.",
    ),
    format: str = typer.Option(  # noqa: B008
        "all",
        help=f"Output format: {' or '.join(OUTPUT_FORMATS)}, or a comma-separated "
        f"subset. Default: all {len(OUTPUT_FORMATS)} formats.",
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
    out: Path | None = typer.Option(  # noqa: B008
        None,
        "--out",
        help="Single output file path; the first requested format is written there.",
    ),
    copy: bool = typer.Option(  # noqa: B008
        False,
        "--copy",
        help="Copy the transcript text to the clipboard (macOS only).",
    ),
    config: Path | None = typer.Option(  # noqa: B008
        None,
        "--config",
        help=(
            "LLM config file (default: layered .vemoizer/config.toml "
            "search — see `vemoizer meeting --help`)."
        ),
    ),
    profile: str = typer.Option(  # noqa: B008
        "dictation",
        "--profile",
        help="Recording profile: dictation (solo memo, fast) or meeting "
        "(far-field multi-speaker; Whisper decode A).",
    ),
    glossary: Path | None = typer.Option(  # noqa: B008
        None,
        "--glossary",
        help="Text file of domain terms and names (one per line); fed to "
        "the recognizer and LLM stages so vocabulary is spelled right.",
    ),
    repair: bool = typer.Option(  # noqa: B008
        False,
        "--repair",
        help="LLM repair pass over the final paragraphs (fixes phonetic "
        "ASR garble; guarded against invention; needs an LLM config).",
    ),
    speakers: str | None = typer.Option(  # noqa: B008
        None,
        "--speakers",
        help="People in the recording: N pins diarization clustering, "
        "MIN-MAX bounds it when people join and leave (e.g. 3-5); only "
        "used with --diarize.",
    ),
    diarize: bool = typer.Option(  # noqa: B008
        False,
        "--diarize",
        help="Run speaker diarization and attach speaker labels "
        "(pyannote.audio; off by default).",
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
    preprocess: str | None = typer.Option(  # noqa: B008
        None,
        "--preprocess",
        help=(
            "Audio preprocessing (opt-in): loudnorm (two-pass loudnorm "
            "normalization for far-field recordings; default: off, the "
            "plain decode). Case-insensitive, like --language."
        ),
    ),
) -> None:
    """Transcribe one or more voice memos and write transcript files."""
    # Battery warning (fail-open: pmset errors are silent)
    _warn_on_battery()

    if verbose:
        logging.basicConfig(level=logging.INFO, stream=sys.stderr)
    _run_log_configure(
        verbose=verbose,
        quiet=quiet,
        config_path=str(config) if config is not None else None,
    )

    from vemoizer.batch import transcribe_batch
    from vemoizer.output.formatters import FORMAT_EXTENSIONS
    from vemoizer.progress_wiring import make_batch_display

    # M4b (issue #105): one display per CLI invocation, constructed before
    # any stderr redirection/logging setup, closed in a finally below.
    # --quiet suppresses the live progress line too (not just the summary).
    display = make_batch_display(quiet=quiet)

    # Resolve and validate formats BEFORE any transcription: an invalid
    # --format must fail in milliseconds, not after minutes of decoding
    # (the default "all" used to reach FORMAT_EXTENSIONS["all"] and crash
    # only once the whole file had been transcribed).
    formats = [f.strip() for f in format.split(",") if f.strip()]
    if formats == ["all"]:
        formats = list(OUTPUT_FORMATS)
    unknown = [f for f in formats if f not in FORMAT_EXTENSIONS]
    if unknown:
        known = ", ".join(OUTPUT_FORMATS)
        typer.echo(
            f"error: unknown format(s): {', '.join(unknown)} (known: {known})",
            err=True,
        )
        raise typer.Exit(code=2)
    speaker_count = _parse_speakers(speakers)
    if out is not None and len(formats) > 1:
        typer.echo(
            "warning: --out takes a single file; only the first format "
            f"({formats[0]}) is written to it",
            err=True,
        )

    # Two or more files: M3 split-recording grouping (issue #77) — natural
    # sort, 20s boundary decodes, confirmation (--yes / --no-group /
    # interactive), concat, one decode per group, part markers. A single
    # file stays on the plain per-file loop (no grouping work at all).
    # Issue #135: --preprocess loudnorm is validated here (case-insensitive,
    # like --language) and threaded through run_batch / transcribe_batch
    # (the RunOptions.expert_transcribe field covers both paths).
    lowered_preprocess = preprocess.strip().lower() if preprocess is not None else None
    if lowered_preprocess is not None and lowered_preprocess != "loudnorm":
        typer.echo(
            f"error: unknown preprocess {lowered_preprocess!r} (known: loudnorm)",
            err=True,
        )
        raise typer.Exit(code=2)
    try:
        if len(files) > 1:
            from vemoizer.batch import run_batch
            from vemoizer.presets import RunOptions

            if copy:
                # The plain loop honors --copy; the batch loop does not (one
                # clipboard per group is not a sensible multi-file contract),
                # so the narrowing is made explicit rather than silent.
                typer.echo(
                    "warning: --copy is only honored for a single file; "
                    "skipped for multi-file runs",
                    err=True,
                )
            batch_options = RunOptions.expert_transcribe(
                profile=profile,
                diarize=diarize,
                repair=repair,
                speakers=speaker_count,
                glossary_path=str(glossary) if glossary is not None else None,
                config_path=str(config) if config is not None else None,
            )
            from vemoizer.presets import replace as _replace

            batch_options = _replace(batch_options, preprocess=lowered_preprocess)
            exit_code = run_batch(
                files,
                batch_options,
                formats=formats,
                out=out,
                quiet=quiet,
                yes=yes,
                no_group=no_group,
                display=display,
            )
        else:
            exit_code = transcribe_batch(
                files,
                formats=formats,
                config_path=str(config) if config is not None else None,
                profile=profile,
                repair=repair,
                glossary_path=str(glossary) if glossary is not None else None,
                speakers=speaker_count,
                diarize=diarize,
                out=out,
                quiet=quiet,
                copy=copy,
                display=display,
                preprocess=lowered_preprocess,
            )
    finally:
        if display is not None:
            display.close()
    if exit_code:
        raise typer.Exit(code=exit_code)


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
            "plain decode). Case-insensitive, like --language."
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
    lowered_preprocess = preprocess.strip().lower() if preprocess is not None else None
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
            "plain decode). Case-insensitive, like --language."
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
    lowered_preprocess = preprocess.strip().lower() if preprocess is not None else None
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


def main() -> None:
    """Console-script entry point (``vemoizer`` on PATH)."""
    app()


if __name__ == "__main__":
    main()
