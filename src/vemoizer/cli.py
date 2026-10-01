"""Typer CLI entry point for vemoizer.

Multi-file batch interface: one or more voice-memo paths as positional
arguments, format selection (default: all of txt/json/srt/vtt), and
``--quiet`` / ``--verbose`` verbosity flags.

macOS UX polish (issue #14):
- ``--copy`` — copy transcript text to the clipboard via pbcopy
- Battery warning — warn before long transcription on battery power
- Caffeinate — hold a wake assertion during transcription
- ``--low-memory`` / ``--no-low-memory`` — low-memory model-loading mode

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

from vemoizer.battery import on_battery
from vemoizer.diarization import SpeakerCount
from vemoizer.eval_cli import register_eval
from vemoizer.low_memory import apply_low_memory_mode, default_low_memory

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


def _warn_on_battery() -> None:
    """Emit a battery warning to stderr if running on battery power."""
    if on_battery():
        typer.echo(
            "warning: running on battery power — transcription may take a while",
            err=True,
        )


@models_app.command("pull")
def models_pull() -> None:
    """Pre-download and revision-pin all models, then report cache sizes."""
    from vemoizer.models import MODELS, cache_size, pull_models, render_pull_report

    results = pull_models(MODELS)
    sizes = cache_size(MODELS)
    typer.echo(render_pull_report(results, sizes))
    if any(r.error is not None for r in results):
        raise typer.Exit(code=1)


def _resolve_low_memory(
    low_memory: bool | None,
) -> bool:
    """Resolve the low-memory flag to a final boolean.

    If the user explicitly set --low-memory or --no-low-memory, use that.
    Otherwise, auto-detect based on total system RAM.
    """
    if low_memory is not None:
        return low_memory
    return default_low_memory()


def _parse_speakers(value: str | None) -> SpeakerCount | None:
    """``--speakers`` as an exact count ("4") or bounds ("3-5").

    Validated before any transcription, like --format: a typo must fail in
    milliseconds, not after minutes of decoding.
    """
    if value is None:
        return None
    lo, sep, hi = value.partition("-")
    try:
        bounds = (int(lo), int(hi)) if sep else (int(lo), int(lo))
    except ValueError:
        bounds = (0, 0)
    if bounds[0] < 1 or bounds[1] < bounds[0]:
        typer.echo(f"error: --speakers expects N or MIN-MAX (got {value!r})", err=True)
        raise typer.Exit(code=2)
    return bounds if sep else bounds[0]


@app.command()
def transcribe(
    # B008: typer.Argument/Option in defaults are Typer's documented pattern
    files: list[Path] = typer.Argument(  # noqa: B008
        ...,
        help="One or more audio files (.m4a etc.) to transcribe.",
    ),
    format: str = typer.Option(  # noqa: B008
        "all",
        help="Output format: txt, json, srt, vtt, or a comma-separated subset. "
        "Default: all four formats.",
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
    low_memory: bool | None = typer.Option(  # noqa: B008
        None,
        "--low-memory",
        "--no-low-memory",
        help=(
            "Enable low-memory model-loading mode (auto-detected when "
            "not set; on by default for <=16 GiB RAM)."
        ),
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
) -> None:
    """Transcribe one or more voice memos and write transcript files."""
    # Resolve low-memory mode (auto-detect or explicit flag)
    lm = _resolve_low_memory(low_memory)
    apply_low_memory_mode(lm)

    # Battery warning (fail-open: pmset errors are silent)
    _warn_on_battery()

    if verbose:
        logging.basicConfig(level=logging.INFO, stream=sys.stderr)

    from vemoizer.batch import transcribe_batch
    from vemoizer.output.formatters import FORMAT_EXTENSIONS, OUTPUT_FORMATS

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
        exit_code = run_batch(
            files,
            batch_options,
            formats=formats,
            out=out,
            quiet=quiet,
            yes=yes,
            no_group=no_group,
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
        )
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
    low_memory: bool | None = typer.Option(  # noqa: B008
        None,
        "--low-memory",
        "--no-low-memory",
        help=(
            "Enable low-memory model-loading mode (auto-detected when "
            "not set; on by default for <=16 GiB RAM)."
        ),
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
) -> None:
    """Transcribe a meeting: whisper decode, diarization, repair, .md+.json."""
    lm = _resolve_low_memory(low_memory)
    apply_low_memory_mode(lm)
    _warn_on_battery()

    if verbose:
        logging.basicConfig(level=logging.INFO, stream=sys.stderr)

    from vemoizer.batch import run_preset

    speaker_count = _parse_speakers(speakers)
    exit_code = run_preset(
        files,
        command="meeting",
        config_path=str(config) if config is not None else None,
        glossary_path=str(glossary) if glossary is not None else None,
        repair=repair,
        diarize=False if no_diarize else None,
        speakers=speaker_count,
        quiet=quiet,
        yes=yes,
        no_group=no_group,
    )
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
    low_memory: bool | None = typer.Option(  # noqa: B008
        None,
        "--low-memory",
        "--no-low-memory",
        help=(
            "Enable low-memory model-loading mode (auto-detected when "
            "not set; on by default for <=16 GiB RAM)."
        ),
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
) -> None:
    """Transcribe a memo: whisper decode, no diarization, repair, .md+.json."""
    lm = _resolve_low_memory(low_memory)
    apply_low_memory_mode(lm)
    _warn_on_battery()

    if verbose:
        logging.basicConfig(level=logging.INFO, stream=sys.stderr)

    from vemoizer.batch import run_preset

    exit_code = run_preset(
        files,
        command="memo",
        config_path=str(config) if config is not None else None,
        glossary_path=str(glossary) if glossary is not None else None,
        repair=repair,
        quiet=quiet,
    )
    if exit_code:
        raise typer.Exit(code=exit_code)


def main() -> None:
    """Console-script entry point (``vemoizer`` on PATH)."""
    app()


if __name__ == "__main__":
    main()
