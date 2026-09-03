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

Progress bars render to stderr via rich; transcripts render to stdout
(rich auto-detects TTY and disables progress on non-TTY stderr).
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import typer

from vemoizer.battery import on_battery
from vemoizer.caffeinate import caffeinate_context
from vemoizer.copy import copy_to_clipboard
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
        help="LLM config file (default: ~/.config/vemoizer/config.toml).",
    ),
    profile: str = typer.Option(  # noqa: B008
        "dictation",
        "--profile",
        help="Recording profile: dictation (solo memo, fast) or meeting "
        "(far-field multi-speaker; Whisper decode A).",
    ),
    repair: bool = typer.Option(  # noqa: B008
        False,
        "--repair",
        help="LLM repair pass over the final paragraphs (fixes phonetic "
        "ASR garble; guarded against invention; needs an LLM config).",
    ),
    diarize: bool = typer.Option(  # noqa: B008
        False,
        "--diarize",
        help="Run speaker diarization and attach speaker labels "
        "(pyannote.audio; off by default).",
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

    from vemoizer.output.formatters import FORMAT_EXTENSIONS, OUTPUT_FORMATS
    from vemoizer.output.naming import nfc_stem_and_suffix
    from vemoizer.pipeline import transcribe_file

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
    if out is not None and len(formats) > 1:
        typer.echo(
            "warning: --out takes a single file; only the first format "
            f"({formats[0]}) is written to it",
            err=True,
        )

    exit_code = 0
    with caffeinate_context():
        for file in files:
            result = transcribe_file(
                file,
                diarize=diarize,
                config_path=str(config) if config is not None else None,
                profile=profile,
                repair=repair,
            )
            for warning in result.pop("warnings", []):
                typer.echo(warning, err=True)
            if "error" in result:
                typer.echo(f"error: {result['error']}", err=True)
                exit_code = 1
                continue
            stem, _suffix = nfc_stem_and_suffix(file)
            if out is not None:
                ok = _write_output(out, result, formats[0] if formats else "txt")
            else:
                ok = all(
                    # all() over a list, not a generator: every format must be
                    # attempted even after one fails.
                    [
                        _write_output(
                            Path(f"{stem}{FORMAT_EXTENSIONS[fmt]}"), result, fmt
                        )
                        for fmt in formats
                    ]
                )
            if not ok:
                exit_code = 1
                continue
            if copy:
                copy_to_clipboard(result["text"])
            if not quiet:
                typer.echo(f"wrote transcript for {file.name}")
    if exit_code:
        raise typer.Exit(code=exit_code)


def _write_output(target: Path, result: dict, fmt: str) -> bool:
    """Render *result* in *fmt* and write it to *target* (``-`` = stdout).

    Returns True on success; False after printing the error, so the caller
    can fail the run instead of reporting a transcript that was never
    written.
    """
    from vemoizer.output.formatters import format_transcript

    try:
        rendered = format_transcript(result, fmt)
    except (ValueError, KeyError) as e:
        typer.echo(f"error: {e}", err=True)
        return False
    if str(target) == "-":
        typer.echo(rendered)
        return True
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(rendered, encoding="utf-8")
    except OSError as e:
        typer.echo(f"error: could not write {target}: {e}", err=True)
        return False
    return True


def main() -> None:
    """Console-script entry point (``vemoizer`` on PATH)."""
    app()


if __name__ == "__main__":
    main()
