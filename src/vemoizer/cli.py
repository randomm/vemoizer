"""Typer CLI entry point for vemoizer.

Multi-file batch interface: one or more voice-memo paths as positional
arguments, format selection (default: all of txt/json/srt/vtt), and
``--quiet`` / ``--verbose`` verbosity flags.

macOS UX polish (issue #14):
- ``--copy`` — copy transcript text to the clipboard via pbcopy
- Battery warning — warn before long transcription on battery power
- Caffeinate — hold a wake assertion during transcription
- ``--low-memory`` / ``--no-low-memory`` — low-memory model-loading mode

Progress bars render to stderr via rich; transcripts render to stdout
(rich auto-detects TTY and disables progress on non-TTY stderr).
"""

from __future__ import annotations

from pathlib import Path

import typer

from vemoizer.battery import on_battery
from vemoizer.low_memory import apply_low_memory_mode, default_low_memory

app = typer.Typer(
    name="vemoizer",
    help="Local-first voice memo transcription (Finnish/English consensus).",
    no_args_is_help=True,
)


def _warn_on_battery() -> None:
    """Emit a battery warning to stderr if running on battery power."""
    if on_battery():
        typer.echo(
            "warning: running on battery power — transcription may take a while",
            err=True,
        )


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
        help="Suppress the summary output.",
    ),
    verbose: bool = typer.Option(  # noqa: B008
        False,
        "--verbose",
        help="Emit per-stage progress logging to stderr.",
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
) -> None:
    """Transcribe one or more voice memos and write transcript files."""
    # Resolve low-memory mode (auto-detect or explicit flag)
    lm = _resolve_low_memory(low_memory)
    apply_low_memory_mode(lm)

    # Battery warning (fail-open: pmset errors are silent)
    _warn_on_battery()

    # Placeholder: the full consensus pipeline is not wired yet.
    # When it lands, the transcribe work will run inside
    # caffeinate_context() and the transcript will be copied
    # to the clipboard when --copy is set.
    del files, format, quiet, verbose, copy
    typer.echo("transcribe: not implemented yet", err=True)
    raise typer.Exit(code=1)


def main() -> None:
    """Console-script entry point (``vemoizer`` on PATH)."""
    app()


if __name__ == "__main__":
    main()
