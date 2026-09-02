"""Typer CLI entry point for vemoizer.

Multi-file batch interface: one or more voice-memo paths as positional
arguments, format selection (default: all of txt/json/srt/vtt), and
``--quiet`` / ``--verbose`` verbosity flags.

Progress bars render to stderr via rich; transcripts render to stdout
(rich auto-detects TTY and disables progress on non-TTY stderr).
"""

from __future__ import annotations

from pathlib import Path

import typer

app = typer.Typer(
    name="vemoizer",
    help="Local-first voice memo transcription (Finnish/English consensus).",
    no_args_is_help=True,
)


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
) -> None:
    """Transcribe one or more voice memos and write transcript files."""
    del files, format, quiet, verbose
    typer.echo("transcribe: not implemented yet", err=True)
    raise typer.Exit(code=1)


def main() -> None:
    """Console-script entry point (``vemoizer`` on PATH)."""
    app()


if __name__ == "__main__":
    main()
