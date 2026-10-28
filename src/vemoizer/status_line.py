"""The one stderr line ``vemoizer listen`` shares while recording.

The ``● REC`` tick redraws in place with ``\\r``; draft lines arrive from the
live decode thread and warnings from the main loop. Every writer holds
:data:`STATUS_LOCK`, and :func:`say` clears the REC line first on a TTY, so
no message is glued onto a half-drawn status and the next tick redraws REC.
"""

from __future__ import annotations

import sys
import threading

import typer

STATUS_LOCK = threading.Lock()


def hms(elapsed: float) -> str:
    """``HH:MM:SS`` for a non-negative number of seconds."""
    whole = int(elapsed)
    return f"{whole // 3600:02d}:{whole % 3600 // 60:02d}:{whole % 60:02d}"


def say(line: str) -> None:
    """Print *line* to stderr on its own line, clearing the REC status first."""
    with STATUS_LOCK:
        if sys.stderr.isatty():
            sys.stderr.write("\r\033[K")
            sys.stderr.flush()
        typer.echo(line, err=True)


def rec_tick(elapsed: float, *, rewrite: bool) -> None:
    """Redraw ``● REC`` in place on a TTY; else one plain line a minute."""
    with STATUS_LOCK:
        if rewrite:
            sys.stderr.write(f"\r● REC {hms(elapsed)}")
            sys.stderr.flush()
        elif int(elapsed) % 60 == 0:
            typer.echo(f"● REC {hms(elapsed)}", err=True)
