"""End-of-meeting "Check the terms now?" hook (issue #268 rework).

Asks one final question after the naming hook — ``Check the terms in this
transcript? [y/N]`` — and, on yes, runs the interactive term check
(:func:`vemoizer.term_check_cli.run_terms`) on each written sidecar. A
sibling of :mod:`vemoizer.naming_hook` (kept separate so naming_hook stays
small), with the same skip rules and the same fail-open boundary: the hook
never alters the run's exit code and a per-sidecar failure only warns.

A skip prints NOTHING: the naming hook runs right before this one on the
same written list and already narrates those conditions
(:mod:`vemoizer.naming_hook`).
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path

import typer

from vemoizer.names_cli import _stdout_isatty
from vemoizer.naming_hook import _written_json_names

__all__ = ["ask_terms_hook"]


def ask_terms_hook(
    written_sidecars: list[str],
    *,
    yes: bool,
    input_fn: Callable[[str], str] | None = None,
    tty_isatty: Callable[[], bool] | None = None,
) -> int:
    """The end-of-meeting term-check hook; mirrors ``ask_naming_hook``.

    Skips (returns 0 immediately, SILENTLY — the naming hook already
    narrated those conditions) on *yes*, a non-TTY stdin/stdout, and a
    run that wrote no ``.json`` sidecar. Otherwise prompts once; a
    stripped answer starting with ``y``/``Y`` runs
    :func:`vemoizer.term_check_cli.run_terms` on each written sidecar in
    order. The return code is always ``0`` — the caller ``max``es it with
    the run's own code.
    """
    if yes:
        return 0
    isatty = tty_isatty if tty_isatty is not None else sys.stdin.isatty
    if not isatty() or not _stdout_isatty():
        return 0
    names = _written_json_names(written_sidecars)
    if not names:
        return 0

    ask = input_fn if input_fn is not None else input
    try:
        answer = ask("Check the terms in this transcript? [y/N] ")
    except (EOFError, KeyboardInterrupt):
        return 0
    if not answer.strip().lower().startswith("y"):
        return 0

    from vemoizer.term_check_cli import run_terms

    for name in names:
        try:
            run_terms(Path.cwd() / name, input_fn=input_fn, tty_isatty=tty_isatty)
        except (EOFError, KeyboardInterrupt):
            typer.echo("term check cancelled", err=True)
            break
        except (Exception, SystemExit) as e:  # noqa: BLE001 - per-sidecar fail-open boundary (same as the naming hook): a sys.exit() inside run_terms must not change the meeting run's exit status
            typer.echo(
                f"warning: term check failed for {name}: {type(e).__name__}",
                err=True,
            )
    return 0
