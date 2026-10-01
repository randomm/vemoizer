"""End-of-meeting "Name the speakers now?" hook (issue #95, M5c-2).

After an interactive ``vemoizer meeting`` run finishes (single file, the
``--no-group`` loop, or the grouped run), asks the user one final
question — ``Name the speakers now? [y/N]`` — and, on yes, runs the
existing ``run_names`` flow on each written sidecar that has 2+ labelled
speakers.

Every guard fires BEFORE any ``input_fn`` call, so memo, ``--yes``,
piped/CI (non-TTY stdin or stdout) and runs with no eligible sidecar are
complete no-ops — no prompt, no output, no change to the run's exit
code (the hook never alters it; the call sites take ``max(code, exit)``).

Eligibility is computed from the sidecars on disk (suffix-based scan of
the written names: ``*.json`` resolved against ``Path.cwd()``), never
from the in-memory transcribe result. Missing, unreadable, non-dict, or
non-list-``paragraphs`` sidecars are skipped silently; a sidecar with a
pre-existing ``speaker_names`` is still eligible.

``_stdout_isatty`` is re-exported from :mod:`vemoizer.names_cli` (the
single source of truth, issue #95 d3) so the hook's stdout-TTY gate is
one monkeypatchable seam.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Callable
from pathlib import Path

import typer

from vemoizer.names_cli import _stdout_isatty
from vemoizer.speaker_clips import talk_share

__all__ = ["_eligible_sidecars", "_stdout_isatty", "ask_naming_hook"]


def _eligible_sidecars(written: list[str]) -> list[Path]:
    """The written ``*.json`` sidecars (resolved against ``Path.cwd()``)
    whose on-disk ``paragraphs`` list carries 2+ distinct speaker
    labels. Missing, unreadable, malformed, or single-label sidecars are
    skipped silently."""
    eligible: list[Path] = []
    for name in written:
        if not name.endswith(".json"):
            continue
        path = Path.cwd() / name
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not isinstance(data, dict):
            continue
        paragraphs = data.get("paragraphs")
        if not isinstance(paragraphs, list):
            continue
        if len(talk_share(paragraphs)) >= 2:
            eligible.append(path)
    return eligible


def ask_naming_hook(
    written_sidecars: list[str],
    *,
    yes: bool,
    input_fn: Callable[[str], str] | None = None,
    tty_isatty: Callable[[], bool] | None = None,
) -> int:
    """The end-of-meeting naming hook.

    Skips (returns immediately, no output) when: *yes* is set (``--yes``
    is a complete no-op for the hook), stdin or stdout is not a TTY, or
    no written sidecar is eligible. Otherwise prompts once; a stripped
    answer starting with ``y``/``Y`` runs :func:`vemoizer.names_cli.run_names`
    on each eligible sidecar in order.

    Per-sidecar failures never touch the exit code: any ``Exception``
    or ``SystemExit`` (e.g. a ``sys.exit()`` inside the names flow) prints
    one ``warning: naming failed for <basename>: <ExcClassName>`` line and
    continues; an ``EOFError``/``KeyboardInterrupt`` prints
    ``naming cancelled`` and stops the loop. The return code is always
    ``0`` — the caller ``max``es it with the run's own code.
    """
    if yes:
        return 0
    isatty = tty_isatty if tty_isatty is not None else sys.stdin.isatty
    if not isatty():
        return 0
    if not _stdout_isatty():
        return 0

    from vemoizer.names_cli import run_names

    eligible = _eligible_sidecars(written_sidecars)
    if not eligible:
        return 0

    ask = input_fn if input_fn is not None else input
    try:
        answer = ask("Name the speakers now? [y/N] ")
    except (EOFError, KeyboardInterrupt):
        # Declined/aborted at the prompt itself: nothing is printed.
        return 0
    if not answer.strip().lower().startswith("y"):
        return 0

    for path in eligible:
        try:
            run_names(path, input_fn=input_fn, tty_isatty=tty_isatty)
        except (EOFError, KeyboardInterrupt):
            typer.echo("naming cancelled", err=True)
            break
        except (Exception, SystemExit) as e:  # noqa: BLE001 - per-sidecar fail-open boundary: a sys.exit() inside run_names must not change the meeting run's exit status
            typer.echo(
                f"warning: naming failed for {path.name}: {type(e).__name__}",
                err=True,
            )
    return 0
