"""Batch display wiring helpers (issue #105 M4b).

The CLI constructs ONE :class:`~vemoizer.progress.ProgressDisplay` per
invocation (before any stderr redirection) and threads it down through the
batch layer to ``transcribe_file``.  For multi-file runs the batch layer
prefixes the active stage with ``[i/N] <stem> ·`` (shown only when N > 1)
so the user can see which file is being processed.  The display is closed
in a ``finally`` block by the CLI caller.
"""

from __future__ import annotations

from vemoizer.progress import ProgressDisplay


def make_batch_display(quiet: bool = False) -> ProgressDisplay | None:
    """Construct the run's :class:`ProgressDisplay` (issue #105 M4b).

    Returns ``None`` when *quiet* is true (``--quiet`` suppresses the live
    progress line as well as the summary lines it already suppresses).
    A non-TTY stderr is handled by :class:`ProgressDisplay` itself (its
    ``disable`` flag makes every method a no-op and the tqdm shim a
    pass-through), so the display object is always constructed when not
    quiet and the TTY state only decides whether anything renders.
    ``None`` is the default at every downstream call site, so all existing
    tests and fakes keep working unchanged.
    """
    if quiet:
        return None
    return ProgressDisplay()


def set_batch_prefix(
    display: ProgressDisplay | None, index: int, total: int, stem: str
) -> None:
    """Prefix the active stage with ``[i/N] <stem> ·`` (issue #105 M4b).

    The prefix is **part of** the progress line's description (not a
    separate echo line) and is shown only when *total* > 1; a single-file
    run keeps the plain stage text.  *index* is 1-based.  *stem* is the
    current file's stem (for a grouped meeting: the group's first part's
    NFC stem) — the privacy contract is that progress text contains only
    stage names, minute counts and the file stem (never transcript text).
    A no-op when *display* is ``None`` or disabled.
    """
    if display is None or total <= 1:
        return
    prefix = f"[{index}/{total}] {stem} · "
    display.prefix_active_stage(prefix)
