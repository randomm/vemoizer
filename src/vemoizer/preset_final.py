"""End-of-run reporting for the ``meeting`` / ``memo`` presets (issue #148).

One coherent responsibility: the terminal output a preset run emits after
the last file is written — the per-file ``wrote <path>`` lines and the
single final ``✓ complete`` line. The final line is the only place the
word ``complete`` appears (per-stage markers use the stage name,
``decode ✓`` / ``diarize ✓`` …), so it is gated on the run having succeeded
for **all** files: a partial pair or a failed check means some file's
output is missing, so nothing may read ``complete`` — the successful
files' ``wrote`` lines still print, and ``--quiet`` suppresses both.
"""

from __future__ import annotations

import typer

from vemoizer.batch_output import PRESET_FORMATS

__all__ = ["print_final_line", "print_wrote_lines", "run_went_full"]


def run_went_full(
    written: list[str],
    expected_pairs: int | None = None,
    exit_code: int = 0,
) -> bool:
    """Whether the run succeeded for **all** inputs (the final-line gate).

    The gate is exactly: ``exit_code == 0`` and at least one file written,
    and — when an *independent* pair count is given (``expected_pairs`` is
    not ``None``) — exactly that many pairs fully written
    (``len(written) == expected_pairs * len(PRESET_FORMATS)``; one pair per
    file in the plain per-file path).

    ``expected_pairs=None`` means no independent expectation is available
    (the M3 grouping path: a merged group writes a single pair, so the pair
    count cannot be known without deriving it from ``written`` itself, which
    would make the length check tautological). There the gate rests on
    ``exit_code`` and a non-empty ``written``: a group's partial pair or a
    failed check is suppressed through ``exit_code == 1``, which the write
    seam sets on any failed pair.
    """
    if not written or exit_code != 0:
        return False
    if expected_pairs is None:
        return True
    return len(written) == expected_pairs * len(PRESET_FORMATS)


def print_wrote_lines(written: list[str], quiet: bool) -> None:
    """Print one ``wrote <path>`` line per written file (suppressed when
    *quiet*); these precede the single final line."""
    for name in written:
        if not quiet:
            typer.echo(f"wrote {name}")


def print_final_line(n_files: int, *, quiet: bool) -> None:
    """Print the single final ``✓ complete`` line of the run.

    Plain ``typer.echo`` — no rich markup: ``[green]`` in the string would
    print verbatim because ``typer.echo`` does not render rich markup
    (issue #148 FIX 2), so the operator would see the brackets. Plain
    text is what the operator sees and what the tests assert. ``--quiet``
    suppresses the line.
    """
    if quiet:
        return
    typer.echo(f"\u2713 complete — wrote {n_files} file(s)")
