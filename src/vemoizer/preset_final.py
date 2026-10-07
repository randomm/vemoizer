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


def run_went_full(written: list[str], expected_pairs: int, exit_code: int) -> bool:
    """Whether the run succeeded for **all** inputs (the final-line gate).

    ``expected_pairs`` is the number of output pairs the run is expected to
    write — one per file in the plain per-file path, one per *group* in the
    M3 grouping path (a merged group writes a single pair, not one per file).
    The gate is the run having exited 0 *and* every expected pair being fully
    written (``len(written) == expected_pairs * len(PRESET_FORMATS)``). A
    partial pair or a failed check fails the gate, so no ``complete`` line is
    printed even though some outputs did write.
    """
    return bool(
        written
        and exit_code == 0
        and len(written) == expected_pairs * len(PRESET_FORMATS)
    )


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
