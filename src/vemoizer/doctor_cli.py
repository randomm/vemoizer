"""The ``doctor`` command's ``--preprocess`` handling, extracted from ``cli.py``.

Holds only the ``--preprocess`` validation for the ``doctor`` command
(validates the value against ``"loudnorm"`, then calls ``run_doctor``);
``cli.py`` imports it here so the command signature, the ``--help``
output, and the exit-code contract are unchanged. Split out purely to
restore headroom under the 500-line source cap (issue #135).
"""

from __future__ import annotations

import typer

from vemoizer.doctor import run_doctor


def run_doctor_command(preprocess: str | None) -> None:
    """Validate *preprocess* and run the doctor checks.

    ``None`` (no flag): no loudnorm check. A non-empty value must
    lowercase to ``"loudnorm"``; anything else is a clean exit 2 (one
    line, no traceback) — the same contract as ``--language``.
    """
    lowered = preprocess.strip().lower() if preprocess is not None else None
    if lowered is not None and lowered != "loudnorm":
        typer.echo(f"error: unknown preprocess {lowered!r} (known: loudnorm)", err=True)
        raise typer.Exit(code=2)
    report = run_doctor(echo=lambda line: typer.echo(line), preprocess=lowered or None)
    if not report.ok:
        raise typer.Exit(code=1)
