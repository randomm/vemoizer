"""Shared per-command setup helpers for the Typer CLI (issue #117).

These three helpers are called from ``transcribe``, ``meeting`` and ``memo``
in the same order for every command (battery warning, run-log context,
speaker parsing) and are extracted from ``cli.py`` purely to restore headroom
under the 500-line source cap. They have no CLI command of their own and are
not reachable through the public ``app`` interface.

``cli.py`` imports them into its module namespace so the per-command call
order and the ``--quiet`` display-suppression contract are unchanged.
"""

from __future__ import annotations

import typer

from vemoizer.battery import on_battery
from vemoizer.diarization import SpeakerCount

__all__ = ["parse_speakers", "run_log_configure", "warn_on_battery"]


def run_log_configure(*, verbose: bool, quiet: bool, config_path: str | None) -> None:
    """Set the per-invocation run-log context (issue #111, M4c).
    Resolves the LLM config fail-open to recover ``api_key_env`` (scrubbed by
    the ``run_log`` redaction formatter); a config error is the seam's job.
    """
    from vemoizer.run_log import configure

    api_key_env: str | None = None
    try:
        from vemoizer.batch import _resolve_llm_config

        cfg = _resolve_llm_config(config_path)
        if cfg is not None:
            api_key_env = cfg.api_key_env
    except Exception:  # noqa: BLE001 - fail-open: config errors are the seam's job
        pass
    configure(verbose=verbose, quiet=quiet, llm_api_key_env=api_key_env)


def warn_on_battery() -> None:
    """Emit a battery warning to stderr if running on battery power."""
    if on_battery():
        typer.echo(
            "warning: running on battery power — transcription may take a while",
            err=True,
        )


def parse_speakers(value: str | None) -> SpeakerCount | None:
    """``--speakers`` as an exact count ("4") or bounds ("3-5").

    Validated before any transcription, like --format: a typo must fail in
    milliseconds, not after minutes of decoding.
    """
    if value is None:
        return None
    lo, sep, hi = value.partition("-")
    try:
        bounds = (int(lo), int(hi)) if sep else (int(lo), int(lo))
    except ValueError:
        bounds = (0, 0)
    if bounds[0] < 1 or bounds[1] < bounds[0]:
        typer.echo(f"error: --speakers expects N or MIN-MAX (got {value!r})", err=True)
        raise typer.Exit(code=2)
    return bounds if sep else bounds[0]
