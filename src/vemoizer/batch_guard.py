"""Guarded single-file transcribe helpers (issue #77 merge gate).

Extracted from :mod:`vemoizer.batch` (500-line hard cap).
``_transcribe_guarded`` is the single per-file/per-group guard shared by
the plain loop and both branches of the group loop; ``_transcribe_one``
is the one ``transcribe_file`` call with the fail-loud config check.
Re-exported from ``vemoizer.batch`` for test-seam compatibility.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import typer

from vemoizer.presets import RunOptions

if TYPE_CHECKING:
    from vemoizer.progress import ProgressDisplay


def _transcribe_guarded(
    target: Path,
    options: RunOptions,
    description: str,
    display: ProgressDisplay | None = None,
) -> dict[str, Any] | None:
    """One guarded transcribe (issue #77 merge gate).

    The single per-file/per-group guard shared by the plain loop and both
    branches of the group loop: an unexpected exception degrades to a clean
    one-line ``error:`` naming *description* (never a raw traceback);
    ``KeyboardInterrupt``/``SystemExit`` propagate; returns ``None`` when
    the target failed.  *display* (issue #105 M4b) is the CLI-level
    :class:`~vemoizer.progress.ProgressDisplay`, threaded into
    ``transcribe_file`` so the tqdm shim can drive the decode stage.
    """
    try:
        return _transcribe_one(target, options, display=display)
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as e:  # noqa: BLE001 - per-file fail-loud boundary
        typer.echo(f"error: {description}: {e}", err=True)
        # M4a (issue #100), seam (c): one failure notification per
        # file/group whose transcribe raised; reason = the stderr line above.
        from vemoizer.notify import notify_result

        notify_result(description, "failed", f"error: {description}: {e}")
        return None


def _transcribe_one(
    file: Path,
    options: RunOptions,
    display: ProgressDisplay | None = None,
) -> dict[str, Any]:
    """One ``transcribe_file`` call with the fail-loud config check.

    Returns ``TranscriptionResult``-shaped (``text`` required) — or the
    ``{"text", "segments", "error"}`` triple when the config check fails.
    The ``error`` key is contract: ``_check_result`` turns it into a clean
    error line, so it must stay visible.  *display* (issue #105 M4b) is
    threaded into ``transcribe_file``; ``None`` keeps the default.
    """
    from vemoizer.batch import _resolve_llm_config
    from vemoizer.llm import ConfigError
    from vemoizer.pipeline import transcribe_file

    try:
        # Fail loud on a malformed project config (issue #78): clean error
        # line, never a traceback (issue #78).
        _resolve_llm_config(options.config_path)
    except ConfigError as e:
        return {"text": "", "segments": [], "error": str(e)}
    return transcribe_file(
        file,
        diarize=options.diarize,
        config_path=options.config_path,
        profile=options.profile,
        repair=options.repair,
        glossary_path=options.glossary_path,
        speakers=options.speakers,
        display=display,
        language=None if options.language == "auto" else options.language,
        preprocess=options.preprocess,
    )
