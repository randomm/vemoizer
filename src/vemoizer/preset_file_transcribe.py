"""The preset per-file transcribe core (extracted from :mod:`vemoizer.batch_preset`).

``_transcribe_preset_file`` is the fail-loud per-file transcribe body used
by :func:`vemoizer.batch_preset.run_preset`'s plain per-file loop (single /
memo / ``--no-group``). It was extracted into its own module so the 500-line
cap on ``batch_preset.py`` leaves room for the M4c (issue #111) per-file log
span, and so the M4c seam (b) ``file_log`` block can wrap the transcribe call
plus its per-file result handling without touching ``_transcribe_preset_file``.

``transcribe_file`` is imported from :mod:`vemoizer.pipeline` inside the
function (the same deferred-import patch seam the test suite relies on:
``monkeypatch.setattr(pipeline, "transcribe_file", ...)``); ``_resolve_llm_config``
is resolved through :mod:`vemoizer.batch`'s module namespace so the
``monkeypatch.setattr(batch, "_resolve_llm_config", ...)`` call sites keep
patching the name this function actually calls.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import typer

from vemoizer.presets import RunOptions
from vemoizer.progress import ProgressDisplay

__all__ = ["_transcribe_preset_file"]


def _transcribe_preset_file(
    file: Path,
    options: RunOptions,
    glossary_path: str | None,
    notify_failed: bool = False,
    display: ProgressDisplay | None = None,
) -> dict[str, Any] | None:
    """One guarded preset transcribe (the per-file loop's fail-loud core).

    An unexpected ``transcribe_file`` exception degrades to a clean
    one-line ``error:`` naming the file; ``KeyboardInterrupt``/``SystemExit``
    propagate; ``None`` on failure. A malformed project config
    (``ConfigError``) fails loud with a clean error line (issue #78).
    *display* (issue #105 M4b) is threaded into ``transcribe_file``.
    """
    from vemoizer.batch import _resolve_llm_config
    from vemoizer.llm import ConfigError
    from vemoizer.pipeline import transcribe_file

    try:
        # Fail loud on a malformed project config (issue #78).
        _resolve_llm_config(options.config_path)
    except ConfigError as e:
        typer.echo(f"error: {e}", err=True)
        return None
    try:
        return transcribe_file(
            file,
            diarize=options.diarize,
            config_path=options.config_path,
            profile=options.profile,
            repair=options.repair,
            glossary_path=glossary_path,
            speakers=options.speakers,
            display=display,
            language=None if options.language == "auto" else options.language,
            preprocess=options.preprocess,
        )
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as e:  # noqa: BLE001 - per-file fail-loud boundary
        typer.echo(f"error: {file.name}: {e}", err=True)
        if notify_failed:
            # M4a (issue #100), seam (b): one failure notification per
            # file whose transcribe raised; reason = the stderr line above.
            from vemoizer.notify import notify_result

            notify_result(file, "failed", f"error: {file.name}: {e}")
        return None
