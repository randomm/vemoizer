"""Expert ``transcribe`` per-file loop (extracted from :mod:`vemoizer.batch`).

The ``transcribe_batch`` function used to live in ``batch.py``; it is
extracted here so ``batch.py`` stays under the 500-line hard cap while the
M4a per-file notification hooks (issue #100) take a few lines at the
seams. The function is re-exported from ``vemoizer.batch`` so
``batch.transcribe_batch`` (and every test's import of it) keeps working.

The seams are resolved through ``vemoizer.batch``'s module namespace
(``_resolve_llm_config``, ``_process_result``) so the
``monkeypatch.setattr(batch, ...)`` call sites in the test suite still
patch the names this loop actually calls; ``transcribe_file`` is imported
from ``vemoizer.pipeline`` inside the function, matching the existing
patch convention (``monkeypatch.setattr(pipeline, "transcribe_file", ...)``).
"""

from __future__ import annotations

from pathlib import Path

import typer

from vemoizer.batch_output import check_failure_reason
from vemoizer.caffeinate import caffeinate_context
from vemoizer.diarization import SpeakerCount
from vemoizer.llm import ConfigError

__all__ = ["transcribe_batch"]


def transcribe_batch(
    files: list[Path],
    *,
    formats: list[str],
    config_path: str | None,
    profile: str,
    repair: bool,
    glossary_path: str | None,
    speakers: SpeakerCount | None,
    diarize: bool,
    out: Path | None = None,
    quiet: bool = False,
    copy: bool = False,
) -> int:
    """Transcribe *files* and write output files (the loop from old cli.py).

    Returns 0 on success, 1 if any file failed.
    """
    # Deferred imports so the tests' module-namespace patches keep working
    # (issue #78: ``batch._resolve_llm_config`` is the patched name).
    from vemoizer.batch import _process_result, _resolve_llm_config
    from vemoizer.pipeline import transcribe_file

    exit_code = 0
    with caffeinate_context():
        for file in files:
            try:
                # Fail loud on a malformed project config (issue #78).
                _resolve_llm_config(config_path)
            except ConfigError as e:
                # Consistent with run_preset: stop the batch, no siblings —
                # and no notification: the abort happens before any per-file
                # attempt (issue #100, M4a decision 6).
                typer.echo(f"error: {e}", err=True)
                return 1
            try:
                result = transcribe_file(
                    file,
                    diarize=diarize,
                    config_path=config_path,
                    profile=profile,
                    repair=repair,
                    glossary_path=glossary_path,
                    speakers=speakers,
                )
            except (KeyboardInterrupt, SystemExit):
                # ConfigError is handled by the try above; only the two
                # non-Exception control-flow signals need re-raising here.
                raise
            except Exception as e:
                # A per-file decode/write failure is a clean one-line error.
                typer.echo(f"error: {file.name}: {e}", err=True)
                exit_code = 1
                # M4a (issue #100), seam (a): one failure notification per
                # file whose transcribe raised. The one-line reason is the
                # stderr line above; never changes the exit code.
                from vemoizer.notify import notify_result

                notify_result(file, "failed", f"error: {file.name}: {e}")
                continue
            if not _process_result(
                file,
                result,
                formats=list(formats),
                out=out,
                quiet=quiet,
                # The expert transcribe loop: --copy is honored here (the
                # group path never copies); the diarize flag comes from the
                # function parameter, so both are passed explicitly.
                options=None,
                diarize=diarize,
                copy=copy,
            ):
                # M4a (issue #100), seam (a): one failure notification per
                # file that failed the per-file checks or the output write.
                # The reason is the ``error:`` line the check (or the
                # write) already printed.
                from vemoizer.notify import notify_result

                reason = check_failure_reason(file, result, diarize=diarize)
                notify_result(file, "failed", reason or "")
                exit_code = 1
                continue
            # M4a (issue #100), seam (a): one success notification per file
            # that was transcribed AND its output written. Independent of
            # --quiet (the quiet echo above is display only).
            from vemoizer.notify import notify_result

            notify_result(file, "done")
    return exit_code
