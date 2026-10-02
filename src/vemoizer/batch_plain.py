"""The plain per-file loop for ``run_batch`` (extracted from :mod:`vemoizer.batch`).

``_run_plain`` used to live in ``batch.py``; it is extracted here so the
M4c (issue #111) per-file log span fits under the 500-line hard cap on
``batch.py``. The function is re-exported from ``vemoizer.batch`` so
``batch._run_plain`` (and every test import of it) keeps working.

Seam convention (same as ``batch_guard`` / ``transcribe_loop``):
``_transcribe_guarded`` and ``_process_result`` are imported from
``vemoizer.batch`` INSIDE the loop, so ``monkeypatch.setattr(batch, ...)``
call sites in the test suite keep patching the names this loop actually
calls.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

from vemoizer.batch_output import (
    _check_and_write,
    check_failure_reason,
)
from vemoizer.caffeinate import caffeinate_context
from vemoizer.output.naming import nfc_stem_and_suffix
from vemoizer.run_log import file_log

if TYPE_CHECKING:
    from vemoizer.presets import RunOptions
    from vemoizer.progress import ProgressDisplay

__all__ = ["_run_plain"]


def _run_plain(
    ordered: list[Path],
    options: RunOptions,
    *,
    formats: Sequence[str],
    out: Path | None,
    quiet: bool,
    write_group_fn: Callable[[Path | str, dict[str, Any]], None] | None = None,
    display: ProgressDisplay | None = None,
) -> int:
    """The plain per-file loop (single file / --no-group).

    No --copy; per-file config-error continue. ``write_group_fn``
    (issue #87): when set, each result goes through the preset seam
    (one dated .md/.json pair per file) instead of _process_result.
    ``display`` (issue #105 M4b): the CLI-level display, threaded into
    ``_transcribe_guarded`` and prefixed with ``[i/N] stem`` when N > 1.
    """
    # Deferred imports so the tests' module-namespace patches keep working
    # (issue #78 convention: ``batch._process_result`` / ``batch._transcribe_guarded``
    # / ``batch.set_batch_prefix`` are the patched names).
    from vemoizer.batch import (
        _process_result,
        _transcribe_guarded,
        set_batch_prefix,
    )

    exit_code = 0
    with caffeinate_context():
        for index, file in enumerate(ordered, start=1):
            # M4b (issue #105): prefix the active stage with ``[i/N] stem``
            # for multi-file runs; the prefix is part of the description,
            # not a separate echo line.  No-op when display is None or N=1.
            stem, _ = nfc_stem_and_suffix(file)
            set_batch_prefix(display, index, len(ordered), stem)
            # M4c (issue #111), seam (d): per-file log at
            # .vemoizer/logs/<stem>.log — the FOURTH seam (this is the
            # plain loop behind run_batch's single-file / --no-group
            # short-circuit, which the other three seams do not cover).
            # The span wraps that file's transcribe + result handling +
            # write + notify, mirroring the other seams. A per-file
            # ConfigError fails INSIDE this span (fail-loud), so every
            # attempted file leaves a log.
            with file_log(stem):
                if (
                    result := _transcribe_guarded(
                        file, options, file.name, display=display
                    )
                ) is None:
                    exit_code = 1
                    continue
                if write_group_fn is not None:
                    # The preset write seam (issue #87): shared _check_and_write
                    # helper. M4a (issue #100), seam (c): failure notifications
                    # for this seam live HERE (check failed — the ``error:``
                    # line is on stderr); the SUCCESS notification lives at the
                    # seam's own write point (write_group), so a partial-pair
                    # write failure is a failure, never a success.
                    if not _check_and_write(
                        write_group_fn,
                        file,
                        result,
                        diarize=options.diarize,
                        diarize_label="diarize",
                    ):
                        from vemoizer.notify import notify_result

                        reason = check_failure_reason(
                            file,
                            result,
                            diarize=options.diarize,
                            diarize_label="diarize",
                        )
                        notify_result(file, "failed", reason or "")
                        exit_code = 1
                    continue
                if not _process_result(
                    file,
                    result,
                    formats=list(formats),
                    out=out,
                    quiet=quiet,
                    options=options,
                    diarize=options.diarize,
                ):
                    # M4a (issue #100), seam (a): one failure notification per
                    # file that failed the checks or the output write (expert
                    # plain loop / --no-group; the preset seam path above never
                    # reaches this branch).
                    from vemoizer.notify import notify_result

                    reason = check_failure_reason(
                        file, result, diarize=options.diarize, diarize_label="--diarize"
                    )
                    notify_result(file, "failed", reason or "")
                    exit_code = 1
                    continue
                # M4a (issue #100), seam (a): one success notification per file
                # that was transcribed AND its output written (expert plain /
                # --no-group; the preset seam's write point covers preset runs).
                from vemoizer.notify import notify_result

                notify_result(file, "done")
    return exit_code
