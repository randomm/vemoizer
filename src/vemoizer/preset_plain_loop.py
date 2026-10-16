"""The plain per-file loop of ``run_preset`` (issue #151 extraction).

Extracted verbatim from :func:`vemoizer.batch_preset.run_preset`, which keeps
everything that is not per-file work: the layered glossary, the temp-glossary
seam, the M3 grouping short-circuit, the ``wrote`` lines, the final line, and
the naming hook. This module owns the flow used for a single file of either
preset, for memo always, and for meeting with ``--no-group`` — one dated
``.md`` + ``.json`` pair per file.

Pure move: every statement below is unchanged from ``run_preset`` (issue #87
and the later issues named in the comments). Seam convention (same as
:mod:`vemoizer.batch_plain`, which serves ``run_batch``): the four names the
test suite patches — ``_transcribe_preset_file``, ``_write_preset_output``,
``_mtime_date_str``, ``file_log`` — are imported from ``vemoizer.batch_preset``
inside the function body, so ``monkeypatch.setattr(batch_preset, ...)`` and
``mock.patch.object(batch_preset, ...)`` keep hitting what this loop actually
calls, and those four (only) stay re-exported there. Every other helper the
tests do not patch is imported at module level here; the two ``vemoizer.notify``
posters stay deferred inside the function because the suite stubs them by
name on ``vemoizer.notify`` (see the import site).

Nothing here wraps or swallows exceptions: the temp-glossary ``OSError`` /
``ValueError`` and a ``KeyboardInterrupt`` keep propagating to the handlers
and the ``finally`` in ``run_preset``.
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import suppress
from pathlib import Path
from typing import TYPE_CHECKING

from vemoizer.batch_output import (
    PRESET_FORMATS,
    _check_result,
    _render_quality_report,
    check_failure_reason,
)
from vemoizer.ingest import IngestError
from vemoizer.naming_hook import ask_naming_hook
from vemoizer.output.naming import nfc_stem_and_suffix
from vemoizer.preset_final import (
    print_final_line,
    print_wrote_lines,
    run_went_full,
)
from vemoizer.preset_interrupt import note_written_files
from vemoizer.progress_wiring import _close_run_display
from vemoizer.sidecar import build_sidecar, resolve_run_glossary_files

if TYPE_CHECKING:
    from vemoizer.preset_interrupt import InterruptTracker
    from vemoizer.presets import RunOptions
    from vemoizer.progress import ProgressDisplay

__all__ = ["run_preset_plain_loop"]


def run_preset_plain_loop(
    files: list[Path],
    options: RunOptions,
    *,
    command: str,
    effective_glossary: str | None,
    first_stem: str,
    quiet: bool,
    yes: bool,
    input_fn: Callable[[str], str] | None,
    tty_isatty: Callable[[], bool] | None,
    display: ProgressDisplay | None = None,
    tracker: InterruptTracker | None = None,
) -> int:
    """Run the plain per-file preset flow over *files*; return the exit code.

    One dated pair per file; the date is each file's own modification date
    (issue #87) and the fallback stem is the FIRST file's stem (deterministic,
    unchanged since #82) — both handed in by the caller so this loop carries no
    option-composition context. Returns 0 on success and 1 on any per-file
    failure (transcribe returned nothing, a failed check, or a partial pair).
    """
    # Deferred (``batch_preset`` imports this module, so the direction stays
    # one-way) AND authoritative: these FOUR (and only these) are the names the
    # test suite patches on :mod:`vemoizer.batch_preset` —
    # ``_mtime_date_str`` (test_meeting_grouping,
    # test_meeting_write_failures), ``_write_preset_output`` (test_final_line,
    # test_notify_preset_cli, test_meeting_write_failures),
    # ``_transcribe_preset_file`` (test_final_line, test_preset_interrupt) and
    # ``file_log`` (test_run_log_cli's ``_patch_file_log``). The loop looks them
    # up THERE so those patches take effect; ``batch_preset`` is the patch-target
    # module the suite has always used (issue #78 convention: patch where the
    # name is looked up). Nothing else is looked up on ``batch_preset`` — the
    # module re-exports exactly these four and no others.
    from vemoizer.batch_preset import (
        _mtime_date_str,
        _transcribe_preset_file,
        _write_preset_output,
        file_log,
    )
    from vemoizer.caffeinate import caffeinate_context
    from vemoizer.ingest import pcm_duration_seconds

    # Both notification seams are imported here (not at module level) for the
    # same reason ``batch_plain`` defers its ``notify_result``: the suite
    # stubs the poster by replacing the name on ``vemoizer.notify`` —
    # ``mock.patch.object(notify_module, "notify_write", ...)`` — and a
    # module-level ``from ... import`` binds the function object once at
    # import time, so that patch would never reach this loop.
    from vemoizer.notify import notify_result, notify_write
    from vemoizer.progress_wiring import set_batch_prefix

    exit_code = 0
    written: list[str] = []
    # M5a: the glossary files the run used (for the sidecar hash).
    gfiles = resolve_run_glossary_files(command, options.glossary_path)

    with caffeinate_context():
        for index, file in enumerate(files, start=1):
            # M4b (issue #105): prefix the active stage with ``[i/N]
            # stem`` for multi-file runs.
            stem, _ = nfc_stem_and_suffix(file)
            set_batch_prefix(display, index, len(files), stem)
            # M4c (issue #111), seam (b): per-file log wrapping the
            # entire per-file iteration; a failing transcribe still
            # leaves a log file (decision 6).
            with file_log(stem):
                result = _transcribe_preset_file(
                    file,
                    options,
                    effective_glossary,
                    notify_failed=True,
                    display=display,
                    tracker=tracker,
                )
                if result is None:
                    exit_code = 1
                    continue
                # M5a: stash this file's PCM duration (fail-open) and
                # build the sidecar keys before the seam writes the
                # .md + .json pair.
                with suppress(OSError, IngestError):  # fail-open: skip on ffmpeg error
                    result["_source_durations"] = [
                        pcm_duration_seconds(file, preprocess=options.preprocess)
                    ]
                # Issue #135: stash the opt-in preprocess on the result
                # so build_sidecar records it present-only.
                if options.preprocess:
                    result["preprocess"] = options.preprocess
                build_sidecar(
                    result,
                    command=command,
                    glossary_files=gfiles,
                    source_paths=[file],
                )
                # M6 (issue #75): stash the report-only glossary
                # provenance before the seam writes. Source = the real
                # layer file path(s) the run read (never the composed
                # temp file); term count = the whisper-prompt terms
                # only (@-prefixed LLM-only terms never reached the
                # whisper prompt). Absent when no glossary was read.
                if gfiles:
                    result["glossary_source"] = (", ".join(gfiles)) + (
                        f" ({len(options.whisper_prompt)} terms)"
                    )
                    result["glossary_terms"] = list(options.whisper_prompt)
                # M6 (issue #75): the per-file quality report is computed
                # BEFORE _check_result pops result["warnings"] (a report
                # computed after the pop would see an empty warnings
                # list), then the check prints the warnings to stderr
                # and fails loud on error/no-transcript/no-labels.
                _render_quality_report(result, diarize_requested=bool(options.diarize))
                result.pop("glossary_terms", None)
                if _check_result(
                    file,
                    result,
                    diarize=options.diarize,
                    diarize_label="diarize",
                ):
                    # M4a (issue #100), seam (b): one failure notification
                    # per file that failed the checks; reason = the stderr
                    # ``error:`` line (same source as _check_result).
                    notify_result(
                        file,
                        "failed",
                        check_failure_reason(
                            file,
                            result,
                            diarize=options.diarize,
                            diarize_label="diarize",
                        )
                        or "",
                    )
                    exit_code = 1
                    continue
                pair = _write_preset_output(
                    result,
                    first_stem,
                    Path.cwd(),
                    date_str=_mtime_date_str(file),
                )
                note_written_files(tracker, pair)
                written.extend(pair)
                # M4a (issue #100), seam (b): one notification per file
                # (success = the full pair was written; failure = a
                # partial pair, with the stderr error line as reason).
                if len(pair) < len(PRESET_FORMATS):
                    exit_code = 1
                    notify_write(
                        file,
                        len(pair),
                        len(PRESET_FORMATS),
                        "could not write output",
                    )
                    continue
                notify_write(file, len(pair), len(PRESET_FORMATS))
    # Issue #143: close the display before the wrote lines and the
    # naming prompt (meeting only); close() is idempotent.
    if command == "meeting":
        _close_run_display(display)
    print_wrote_lines(written, quiet)
    # Final line (issue #148): only when the run went full for ALL files.
    # The plain path writes one pair per file, so expected pairs = files
    # (an independent count: the gate verifies every pair fully written).
    if run_went_full(written, len(files), exit_code):
        print_final_line(len(written), quiet=quiet)
    # End-of-meeting naming hook (issue #95): memo never prompts;
    # the hook never alters the run's exit code.
    if command == "meeting":
        ask_naming_hook(
            written, yes=yes, quiet=quiet, input_fn=input_fn, tty_isatty=tty_isatty
        )
    return exit_code
