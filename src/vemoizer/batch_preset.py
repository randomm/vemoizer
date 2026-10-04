"""Preset (``meeting`` / ``memo``) run orchestration.

Extracted from :mod:`vemoizer.batch_output` (500-line cap, issue #87).
``run_preset`` composes the layered glossary, resolves the preset
:class:`~vemoizer.presets.RunOptions`, and runs the plain per-file loop
(single / memo / ``--no-group``) or the M3 grouping flow (meeting 2+).
The dated output name uses the first file's ``st_mtime`` (fallback: today).
"""

from __future__ import annotations

import os
from collections.abc import Callable
from contextlib import suppress
from datetime import date, datetime
from pathlib import Path
from typing import Any

import typer

from vemoizer.batch_output import (
    PRESET_FORMATS,
    _check_result,
    _render_quality_report,
    _write_preset_output,
    check_failure_reason,
)
from vemoizer.diarization import SpeakerCount
from vemoizer.ingest import IngestError
from vemoizer.naming_hook import ask_naming_hook
from vemoizer.output.naming import nfc_stem_and_suffix
from vemoizer.preset_file_transcribe import _transcribe_preset_file
from vemoizer.presets import RunOptions, resolve_options
from vemoizer.progress import ProgressDisplay
from vemoizer.run_log import file_log

__all__ = ["run_preset"]


def _mtime_date_str(path: Path) -> str:
    """The ``YYYY-MM-DD`` of *path*'s modification date (local time).

    ``st_mtime`` — deliberately NOT ``st_birthtime``/``creation_time``:
    on iOS exports the creation time is the copy/export time, not the
    recording date. Falls back to today when the file cannot be
    stat'ed (``OSError`` on special filesystems, deleted-then-replaced
    paths) — a clean fallback, never an exception.
    """
    try:
        mtime = os.stat(path).st_mtime
    except OSError:
        return date.today().isoformat()
    return datetime.fromtimestamp(mtime).date().isoformat()


def _run_preset_groups(
    files: list[Path],
    options: RunOptions,
    *,
    quiet: bool,
    yes: bool,
    no_group: bool,
    transcribe_fn: Callable | None,
    input_fn: Callable[[str], str] | None,
    print_fn: Callable[[str], None] | None,
    tty_isatty: Callable[[], bool] | None,
    effective_glossary: str | None,
    command: str,
    display: ProgressDisplay | None = None,
) -> int:
    """The meeting 2+ files path: the M3 flow via ``run_batch``.

    ``run_batch`` owns the full grouping logic; this function only hands
    over the meeting write seam — one dated ``.md`` + ``.json`` pair per
    group in the CWD — and lets ``run_batch``'s single-file and
    ``--no-group`` short-circuits keep the per-file behaviour.
    """
    from vemoizer.batch import run_batch
    from vemoizer.presets import replace

    # RunOptions is frozen; rebuild with the composed effective_glossary so
    # run_batch's per-group transcribe reads the right glossary file.
    group_options = replace(options, glossary_path=effective_glossary)

    written: list[str] = []
    exit_code = 0
    from vemoizer.sidecar import resolve_run_glossary_files  # noqa: E402

    gfiles = resolve_run_glossary_files(command, options.glossary_path)
    glossary_terms = list(options.whisper_prompt)
    if gfiles:
        glossary_source = ", ".join(gfiles) + f" ({len(glossary_terms)} terms)"
    else:
        glossary_source = None

    def write_group(label: Path | str, result: dict[str, Any]) -> None:
        # One dated pair per group: the date is the first part's mtime
        # (group[0] in natural-sort order); the fallback stem is the same
        # first part's stem.
        # M5a: stash the per-part PCM durations (fail-open) and the real
        # part paths, then build the sidecar keys.
        from vemoizer.sidecar import (
            build_sidecar,
            group_durations,
            group_part_paths,
        )

        parts = group_part_paths(label, files)
        with suppress(OSError, IngestError):  # fail-open: skip on ffmpeg error
            result["_source_durations"] = group_durations(
                parts, preprocess=options.preprocess
            )
        # Issue #135: stash the opt-in preprocess on the result so
        # build_sidecar records it present-only.
        if options.preprocess:
            result["preprocess"] = options.preprocess
        build_sidecar(
            result,
            command=command,
            glossary_files=gfiles,
            source_paths=parts,
        )
        # M6: stash the glossary provenance (report-only; the run dict is
        # the interface, never stored on the pipeline result).
        if glossary_source is not None:
            result["glossary_source"] = glossary_source
            result["glossary_terms"] = glossary_terms
        # The first part's path: the label's own path (single-part) or its
        # first name resolved against the original files (the group is
        # built over natural_sort(files), so the first part is a member).
        first = parts[0]
        stem, _ = nfc_stem_and_suffix(first)
        pair = _write_preset_output(
            result,
            stem,
            Path.cwd(),
            date_str=_mtime_date_str(first),
        )
        written.extend(pair)
        # A partial pair (fewer paths than the preset's formats — e.g.
        # the .json write failed) means this group's run failed; the error
        # line was already printed by _write_output.
        nonlocal exit_code
        if len(pair) < len(PRESET_FORMATS):
            exit_code = 1
        # M4a (issue #100), seam (c): one notification per group, at the
        # seam's own write point (check-/decode-failure notifications fire
        # earlier in run_batch, so a group is never double-notified).
        # Success = the full pair written; a partial pair is a FAILURE even
        # though one file landed; the stem is the group's first part.
        from vemoizer.notify import notify_write

        notify_write(
            first,
            len(pair),
            len(PRESET_FORMATS),
            reason="could not write output" if len(pair) < len(PRESET_FORMATS) else "",
        )

    code = run_batch(
        files,
        group_options,
        write_group_fn=write_group,
        quiet=quiet,
        yes=yes,
        no_group=no_group,
        transcribe_fn=transcribe_fn,
        input_fn=input_fn,
        print_fn=print_fn,
        tty_isatty=tty_isatty,
        display=display,
    )

    for name in written:
        if not quiet:
            typer.echo(f"wrote {name}")
    # End-of-meeting naming hook (issue #95): the prompt is the last
    # interactive output (after the wrote lines); the command guard
    # (meeting only — memo never prompts) is enforced at this call
    # site; the hook never alters the run's exit code.
    if command == "meeting":
        ask_naming_hook(written, yes=yes, input_fn=input_fn, tty_isatty=tty_isatty)
    return max(code, exit_code)


def run_preset(
    files: list[Path],
    *,
    command: str,
    config_path: str | None,
    glossary_path: str | None,
    repair: bool | None = None,
    diarize: bool | None = None,
    speakers: SpeakerCount | None = None,
    language: str | None = None,
    preprocess: str | None = None,
    quiet: bool = False,
    yes: bool = False,
    no_group: bool = False,
    transcribe_fn: Callable | None = None,
    input_fn: Callable[[str], str] | None = None,
    print_fn: Callable[[str], None] | None = None,
    tty_isatty: Callable[[], bool] | None = None,
    display: ProgressDisplay | None = None,
) -> int:
    """Run the *meeting* or *memo* preset over *files*.

    ``preprocess`` (issue #135): ``"loudnorm"`` threads the two-pass
    loudnorm normalization through the option (the CLI validates the
    value; ``None`` keeps the plain decode).

    Composes the layered glossary, resolves the preset options, then runs
    the plain per-file loop (single file / memo / ``--no-group``) or, for
    meeting with 2+ files, the M3 grouping flow via
    :func:`vemoizer.batch.run_batch` with the meeting write seam.
    ``display`` (issue #105 M4b) is threaded to ``transcribe_file``.
    Returns 0 on success, 1 on any failure, 2 on a bad flag combination.
    """
    # Deferred import so run_preset (defined here) and _write_temp_glossary
    # (defined in batch.py, which re-exports run_preset from here) do not
    # create an import cycle.
    from vemoizer.batch import _write_temp_glossary

    if command not in ("meeting", "memo"):
        typer.echo(f"error: unknown preset {command!r}", err=True)
        return 2

    # Layered glossary: load_layers (I/O) → merge (pure) — skipped when
    # --glossary explicitly replaces both layers entirely (no merging).
    merged_terms: list[str] = []
    merged_corrections: dict[str, str] = {}
    if glossary_path is None:
        from vemoizer.glossary_layers import load_layers, merge

        home_terms, home_corr, project_terms, project_corr = load_layers()
        merged_terms, merged_corrections, notices = merge(
            project_terms,
            project_corr,
            home_terms,
            home_corr,
        )
        for notice in notices:
            typer.echo(notice, err=True)

    # Pure core: resolve the preset options from the merged layers and
    # the CLI overrides (CLI > layers > preset defaults).
    options = resolve_options(
        command,
        layers=None,
        cli_overrides={
            "glossary": glossary_path,
            "config": config_path,
            "glossary_terms": merged_terms,
            "glossary_corrections": merged_corrections,
            "repair": repair,
            "diarize": diarize,
            "speakers": speakers,
            "language": language,
            "preprocess": preprocess,
        },
    )

    # Temp-file seam: without --glossary, write the composed glossary to
    # a temp file and pass it through the existing glossary_path argument
    # (no new pipeline parameter). Memo: correction pairs ONLY (the
    # whisper prompt stays empty) — meeting: merged terms plus the merged
    # correction pairs.
    temp_path: Path | None = None
    effective_glossary: str | None = None

    # The temp file is created INSIDE the protected region so a write
    # failure neither leaks the file nor raises a raw traceback
    # (issue #82 review): a clean error line and exit 1 instead.
    try:
        if options.glossary_path is None:
            if command == "memo":
                lines = [f"{w} => {r}" for w, r in options.corrections.items()]
            else:
                lines = [
                    *options.whisper_prompt,
                    *[f"@{t}" for t in options.llm_terms],
                    *[f"{w} => {r}" for w, r in options.corrections.items()],
                ]
            if lines:
                temp_path = Path(_write_temp_glossary(lines))
                effective_glossary = str(temp_path)
        else:
            effective_glossary = options.glossary_path
            # Memo explicit --glossary (issue #82): correction pairs only
            # (whisper prompt stays empty). A non-UTF-8 file raises
            # ValueError, caught below (one clean line; finally cleans up).
            if command == "memo":
                from vemoizer.glossary import load_corrections

                corr = load_corrections(effective_glossary)
                lines = [f"{w} => {r}" for w, r in corr.items()]
                if lines:
                    temp_path = Path(_write_temp_glossary(lines))
                    effective_glossary = str(temp_path)
                else:
                    effective_glossary = None

        # M3 grouping (issue #87): meeting with 2+ files runs the full
        # grouping flow through run_batch (the single source of the M3
        # contract — TTY guard, boundary decodes, concat, part markers).
        # The mutual-exclusion check also fires for a single file (before
        # the single-file short-circuit, matching run_batch's existing
        # order) — run_preset must check it here too.
        if command == "meeting" and yes and no_group:
            typer.echo("error: --yes and --no-group are mutually exclusive", err=True)
            return 2
        if command == "meeting" and len(files) > 1:
            return _run_preset_groups(
                files,
                options,
                quiet=quiet,
                yes=yes,
                no_group=no_group,
                transcribe_fn=transcribe_fn,
                input_fn=input_fn,
                print_fn=print_fn,
                tty_isatty=tty_isatty,
                effective_glossary=effective_glossary,
                command=command,
                display=display,
            )

        # Plain per-file loop: single file (either preset), memo (always),
        # or meeting with --no-group. One dated pair per file; the date is
        # the file's own modification date (issue #87); the fallback stem
        # is the FIRST file's stem (deterministic, unchanged since #82).
        from vemoizer.caffeinate import caffeinate_context
        from vemoizer.progress_wiring import set_batch_prefix

        first_stem, _ = nfc_stem_and_suffix(files[0])
        exit_code = 0
        written: list[str] = []
        # M5a: the glossary files the run used (for the sidecar hash).
        from vemoizer.sidecar import resolve_run_glossary_files

        gfiles = resolve_run_glossary_files(command, options.glossary_path)
        with caffeinate_context():
            for index, file in enumerate(files, start=1):
                # M4b (issue #105): prefix the active stage with ``[i/N]
                # stem`` for multi-file runs.
                stem, _ = nfc_stem_and_suffix(file)
                set_batch_prefix(display, index, len(files), stem)
                # M4c (issue #111), seam (b): per-file log wrapping the entire
                # per-file iteration (transcribe through notify_write); a
                # failing transcribe still leaves a log file (decision 6).
                with file_log(stem):
                    result = _transcribe_preset_file(
                        file,
                        options,
                        effective_glossary,
                        notify_failed=True,
                        display=display,
                    )
                    if result is None:
                        exit_code = 1
                        continue
                    # M5a: stash this file's PCM duration (fail-open) and
                    # build the sidecar keys before the seam writes the
                    # .md + .json pair.
                    from vemoizer.ingest import pcm_duration_seconds
                    from vemoizer.sidecar import build_sidecar

                    with suppress(
                        OSError, IngestError
                    ):  # fail-open: skip on ffmpeg error
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
                    # temp file — deleted in the finally) or the explicit
                    # --glossary; term count = the whisper-prompt terms only
                    # (@-prefixed LLM-only terms never reached the whisper
                    # prompt). Absent when the run read no glossary at all —
                    # then the md header and the report omit the line, never
                    # a blank one.
                    if gfiles:
                        result["glossary_source"] = (", ".join(gfiles)) + (
                            f" ({len(options.whisper_prompt)} terms)"
                        )
                        result["glossary_terms"] = list(options.whisper_prompt)
                    # M6 (issue #75): compute the per-file quality report
                    # BEFORE _check_result pops result["warnings"] (a report
                    # computed after the pop would see an empty warnings
                    # list), then let the check print the warnings to stderr
                    # and fail loud on error/no-transcript/no-labels.
                    _render_quality_report(
                        result, diarize_requested=bool(options.diarize)
                    )
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
                        from vemoizer.notify import notify_result

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
                    written.extend(pair)
                    if len(pair) < len(PRESET_FORMATS):
                        # A partial pair (e.g. the .json write failed) means
                        # this file's run failed; the error line was already
                        # printed by _write_output. A partial pair is a
                        # FAILURE notification (one file landed, the pair did
                        # not). M4a (issue #100), seam (b).
                        exit_code = 1
                        from vemoizer.notify import notify_write

                        notify_write(
                            file,
                            len(pair),
                            len(PRESET_FORMATS),
                            "could not write output",
                        )
                        continue
                    # M4a (issue #100), seam (b): one success notification
                    # per file that was transcribed AND the full pair was
                    # written (meeting single / --no-group / memo; independent
                    # of --quiet).
                    from vemoizer.notify import notify_write

                    notify_write(file, len(pair), len(PRESET_FORMATS))
        for name in written:
            if not quiet:
                typer.echo(f"wrote {name}")
        # End-of-meeting naming hook (issue #95): memo never prompts
        # (the command guard lives in the hook's call site), and the
        # hook never alters the run's exit code.
        if command == "meeting":
            ask_naming_hook(written, yes=yes, input_fn=input_fn, tty_isatty=tty_isatty)
        return exit_code
    except OSError as e:
        # Temp-glossary write failure: clean error, non-zero exit, no
        # leaked file (the finally still cleans up what exists).
        typer.echo(f"error: could not write glossary: {e}", err=True)
        return 1
    except ValueError as e:
        # Non-UTF-8 layered glossary (loader re-raises UnicodeDecodeError
        # as ValueError, issue #79): one clean line, no traceback, no
        # leaked file (the finally still cleans up).
        typer.echo(f"error: glossary {glossary_path}: {e}", err=True)
        return 1
    finally:
        # The temp glossary file is deleted after the run (issue #82).
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
