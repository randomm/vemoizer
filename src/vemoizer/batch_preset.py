"""Preset (``meeting`` / ``memo``) run orchestration (issue #87).

``run_preset`` composes the layered glossary, resolves the preset
:class:`~vemoizer.presets.RunOptions`, and runs the plain per-file loop
or the M3 grouping flow (meeting 2+). The dated output name uses the
first file's ``st_mtime`` (fallback: today).
"""

from __future__ import annotations

import os
import sys
from collections.abc import Callable
from contextlib import suppress
from datetime import date, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import typer

from vemoizer.batch_output import PRESET_FORMATS, _write_preset_output
from vemoizer.diarization import SpeakerCount
from vemoizer.ingest import IngestError
from vemoizer.naming_hook import ask_naming_hook
from vemoizer.output.naming import nfc_stem_and_suffix
from vemoizer.owner_config import resolve_run_owner
from vemoizer.preset_file_transcribe import (
    _transcribe_preset_file as _transcribe_preset_file,
)
from vemoizer.preset_interrupt import note_written_files, set_interrupt_stage
from vemoizer.presets import RunOptions, resolve_options
from vemoizer.progress import ProgressDisplay
from vemoizer.progress_wiring import _close_run_display
from vemoizer.run_log import file_log as file_log
from vemoizer.sidecar import resolve_run_glossary_files
from vemoizer.tracks import TrackPaths, tracks_error

# The seam invariant this module keeps for the moved plain loop
# (:mod:`vemoizer.preset_plain_loop`, issue #151): a name is re-exported here,
# with an ``as``-binding (ruff F401), only because that loop looks it up on
# THIS module — which is exactly where the long-standing test seams patch. A
# name no code looks up here is never reached by a patch and only manufactures
# vacuously-green tests, so it is not re-exported. The invariant is asserted,
# not just claimed: ``tests/test_preset_plain_loop_seams.py`` parses both
# modules and fails on a dead or missing seam re-export.
from .preset_plain_loop import run_preset_plain_loop

if TYPE_CHECKING:
    from vemoizer.preset_interrupt import InterruptTracker

__all__ = ["run_preset"]


def _mtime_date_str(path: Path) -> str:
    """The ``YYYY-MM-DD`` of *path*'s modification date (local time).

    ``st_mtime`` (not ``st_birthtime``): on iOS exports the creation time
    is the copy/export time, not the recording date. Falls back to today
    when the file cannot be stat'ed.
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
    tracker: InterruptTracker | None = None,
) -> int:
    """The meeting 2+ files path: the M3 flow via ``run_batch``.

    ``run_batch`` owns the grouping logic; this function only hands over
    the meeting write seam. A ``KeyboardInterrupt`` raised inside
    propagates untouched (it is neither ``OSError`` nor ``ValueError``)
    to the command's handler.
    """
    from vemoizer.batch import run_batch

    # Deferred for the same reason as the module-level seam rule above: this is
    # the only path in THIS module that calls the ``preset_final`` trio, and the
    # plain loop imports the same three from its own module. A re-export here
    # would be a name nothing looks up — never reached by a patch.
    from vemoizer.preset_final import (
        print_final_line,
        print_wrote_lines,
        run_went_full,
    )
    from vemoizer.presets import replace

    # RunOptions is frozen; rebuild with the composed effective_glossary so
    # run_batch's per-group transcribe reads the right glossary file.
    group_options = replace(options, glossary_path=effective_glossary)

    written: list[str] = []
    exit_code = 0
    gfiles = resolve_run_glossary_files(command, options.glossary_path)
    glossary_terms = list(options.whisper_prompt)
    if gfiles:
        glossary_source = ", ".join(gfiles) + f" ({len(glossary_terms)} terms)"
    else:
        glossary_source = None

    def write_group(label: Path | str, result: dict[str, Any]) -> None:
        # One dated pair per group; stash per-part PCM durations and
        # build the sidecar keys (M5a).
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
        note_written_files(tracker, pair)
        written.extend(pair)
        # A partial pair (fewer paths than the preset's formats — e.g.
        # the .json write failed) means this group's run failed; the error
        # line was already printed by _write_output.
        nonlocal exit_code
        if len(pair) < len(PRESET_FORMATS):
            exit_code = 1
        # M4a (issue #100), seam (c): one notification per group; a
        # partial pair is a FAILURE even though one file landed.
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

    # Issue #143: close the display before the wrote lines and the naming
    # prompt so neither is overdrawn by the live rich Progress.
    _close_run_display(display)
    print_wrote_lines(written, quiet)
    # Final line (issue #148): only when the run went full. No independent
    # pair count is passed on the group path (counting pairs from
    # ``written`` itself would make the length check tautological); a
    # group's partial pair or failed check sets exit_code = 1 in the write
    # seam above — the gate rests on that plus a non-empty ``written``.
    if not quiet and run_went_full(written, exit_code=exit_code):
        print_final_line(len(written), quiet=quiet)
    # End-of-meeting naming hook (issue #95): the prompt is the last
    # interactive output; the hook never alters the run's exit code.
    if command == "meeting":
        ask_naming_hook(
            written, yes=yes, quiet=quiet, input_fn=input_fn, tty_isatty=tty_isatty
        )
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
    tracker: InterruptTracker | None = None,
    tracks: TrackPaths | None = None,
    me: str | None = None,
) -> int:
    """Run the *meeting* or *memo* preset over *files*.

    ``preprocess`` (issue #135): ``"loudnorm"`` threads the two-pass
    loudnorm normalization through the option (the CLI validates the
    value; ``None`` keeps the plain decode).

    ``tracks`` / ``me`` (issue #171): the two-track call's stems and the
    ``--me`` owner name. A tracks request that is not one meeting recording
    exits 2 before any decode; the owner name is resolved (and, on the first
    interactive run, asked for and saved) before the glossary is composed.

    Composes the layered glossary, resolves the preset options, then runs
    the plain per-file loop or, for meeting with 2+ files, the M3
    grouping flow via :func:`vemoizer.batch.run_batch`. ``display`` is
    threaded to ``transcribe_file``; on the success path ``run_preset``
    closes it (meeting runs, before the wrote lines / naming prompt) so
    the CLI's finally-close is a no-op (issue #143).
    Returns 0 on success, 1 on any failure, 2 on a bad flag combination.
    """
    # Deferred import so run_preset (defined here) and _write_temp_glossary
    # (defined in batch.py, which re-exports run_preset from here) do not
    # create an import cycle.
    from vemoizer.batch import _write_temp_glossary

    if command not in ("meeting", "memo"):
        typer.echo(f"error: unknown preset {command!r}", err=True)
        return 2
    owner: str | None = None
    if tracks is not None:
        error = tracks_error(command, files, speakers)
        if error is not None:
            typer.echo(f"error: {error}", err=True)
            return 2
        interactive = not yes and (tty_isatty or sys.stdin.isatty)()
        owner = resolve_run_owner(
            me,
            prompt=interactive,
            input_fn=input_fn or input,
            echo=lambda line: typer.echo(line, err=True),
        )

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
            "tracks": tracks,
            "owner": owner,
        },
    )

    # Temp-file seam: without --glossary, write the composed glossary to
    # a temp file (memo: correction pairs ONLY — the whisper prompt stays
    # empty; meeting: merged terms + ``@`` lines + correction pairs).
    # The temp file is created INSIDE the protected region so a write
    # failure neither leaks the file nor raises a raw traceback
    # (issue #82 review): a clean error line and exit 1 instead.
    temp_path: Path | None = None
    effective_glossary: str | None = None

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
        # grouping flow through run_batch. The mutual-exclusion check
        # fires before the single-file short-circuit (matching run_batch's
        # order) — run_preset must check it here too.
        if command == "meeting" and yes and no_group:
            typer.echo("error: --yes and --no-group are mutually exclusive", err=True)
            return 2
        if command == "meeting" and len(files) > 1:
            set_interrupt_stage(tracker, "decoding")
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
                tracker=tracker,
            )

        # Plain per-file loop: single file (either preset), memo (always),
        # or meeting with --no-group. One dated pair per file; the date is
        # the file's own modification date (issue #87); the fallback stem
        # is the FIRST file's stem (deterministic, unchanged since #82).
        # The loop body itself lives in
        # :func:`vemoizer.preset_plain_loop.run_preset_plain_loop` (issue #151)
        # — a pure move. The four names the loop looks up on THIS module
        # (``_transcribe_preset_file``, ``_write_preset_output``,
        # ``_mtime_date_str``, ``file_log``) stay re-exported here, so the
        # long-standing tests' patch targets are unchanged; the check/report
        # helpers and the echoes are not — the loop imports those itself.
        set_interrupt_stage(tracker, "decoding")
        first_stem, _ = nfc_stem_and_suffix(files[0])
        return run_preset_plain_loop(
            files,
            options,
            command=command,
            effective_glossary=effective_glossary,
            first_stem=first_stem,
            quiet=quiet,
            yes=yes,
            input_fn=input_fn,
            tty_isatty=tty_isatty,
            display=display,
            tracker=tracker,
        )
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
