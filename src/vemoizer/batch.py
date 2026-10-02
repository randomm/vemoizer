"""Batch transcription orchestration (issue #82, M2).

The per-file loop that used to live inside ``cli.transcribe`` (lines
222-259 of the old cli.py) moved here so the ``meeting`` and ``memo``
preset commands can reuse the same transcribe → format → write pipeline
without duplicating the loop.

``run_preset`` composes the meeting and memo presets:

- **meeting**: profile=meeting, diarize, repair, default speakers 2-6,
  output is ``.md`` + ``.json`` to the CWD with a dated, sanitised
  title (``YYYY-MM-DD <title>.md``) and an NFC collision suffix.
- **memo**: profile=meeting (whisper decode), no diarization, repair
  on.  Output is ``.md`` + ``.json`` to the CWD with the same dated
  title naming.

Both presets call ``transcribe_file`` with the existing signature —
no new pipeline parameter.  The layered glossary is composed into a
temp file passed through ``glossary_path``; temp files deleted after.
M3 grouping (issue #77): ``run_batch`` takes ``RunOptions`` and, for 2+
files, adds natural sort, boundary decodes, confirm, concat, and a
clean one-line error per failed group (keep going, exit 1).

The output-writing helpers (``_write_output``, ``_write_preset_output``,
``_write_temp_glossary``, ``_check_result``) live in
:mod:`vemoizer.batch_output` and are re-exported here.
"""

from __future__ import annotations

import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vemoizer.llm import LLMConfig

import numpy as np
import typer

from vemoizer.caffeinate import caffeinate_context
from vemoizer.grouping_common import with_part_markers
from vemoizer.ingest import IngestError
from vemoizer.output.naming import (  # noqa: F401
    collision_free_paths,
    dated_basename,
    nfc_stem_and_suffix,
)
from vemoizer.presets import RunOptions
from vemoizer.progress_wiring import set_batch_prefix


def _write_temp_glossary(lines: list[str]) -> str:
    """Indirection for batch_output._write_temp_glossary so
    monkeypatch.setattr(batch, "_write_temp_glossary", fake) patches
    the name run_preset actually calls.
    """
    from vemoizer.batch_output import _write_temp_glossary as _wgt

    return _wgt(lines)


# Re-export the output-writing helpers (now in batch_output.py) so
# existing imports from vemoizer.batch continue to work.
from vemoizer.batch_output import (  # noqa: F401,E402
    PRESET_FORMATS,
    _call_write_seam,
    _check_and_write,
    _check_result,
    _process_result,
    _write_output,
    _write_preset_output,
    check_failure_reason,
)

# Re-export run_preset from its new home (batch_preset.py, issue #87) so
# existing imports from vemoizer.batch continue to work.
from vemoizer.batch_preset import run_preset  # noqa: F401,E402

# Re-export the grouping helpers (now in grouping_*.py submodules) so
# existing imports and monkeypatch.setattr("vemoizer.grouping.X", ...)
# continue to work.
from vemoizer.grouping import (  # noqa: F401,E402
    GroupingError,
    GroupProposal,
    PartMarker,
    PartOffset,
    natural_sort,
    stems_of,
)
from vemoizer.grouping_concat import (  # noqa: F401,E402
    concat_groups,
    part_offsets,
    remove_concat_output,
)
from vemoizer.grouping_decode import (  # noqa: F401,E402
    _decode_edge_window,
    decode_boundaries,
)
from vemoizer.grouping_probe import (  # noqa: F401,E402
    _probe_stream,
    probe_duration_seconds,
)


def _resolve_llm_config(config_path: str | None) -> LLMConfig | None:
    """Resolve the LLM config for a batch run (issue #82 review).

    Explicit ``--config`` short-circuits to ``load_config`` (fail-open).
    Without one, the strict layered search runs; its ``ConfigError`` is
    caught by the caller and becomes a clean error line — never a raw
    traceback (the fail-open ``load_default_config`` is NOT used here,
    because its ``except Exception`` swallows ``ConfigError`` silently).
    ``None`` (no config found) is fine.
    """
    from vemoizer.llm import _default_search, load_config

    if config_path is not None:
        return load_config(config_path)
    return _default_search()


# The expert transcribe loop now lives in transcribe_loop.py (the 500-line
# hard cap on this file, issue #100 M4a); re-exported here so
# ``batch.transcribe_batch`` (and every test's import of it) keeps working.
# The loop resolves its seams through this module's namespace
# (``_resolve_llm_config`` / ``_process_result``), so the
# ``batch._resolve_llm_config`` test patches still patch the name it calls.
# Re-export the guarded transcribe helpers (now in batch_guard.py) so
# existing imports from vemoizer.batch continue to work.
from vemoizer.batch_guard import (  # noqa: F401,E402
    _transcribe_guarded,
    _transcribe_one,
)
from vemoizer.transcribe_loop import transcribe_batch  # noqa: F401,E402


def _run_plain(
    ordered: list[Path],
    options: RunOptions,
    *,
    formats: Sequence[str],
    out: Path | None,
    quiet: bool,
    write_group_fn: Callable[[Path | str, dict[str, Any]], None] | None = None,
    display: Any | None = None,
) -> int:
    """The plain per-file loop (single file / --no-group).

    No --copy; per-file config-error continue. ``write_group_fn``
    (issue #87): when set, each result goes through the preset seam
    (one dated .md/.json pair per file) instead of _process_result.
    ``display`` (issue #105 M4b): the CLI-level display, threaded into
    ``_transcribe_guarded`` and prefixed with ``[i/N] stem`` when N > 1.
    """
    exit_code = 0
    with caffeinate_context():
        for index, file in enumerate(ordered, start=1):
            # M4b (issue #105): prefix the active stage with ``[i/N] stem``
            # for multi-file runs; the prefix is part of the description,
            # not a separate echo line.  No-op when display is None or N=1.
            set_batch_prefix(display, index, len(ordered), file.stem)
            if (
                result := _transcribe_guarded(file, options, file.name, display=display)
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
                        file, result, diarize=options.diarize, diarize_label="diarize"
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


def run_batch(
    files: list[Path],
    options: RunOptions,
    *,
    formats: Sequence[str] = ("txt", "json", "srt", "vtt", "md"),
    out: Path | None = None,
    quiet: bool = False,
    yes: bool = False,
    no_group: bool = False,
    transcribe_fn: Callable[[np.ndarray], dict[str, Any]] | None = None,
    input_fn: Callable[[str], str] | None = None,
    print_fn: Callable[[str], None] | None = None,
    tty_isatty: Callable[[], bool] | None = None,
    write_group_fn: Callable[[Path | str, dict[str, Any]], None] | None = None,
    display: Any | None = None,
) -> int:
    """Transcribe *files* with M3 split-recording grouping (issue #77).

    Takes the M2 :class:`~vemoizer.presets.RunOptions` (not an ad-hoc
    kwargs dict). Single file: no grouping work at all — the plain
    per-file loop. Multi-file: natural sort, then ``--no-group``
    (each file standalone), ``--yes`` (boundary decodes run, every
    proposal accepted without a prompt), or interactive (Enter accept,
    ``e`` edit, ``q`` quit).

    Multi-part groups are joined with the ffmpeg concat demuxer (``-c
    copy``) and decoded ONCE (invariant 6); the part offsets (decoded
    PCM, never ffprobe) become ``transcript["part_markers"]``. Single-part
    groups carry no ``part_markers`` key.

    ``write_group_fn`` (issue #87) overrides the per-group write seam:
    when set, called via ``_call_write_seam`` after the result passes
    ``_check_result`` (instead of ``_process_result``); an unexpected
    seam exception becomes a clean one-line error (exit 1, keep going)
    while ``KeyboardInterrupt``/``SystemExit`` propagate. The meeting
    preset uses it for one dated ``.md``/``.json`` pair per group in the
    CWD; ``None`` keeps the expert behaviour. The seam runs inside the
    ``caffeinate_context``. The ``_run_plain`` short-circuit (single
    file / ``--no-group``) honours it too: each file's result goes
    through the seam instead of ``_process_result``.

    ``display`` (issue #105 M4b) is the CLI-level
    :class:`~vemoizer.progress.ProgressDisplay`, threaded through the
    plain loop and the per-group transcribe calls; ``None`` keeps the
    default.

    ``--out`` with 2+ files is only honored for a single group (else
    every group would overwrite the same target — fail up front, 2)
    except ``--out -`` (stdout).

    Returns 0 on success, 1 if any group failed, 2 on a bad combination
    of group flags.
    """
    from vemoizer.grouping import confirm_groups, propose_groups

    ordered = natural_sort(files)

    # Mutually exclusive group-decision flags.
    if yes and no_group:
        typer.echo("error: --yes and --no-group are mutually exclusive", err=True)
        return 2

    # Single file: no grouping work at all — no boundary slice, no
    # boundary model load, no prompt — the plain per-file loop. Same for
    # --no-group: each file is transcribed standalone, no boundary decode,
    # no concat, no part markers.
    if len(ordered) < 2 or no_group:
        return _run_plain(
            ordered,
            options,
            formats=formats,
            out=out,
            quiet=quiet,
            write_group_fn=write_group_fn,
            display=display,
        )

    # The TTY guard is BEFORE any boundary decode or model load, so a
    # piped/CI invocation fails in milliseconds instead of hanging on
    # input() — and --yes never pays the prompt cost it skips.
    isatty = tty_isatty if tty_isatty is not None else sys.stdin.isatty
    if not yes and input_fn is None and not isatty():
        typer.echo(
            "error: group confirmation requires a TTY; use --yes or --no-group",
            err=True,
        )
        return 2
    # --yes still runs the boundary decodes (they feed the proposal);
    # --no-group is the flag that skips them (it already returned above).
    # Deferred import so that monkeypatch.setattr(grouping, "decode_boundaries", ...)
    # in tests patches the name that run_batch actually calls.
    from .grouping import decode_boundaries as _db

    # The boundary decode owns the default WhisperTranscriber's lazy
    # model load (a download): _edge_text degrades per-EDGE failures
    # (IngestError/RuntimeError/OSError) to "", but a model LOAD failure
    # can be any exception type (HuggingFace/network) — none of those
    # may escape as a raw traceback. KeyboardInterrupt still propagates.
    try:
        tail_texts, head_texts = _db(ordered, transcribe_fn)
    except KeyboardInterrupt:
        raise
    except Exception as e:  # noqa: BLE001 - intentional: any load failure -> clean line
        typer.echo(f"error: could not decode boundaries: {e}", err=True)
        return 1
    proposals = propose_groups(ordered, tail_texts, head_texts)
    try:
        groups = confirm_groups(
            ordered,
            proposals,
            yes=yes,
            input_fn=input_fn or input,
            print_fn=(
                print_fn
                if print_fn is not None
                else (lambda s: typer.echo(s, err=True))
            ),
        )
    except GroupingError as e:
        typer.echo(f"error: {e}", err=True)
        return 1

    exit_code = 0
    if out is not None and str(out) != "-" and len(groups) > 1:
        # Multi-group, explicit --out target: every group would overwrite
        # the same file (only the last group would survive). Fail up front
        # rather than silently losing a transcript — one --out per run.
        typer.echo(
            f"error: --out {out} with {len(groups)} groups would overwrite "
            "itself; drop --out (one file per group) or use --out - (stdout) "
            "or --no-group",
            err=True,
        )
        return 2
    with caffeinate_context():
        for index, group in enumerate(groups, start=1):
            # M4b (issue #105): prefix the active stage with ``[i/N]`` where
            # N is the number of transcribe invocations (groups), not source
            # files; the stem is the group's first part (deterministic).
            set_batch_prefix(display, index, len(groups), group[0].stem)
            if len(group) == 1:
                # A per-file decode/write failure is a clean one-line error.
                result = _transcribe_guarded(
                    group[0], options, group[0].name, display=display
                )
                if result is None:
                    exit_code = 1
                    continue
                label = group[0].name
            else:
                try:
                    merged = concat_groups(group)
                except GroupingError as e:
                    # Fail this group, continue with the rest (the
                    # _check_result pattern): the remaining groups are
                    # transcribed, and the failure is visible.
                    typer.echo(f"error: {e}", err=True)
                    exit_code = 1
                    continue
                merged_is_temp = merged != group[0]
                try:
                    # A per-part ingest failure (missing/corrupt file) is
                    # a group failure, not a crash: clean error line and
                    # continue with the remaining groups.
                    offsets = part_offsets(group)
                except IngestError as e:
                    if merged_is_temp:
                        remove_concat_output(merged)
                    typer.echo(f"error: {e}", err=True)
                    exit_code = 1
                    continue
                try:
                    # An unexpected per-group decode failure is a clean
                    # one-line error, not a traceback mid-batch: name the
                    # group's first part's file, mark the group failed,
                    # keep going.
                    result = _transcribe_guarded(
                        merged,
                        options,
                        f"group {group[0].name} (+{len(group) - 1} more part(s))",
                        display=display,
                    )
                finally:
                    if merged_is_temp:
                        remove_concat_output(merged)
                if result is None:
                    # The transcribe for this group failed: the finally
                    # already cleaned the temp concat file, so skip the
                    # write and move on to the next group.
                    exit_code = 1
                    continue
                if "error" not in result:
                    # Multi-part groups only: single-part groups get no
                    # part_markers key at all (issue #77). with_part_markers
                    # returns a NEW dict, so the pipeline result is never
                    # mutated in place.
                    result = with_part_markers(result, offsets)
                label = "+".join(p.name for p in group)

            if write_group_fn is not None:
                # Preset write seam (issue #87): shared _check_and_write
                # helper — fail-loud _check_result first, then the seam (an
                # unexpected seam exception degrades per-group via
                # _call_write_seam). M4a (issue #100), seam (c): failure
                # notifications for the grouped preset seam live HERE
                # (check failed — the ``error:`` line is on stderr); the
                # SUCCESS notification lives at the seam's own write point
                # (write_group).
                if not _check_and_write(
                    write_group_fn,
                    label,
                    result,
                    diarize=options.diarize,
                    diarize_label="diarize",
                ):
                    from vemoizer.notify import notify_result

                    reason = check_failure_reason(
                        label, result, diarize=options.diarize, diarize_label="diarize"
                    )
                    notify_result(label, "failed", reason or "")
                    exit_code = 1
                continue
            if not _process_result(
                label,
                result,
                formats=list(formats),
                out=out,
                quiet=quiet,
                options=options,
                diarize=options.diarize,
            ):
                # M4a (issue #100), seam (a): one failure notification per
                # group that failed the checks or the output write (expert
                # group path, no preset seam; a multi-part group's label
                # names its first part).
                from vemoizer.notify import notify_result

                reason = check_failure_reason(
                    label, result, diarize=options.diarize, diarize_label="--diarize"
                )
                notify_result(label, "failed", reason or "")
                exit_code = 1
                continue
            # M4a (issue #100), seam (a): one success notification per group
            # that was transcribed AND its output written (expert group
            # loop, no preset seam).
            from vemoizer.notify import notify_result

            notify_result(label, "done")
    return exit_code
