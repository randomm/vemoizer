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
from vemoizer.diarization import SpeakerCount
from vemoizer.grouping_common import with_part_markers
from vemoizer.ingest import IngestError
from vemoizer.llm import ConfigError
from vemoizer.output.naming import (  # noqa: F401
    collision_free_paths,
    dated_basename,
    nfc_stem_and_suffix,
)
from vemoizer.presets import RunOptions


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
    _check_result,
    _process_result,
    _write_output,
    _write_preset_output,
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
    from vemoizer.pipeline import transcribe_file

    exit_code = 0
    with caffeinate_context():
        for file in files:
            try:
                # Fail loud on a malformed project config (issue #78).
                _resolve_llm_config(config_path)
            except ConfigError as e:
                # Consistent with run_preset: stop the batch, no siblings.
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
                exit_code = 1
                continue
    return exit_code


def _transcribe_guarded(
    target: Path,
    options: RunOptions,
    description: str,
) -> dict[str, Any] | None:
    """One guarded transcribe (issue #77 merge gate).

    The single per-file/per-group guard shared by the plain loop and both
    branches of the group loop: an unexpected exception degrades to a clean
    one-line ``error:`` naming *description* (never a raw traceback);
    ``KeyboardInterrupt``/``SystemExit`` propagate; returns ``None`` when
    the target failed.
    """
    try:
        return _transcribe_one(target, options)
    except (KeyboardInterrupt, SystemExit):
        # Control-flow signals only — everything else degrades per target
        # (same contract as transcribe_batch / the group path).
        raise
    except Exception as e:  # noqa: BLE001 - per-file fail-loud boundary
        typer.echo(f"error: {description}: {e}", err=True)
        return None


def _transcribe_one(file: Path, options: RunOptions) -> dict[str, Any]:
    """One ``transcribe_file`` call with the fail-loud config check.

    Returns ``TranscriptionResult``-shaped (``text`` required) — or the
    ``{"text", "segments", "error"}`` triple when the config check fails.
    The ``error`` key is contract: ``_check_result`` turns it into a clean
    error line, so it must stay visible.
    """
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
    )


def _run_plain(
    ordered: list[Path],
    options: RunOptions,
    *,
    formats: Sequence[str],
    out: Path | None,
    quiet: bool,
    write_group_fn: Callable[[Path | str, dict[str, Any]], None] | None = None,
) -> int:
    """The plain per-file loop (single file / --no-group).

    No --copy; per-file config-error continue. ``write_group_fn``
    (issue #87): when set, each result goes through the preset seam
    (one dated .md/.json pair per file) instead of _process_result.
    """
    exit_code = 0
    with caffeinate_context():
        for file in ordered:
            if (result := _transcribe_guarded(file, options, file.name)) is None:
                exit_code = 1
                continue
            if write_group_fn is not None:
                if _check_result(
                    file,
                    result,
                    diarize=options.diarize,
                    diarize_label="diarize",
                ):
                    exit_code = 1
                    continue
                if not _call_write_seam(write_group_fn, file, result):
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
                exit_code = 1
                continue
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
        for group in groups:
            if len(group) == 1:
                # A per-file decode/write failure is a clean one-line error.
                result = _transcribe_guarded(group[0], options, group[0].name)
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
                # Preset write seam (issue #87): shared _check_result so
                # the fail-loud contract matches _process_result; an
                # unexpected seam exception degrades per-group via
                # _call_write_seam (clean one-line error, keep going).
                if _check_result(
                    label,
                    result,
                    diarize=options.diarize,
                    diarize_label="diarize",
                ):
                    exit_code = 1
                    continue
                if not _call_write_seam(write_group_fn, label, result):
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
                exit_code = 1
                continue
    return exit_code
