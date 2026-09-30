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
no new pipeline parameter.  The layered glossary (``glossary_layers``:
``load_layers`` → ``merge``) is composed into a temporary glossary file
and passed through the existing ``glossary_path`` argument: for meeting
the temp file carries the merged terms (bare + ``@`` lines) and merged
correction pairs; for memo it carries the merged correction pairs only,
so the whisper ``initial_prompt`` stays empty while
``apply_corrections`` still fires on the deterministic pairs (issue #82,
DESIGN DECISION "memo seam").  An explicit ``--glossary`` replaces the
layers entirely for meeting (the file is passed straight through); for
memo it is filtered to that file's correction pairs via a second temp
file (the whisper prompt stays empty, mirroring the layered memo seam).
Temp files are deleted after the run.

M3 split-recording grouping (issue #77): ``run_batch`` takes the
M2-defined :class:`~vemoizer.presets.RunOptions` and, for 2+ input
files, adds the grouping layer (``grouping`` module): natural sort,
20 s boundary decodes, the confirm step (``--yes`` / ``--no-group`` /
interactive), ffmpeg concat of multi-part groups, one decode per group
(invariant 6), and the ``part_markers`` sidecar record.

The output-writing helpers (``_write_output``, ``_write_preset_output``,
``_write_temp_glossary``, ``_check_result``) live in
:mod:`vemoizer.batch_output` and are re-exported here for backwards
compatibility with existing tests.
"""

from __future__ import annotations

import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import typer

from vemoizer.caffeinate import caffeinate_context
from vemoizer.diarization import SpeakerCount
from vemoizer.ingest import IngestError
from vemoizer.llm import ConfigError
from vemoizer.output.naming import (  # noqa: F401
    collision_free_paths,
    dated_basename,
    nfc_stem_and_suffix,
)
from vemoizer.presets import RunOptions


def _write_temp_glossary(lines: list[str]) -> str:
    """Indirection for :func:`batch_output._write_temp_glossary`.

    Defined here (not imported) so that ``monkeypatch.setattr(batch,
    "_write_temp_glossary", fake)`` in tests patches the name that
    ``run_preset`` actually calls.
    """
    from vemoizer.batch_output import _write_temp_glossary as _wgt

    return _wgt(lines)


# Re-export the output-writing helpers (now in batch_output.py) so
# existing imports from vemoizer.batch continue to work.
from vemoizer.batch_output import (  # noqa: F401,E402
    PRESET_FORMATS,
    _check_result,
    _write_output,
    _write_preset_output,
    run_preset,
)

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


def _resolve_llm_config(config_path: str | None):
    """Resolve the LLM config for a batch run (issue #82 review).

    Explicit ``--config`` short-circuits to ``load_config`` (fail-open).
    Without one, the strict layered search runs; its ``ConfigError``
    (malformed ``./.vemoizer/config.toml``) is caught by the caller and
    becomes a clean ``error: ...`` line — never a raw traceback (the
    fail-open ``load_default_config`` is NOT used here, because its
    ``except Exception`` swallows ``ConfigError`` silently, which would
    hide a broken project config).``None`` (no config found) is fine.
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
                # Fail loud on a malformed project config (issue #78):
                # clean error line, never a traceback.
                _resolve_llm_config(config_path)
            except ConfigError as e:
                # A malformed .vemoizer/config.toml must fail loud with a
                # clean error line, not a traceback (issue #78); consistent
                # with run_preset: stop the batch, no sibling files.
                typer.echo(f"error: {e}", err=True)
                return 1
            result = transcribe_file(
                file,
                diarize=diarize,
                config_path=config_path,
                profile=profile,
                repair=repair,
                glossary_path=glossary_path,
                speakers=speakers,
            )
            if _check_result(file, result, diarize=diarize, diarize_label="--diarize"):
                exit_code = 1
                continue
            stem, _suffix = nfc_stem_and_suffix(file)
            if out is not None:
                ok = _write_output(out, result, formats[0] if formats else "txt")
            else:
                from vemoizer.output.formatters import FORMAT_EXTENSIONS

                ok = all(
                    [
                        _write_output(
                            Path(f"{stem}{FORMAT_EXTENSIONS[fmt]}"), result, fmt
                        )
                        for fmt in formats
                    ]
                )
            if not ok:
                exit_code = 1
                continue
            if copy:
                from vemoizer.copy import copy_to_clipboard

                copy_to_clipboard(result["text"])
            if not quiet:
                typer.echo(f"wrote transcript for {file.name}")
    return exit_code


def _transcribe_one(file: Path, options: RunOptions) -> dict[str, Any]:
    """One ``transcribe_file`` call with the fail-loud config check.

    The return is ``TranscriptionResult``-shaped (``text`` required;
    ``segments`` / ``part_markers`` / ``notes`` optional) — or the
    ``{"text", "segments", "error"}`` triple when the project config
    check fails. The ``error`` key is part of the contract: ``_check_result`
    turns it into a clean ``error: ...`` line, so it must be visible in the
    type, not elided (round 3 finding 2).
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


def _write_group_outputs(
    group: list[Path],
    result: dict,
    *,
    formats: list[str],
    out: Path | None,
) -> bool:
    """Write one group's outputs; True on success.

    The ``--out`` override applies only for single-group runs (or stdout,
    where each group streams in order) — a multi-group run with an explicit
    file target is rejected up front in :func:`run_batch` before any
    decode, so it never reaches this loop.
    """
    if out is not None:
        return _write_output(out, result, formats[0] if formats else "txt")
    stem, _ = nfc_stem_and_suffix(group[0])
    from vemoizer.output.formatters import FORMAT_EXTENSIONS

    return all(
        [
            _write_output(Path(f"{stem}{FORMAT_EXTENSIONS[fmt]}"), result, fmt)
            for fmt in formats
        ]
    )


def _run_plain(
    ordered: list[Path],
    options: RunOptions,
    *,
    formats: Sequence[str],
    out: Path | None,
    quiet: bool,
) -> int:
    """The plain per-file loop (single-file runs and --no-group groups).

    Note: intentionally diverges from ``transcribe_batch`` — no ``--copy``
    support and per-file config-error continue (the batch loop honors
    ``--copy`` and aborts the run on a malformed project config). The
    divergence is deliberate: the plain loop feeds the grouping path
    where ``--copy`` is a single-file-only concern and a broken config
    should fail that file, not the whole multi-file run.
    """
    exit_code = 0
    with caffeinate_context():
        for file in ordered:
            result = _transcribe_one(file, options)
            if _check_result(file, result, diarize=options.diarize):
                exit_code = 1
                continue
            if not _write_group_outputs(
                [file],
                result,
                formats=list(formats),
                out=out,
            ):
                exit_code = 1
                continue
            if not quiet:
                typer.echo(f"wrote transcript for {file.name}")
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
) -> int:
    """Transcribe *files* with M3 split-recording grouping (issue #77).

    Takes the M2 :class:`~vemoizer.presets.RunOptions` (not an ad-hoc
    kwargs dict). Single file: no grouping work at all (no boundary
    slice, no Whisper boundary model load) — the plain per-file loop.
    Multi-file: natural sort, then:

    - ``--no-group``: each file transcribed standalone (no boundary
      decode, no concat, no part markers);
    - ``--yes``: the boundary decodes still run (they feed the proposal)
      and every proposal is accepted without a prompt;
    - interactive: one confirmation prompt per boundary (Enter accept,
      ``e`` edit — any valid partition, ``q`` quit).

    Multi-part groups are joined with the ffmpeg concat demuxer (``-c
    copy``) into a temp file and decoded ONCE (invariant 6); the part
    offsets (decoded-PCM, never ffprobe) become
    ``transcript["part_markers"]`` so the JSON sidecar and Markdown
    carry the ``— osa N (äänitys X) —`` markers. Single-part groups
    carry no ``part_markers`` key at all.

    ``--out`` with 2+ files is only honored when the run is a single
    group (one combined transcript) — otherwise every group would
    overwrite the same target, so the call fails up front (2) before
    any decode (``--out -`` for stdout is always fine).

    Returns 0 on success, 1 if any group failed, 2 on a bad
    combination of group flags.
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

    tail_texts, head_texts = _db(ordered, transcribe_fn)
    proposals = propose_groups(ordered, tail_texts, head_texts)
    try:
        groups = confirm_groups(
            ordered,
            proposals,
            yes=yes,
            no_group=False,
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
                result = _transcribe_one(group[0], options)
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
                    result = _transcribe_one(merged, options)
                finally:
                    if merged_is_temp:
                        remove_concat_output(merged)
                if "error" not in result:
                    # Multi-part groups only: single-part groups get no
                    # part_markers key at all (issue #77). PartMarker gives
                    # each {offset, label} entry an explicit type (finding 5).
                    markers: list[PartMarker] = [
                        {
                            "offset": off.start_offset,
                            "label": f"— osa {off.part_number} "
                            f"(äänitys {off.source_filename})",
                        }
                        for off in offsets
                    ]
                    result["part_markers"] = markers
                label = "+".join(p.name for p in group)

            if _check_result(label, result, diarize=options.diarize):
                exit_code = 1
                continue
            if not _write_group_outputs(
                group,
                result,
                formats=list(formats),
                out=out,
            ):
                exit_code = 1
                continue
            if not quiet:
                typer.echo(f"wrote transcript for {label}")
    return exit_code
