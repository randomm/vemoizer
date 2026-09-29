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
"""

from __future__ import annotations

import os
import sys
import tempfile
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import typer

from vemoizer.caffeinate import caffeinate_context
from vemoizer.diarization import SpeakerCount
from vemoizer.ingest import IngestError
from vemoizer.llm import ConfigError
from vemoizer.output.naming import (
    collision_free_paths,
    dated_basename,
    nfc_stem_and_suffix,
)
from vemoizer.presets import RunOptions

#: The two output formats the meeting and memo presets write.
PRESET_FORMATS = ("md", "json")


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


def _write_output(target: Path, result: dict, fmt: str) -> bool:
    """Render *result* in *fmt* and write it to *target* (``-`` = stdout).

    Returns True on success; False after printing the error, so the caller
    can fail the run instead of reporting a transcript that was never
    written.
    """
    from vemoizer.output.formatters import format_transcript

    try:
        rendered = format_transcript(result, fmt)
    except (ValueError, KeyError) as e:
        typer.echo(f"error: {e}", err=True)
        return False
    if str(target) == "-":
        typer.echo(rendered)
        return True
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(rendered, encoding="utf-8")
    except OSError as e:
        typer.echo(f"error: could not write {target}: {e}", err=True)
        return False
    return True


def _write_preset_output(
    result: dict,
    first_stem: str,
    out_dir: Path,
) -> list[str]:
    """Write the meeting/memo output pair (``.md`` + ``.json``) to *out_dir*.

    The base name is ``YYYY-MM-DD <title>`` where *title* comes from
    ``result["notes"]["title"]`` and falls back to *first_stem* (the
    first source file's stem) when the LLM produced no title.  Collision
    suffixes `` (2)``, `` (3)``, … are checked against the real
    filesystem via ``collision_free_path``.

    Returns the list of written relative paths (for the ``wrote <path>``
    summary lines).
    """
    notes = result.get("notes")
    title = ""
    if isinstance(notes, dict):
        t = notes.get("title")
        if isinstance(t, str) and t.strip():
            title = t

    base = dated_basename(title, fallback_stem=first_stem)
    # The .md/.json pair is probed as a unit so both files always share
    # one stem (never ``X.md`` + ``X (2).json``) — issue #82 review.
    paths = collision_free_paths(out_dir, base, [f".{fmt}" for fmt in PRESET_FORMATS])
    written: list[str] = []
    for path, fmt in zip(paths, PRESET_FORMATS, strict=True):
        if _write_output(path, result, fmt):
            written.append(path.name)
    return written


def _check_result(
    file: Path | str, result: dict, *, diarize: bool, diarize_label: str = "--diarize"
) -> int:
    """The M1 fail-loud checks over one file's result (issue #78).

    *file* may be a path or a string label (a multi-part group's "a.m4a+
    b.m4a"); only display matters, so a bare label string is accepted
    directly. Prints the warnings channel, then fails (1) on an ``error``
    key, an empty transcript, or ``--diarize`` without speaker labels;
    returns 0 when the result looks like a real transcript. The messages
    are pinned by tests — both ``transcribe_batch`` and ``run_preset`` go
    through here (one implementation); ``diarize_label`` preserves each
    entry point's pre-M2 wording for the no-labels line.
    """
    for warning in result.pop("warnings", []):
        typer.echo(warning, err=True)
    if "error" in result:
        typer.echo(f"error: {result['error']}", err=True)
        return 1
    if not result.get("text") and not result.get("segments"):
        typer.echo(
            f"error: no transcript produced for {file} (empty transcript)",
            err=True,
        )
        return 1
    if (
        diarize
        and result.get("segments")
        and not any("speaker" in seg for seg in result["segments"])
    ):
        typer.echo(
            f"error: {diarize_label} requested but "
            f"no speaker labels returned for {file}",
            err=True,
        )
        return 1
    return 0


def _write_temp_glossary(lines: list[str]) -> str:
    """Write *lines* to a fresh temp glossary file; return its path.

    Raises ``OSError`` on a write failure so the caller can fail with a
    clean error line inside the protected region (no leaked file).
    """
    fd, name = tempfile.mkstemp(prefix="vemoizer-glossary-", suffix=".txt")
    os.close(fd)
    path = Path(name)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)


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
    check fails (a clean error line downstream, never a traceback).
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

    Returns 0 on success, 1 if any group failed, 2 on a bad
    combination of group flags.
    """
    from vemoizer.grouping import (
        GroupingError,
        concat_groups,
        confirm_groups,
        decode_boundaries,
        natural_sort,
        part_offsets,
        propose_groups,
    )

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
    tail_texts, head_texts = decode_boundaries(ordered, transcribe_fn)
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
                        from vemoizer.grouping import remove_concat_output

                        remove_concat_output(merged)
                    typer.echo(f"error: {e}", err=True)
                    exit_code = 1
                    continue
                try:
                    result = _transcribe_one(merged, options)
                finally:
                    if merged_is_temp:
                        from vemoizer.grouping import remove_concat_output

                        remove_concat_output(merged)
                if "error" not in result:
                    # Multi-part groups only: single-part groups get no
                    # part_markers key at all (issue #77).
                    result["part_markers"] = [
                        {
                            "offset": off.start_offset,
                            "label": f"— osa {off.part_number} "
                            f"(äänitys {off.source_filename})",
                        }
                        for off in offsets
                    ]
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


def _write_group_outputs(
    group: list[Path],
    result: dict,
    *,
    formats: list[str],
    out: Path | None,
) -> bool:
    """Write one group's outputs; True on success (the --out override
    still applies, as in the plain loop)."""
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


def run_preset(
    files: list[Path],
    *,
    command: str,
    config_path: str | None,
    glossary_path: str | None,
    repair: bool | None = None,
    diarize: bool | None = None,
    speakers: SpeakerCount | None = None,
    quiet: bool = False,
) -> int:
    """Run the *meeting* or *memo* preset over *files*.

    Composition (issue #82, DESIGN DECISION): calls
    ``glossary_layers.load_layers`` + ``merge`` (no merge when an
    explicit ``--glossary`` is given), ``presets.resolve_options`` for
    the resolved options, then ``transcribe_file`` per file with the
    composed glossary.  Without ``--glossary`` the merged/filtered
    glossary is written to a temp file passed as ``glossary_path`` —
    meeting: merged terms (bare + ``@`` lines) and merged correction
    pairs; memo: merged correction pairs ONLY (whisper prompt stays
    empty, ``apply_corrections`` still fires).  An explicit
    ``--glossary`` is passed through as-is for meeting; for memo it is
    filtered to correction pairs only, via a temp file (same
    empty-whisper-prompt invariant).  ``quiet`` suppresses the final
    ``wrote <path>`` summary lines.  Temp files are deleted after the
    run.

    Both presets:

    - use ``profile="meeting"`` (whisper decode, skip consensus).
    - write ``.md`` + ``.json`` to the CWD with a dated, sanitised
      title and NFC collision suffix.
    - print one ``wrote <relative path>`` line per written file at the
      end.

    Meeting additionally enables diarization (default 2-6 speakers)
    and repair.  Memo disables diarization but keeps repair on.

    Returns 0 on success, 1 on any failure.
    """
    from vemoizer.pipeline import transcribe_file
    from vemoizer.presets import resolve_options

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
        },
    )

    # Temp-file seam: without --glossary, write the composed glossary to
    # a temp file and pass it through the existing glossary_path argument
    # (no new pipeline parameter).  Memo: correction pairs ONLY (the
    # whisper prompt stays empty) — meeting: merged terms (bare + @
    # lines) plus the merged correction pairs.
    temp_path: Path | None = None
    effective_glossary: str | None = None
    exit_code = 0
    written: list[str] = []

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
            # Memo seam with an explicit --glossary (issue #82): the file
            # replaces both layers, but the whisper initial_prompt must
            # stay empty — so filter it to correction pairs only (same
            # invariant as the layered memo path) via a temp file.
            if command == "memo":
                from vemoizer.glossary import load_corrections

                lines = [
                    f"{w} => {r}"
                    for w, r in load_corrections(effective_glossary).items()
                ]
                if lines:
                    temp_path = Path(_write_temp_glossary(lines))
                    effective_glossary = str(temp_path)
                else:
                    effective_glossary = None

        with caffeinate_context():
            for file in files:
                try:
                    # Fail loud on a malformed project config (issue #78).
                    _resolve_llm_config(options.config_path)
                except ConfigError as e:
                    # A malformed .vemoizer/config.toml must fail loud
                    # with a clean error line, not a traceback (issue #78).
                    typer.echo(f"error: {e}", err=True)
                    return 1
                result = transcribe_file(
                    file,
                    diarize=options.diarize,
                    config_path=options.config_path,
                    profile=options.profile,
                    repair=options.repair,
                    glossary_path=effective_glossary,
                    speakers=options.speakers,
                )
                if _check_result(
                    file, result, diarize=options.diarize, diarize_label="diarize"
                ):
                    exit_code = 1
                    continue

                # Determine the fallback stem from the FIRST file in the
                # argument list (deterministic), not the current iteration.
                first_stem, _ = nfc_stem_and_suffix(files[0])
                written.extend(_write_preset_output(result, first_stem, Path.cwd()))
    except OSError as e:
        # Temp-glossary write failure: clean error, non-zero exit, no
        # leaked file (the finally still cleans up what exists).
        typer.echo(f"error: could not write glossary: {e}", err=True)
        return 1
    finally:
        # The temp glossary file is deleted after the run (issue #82).
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)

    # Print one "wrote <relative path>" line per written file at the end
    # (--quiet suppresses them).
    for name in written:
        if not quiet:
            typer.echo(f"wrote {name}")

    return exit_code
