"""Preset (``meeting`` / ``memo``) run orchestration.

Extracted from :mod:`vemoizer.batch_output` (the 500-line hard limit,
issue #87) so the meeting preset can gain the M3 split-recording
grouping seam without pushing either module over the ceiling.

``run_preset`` composes the layered glossary, resolves the preset
:class:`~vemoizer.presets.RunOptions`, and then:

- **single file or ``memo``**: the plain per-file loop (unchanged since
  issue #82 — per-file decode guard, ``_check_result`` fail-loud, dated
  ``.md`` + ``.json`` pair per file);
- **``meeting`` with 2+ files**: the M3 grouping flow
  (:func:`vemoizer.batch.run_batch`) — natural sort, mutually exclusive
  ``--yes``/``--no-group`` (exit 2), the pre-decode TTY guard (exit 2),
  boundary decodes → proposals → confirmation, ffmpeg concat per
  multi-part group, one decode per group (invariant 6), part markers —
  with the write seam handed over as a callable (``run_batch``'s
  ``write_group_fn``) so each group yields its dated ``.md`` + ``.json``
  pair in the CWD instead of stem-named per-format files.

The dated output name uses the **modification date** of the first
source file (``st_mtime``; for a group, the first part in natural-sort
order) rather than today, falling back to today when the file cannot be
stat'ed. The file's ``creation_time`` metadata is deliberately NOT used
(it is the export/copy time on iOS exports, not the recording date).

The temp glossary lifecycle (write once before the run, delete in a
``finally``), the ``@``-term split, the layered config search, and the
per-file guard all behave exactly as before for the no-group and
single-file cases.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from datetime import date, datetime
from pathlib import Path
from typing import Any

import typer

from vemoizer.batch_output import PRESET_FORMATS, _check_result, _write_preset_output
from vemoizer.diarization import SpeakerCount
from vemoizer.llm import ConfigError
from vemoizer.output.naming import nfc_stem_and_suffix
from vemoizer.presets import RunOptions, resolve_options

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


def _transcribe_preset_file(
    file: Path,
    options: RunOptions,
    glossary_path: str | None,
) -> dict[str, Any] | None:
    """One guarded preset transcribe (the per-file loop's fail-loud core).

    An unexpected ``transcribe_file`` exception degrades to a clean
    one-line ``error:`` naming the file (never a raw traceback mid-
    run); ``KeyboardInterrupt``/``SystemExit`` propagate; ``None`` on
    failure. A malformed project config (``ConfigError``) fails loud
    with a clean error line (issue #78).
    """
    from vemoizer.batch import _resolve_llm_config
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
        )
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as e:  # noqa: BLE001 - per-file fail-loud boundary
        typer.echo(f"error: {file.name}: {e}", err=True)
        return None


def _first_part_path(label: Path | str, files: list[Path]) -> Path:
    """The first part's path for a group label.

    A single-part group's label IS the part's path (``run_batch`` passes
    the group's ``Path``). A multi-part label joins part filenames with
    '+'; the first part is matched by name against the original
    *files* list (the group is built over ``natural_sort(files)``, so
    the first part is always a member of *files*).
    """
    if isinstance(label, Path):
        return label
    first_name = label.split("+", 1)[0]
    for f in files:
        if f.name == first_name:
            return f
    return Path(first_name)


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
) -> int:
    """The meeting 2+ files path: the M3 flow via ``run_batch``.

    ``run_batch`` owns the full grouping logic (natural sort, the
    mutually-exclusive flag check, the pre-decode TTY guard, boundary
    decodes, proposals, confirmation, concat, part offsets, part
    markers); this function only hands over the meeting write seam —
    one dated ``.md`` + ``.json`` pair per group in the CWD — and lets
    ``run_batch``'s single-file and ``--no-group`` short-circuits keep
    today's per-file behaviour (per-file pair, per-file mtime date).
    """
    from vemoizer.batch import run_batch
    from vemoizer.presets import replace

    # RunOptions is frozen; rebuild it with the composed effective_glossary
    # so run_batch's per-group transcribe reads the right glossary file.
    group_options = replace(options, glossary_path=effective_glossary)

    written: list[str] = []
    exit_code = 0

    def write_group(label: Path | str, result: dict[str, Any]) -> None:
        # One dated pair per group. The date is the first part's mtime
        # (group[0] in natural-sort order); the fallback stem is the
        # same first part's stem (mirrors the plain loop's files[0]).
        first = _first_part_path(label, files)
        stem, _ = nfc_stem_and_suffix(first)
        pair = _write_preset_output(
            result,
            stem,
            Path.cwd(),
            date_str=_mtime_date_str(first),
        )
        written.extend(pair)
        # A partial pair (fewer paths than the preset's formats — e.g.
        # the .json write failed) means this group's run failed; the
        # error line was already printed by _write_output.
        nonlocal exit_code
        if len(pair) < len(PRESET_FORMATS):
            exit_code = 1

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
    )

    for name in written:
        if not quiet:
            typer.echo(f"wrote {name}")
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
    quiet: bool = False,
    yes: bool = False,
    no_group: bool = False,
    transcribe_fn: Callable | None = None,
    input_fn: Callable[[str], str] | None = None,
    print_fn: Callable[[str], None] | None = None,
    tty_isatty: Callable[[], bool] | None = None,
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

    Meeting with 2+ files (issue #87): routes through the M3
    split-recording grouping flow (``run_batch``) — natural sort,
    ``--yes``/``--no-group`` (mutually exclusive, exit 2), the pre-
    decode TTY guard (exit 2), boundary decodes → proposals →
    confirmation, concat, one decode per group, part markers — with a
    group write seam so each group yields one dated ``.md`` + ``.json``
    pair in the CWD. A single file (either preset) keeps the plain
    per-file loop and needs no TTY. ``memo`` never groups.

    The dated output base name ``YYYY-MM-DD <title>`` uses the first
    source file's *modification date* (for a group: the first part in
    natural-sort order), falling back to today when it cannot be read;
    ``creation_time`` metadata is NOT used (issue #87).

    Returns 0 on success, 1 on any failure, 2 on a bad combination of
    group flags.
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
        },
    )

    # Temp-file seam: without --glossary, write the composed glossary to
    # a temp file and pass it through the existing glossary_path argument
    # (no new pipeline parameter).  Memo: correction pairs ONLY (the
    # whisper prompt stays empty) — meeting: merged terms (bare + @
    # lines) plus the merged correction pairs.
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

        # M3 grouping (issue #87): meeting with 2+ files runs the full
        # grouping flow through run_batch (the single source of the M3
        # contract — TTY guard, boundary decodes, concat, part markers).
        # run_batch handles the --yes/--no-group mutual-exclusion check
        # (exit 2) BEFORE the no_group short-circuit, so both flags
        # together always fail fast. The meeting write seam writes one
        # dated .md/.json pair per group.
        # The mutual-exclusion check also fires for a single file (before
        # the single-file short-circuit, matching run_batch's existing
        # order) — run_preset must check it here too, since a single file
        # does not route through run_batch.
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
            )

        # Plain per-file loop: single file (either preset), memo (always),
        # or meeting with --no-group. One dated pair per file; the date is
        # the file's own modification date (issue #87); the fallback stem
        # is the FIRST file's stem (deterministic, unchanged since #82).
        from vemoizer.caffeinate import caffeinate_context

        first_stem, _ = nfc_stem_and_suffix(files[0])
        exit_code = 0
        written: list[str] = []
        with caffeinate_context():
            for file in files:
                result = _transcribe_preset_file(file, options, effective_glossary)
                if result is None:
                    exit_code = 1
                    continue
                if _check_result(
                    file,
                    result,
                    diarize=options.diarize,
                    diarize_label="diarize",
                ):
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
                    # printed by _write_output.
                    exit_code = 1
        for name in written:
            if not quiet:
                typer.echo(f"wrote {name}")
        return exit_code
    except OSError as e:
        # Temp-glossary write failure: clean error, non-zero exit, no
        # leaked file (the finally still cleans up what exists).
        typer.echo(f"error: could not write glossary: {e}", err=True)
        return 1
    finally:
        # The temp glossary file is deleted after the run (issue #82).
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
