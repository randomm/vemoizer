"""Output-writing and temp-glossary helpers for batch transcription.

Split from ``batch.py`` so the orchestration loops (``transcribe_batch``,
``run_batch``, ``run_preset``) have a single responsibility: the
transcribe → check → write pipeline. The output-format rendering,
collision-free path selection, and temp-glossary file management live
here.

``_write_output`` renders a result dict in the requested format and
writes it to the target path (``-`` = stdout). ``_write_preset_output``
writes the meeting/memo ``.md`` + ``.json`` pair. ``_write_temp_glossary``
creates a temp glossary file for the layered-glossary seam.

All functions are importable from ``vemoizer.batch`` (re-exported) for
backwards compatibility with existing tests.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import typer

from vemoizer.caffeinate import caffeinate_context
from vemoizer.diarization import SpeakerCount
from vemoizer.llm import ConfigError
from vemoizer.output.naming import (
    collision_free_paths,
    nfc_stem_and_suffix,
)
from vemoizer.presets import RunOptions


def dated_basename(title: str, **kwargs) -> str:
    """Indirection for :func:`vemoizer.output.naming.dated_basename`.

    Defined here (not imported) so that ``monkeypatch.setattr(batch,
    "dated_basename", fake)`` in tests patches the name that
    ``_write_preset_output`` actually calls.
    """
    from vemoizer.output.naming import dated_basename as _db

    return _db(title, **kwargs)


#: The two output formats the meeting and memo presets write.
PRESET_FORMATS = ("md", "json")


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
    entry point's pre-M2 wording for the no-labels line. The ``warnings``
    channel is defensively normalised: a lone string becomes a
    one-element list (``_part_markers``'s same contract), anything else
    non-list-like is treated as no warnings — never a TypeError.
    """
    raw = result.pop("warnings", [])
    warnings: list[str] = []
    if isinstance(raw, str):
        warnings = [raw]
    elif isinstance(raw, (list, tuple)):
        warnings = [str(w) for w in raw]
    for warning in warnings:
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


def _process_result(
    label: Path | str,
    result: dict,
    *,
    formats: list[str],
    out: Path | None,
    quiet: bool,
    options: RunOptions | None,
    diarize: bool | None = None,
) -> bool:
    """One decoded result's M1 checks + output write + quiet echo.

    The single per-file/per-group core shared by ``transcribe_batch``
    (options=None — ``--copy`` is honored, one ``wrote transcript`` echo
    per file; ``diarize`` passed explicitly from the function parameter)
    and ``run_batch``/``_run_plain`` (a ``RunOptions`` — no ``--copy`` on
    the group path; ``diarize`` read from the options). The output target
    mirrors the old ``_write_group_outputs`` contract: an explicit *out*
    gets only the first format (``-`` = stdout); otherwise every format
    is written from the first path's stem. Returns True when the result
    passed ``_check_result`` AND every write succeeded — False means the
    caller must set exit 1. The ``--diarize`` no-labels wording is the
    pre-refactor wording for every path (pinned by tests); the preset
    path (``run_preset``) keeps its own ``diarize_label="diarize"``
    call to ``_check_result`` directly.
    """
    effective_diarize = (
        diarize if diarize is not None else (options.diarize if options else False)
    )
    if _check_result(
        label,
        result,
        diarize=effective_diarize,
    ):
        return False
    if out is not None:
        ok = _write_output(out, result, formats[0] if formats else "txt")
    else:
        from vemoizer.output.formatters import FORMAT_EXTENSIONS

        stem, _ = nfc_stem_and_suffix(Path(str(label)))
        ok = all(
            [
                _write_output(Path(f"{stem}{FORMAT_EXTENSIONS[fmt]}"), result, fmt)
                for fmt in formats
            ]
        )
    if not ok:
        return False
    if options is None:
        # The transcribe_batch loop honors --copy (the group path does
        # not — one clipboard per group is not a sensible contract).
        from vemoizer.copy import copy_to_clipboard

        copy_to_clipboard(result["text"])
    if not quiet:
        name = label.name if isinstance(label, Path) else label
        typer.echo(f"wrote transcript for {name}")
    return True


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
    # Deferred import so run_preset (defined here) and _resolve_llm_config /
    # _write_temp_glossary (defined in batch.py, which re-exports run_preset
    # from here) do not create an import cycle.
    from vemoizer.batch import _resolve_llm_config, _write_temp_glossary
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
