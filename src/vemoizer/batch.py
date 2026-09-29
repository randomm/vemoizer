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
    collision_free_path,
    dated_basename,
    nfc_stem_and_suffix,
)

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
    written: list[str] = []
    for fmt in PRESET_FORMATS:
        suffix = f".{fmt}"
        path = collision_free_path(out_dir, base, suffix)
        if _write_output(path, result, fmt):
            written.append(path.name)
    return written


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
            result = transcribe_file(
                file,
                diarize=diarize,
                config_path=config_path,
                profile=profile,
                repair=repair,
                glossary_path=glossary_path,
                speakers=speakers,
            )
            for warning in result.pop("warnings", []):
                typer.echo(warning, err=True)
            if "error" in result:
                typer.echo(f"error: {result['error']}", err=True)
                exit_code = 1
                continue
            # Fail loud (issue #78): an empty transcript must not look
            # like success.
            if (
                not result.get("text")
                and not result.get("segments")
                and "error" not in result
            ):
                typer.echo(
                    f"error: no transcript produced for {file.name} (empty transcript)",
                    err=True,
                )
                exit_code = 1
                continue
            elif (
                diarize
                and result.get("segments")
                and not any("speaker" in seg for seg in result["segments"])
            ):
                typer.echo(
                    f"error: --diarize requested but no speaker labels "
                    f"returned for {file.name}",
                    err=True,
                )
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
    filtered to that file's correction pairs via a temp file (same
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
    # whisper prompt stays empty); meeting: merged terms (bare + @ lines)
    # plus the merged correction pairs.
    temp_path: Path | None = None
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
            fd, name = tempfile.mkstemp(prefix="vemoizer-glossary-", suffix=".txt")
            os.close(fd)
            temp_path = Path(name)
            temp_path.write_text(
                "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
            )
            effective_glossary = str(temp_path)
        else:
            effective_glossary = None
    else:
        effective_glossary = options.glossary_path
        # Memo seam with an explicit --glossary (issue #82): the file
        # replaces both layers, but the whisper initial_prompt must stay
        # empty — so filter it to correction pairs only (same invariant
        # as the layered memo path) via a temp file.
        if command == "memo":
            from vemoizer.glossary import load_corrections

            lines = [
                f"{w} => {r}" for w, r in load_corrections(effective_glossary).items()
            ]
            if lines:
                fd, name = tempfile.mkstemp(prefix="vemoizer-glossary-", suffix=".txt")
                os.close(fd)
                temp_path = Path(name)
                temp_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
                effective_glossary = str(temp_path)
            else:
                effective_glossary = None

    exit_code = 0
    written: list[str] = []

    try:
        with caffeinate_context():
            for file in files:
                try:
                    result = transcribe_file(
                        file,
                        diarize=options.diarize,
                        config_path=options.config_path,
                        profile=options.profile,
                        repair=options.repair,
                        glossary_path=effective_glossary,
                        speakers=options.speakers,
                    )
                except ConfigError as e:
                    # A malformed .vemoizer/config.toml must fail loud
                    # with a clean error line, not a traceback (issue #78).
                    typer.echo(f"error: {e}", err=True)
                    exit_code = 1
                    continue
                for warning in result.pop("warnings", []):
                    typer.echo(warning, err=True)
                if "error" in result:
                    typer.echo(f"error: {result['error']}", err=True)
                    exit_code = 1
                    continue
                if (
                    not result.get("text")
                    and not result.get("segments")
                    and "error" not in result
                ):
                    typer.echo(
                        "error: no transcript produced for "
                        f"{file.name} (empty transcript)",
                        err=True,
                    )
                    exit_code = 1
                    continue
                elif (
                    options.diarize
                    and result.get("segments")
                    and not any("speaker" in seg for seg in result["segments"])
                ):
                    typer.echo(
                        f"error: diarize requested but no speaker labels "
                        f"returned for {file.name}",
                        err=True,
                    )
                    exit_code = 1
                    continue

                # Determine the fallback stem from the FIRST file in the
                # argument list (deterministic), not the current iteration.
                first_stem, _ = nfc_stem_and_suffix(files[0])
                written.extend(_write_preset_output(result, first_stem, Path.cwd()))
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
