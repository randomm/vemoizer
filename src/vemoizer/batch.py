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
no new pipeline parameter.  The glossary path flows through the
existing ``glossary_path`` argument; the temp-file seam for layered
glossary composition (corrections-only for memo, merged terms+@ for
meeting) will land with the glossary_layers workstream.
"""

from __future__ import annotations

from pathlib import Path

import typer

from vemoizer.caffeinate import caffeinate_context
from vemoizer.diarization import SpeakerCount
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
) -> int:
    """Run the *meeting* or *memo* preset over *files*.

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

    # Resolve preset defaults.
    if command == "meeting":
        do_diarize = diarize if diarize is not None else True
        do_repair = repair if repair is not None else True
        do_speakers = speakers if speakers is not None else (2, 6)
    elif command == "memo":
        do_diarize = diarize if diarize is not None else False
        do_repair = repair if repair is not None else True
        do_speakers = None
    else:
        typer.echo(f"error: unknown preset {command!r}", err=True)
        return 2

    exit_code = 0
    written: list[str] = []

    with caffeinate_context():
        for file in files:
            result = transcribe_file(
                file,
                diarize=do_diarize,
                config_path=config_path,
                profile="meeting",
                repair=do_repair,
                glossary_path=glossary_path,
                speakers=do_speakers,
            )
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
                    f"error: no transcript produced for {file.name} (empty transcript)",
                    err=True,
                )
                exit_code = 1
                continue
            elif (
                do_diarize
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

    # Print one "wrote <relative path>" line per written file at the end.
    for name in written:
        typer.echo(f"wrote {name}")

    return exit_code
