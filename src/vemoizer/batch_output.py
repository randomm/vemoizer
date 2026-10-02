"""Output-writing and temp-glossary helpers for batch transcription.

Split from ``batch.py`` so the orchestration loops (``transcribe_batch``,
``run_batch`` in :mod:`vemoizer.batch`, ``run_preset`` in
:mod:`vemoizer.batch_preset`) have a single responsibility: the
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
from collections.abc import Callable
from pathlib import Path
from typing import Any

import typer

from vemoizer.output.naming import (
    collision_free_paths,
    nfc_stem_and_suffix,
)
from vemoizer.presets import RunOptions
from vemoizer.report import build_quality_report


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


def _write_output(target: Path, result: dict[str, Any], fmt: str) -> bool:
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
    result: dict[str, Any],
    first_stem: str,
    out_dir: Path,
    *,
    date_str: str | None = None,
) -> list[str]:
    """Write the meeting/memo output pair (``.md`` + ``.json``) to *out_dir*.

    The base name is ``YYYY-MM-DD <title>`` where *title* comes from
    ``result["notes"]["title"]`` and falls back to *first_stem* (the
    first source file's stem) when the LLM produced no title.  The date
    is *date_str* when given (issue #87: the first source file's
    modification date), else today.  Collision suffixes `` (2)``,
    `` (3)``, … are checked against the real filesystem via
    ``collision_free_path``.

    Returns the list of written relative paths (for the ``wrote <path>``
    summary lines).
    """
    notes = result.get("notes")
    title = ""
    if isinstance(notes, dict):
        t = notes.get("title")
        if isinstance(t, str) and t.strip():
            title = t

    base = dated_basename(title, fallback_stem=first_stem, date_str=date_str)
    # The .md/.json pair is probed as a unit so both files always share
    # one stem (never ``X.md`` + ``X (2).json``) — issue #82 review.
    paths = collision_free_paths(out_dir, base, [f".{fmt}" for fmt in PRESET_FORMATS])
    written: list[str] = []
    for path, fmt in zip(paths, PRESET_FORMATS, strict=True):
        if _write_output(path, result, fmt):
            written.append(path.name)
    return written


def _call_write_seam(
    fn: Callable[[Path | str, dict[str, Any]], object],
    label: Path | str,
    result: dict[str, Any],
) -> bool:
    """Call the per-group preset write seam with a fail-loud boundary.

    Contract: the seam is called with exactly ``(label, result)`` —
    two positional arguments, nothing more. Its return value is
    ignored (bool/None both fine); it may raise, including from steps
    that run before the actual file write (first-part path lookup,
    mtime-date lookup). This helper turns any ``Exception`` into one
    clean ``error: <label>: could not produce output: <reason>`` line
    and ``False``; ``KeyboardInterrupt``/``SystemExit`` propagate; a
    normal call returns True (the caller sets exit 1 on ``False`` and
    keeps going — the seam must never escape as a raw traceback after
    minutes of decoding).
    """
    try:
        fn(label, result)
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as e:  # noqa: BLE001 - per-group fail-loud boundary
        typer.echo(f"error: {label}: could not produce output: {e}", err=True)
        return False
    return True


def _check_and_write(
    write_group_fn: Callable[[Path | str, dict[str, Any]], object] | None,
    label: Path | str,
    result: dict[str, Any],
    *,
    diarize: bool,
    diarize_label: str = "--diarize",
) -> bool:
    """The preset write-seam block shared by both of ``run_batch``'s loops.

    When *write_group_fn* is set: first the fail-loud result checks
    (``_check_result``), then the seam via ``_call_write_seam``. Returns
    True on success and also True when no seam is set (nothing to do);
    returns False when the caller must set exit 1 and keep going. Both
    current call sites guard ``write_group_fn is not None`` before
    calling. Byte-identical to the inline blocks it replaced (issue #87
    lens review: one implementation).
    """
    if write_group_fn is None:
        return True
    if _check_result(
        label,
        result,
        diarize=diarize,
        diarize_label=diarize_label,
    ):
        return False
    return _call_write_seam(write_group_fn, label, result)


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


def _render_quality_report(
    result: dict[str, Any],
    *,
    diarize_requested: bool,
) -> None:
    """Compute the per-file quality report BEFORE the warnings pop.

    M6 (issue #75): the report reads ``result["warnings"]`` and rides on
    the run dict as ``quality_report`` (the md embeds it as a
    ``<details>`` block) AND is returned to the caller for stdout
    printing (suppressed by ``--quiet``). It must be computed HERE — not
    in the write seam — because ``_check_result``'s destructive
    ``result.pop("warnings")`` runs after this call: a report computed
    later would see an empty warnings list in every real run.
    Fail-open (invariant #5): ``build_quality_report`` never raises for
    any run-dict shape the pipeline produces; a blank report (nothing to
    report) stores an empty string the md formatter omits.
    """
    glossary_source = result.get("glossary_source")
    glossary_source = (
        glossary_source
        if isinstance(glossary_source, str) and glossary_source
        else None
    )
    glossary_terms = result.get("glossary_terms")
    glossary_terms = (
        [str(t) for t in glossary_terms] if isinstance(glossary_terms, list) else None
    )
    language = result.get("language")
    language = language if isinstance(language, str) and language else "fi"
    report = build_quality_report(
        result,
        diarize_requested=diarize_requested,
        glossary_source=glossary_source,
        glossary_terms=glossary_terms,
        language=language,
    )
    result["quality_report"] = report


def check_failure_reason(
    file: Path | str,
    result: dict[str, Any],
    *,
    diarize: bool,
    diarize_label: str = "--diarize",
) -> str | None:
    """The one-line failure reason for a failed result check (issue #100).

    The single source of the failure reason shared by ``_check_result``
    (which prints ``error: <reason>``) and the M4a failure notifications
    (which must carry the same reason, with the ``error: `` prefix
    stripped by the notify helper). Returns ``None`` when the result is
    fine, or the exact message for each of the three failure conditions:
    an ``error`` key, an empty transcript (no text and no segments), or
    *diarize* requested but no speaker labels.
    """
    if "error" in result:
        return str(result["error"])
    if not result.get("text") and not result.get("segments"):
        return f"no transcript produced for {file} (empty transcript)"
    if (
        diarize
        and result.get("segments")
        and not any("speaker" in seg for seg in result["segments"])
    ):
        return f"{diarize_label} requested but no speaker labels returned for {file}"
    return None


def _check_result(
    file: Path | str,
    result: dict[str, Any],
    *,
    diarize: bool,
    diarize_label: str = "--diarize",
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
    one-element list (``_part_markers``'s same contract). A payload of
    any other type cannot be represented as warnings — it is still
    dropped without a crash, but the degradation is observable as
    exactly one bounded stderr line
    (``warning: ignored an unparseable warnings payload of type
    <TypeName>``). Non-string entries inside a list/tuple are coerced
    with ``str()`` rather than dropped.
    """
    raw = result.pop("warnings", [])
    warnings: list[str] = []
    if isinstance(raw, str):
        warnings = [raw]
    elif isinstance(raw, (list, tuple)):
        warnings = [str(w) for w in raw]
    elif raw is not None:
        typer.echo(
            f"warning: ignored an unparseable warnings payload of "
            f"type {type(raw).__name__}",
            err=True,
        )
    for warning in warnings:
        typer.echo(warning, err=True)
    reason = check_failure_reason(
        file, result, diarize=diarize, diarize_label=diarize_label
    )
    if reason is not None:
        typer.echo(f"error: {reason}", err=True)
        return 1
    return 0


def _process_result(
    label: Path | str,
    result: dict[str, Any],
    *,
    formats: list[str],
    out: Path | None,
    quiet: bool,
    options: RunOptions | None,
    diarize: bool | None = None,
    copy: bool = False,
) -> bool:
    """One decoded result's M1 checks + output write + quiet echo.

    The single per-file/per-group core shared by ``transcribe_batch``
    (the expert ``transcribe`` loop; ``diarize``/``copy`` come from the
    function parameters) and ``run_batch``/``_run_plain``/``run_preset``
    (the group/preset paths; a ``RunOptions`` is passed, ``copy`` is
    never set, ``diarize`` is read from the options).

    ``--copy`` is honored ONLY by the expert ``transcribe`` loop
    (``copy=True``, passed by ``transcribe_batch``). The group path
    intentionally never copies: one clipboard per group is not a
    sensible multi-file contract, so it passes ``copy=False`` (the
    CLI warns when ``--copy`` is combined with 2+ files).

    The output target mirrors the old ``_write_group_outputs`` contract:
    an explicit *out* gets only the first format (``-`` = stdout);
    otherwise every format is written from the first path's stem. Returns
    True when the result passed ``_check_result`` AND every write
    succeeded — False means the caller must set exit 1. The ``--diarize``
    no-labels wording is the pre-refactor wording for every path (pinned
    by tests); the preset path (``run_preset``) keeps its own
    ``diarize_label="diarize"`` call to ``_check_result`` directly.
    """
    effective_diarize = (
        diarize if diarize is not None else (options.diarize if options else False)
    )
    # M6 (issue #75): the quality report is computed BEFORE _check_result
    # pops result["warnings"] — after that pop the report would see an
    # empty warnings list in every real run (the CLI's old line-233 pop
    # is now inside _check_result).
    _render_quality_report(result, diarize_requested=effective_diarize)
    # glossary_terms is the report's matching input, not an output field:
    # it is stashed by the preset seam and consumed here — never mirrored
    # into a json/srt output file (the header shows the formatted
    # glossary_source string only).
    result.pop("glossary_terms", None)
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
    if copy:
        # Only the expert transcribe loop passes copy=True; the group path
        # never copies (see docstring).
        from vemoizer.copy import copy_to_clipboard

        copy_to_clipboard(result["text"])
    if not quiet:
        name = label.name if isinstance(label, Path) else label
        typer.echo(f"wrote transcript for {name}")
        # M6 (issue #75): the per-file quality report goes to stdout
        # (suppressed by --quiet, per the ticket); the <details> block in
        # the md already carries it for the file output.
        report = result.get("quality_report")
        if isinstance(report, str) and report.strip():
            typer.echo(report)
    return True
