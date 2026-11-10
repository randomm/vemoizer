"""``vemoizer terms X.json``: interactive hear-and-fix term check (issue #268).

Whisper's ``p`` alone does not find the misrecognitions (a real 48-min call
had several at p 0.7-0.96), so the operator — not a threshold — decides:
list the term-shaped words least-sure first, pick any number, HEAR the word
in context, correct the spelling if needed. Confirmed and corrected terms
are appended to the project glossary and the ``.md`` is re-rendered once.

Mirror of :func:`vemoizer.names_cli.run_names`: same TTY gate, sidecar
reader, source resolution and clip machinery. ``extract_fn``/``play_fn`` are
injectable seams (tests never play audio). Exit codes: 0 success, 1
unreadable sidecar or failed write, 2 non-TTY stdin.
"""

from __future__ import annotations

import json
import logging
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

import typer

from vemoizer.glossary_write import (
    HomeGlossaryWriteError,
    append_to_project_glossary,
    project_glossary_path,
)
from vemoizer.names_cli import _discard_pending_input, _resolve_source_entry
from vemoizer.render import render_markdown
from vemoizer.render_cli import (
    _atomic_write_text,
    _load_corrections,
    _read_sidecar,
    _resolve_glossary_files,
)
from vemoizer.speaker_clips import (
    ClipWindow,
    clip_session,
    extract_clips,
    play,
    source_duration,
)
from vemoizer.term_review import (
    LIST_LIMIT,
    context_line,
    known_terms,
    select_terms,
    terms_grid,
)

logger = logging.getLogger(__name__)

#: Seconds of padding around an occurrence when cutting its clip.
CLIP_PAD_S = 1.5
_LIST_HEADER = "Terms in this call, least sure first:"
_PICK_PROMPT = "Number to hear and fix (Enter when done): "
_SPELL_PROMPT = "  Right spelling (Enter = it is right, r = replay, - = skip): "
_KEEP_PROMPT = "  Use for later calls too (glossary)? [Y/n] "
#: Replies at the keep prompt that mean "yes, a glossary pair" (case-insensitive).
_KEEP_GLOSSARY = ("", "y", "yes", "k", "kyllä")
#: Replies that mean "this call only" (case-insensitive).
_KEEP_CALL_ONLY = ("n", "no", "ei")
_BAD_NUMBER = "type a number from the list, or Enter when done"
_NO_CLIP = "  no clip to replay"
_HOME_REFUSAL = "not saved: run this from the meeting's folder, not your home folder"


class _Abort:
    """Sentinel: EOFError/KeyboardInterrupt at a prompt — keep what we have."""


def _ask_keep(ask: Callable[[str], str | _Abort]) -> bool | _Abort:
    """Ask :data:`_KEEP_PROMPT` until a valid reply.

    True means "a glossary pair", False "this call only"; anything else
    re-asks the same prompt, so a typo cannot silently become a glossary
    write (or a call-only fix).
    """
    while True:
        reply = ask(_KEEP_PROMPT)
        if isinstance(reply, _Abort):
            return reply
        verdict = reply.strip().lower()
        if verdict in _KEEP_CALL_ONLY:
            return False
        if verdict in _KEEP_GLOSSARY:
            return True


def _save(
    ask: Callable[[str], str | _Abort],
    sidecar_path: Path,
    data: dict[str, Any],
    speaker_names: dict[str, str],
    answers: dict[str, list[str]],
    call_only: dict[str, tuple[str, str]],
    heard: int,
    listed: int,
) -> int:
    """The save step: ask, append to the glossary, reconcile the sidecar's
    call-only fixes with this session's answers, re-render the .md once."""
    if not answers and not call_only:
        return 0
    lines = [line for entry in answers.values() for line in entry]
    target = project_glossary_path()
    try:
        shown = str(target.relative_to(Path.cwd()))
    except ValueError:
        shown = str(target)
    if lines:
        typer.echo(f"Save to {shown}?")
        for line in lines:
            typer.echo(f"  {line}")
    else:
        typer.echo("Save?")
    if call_only:
        typer.echo("This call only:")
        for stem, spelling in call_only.values():
            typer.echo(f"  {stem} => {spelling}")
    reply = ask("[Y/n] ")
    if isinstance(reply, _Abort) or reply.strip().lower() not in ("", "y", "yes"):
        typer.echo("not saved")
        return 0
    if lines:
        try:
            append_to_project_glossary(lines)
        except HomeGlossaryWriteError:
            # cwd is $HOME and no project glossary was found: the fallback
            # target would be the home layer. Only a GLOSSARY write is
            # refused, so with glossary lines nothing is saved here; a
            # call-only-only save (no glossary lines) never reaches this
            # handler and goes ahead.
            typer.echo(_HOME_REFUSAL)
            return 0
        except OSError as e:
            typer.echo(f"error: could not write {target}: {e}", err=True)
            return 1
    # A word answered in THIS session (either kind) supersedes whatever the
    # sidecar still carries for it, so a stored fix cannot outlive the
    # answer that replaced it; the new call-only fixes go on top.
    stored = data.get("call_corrections")
    stored_map: dict[str, Any] = dict(stored) if isinstance(stored, dict) else {}
    answered = {key.lower() for key in answers} | {key.lower() for key in call_only}
    merged: dict[str, Any] = {
        k: v for k, v in stored_map.items() if str(k).lower() not in answered
    }
    merged.update({stem: spelling for stem, spelling in call_only.values()})
    if merged != stored_map:
        if merged:
            data["call_corrections"] = merged
        else:
            data.pop("call_corrections", None)
        try:
            _atomic_write_text(
                sidecar_path,
                json.dumps(data, ensure_ascii=False, indent=2) + "\n",
            )
        except OSError as e:
            typer.echo(f"error: could not write {sidecar_path}: {e}", err=True)
            return 1
    corrections = _load_corrections(_resolve_glossary_files(None))
    markdown = render_markdown(
        data, corrections=corrections, speaker_names=speaker_names
    )
    md_path = sidecar_path.with_suffix(".md")
    try:
        _atomic_write_text(md_path, markdown)
    except OSError as e:
        typer.echo(f"error: could not write {md_path}: {e}", err=True)
        return 1
    typer.echo(f"wrote {md_path.name}")
    corrected = sum(1 for a in answers.values() if any("=>" in ln for ln in a))
    logger.info(
        "term check: listed %d, heard %d, corrected %d, confirmed %d, call-only %d",
        listed,
        heard,
        corrected,
        len(answers) - corrected,
        len(call_only),
    )
    return 0


def run_terms(
    sidecar_path: Path,
    *,
    no_play: bool = False,
    input_fn: Callable[[str], str] | None = None,
    tty_isatty: Callable[[], bool] | None = None,
    play_fn: Callable[[Path], Any] | None = None,
    extract_fn: Callable[..., dict[ClipWindow, Path | None]] | None = None,
) -> int:
    """Interactive term check on one sidecar; see the module docstring."""
    isatty = tty_isatty if tty_isatty is not None else sys.stdin.isatty
    if not isatty():
        msg = "error: terms requires an interactive TTY; pipe or cron are not supported"
        typer.echo(msg, err=True)
        return 2
    data = _read_sidecar(sidecar_path)
    if data is None:
        return 1

    words: list[dict[str, Any]] = data.get("words") or []
    paragraphs: list[dict[str, Any]] = data.get("paragraphs") or []
    segments: list[dict[str, Any]] = data.get("segments") or []
    raw_names = data.get("speaker_names")
    speaker_names: dict[str, str] = raw_names if isinstance(raw_names, dict) else {}
    seg_starts = [float(s.get("start", 0.0)) for s in segments if isinstance(s, dict)]
    candidates = select_terms(
        words,
        paragraphs,
        known_terms(data),
        limit=LIST_LIMIT,
        segment_starts=seg_starts,
    )
    if not candidates:
        typer.echo("no terms to check in this transcript")
        return 0
    typer.echo(_LIST_HEADER)
    for row in terms_grid(candidates):
        typer.echo(row)

    resolved_source: list[dict[str, Any]] = []
    for entry in data.get("source") or []:
        if isinstance(entry, dict):
            resolved = _resolve_source_entry(entry, sidecar_path.parent, Path.cwd())
            if resolved is not None:
                resolved_source.append(resolved)
    duration = source_duration(resolved_source, words)
    prompt_fn = input_fn if input_fn is not None else input
    player = play_fn if play_fn is not None else play
    extractor = extract_fn if extract_fn is not None else extract_clips

    def _ask(text: str) -> str | _Abort:
        _discard_pending_input()
        try:
            return prompt_fn(text)
        except (EOFError, KeyboardInterrupt):
            return _Abort()

    def _hear(clip: Path) -> bool:
        """Play *clip*; False on Ctrl-C — like Ctrl-D at a prompt: stop the
        loop and fall through to the save step with the answers so far."""
        nonlocal aborted, heard
        try:
            player(clip)
        except KeyboardInterrupt:
            aborted = True
            return False
        heard += 1
        return True

    # stem (lower) -> glossary lines for the word; the last answer wins.
    # ``call_only`` holds the same-keyed this-call fixes ({STEM: T}); a word
    # cannot be in both kinds at once — the last answer replaces the other.
    answers: dict[str, list[str]] = {}
    call_only: dict[str, tuple[str, str]] = {}
    heard = 0
    aborted = False
    with clip_session() as tmp_dir:
        while not aborted:
            pick = _ask(_PICK_PROMPT)
            if isinstance(pick, _Abort):
                break
            text = pick.strip()
            if not text:
                break
            if not text.isdigit() or not 1 <= int(text) <= len(candidates):
                typer.echo(_BAD_NUMBER)
                continue
            cand = candidates[int(text) - 1]
            typer.echo(context_line(cand, words, paragraphs, speaker_names))
            clip_path: Path | None = None
            if resolved_source and not no_play:
                win = ClipWindow(
                    max(0.0, cand.start - CLIP_PAD_S),
                    min(duration, cand.end + CLIP_PAD_S),
                    cand.word,
                )
                if win.start_s < win.end_s:
                    try:
                        got = extractor(resolved_source, [win], tmp_dir).get(win)
                    except KeyboardInterrupt:
                        # Ctrl-C mid-extraction: same contract as Ctrl-D.
                        aborted = True
                        got = None
                    if got is not None and got.is_file():
                        clip_path = got
            if aborted:
                break
            if clip_path is not None and not _hear(clip_path):
                break
            warned_no_clip = False
            while True:
                spell = _ask(_SPELL_PROMPT)
                if isinstance(spell, _Abort):
                    aborted = True
                    break
                answer = spell.strip()
                if answer.lower() == "r":
                    # "r" is a command, never a spelling: without a clip it
                    # re-prompts, announcing the missing clip only once.
                    if clip_path is None:
                        if not warned_no_clip:
                            typer.echo(_NO_CLIP)
                            warned_no_clip = True
                    elif not _hear(clip_path):
                        break
                    continue
                if answer != "-":
                    # The glossary line carries the STEM, not the surface:
                    # the most frequent surface may wear a case suffix or
                    # punctuation (``AVS:n``); the correction maps the stem
                    # and the render re-attaches the suffix.
                    key = cand.stem.lower()
                    if not answer:
                        answers[key] = [cand.stem]
                        call_only.pop(key, None)
                    else:
                        keep = _ask_keep(_ask)
                        if isinstance(keep, _Abort):
                            # Ctrl-D/Ctrl-C here drops THIS pick's new
                            # answer only — an earlier answer for the word
                            # (either kind) stays — and stops the loop like
                            # Ctrl-D at the spelling prompt, keeping the
                            # earlier answers for the save step.
                            aborted = True
                            break
                        if keep:
                            answers[key] = [f"{cand.stem} => {answer}", answer]
                            call_only.pop(key, None)
                        else:
                            call_only[key] = (cand.stem, answer)
                            answers.pop(key, None)
                break

    return _save(
        _ask,
        sidecar_path,
        data,
        speaker_names,
        answers,
        call_only,
        heard,
        len(candidates),
    )


def register_terms(app: typer.Typer) -> None:
    """Attach the ``terms`` command to *app* (the main Typer instance)."""

    @app.command("terms")
    def terms(  # noqa: A001, A002 - mirrors vemoizer CLI subcommand name
        sidecar_path: Path = typer.Argument(  # noqa: B008 - typer argument default
            ...,
            help="The .json sidecar written by a meeting or memo run.",
        ),
        no_play: bool = typer.Option(  # noqa: B008 - typer option default
            False,
            "--no-play",
            help="Skip clip extraction and playback (context lines only).",
        ),
    ) -> None:
        """Interactively hear and fix the term-like words in a sidecar."""
        rc = run_terms(sidecar_path, no_play=no_play)
        if rc != 0:
            raise typer.Exit(code=rc)
