"""``vemoizer names X.json`` Typer command (issue #93, M5c-1).

Interactive speaker-naming: for each labelled speaker (ordered by talk
share) show talk share and up to 3 quotes, optionally play voice clips,
prompt for a name, then persist into the sidecar ``speaker_names`` and
re-render the Markdown.

A ``people`` list in the layered ``.vemoizer/config.toml`` provides
tab-completion suggestions at the prompt and is written back only when
the user explicitly confirms.

Exit codes:
- 0: success
- 1: unreadable or malformed sidecar
- 2: non-TTY stdin (requires an interactive terminal)
"""

from __future__ import annotations

import sys
import termios
from collections.abc import Callable
from pathlib import Path
from typing import Any

import typer

from vemoizer.people_config import (
    find_people_config_path,
    read_people_list,
    write_people_list,
)
from vemoizer.render import render_markdown
from vemoizer.render_cli import (
    _atomic_write_text,
    _persist_speaker_names,
    _read_sidecar,
)
from vemoizer.speaker_clips import (
    ClipWindow,
    clip_session,
    extract_clips,
    play,
    select_clips,
    talk_share,
)

# ---------------------------------------------------------------------------
# TTY probe (monkeypatchable seam for the meeting naming hook, issue #95)
# ---------------------------------------------------------------------------


def _stdout_isatty() -> bool:
    """The hook's stdout-TTY probe.

    A module global rather than an inlined ``sys.stdout.isatty()`` so the
    end-of-meeting naming hook can gate on it and tests can monkeypatch it
    without touching ``sys.stdout``.
    """
    return sys.stdout.isatty()


def _discard_pending_input() -> None:
    """Drop keystrokes typed while clips were cut or played (issue #202).

    Type-ahead typed during the wait was read as the answer to the next name
    prompt. Flushing the terminal input queue right before each prompt stops
    that. A stdin with no terminal queue has nothing to flush.
    """
    try:
        termios.tcflush(sys.stdin.fileno(), termios.TCIFLUSH)
    except (OSError, ValueError, termios.error):
        return


# ---------------------------------------------------------------------------
# source path resolution
# ---------------------------------------------------------------------------


def _resolve_source_entry(
    entry: dict[str, Any], sidecar_dir: Path, cwd: Path
) -> dict[str, Any] | None:
    """Resolve a relative source path: sidecar dir first, then CWD.

    Returns a new entry dict with the resolved absolute path, or ``None``
    when the path cannot be resolved to an existing file.
    """
    path_str = entry.get("path")
    if not isinstance(path_str, str) or not path_str:
        return None
    p = Path(path_str)
    if p.is_absolute():
        resolved = p
    else:
        candidate = sidecar_dir / p
        if candidate.is_file():
            resolved = candidate
        else:
            candidate = cwd / p
            if candidate.is_file():
                resolved = candidate
            else:
                return None
    if not resolved.is_file():
        return None
    return {**entry, "path": str(resolved)}


def _owner_track_source(
    options: Any, owner: str | None, sidecar_dir: Path, cwd: Path
) -> list[dict[str, Any]]:
    """The owner's mic stem as a one-entry clip source, or ``[]``.

    Recorded in ``options.owner_track`` by a two-track run (issue #202); the
    stem starts with the mix, so its timeline needs no offset.
    """
    track = options.get("owner_track") if isinstance(options, dict) else None
    if owner is None or not isinstance(track, str):
        return []
    resolved = _resolve_source_entry({"path": track}, sidecar_dir, cwd)
    return [resolved] if resolved is not None else []


def _prompt_add_to_people(
    answer: str,
    resolved_config: Path | None,
    people_list: list[str],
    ask: Callable[[str], str | _NotFound],
) -> bool:
    """Ask to add *answer* to the people list; write it on yes.

    Shared by the stored-name and new-name branches. Returns ``True`` when
    EOF aborted the prompt (caller must stop the run). The ask callable is
    passed in so the closure (and its ``_NotFound`` sentinel handling) stays
    with the caller.
    """
    cfg_display = str(resolved_config) if resolved_config else "(no config)"
    add_result = ask(f"Add {answer} to people in {cfg_display}? [y/N] ")
    if isinstance(add_result, _NotFound):
        return True
    if add_result.strip().lower() != "y" or resolved_config is None:
        return False
    new_people = list(people_list)
    if answer not in new_people:
        new_people.append(answer)
    write_people_list(resolved_config, new_people)
    return False


# ---------------------------------------------------------------------------
# readline tab-completion
# ---------------------------------------------------------------------------


def _install_completer(names: list[str]) -> Callable[[], None] | None:
    """Install a readline completer offering *names*.

    Only when ``readline`` is importable AND both stdin and stdout are TTYs.
    Returns an undo callable (restoring the previous completer) when the
    completer was installed, or ``None`` when it was not.
    """
    if not (sys.stdin.isatty() and sys.stdout.isatty()):
        return None
    try:
        import readline
    except ImportError:
        return None

    previous = readline.get_completer()

    def _completer(text: str, state: int) -> str | None:
        options = [n for n in names if n.lower().startswith(text.lower())]
        if state < len(options):
            return options[state]
        return None

    readline.set_completer(_completer)
    readline.parse_and_bind("tab: complete")

    def _restore() -> None:
        readline.set_completer(previous)

    return _restore


# ---------------------------------------------------------------------------
# core run logic
# ---------------------------------------------------------------------------


class _NotFound:
    """Sentinel: caller did not inject a config path (search layered)."""


_NOT_FOUND = _NotFound()


def run_names(
    sidecar_path: Path,
    *,
    no_play: bool = False,
    input_fn: Callable[[str], str] | None = None,
    tty_isatty: Callable[[], bool] | None = None,
    config_path: Path | _NotFound = _NOT_FOUND,
    discard_input: Callable[[], None] | None = None,
) -> int:
    """Interactive speaker-naming loop.

    Returns 0 on success, 1 on sidecar error, 2 on non-TTY. A two-track
    call's owner (``options.owner``) is already named: it gets its quote and
    clip, but is never prompted for. ``discard_input`` runs before each
    prompt (default: flush the terminal input queue).
    """
    # --- Non-TTY guard (FIRST, before any other logic) ---
    isatty = tty_isatty if tty_isatty is not None else sys.stdin.isatty
    if not isatty():
        typer.echo(
            "error: names requires an interactive TTY; pipe or cron are not supported",
            err=True,
        )
        return 2

    # --- Load sidecar ---
    data = _read_sidecar(sidecar_path)
    if data is None:
        return 1

    # --- Locate the people config (nearest layered config) ---
    if config_path is _NOT_FOUND:
        resolved_config: Path | None = find_people_config_path()
    else:
        assert not isinstance(config_path, _NotFound)  # narrows the union
        resolved_config = config_path

    raw_sn = data.get("speaker_names")
    stored_names: dict[str, str] = raw_sn if isinstance(raw_sn, dict) else {}

    paragraphs: list[dict[str, Any]] = data.get("paragraphs") or []
    segments: list[dict[str, Any]] = data.get("segments") or []

    # --- Talk share ordering ---
    shares = talk_share(paragraphs)
    if not shares:
        typer.echo("no labelled speakers found in sidecar", err=True)
        return 0

    # --- Select clips ---
    clips = select_clips(paragraphs, segments)

    # --- Read people list (fail-open to []) ---
    people_list = read_people_list(resolved_config)

    # --- Resolve source entries ---
    raw_source: list[dict[str, Any]] = data.get("source") or []
    sidecar_dir = sidecar_path.parent
    cwd = Path.cwd()
    resolved_source: list[dict[str, Any]] = []
    for entry in raw_source:
        if isinstance(entry, dict):
            resolved = _resolve_source_entry(entry, sidecar_dir, cwd)
            if resolved is not None:
                resolved_source.append(resolved)

    # ``degraded`` encodes the no-clips state (True when no source entry
    # resolved); clip playback gates on ``not degraded`` rather than a
    # separate ``has_clips`` flag.
    degraded = not resolved_source

    # The owner's voice comes from the mic stem (issue #202): the mix also
    # carries whatever the far side played back over the owner's speech.
    options = data.get("options")
    owner = options.get("owner") if isinstance(options, dict) else None
    owner = owner if isinstance(owner, str) and owner else None
    owner_source = _owner_track_source(options, owner, sidecar_dir, cwd)

    # --- Prompt loop (inside clip_session for temp dir cleanup) ---
    new_names: dict[str, str] = {}
    prompt_aborted = False
    prompt_fn = input_fn if input_fn is not None else input
    discard = discard_input if discard_input is not None else _discard_pending_input

    def _ask(prompt_text: str) -> str | _NotFound:
        """Discard type-ahead, then call the input function; EOFError aborts."""
        discard()
        try:
            return prompt_fn(prompt_text)
        except EOFError:
            return _NotFound()

    with clip_session() as tmp_dir:
        # Install readline completer (if applicable); restores the previous
        # completer on every exit path, including KeyboardInterrupt.
        restore_completer = _install_completer(people_list)
        known = [p.lower() for p in people_list]
        try:
            if (
                not no_play
                and (resolved_source or owner_source)
                and any(clips.get(label) for label in shares)
            ):
                typer.echo("Preparing audio clips…")
            for label in sorted(shares, key=lambda x: shares[x], reverse=True):
                share = shares[label]
                label_clips: list[ClipWindow] = clips.get(label, [])

                typer.echo(f"\n{label}  ({share:.0%} of talk)")
                for w in label_clips[:3]:
                    typer.echo(f'  "{w.quote}"')

                # Play clips (skipped under --no-play; quotes only then)
                is_owner = label == owner
                source = owner_source if is_owner and owner_source else resolved_source
                if source and not no_play and label_clips:
                    clip_map = extract_clips(source, label_clips[:3], tmp_dir)
                    for w in label_clips[:3]:
                        path = clip_map.get(w)
                        if path is not None and path.is_file():
                            play(path)
                        else:
                            degraded = True

                # The owner of a two-track call is already named: no prompt,
                # and no offer to add them to the people list.
                if is_owner:
                    continue

                # Prompt for name. A label with a stored name shows it as a
                # bracketed default; empty keeps it, non-empty replaces.
                stored = stored_names.get(label)
                if stored:
                    prompt = f"Name for {label} [{stored}]: "
                else:
                    prompt = f"Name for {label} (empty to skip): "
                result = _ask(prompt)
                if isinstance(result, _NotFound):
                    prompt_aborted = True
                    break
                answer = result.strip()

                # Store the name. Stored-label with empty answer keeps the
                # stored name; new-label with empty answer skips it.
                keep_stored = bool(stored) and not answer
                if keep_stored:
                    new_names[label] = stored
                elif answer:
                    new_names[label] = answer
                else:
                    continue

                # People list: ask to add a NEW name (replacing or fresh)
                # that is not already there (case-insensitive). A kept
                # stored name is skipped (it was added when originally named).
                if not keep_stored and (
                    answer.lower() not in known
                    and _prompt_add_to_people(
                        answer, resolved_config, people_list, _ask
                    )
                ):
                    prompt_aborted = True
                    break
        finally:
            if restore_completer is not None:
                restore_completer()

    # --- EOFError mid-prompt: stop cleanly, persist what was entered ---
    if prompt_aborted:
        typer.echo("input ended (Ctrl-D); persisting names entered so far", err=True)

    # --- Persist names (only after loop completes) ---
    if new_names:
        try:
            _persist_speaker_names(sidecar_path, data, new_names)
        except OSError as e:
            typer.echo(f"error: could not persist speaker names: {e}", err=True)
            return 1

        # --- Re-render Markdown in place (issue #202: one .md per sidecar) ---
        markdown = render_markdown(data, corrections={}, speaker_names=new_names)
        md_path = sidecar_path.with_suffix(".md")
        try:
            _atomic_write_text(md_path, markdown)
        except OSError as e:
            typer.echo(f"error: could not write {md_path}: {e}", err=True)
            return 1
        typer.echo(f"wrote {md_path.name}")
    else:
        typer.echo("no names entered")

    # --- One aggregate notice for degraded clips ---
    if degraded and not no_play:
        notice = "note: some voice clips were unavailable; quotes only for those spans"
        typer.echo(notice, err=True)

    return 0


# ---------------------------------------------------------------------------
# Typer registration
# ---------------------------------------------------------------------------


def register_names(app: typer.Typer) -> None:
    """Attach the ``names`` command to *app* (the main Typer instance)."""

    @app.command("names")
    def names(  # noqa: A001, A002 - mirrors vemoizer CLI subcommand name
        sidecar_path: Path = typer.Argument(  # noqa: B008 - typer argument default
            ...,
            help="The .json sidecar written by a meeting or memo run.",
        ),
        no_play: bool = typer.Option(  # noqa: B008 - typer option default
            False,
            "--no-play",
            help="Skip clip extraction and playback (quotes only).",
        ),
    ) -> None:
        """Interactively name the speakers in a sidecar."""
        rc = run_names(sidecar_path, no_play=no_play)
        if rc != 0:
            raise typer.Exit(code=rc)
