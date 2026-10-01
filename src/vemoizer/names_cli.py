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
from collections.abc import Callable
from pathlib import Path
from typing import Any

import typer

from vemoizer.output.naming import collision_free_path
from vemoizer.people_config import (
    find_people_config_path,
    read_people_list,
    write_people_list,
)
from vemoizer.render import render_markdown
from vemoizer.render_cli import _persist_speaker_names, _read_sidecar
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
) -> int:
    """Interactive speaker-naming loop.

    Returns 0 on success, 1 on sidecar error, 2 on non-TTY.
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

    # --- Prompt loop (inside clip_session for temp dir cleanup) ---
    new_names: dict[str, str] = {}
    prompt_aborted = False
    prompt_fn = input_fn if input_fn is not None else input

    def _ask(prompt_text: str) -> str | _NotFound:
        """Call the input function; treat EOFError as an abort sentinel."""
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
            for label in sorted(shares, key=lambda x: shares[x], reverse=True):
                share = shares[label]
                label_clips: list[ClipWindow] = clips.get(label, [])

                typer.echo(f"\n{label}  ({share:.0%} of talk)")
                for w in label_clips[:3]:
                    typer.echo(f'  "{w.quote}"')

                # Play clips (skipped under --no-play; quotes only then)
                if not degraded and not no_play and label_clips:
                    clip_map = extract_clips(resolved_source, label_clips[:3], tmp_dir)
                    for w in label_clips[:3]:
                        path = clip_map.get(w)
                        if path is not None and path.is_file():
                            play(path)
                        else:
                            degraded = True

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

        # --- Re-render Markdown ---
        markdown = render_markdown(data, corrections={}, speaker_names=new_names)
        md_path = collision_free_path(sidecar_path.parent, sidecar_path.stem, ".md")
        try:
            md_path.write_text(markdown, encoding="utf-8")
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
