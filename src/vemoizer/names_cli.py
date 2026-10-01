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

import os
import sys
import tomllib
from collections.abc import Callable
from pathlib import Path
from typing import Any

import typer

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
# people config helpers (list of strings, fail-open)
# ---------------------------------------------------------------------------


def _find_people_config_path(
    home: Path | None = None,
    cwd: Path | None = None,
    legacy_paths: tuple[Path, ...] | None = None,
) -> Path | None:
    """Nearest layered config path for the ``people`` key.

    Same order as ``llm._default_search``: nearest ``./.vemoizer`` →
    ``~/.vemoizer`` → legacy. ``None`` when nothing exists. ``legacy_paths``
    is injectable so tests can isolate from a real dev-machine legacy
    config (``llm._LEGACY_CONFIG_PATHS`` is computed at import time from
    the real ``Path.home()`` and would otherwise leak into tests).
    """
    from vemoizer.llm import _LEGACY_CONFIG_PATHS, _find_nearest_vemoizer_config

    if cwd is None:
        cwd = Path.cwd()
    if home is None:
        home = Path.home()
    if legacy_paths is None:
        legacy_paths = _LEGACY_CONFIG_PATHS

    nearest = _find_nearest_vemoizer_config(cwd)
    if nearest is not None:
        return nearest
    home_config = home / ".vemoizer" / "config.toml"
    if home_config.is_file():
        return home_config
    for candidate in legacy_paths:
        if candidate.is_file():
            return candidate
    return None


def _read_people_list(config_path: Path | None) -> list[str]:
    """Read the ``people`` key as a list of strings; fail-open to []."""
    if config_path is None or not config_path.is_file():
        return []
    try:
        with config_path.open("rb") as f:
            raw = tomllib.load(f)
    except (OSError, tomllib.TOMLDecodeError, ValueError):
        return []
    if not isinstance(raw, dict):
        return []
    value = _top_level_people(raw)
    return value if value is not None else []


def _top_level_people(raw: dict[str, Any]) -> list[str] | None:
    """The top-level ``people`` value, or ``None`` when it is not a list.

    In TOML, a bare ``people`` key placed after a ``[table]`` header belongs
    to that table, so only a top-level value (``raw["people"]``) is a valid
    ``people`` list. A ``people`` value nested inside a table is ignored here
    and stripped by :func:`_write_people_list`.
    """
    value = raw.get("people")
    if not isinstance(value, list):
        return None
    return [s for s in value if isinstance(s, str)]


def _write_people_list(config_path: Path, new_people: list[str]) -> None:
    """Atomically write *new_people* as the top-level ``people`` key.

    All other keys are preserved verbatim. Any pre-existing ``people`` key
    *inside* a table (e.g. under ``[llm]``) is dropped: a ``people`` key is
    only valid as a top-level list of strings, and leaving a stale one inside
    a table would make the next strict config load fail (issue #93).
    """
    raw: dict[str, Any] = {}
    if config_path.is_file():
        try:
            with config_path.open("rb") as f:
                loaded = tomllib.load(f)
        except (OSError, tomllib.TOMLDecodeError, ValueError):
            loaded = None
        if isinstance(loaded, dict):
            raw = loaded

    for table in raw.values():
        if isinstance(table, dict):
            table.pop("people", None)
    raw["people"] = new_people

    lines = _emit_toml(raw)
    tmp = config_path.with_name(f"{config_path.name}.tmp-{os.getpid()}")
    config_path.parent.mkdir(parents=True, exist_ok=True)
    tmp.write_text(lines, encoding="utf-8")
    os.replace(str(tmp), str(config_path))


def _emit_toml(raw: dict[str, Any]) -> str:
    """Serialize *raw* as TOML: scalars as top-level, dicts as tables."""
    out: list[str] = []
    tables: dict[str, Any] = {}
    for key, value in raw.items():
        if isinstance(value, dict):
            tables[key] = value
        else:
            out.append(f"{key} = {_toml_value(value)}")
    for name, table in tables.items():
        out.append(f"[{name}]")
        for key, value in table.items():
            out.append(f"{key} = {_toml_value(value)}")
    return "\n".join(out) + ("\n" if out else "")


def _escape_basic_string(value: str) -> str:
    """Escape *value* for a TOML basic string: backslash, quote, newline,
    tab, carriage return and every other control character (plus ``U+007F``
    as a TOML basic string forbids raw control characters)."""
    out: list[str] = []
    for ch in value:
        if ch == "\\":
            out.append("\\\\")
        elif ch == '"':
            out.append('\\"')
        elif ch == "\n":
            out.append("\\n")
        elif ch == "\t":
            out.append("\\t")
        elif ch == "\r":
            out.append("\\r")
        elif ord(ch) < 0x20 or ch == "\x7f":
            out.append(f"\\u{ord(ch):04x}")
        else:
            out.append(ch)
    return "".join(out)


def _toml_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, str):
        return f'"{_escape_basic_string(value)}"'
    if isinstance(value, list):
        return "[" + ", ".join(_toml_value(item) for item in value) + "]"
    # ``tomllib`` only yields str/int/float/bool/list; this is unreachable
    # via the config round-trip (``_read_people_list`` / ``_write_people_list``)
    # and kept as an explicit empty string rather than an invalid TOML blob.
    return '""'


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
    # The ``_NOT_FOUND`` sentinel (not ``None``) distinguishes "caller did
    # not inject a path" from "caller injected an isolated path".
    if config_path is _NOT_FOUND:
        resolved_config: Path | None = _find_people_config_path()
    else:
        assert not isinstance(config_path, _NotFound)  # narrows the union
        resolved_config = config_path

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
    people_list = _read_people_list(resolved_config)

    # --- Resolve source entries ---
    raw_source: list[dict[str, Any]] = data.get("source") or []
    sidecar_dir = sidecar_path.parent
    cwd = Path.cwd()
    resolved_source: list[dict[str, Any]] = []
    degraded = False
    for entry in raw_source:
        if not isinstance(entry, dict):
            degraded = True
            continue
        resolved = _resolve_source_entry(entry, sidecar_dir, cwd)
        if resolved is not None:
            resolved_source.append(resolved)
        else:
            degraded = True

    if no_play:
        # Quotes only; no extraction or playback.
        has_clips = False
    elif not raw_source:
        has_clips = False
        degraded = True
    else:
        has_clips = True

    # --- Prompt loop (inside clip_session for temp dir cleanup) ---
    new_names: dict[str, str] = {}

    # Stored names (existing sidecar speaker_names). A label with a stored
    # name shows it as the prompt default; an empty answer keeps it, a
    # non-empty answer replaces it. A kept stored name never triggers the
    # Add-to-people prompt. (issue #93, decision 4.)
    stored_names: dict[str, str] = data.get("speaker_names") or {}

    with clip_session() as tmp_dir:
        # Install readline completer (if applicable); restores the previous
        # completer on every exit path, including KeyboardInterrupt.
        restore_completer = _install_completer(people_list)
        try:
            for label in sorted(shares, key=lambda x: shares[x], reverse=True):
                share = shares[label]
                label_clips: list[ClipWindow] = clips.get(label, [])

                typer.echo(f"\n{label}  ({share:.0%} of talk)")
                for w in label_clips[:3]:
                    typer.echo(f'  "{w.quote}"')

                # Play clips (if not --no-play and clips available)
                if has_clips and not no_play and label_clips:
                    clip_map = extract_clips(resolved_source, label_clips[:3], tmp_dir)
                    for w in label_clips[:3]:
                        path = clip_map.get(w)
                        if path is not None and path.is_file():
                            play(path)
                        else:
                            degraded = True

                # Prompt for name. A label with a stored name shows it as a
                # bracketed default; empty keeps it, non-empty replaces.
                prompt_fn = input_fn if input_fn is not None else input
                stored = stored_names.get(label)
                if stored:
                    prompt = f"Name for {label} [{stored}]: "
                else:
                    prompt = f"Name for {label} (empty to skip): "
                answer = prompt_fn(prompt).strip()

                if stored:
                    if not answer:
                        # Empty keeps the stored name; it already exists so
                        # it never triggers the Add-to-people prompt.
                        new_names[label] = stored
                        continue
                    new_names[label] = answer
                    # People list: add new (replacing) name?
                    if answer.lower() not in [p.lower() for p in people_list]:
                        cfg_display = (
                            str(resolved_config) if resolved_config else "(no config)"
                        )
                        yes = (
                            prompt_fn(
                                f"Add {answer} to people in {cfg_display}? [y/N] "
                            )
                            .strip()
                            .lower()
                        )
                        if yes == "y" and resolved_config is not None:
                            new_people = list(people_list)
                            if answer not in new_people:
                                new_people.append(answer)
                            _write_people_list(resolved_config, new_people)
                else:
                    if not answer:
                        continue
                    new_names[label] = answer
                    # People list: add new name?
                    if answer.lower() not in [p.lower() for p in people_list]:
                        cfg_display = (
                            str(resolved_config) if resolved_config else "(no config)"
                        )
                        yes = (
                            prompt_fn(
                                f"Add {answer} to people in {cfg_display}? [y/N] "
                            )
                            .strip()
                            .lower()
                        )
                        if yes == "y" and resolved_config is not None:
                            new_people = list(people_list)
                            if answer not in new_people:
                                new_people.append(answer)
                            _write_people_list(resolved_config, new_people)
        finally:
            if restore_completer is not None:
                restore_completer()

    # --- Persist names (only after loop completes) ---
    if new_names:
        try:
            _persist_speaker_names(sidecar_path, data, new_names)
        except OSError as e:
            typer.echo(f"error: could not persist speaker names: {e}", err=True)
            return 1

        # --- Re-render Markdown ---
        from vemoizer.output.naming import collision_free_path
        from vemoizer.render import render_markdown

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
        typer.echo(
            "note: some voice clips were unavailable; quotes only for those spans",
            err=True,
        )

    return 0


# ---------------------------------------------------------------------------
# Typer registration
# ---------------------------------------------------------------------------


def register_names(app) -> None:
    """Attach the ``names`` command to *app* (the main Typer instance)."""

    @app.command("names")
    def names(  # noqa: A001, A002
        sidecar_path: Path = typer.Argument(  # noqa: B008
            ...,
            help="The .json sidecar written by a meeting or memo run.",
        ),
        no_play: bool = typer.Option(  # noqa: B008
            False,
            "--no-play",
            help="Skip clip extraction and playback (quotes only).",
        ),
    ) -> None:
        """Interactively name the speakers in a sidecar."""
        rc = run_names(sidecar_path, no_play=no_play)
        if rc != 0:
            raise typer.Exit(code=rc)
