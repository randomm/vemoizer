"""``vemoizer render X.json`` Typer command (issue #89, M5a).

Re-applies the CURRENT glossary correction pairs and any ``--name``
values to a stored meeting/memo sidecar and re-emits the Markdown —
so renaming a speaker or adding a correction pair never requires a
re-transcribe. The LLM is never invoked.

Glossary resolution mirrors ``run_preset``: the layered glossary
(project + home layers, in that order) is used unless ``--glossary``
replaces both layers entirely. The stored ``options.glossary_sha256``
is the sha256 over the run's **prompt-term set** — the non-correction,
non-``@`` lines after layer merge, deduped case-insensitively (first-seen
spelling wins), i.e. the canonical deduplicated prompt-term set that is
the input to ``glossary_prompt`` before its token-budget truncation —
and render recomputes the same hash (via the shared
:func:`vemoizer.sidecar.prompt_term_set_hash`) over the current
glossary files. Only a change to that set prints a drift warning;
correction pairs (``wrong => right``) and ``@``-prefixed LLM-only names
are render-safe and never warn.
A missing glossary file warns and proceeds (fail-open).

By default the rendered Markdown is written to ``<sidecar-stem>.md``
next to the sidecar, **overwriting** any existing file — the output is
fully derived from the sidecar, so the previous file is replaced, not
accumulated. ``--out`` is the way to write elsewhere.

``--name LABEL=NAME`` values are persisted into the sidecar's
``speaker_names`` by rewriting the JSON in place atomically: the
updated dict is written to a temp file in the **same directory** as the
sidecar, then ``os.replace``-d over it. This keeps the rename on the
same filesystem (atomic on APFS and ext4 alike).

Exit codes:
- 0: success (output .md written)
- 1: unreadable or malformed sidecar
- 2: malformed --name value
"""

from __future__ import annotations

import contextlib
import json
import os
import stat
from pathlib import Path
from typing import Any

import typer

from vemoizer.sidecar import prompt_term_set_hash


def _read_sidecar(path: Path) -> dict[str, Any] | None:
    """Read and validate a sidecar JSON file.

    Returns the parsed dict, or ``None`` after printing a clean error
    line (the caller exits 1). The sidecar must be valid JSON and must
    carry at least a ``text`` key (the minimum for a renderable result).
    """
    if not path.is_file():
        typer.echo(f"error: sidecar not found: {path}", err=True)
        return None
    try:
        raw = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as e:
        typer.echo(f"error: could not read {path}: {e}", err=True)
        return None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        typer.echo(f"error: malformed JSON in {path}: {e}", err=True)
        return None
    if not isinstance(data, dict):
        typer.echo(
            f"error: malformed sidecar in {path}: top-level must be a JSON object",
            err=True,
        )
        return None
    # Minimum renderable shape: a text field. Old sidecars without M5a
    # keys still have text and render fine.
    if "text" not in data:
        typer.echo(f"error: malformed sidecar in {path}: missing 'text' key", err=True)
        return None
    return data


def _resolve_glossary_files(glossary: Path | None) -> list[Path]:
    """Resolve the glossary files for hash comparison and correction loading.

    Mirrors ``run_preset``: ``--glossary`` replaces both layers entirely;
    otherwise the layered path (project first, then home) is used.
    Only files that exist are returned (fail-open over missing files).
    """
    if glossary is not None:
        return [glossary] if glossary.is_file() else []
    from vemoizer.sidecar import glossary_layer_files

    return glossary_layer_files()


def _load_corrections(files: list[Path]) -> dict[str, str]:
    """Load correction pairs from all glossary files, project-first.

    Mirrors ``glossary_layers.merge``: for the same wrong-side key the
    project layer's (first file's) right side wins over the home layer's.
    Files are processed in list order (project first); a key already set
    by a higher-priority file is never overwritten by a lower-priority one.
    Missing files are skipped (fail-open).
    """
    from vemoizer.glossary import load_corrections

    corrections: dict[str, str] = {}
    for path in files:
        try:
            file_corrections = load_corrections(path)
        except (OSError, UnicodeDecodeError, ValueError) as e:
            # UnicodeDecodeError is re-raised as ValueError by the loader
            # (fail-loud for transcribe); render degrades to a warning.
            typer.echo(
                f"warning: could not read glossary {path}: "
                f"{type(e).__name__} (proceeding without)",
                err=True,
            )
            continue
        # Later files (lower priority) only fill in keys not already set.
        for k, v in file_corrections.items():
            if k not in corrections:
                corrections[k] = v
    return corrections


def _existing_target_mode(target: Path) -> int | None:
    """Mode bits of *target* as a regular file, or ``None``.

    Follows a symlink for the regular-file test (a symlink whose pointee
    is a regular file keeps its pointee's mode — the old ``write_text``
    did too). ``None`` when the target does not exist (the writer uses
    the umask default then) or is not a regular file (the writer falls
    back to in-place).
    """
    try:
        st = os.stat(target)  # follows symlinks, as the old write_text did
    except OSError:
        return None
    if not stat.S_ISREG(st.st_mode):
        return None
    return stat.S_IMODE(st.st_mode)


def _atomic_write_text(target: Path, content: str) -> None:
    """Write *content* to *target* atomically (temp file + ``os.replace``).

    The temp file is created in the **same directory** as *target* so the
    replace is atomic on the same filesystem, and it is opened with mode
    0600 (0666 & ~umask for a new target, matching ``Path.write_text``)
    so the temp is private from its first byte — no window in which the
    new content is world-readable before the final ``os.chmod``.
    ``os.replace`` over a symlink whose pointee is a regular file replaces
    the symlink itself (not the pointed-to file); a symlink to a
    non-regular file (``/dev/null``, a FIFO, …) is not treated as regular
    and is written in place instead. On failure the temp file is cleaned
    up.

    An existing *regular* target keeps its mode: the temp file is
    ``os.chmod``-ed to the old ``st_mode & 0o7777`` before the replace,
    so a user's ``chmod 600`` on a transcript survives a re-render. A
    *new* target gets the same mode ``Path.write_text`` would have
    produced (0666 & ~umask). A *non-regular* existing target is written
    in place via ``write_text`` instead — the temp+replace path cannot
    represent such a file (the old, pre-atomic behaviour for those).
    """
    mode = _existing_target_mode(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    if mode is not None:
        tmp = target.with_name(f"{target.name}.tmp-{os.getpid()}")
        try:
            fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(content)
            os.chmod(tmp, mode)
            os.replace(str(tmp), str(target))
        except OSError:
            with contextlib.suppress(OSError):  # cleanup is best-effort
                tmp.unlink(missing_ok=True)
            raise
    else:
        target.write_text(content, encoding="utf-8")


def _persist_speaker_names(
    sidecar_path: Path, sidecar: dict[str, Any], names: dict[str, str]
) -> None:
    """Atomically persist *names* into the sidecar's ``speaker_names``.

    Writes the updated JSON to a temp file in the **same directory** as
    the sidecar (same filesystem, so ``os.replace`` is atomic) and
    ``os.replace``-s it over the sidecar. All other keys are preserved
    verbatim.
    """
    existing = sidecar.get("speaker_names")
    if not isinstance(existing, dict):
        sidecar["speaker_names"] = {}
    sidecar["speaker_names"].update(names)

    payload = json.dumps(sidecar, ensure_ascii=False, indent=2) + "\n"
    _atomic_write_text(sidecar_path, payload)


def register_render(app) -> None:
    """Attach the ``render`` command to *app* (the main Typer instance)."""

    @app.command("render")
    def render(  # noqa: A001, A002 - mirrors vemoizer CLI subcommand name
        sidecar_path: Path = typer.Argument(  # noqa: B008
            ...,
            help="The .json sidecar written by a meeting or memo run.",
        ),
        glossary: Path | None = typer.Option(  # noqa: B008
            None,
            "--glossary",
            help="Explicit glossary file (replaces both .vemoizer layers).",
        ),
        name: list[str] = typer.Option(  # noqa: B008
            [],
            "--name",
            help="Set speaker name: LABEL=NAME (repeatable; persisted into "
            "the sidecar).",
        ),
        out: Path | None = typer.Option(  # noqa: B008
            None,
            "--out",
            help="Write Markdown to this path instead of next to the sidecar.",
        ),
    ) -> None:
        """Re-apply glossary corrections and speaker names; re-emit Markdown."""
        # --- Load sidecar ---
        data = _read_sidecar(sidecar_path)
        if data is None:
            raise typer.Exit(code=1)

        # --- Resolve glossary files and load corrections ---
        # Warn about missing glossary files (fail-open).
        if glossary is not None and not glossary.is_file():
            typer.echo(
                f"warning: glossary file not found: {glossary} (proceeding without)",
                err=True,
            )
        files = _resolve_glossary_files(glossary)

        # Fail-open: a missing layered file warns and proceeds.
        if glossary is None:
            for f in files:
                if not f.is_file():
                    typer.echo(
                        f"warning: glossary file not found: {f} (proceeding without)",
                        err=True,
                    )

        corrections = _load_corrections(files)

        # --- Hash comparison (prompt-term set only) ---
        options = data.get("options")
        if isinstance(options, dict):
            stored_hash = options.get("glossary_sha256")
            current_hash = prompt_term_set_hash(files) if files else None
            if (
                stored_hash is not None
                and current_hash is not None
                and stored_hash != current_hash
            ):
                typer.echo(
                    "warning: glossary has changed since the run — new PROMPT "
                    "terms need a re-transcribe (corrections and names applied anyway)",
                    err=True,
                )

        # --- Parse --name values ---
        name_map: dict[str, str] = {}
        for item in name:
            label, sep, display = item.partition("=")
            label, display = label.strip(), display.strip()
            if not label or not display or not sep:
                typer.echo(
                    f"error: --name expects LABEL=NAME (got {item!r})",
                    err=True,
                )
                raise typer.Exit(code=2)
            name_map[label] = display

        # --- Render ---
        from vemoizer.render import render_markdown

        markdown = render_markdown(
            data, corrections=corrections, speaker_names=name_map
        )

        # --- Persist --name values into the sidecar ---
        if name_map:
            try:
                _persist_speaker_names(sidecar_path, data, name_map)
            except OSError as e:
                typer.echo(f"error: could not persist speaker names: {e}", err=True)
                raise typer.Exit(code=1) from e

        # --- Write output ---
        if out is not None:
            try:
                _atomic_write_text(out, markdown)
            except OSError as e:
                typer.echo(f"error: could not write {out}: {e}", err=True)
                raise typer.Exit(code=1) from e
            typer.echo(f"wrote {out}")
        else:
            md_path = sidecar_path.with_suffix(".md")
            try:
                _atomic_write_text(md_path, markdown)
            except OSError as e:
                typer.echo(f"error: could not write {md_path}: {e}", err=True)
                raise typer.Exit(code=1) from e
            typer.echo(f"wrote {md_path.name}")
