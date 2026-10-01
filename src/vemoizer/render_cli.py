"""``vemoizer render X.json`` Typer command (issue #89, M5a).

Re-applies the CURRENT glossary correction pairs and any ``--name``
values to a stored meeting/memo sidecar and re-emits the Markdown —
so renaming a speaker or adding a correction pair never requires a
re-transcribe. The LLM is never invoked.

Glossary resolution mirrors ``run_preset``: the layered glossary
(project + home layers, in that order) is used unless ``--glossary``
replaces both layers entirely. The stored ``options.glossary_sha256``
is compared against the hash of the current glossary files; a mismatch
prints exactly one stderr line (corrections and names are applied
regardless). A missing glossary file warns and proceeds (fail-open).

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
from pathlib import Path
from typing import Any

import typer

from vemoizer.sidecar import sha256_over_files


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
        except (OSError, UnicodeDecodeError) as e:
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
    tmp = sidecar_path.with_name(f"{sidecar_path.name}.tmp-{os.getpid()}")
    try:
        tmp.write_text(payload, encoding="utf-8")
        os.replace(str(tmp), str(sidecar_path))
    except OSError:
        with contextlib.suppress(OSError):  # cleanup is best-effort
            tmp.unlink(missing_ok=True)
        raise


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

        # --- Hash comparison ---
        options = data.get("options")
        if isinstance(options, dict):
            stored_hash = options.get("glossary_sha256")
            current_hash = sha256_over_files([str(f) for f in files]) if files else None
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
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text(markdown, encoding="utf-8")
            except OSError as e:
                typer.echo(f"error: could not write {out}: {e}", err=True)
                raise typer.Exit(code=1) from e
            typer.echo(f"wrote {out}")
        else:
            from vemoizer.output.naming import collision_free_path

            md_path = collision_free_path(sidecar_path.parent, sidecar_path.stem, ".md")
            try:
                md_path.write_text(markdown, encoding="utf-8")
            except OSError as e:
                typer.echo(f"error: could not write {md_path}: {e}", err=True)
                raise typer.Exit(code=1) from e
            typer.echo(f"wrote {md_path.name}")
