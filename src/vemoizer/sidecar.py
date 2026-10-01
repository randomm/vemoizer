"""M5a JSON sidecar assembly (issue #89).

The meeting/memo ``.json`` written next to the ``.md`` carries four extra
keys — ``notes``, ``source``, ``options``, ``speaker_names`` — that let a
model-free ``vemoizer render`` (M5a, later workstreams) re-apply glossary
corrections and speaker names to the stored transcript without a
re-transcribe. This module is the single place those keys are built so the
preset write seam (``batch_output._write_preset_output``) stays a couple of
lines.

Design notes
------------

* The sidecar is built **before** ``format_json`` mirrors the result dict.
  ``format_json`` mirrors only keys *present* on the result (and the
  ``source``/``options``/``speaker_names`` keys are new to the mirror), so
  the expert ``transcribe`` JSON — which never carries ``_source_durations``
  or ``options``/``speaker_names`` — is byte-identical to before.

* ``source[]`` per-part ``path`` is the real source file path the seam
  passes as ``source_paths`` (single-file: ``[file]``; a group: one Path per
  part, resolved from the group label against the run's file list) — so
  ``render`` can re-apply per-part offsets to the actual files. Per-part
  ``part_offset_s`` comes from ``result["part_markers"]`` (the only per-part
  offsets that survive to the seam); ``duration_s`` comes from the per-part
  PCM durations the seam measures and stashes on
  ``result["_source_durations"]`` (single-file: ``[d]``; a group: one entry
  per part, in marker order). ``duration_s`` is optional — it is included
  only when a measured PCM duration is available for that part (the
  measurement is fail-open: an ffmpeg error leaves the key omitted).

* ``options.glossary_sha256`` is the sha256 over the concatenated raw bytes
  of the exact glossary files the run used (project layer first, then home,
  or the single explicit ``--glossary`` file), and is ``null`` when no
  glossary file existed at run time. ``render`` recomputes the same hash
  over the current layers for the drift warning, so the file *list* is the
  shared contract — :func:`glossary_layer_files` resolves it exactly the way
  ``run_preset`` does.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

__all__ = [
    "build_sidecar",
    "glossary_layer_files",
    "group_durations",
    "group_part_paths",
    "sha256_over_files",
]


def build_sidecar(
    result: dict[str, Any],
    *,
    command: str,
    glossary_files: list[str] | None,
    source_paths: list[Path] | None = None,
) -> dict[str, Any]:
    """Build the four M5a sidecar keys for a meeting/memo *result*.

    Mutates *result* in place (attaching ``notes``/``source``/``options``/
    ``speaker_names`` as needed) and returns it, so the seam can hand the
    same dict to both the ``.md`` and the ``.json`` formatter. The ``.md``
    formatter ignores the new keys; ``format_json`` mirrors them
    present-only, so this is a no-op on the Markdown output.

    * ``notes``: stored verbatim — already present on the result when the LLM
      tail produced it, absent (or ``None``) when it failed open. Never
      fabricated here.
    * ``source``: one entry per recorded part — single-file runs a single
      ``{path, part_offset_s: 0.0, duration_s?}``; a grouped run one entry
      per ``part_markers`` part, with ``path`` from the corresponding
      *source_paths* entry (falling back to the marker label when absent),
      ``part_offset_s`` from the marker, and ``duration_s`` from the
      corresponding measured PCM duration. ``duration_s`` is omitted when
      no duration was measured for that part.
    * ``options``: ``{command, glossary_files, glossary_sha256}``.
      ``glossary_files`` is the list of files actually used (``[]`` when
      none); ``glossary_sha256`` is the single sha256 over their
      concatenated raw bytes in layer order, or ``None`` when
      ``glossary_files`` is empty.
    * ``speaker_names``: ``{}`` by default (``render``'s ``--name`` persists
      into it later). Never a ``clips`` key.

    ``result["_source_durations"]`` (the per-part PCM durations the seam
    stashed) is consumed and removed so it never leaks into the output.
    """
    notes = result.get("notes")
    if notes:
        result["notes"] = notes
    elif notes is None and "notes" in result:
        # notes=None (LLM fail-open) → omit the key (format_json's
        # present-only mirror would include it as null otherwise).
        result.pop("notes")

    source = _build_source(result, source_paths)
    if source:
        result["source"] = source

    files = list(glossary_files) if glossary_files else []
    result["options"] = {
        "command": command,
        "glossary_files": files,
        "glossary_sha256": sha256_over_files(files) if files else None,
    }

    result.setdefault("speaker_names", {})

    result.pop("_source_durations", None)
    return result


def _build_source(
    result: dict[str, Any], source_paths: list[Path] | None
) -> list[dict[str, Any]]:
    """The ``source[]`` list, or ``[]`` when no part path is known.

    ``duration_s`` is optional: it is included only when a measured PCM
    duration is available for that part (``_source_durations``). When the
    measurement failed (fail-open, e.g. an ffmpeg error), the entry is
    emitted without the key.
    """
    durations = result.get("_source_durations")
    if not isinstance(durations, list):
        durations = []

    paths = [str(p) for p in source_paths] if source_paths else []

    markers = result.get("part_markers")
    if isinstance(markers, list) and markers:
        # Grouped run: one entry per part. path is the real part path from
        # source_paths (falling back to the marker label when the seam did
        # not pass a path for that part); part_offset_s is the marker's
        # cumulative decoded-PCM start offset; duration_s is the measured
        # PCM duration of that same part (aligned by position).
        entries: list[dict[str, Any]] = []
        for i, m in enumerate(markers):
            if not isinstance(m, dict):
                continue
            entry: dict[str, Any] = {
                "path": paths[i] if i < len(paths) else str(m.get("label", "")),
                "part_offset_s": float(m.get("offset", 0.0)),
            }
            if i < len(durations):
                entry["duration_s"] = float(durations[i])
            entries.append(entry)
        return entries

    # Single-file run: no part_markers. path is the file path the seam
    # passed (falling back to result["source_path"]); part_offset_s is 0.0
    # by definition; duration_s is the measured PCM duration, or omitted.
    path = paths[0] if paths else result.get("source_path")
    if not path:
        return []
    entry: dict[str, Any] = {
        "path": str(path),
        "part_offset_s": 0.0,
    }
    if durations:
        entry["duration_s"] = float(durations[0])
    return [entry]


def glossary_layer_files() -> list[Path]:
    """The glossary layer files the run would use, in hash order.

    Mirrors ``run_preset``: the project layer (nearest ``./.vemoizer/
    glossary.txt`` walking up from CWD) first, then the home layer
    (``~/.vemoizer/glossary.txt``). Only files that actually exist are
    returned, so the stored hash is over the files the run actually hashed
    (``load_layers`` is fail-open over missing files). An explicit
    ``--glossary`` replaces both layers entirely — that single file is passed
    through ``build_sidecar`` directly, not via this resolver.
    """
    from vemoizer.glossary_layers import _home_glossary_path, _nearest_project_glossary

    files: list[Path] = []
    project = _nearest_project_glossary()
    if project is not None:
        files.append(project)
    home = _home_glossary_path()
    if home.is_file():
        files.append(home)
    return files


def group_durations(label: Path | str) -> list[float]:
    """Per-part decoded-PCM durations for a group's sidecar ``source``.

    A single-part group's label is that part's path; a multi-part label
    (``a.m4a+b.m4a``) resolves each part's filename. Durations come from
    ``pcm_duration_seconds`` — the same measurement the sidecar's
    ``part_offset_s`` (derived from ``part_markers``) is based on, so the
    two stay consistent.
    """
    from vemoizer.ingest import pcm_duration_seconds

    parts = [label] if isinstance(label, Path) else [Path(n) for n in label.split("+")]
    return [pcm_duration_seconds(p) for p in parts]


def group_part_paths(label: Path | str, files: list[Path]) -> list[Path]:
    """The group's per-part source paths, in label order.

    A single-part group's label is that part's path; a multi-part label
    (``a.m4a+b.m4a``) resolves each part's filename against the original
    *files* list (the group is built over ``natural_sort(files)``, so each
    part is a member of *files*). The sidecar's ``source[].path`` carries
    these real on-disk paths (not the decorative marker label) so
    ``render`` can re-apply per-part offsets to the actual files.
    """
    if isinstance(label, Path):
        return [label]
    parts: list[Path] = []
    for name in label.split("+"):
        for f in files:
            if f.name == name:
                parts.append(f)
                break
        else:
            parts.append(Path(name))
    return parts


def sha256_over_files(files: list[str] | list[Path]) -> str | None:
    """sha256 over the concatenated raw bytes of *files*, in order.

    ``None`` when no file can be read (fail-open, matching ``render``'s
    missing-file warning path). Empty *files* yields ``None``.
    """
    if not files:
        return None
    h = hashlib.sha256()
    for f in files:
        try:
            h.update(Path(f).read_bytes())
        except OSError:
            return None
    return h.hexdigest()
