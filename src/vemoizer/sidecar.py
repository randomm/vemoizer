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

* ``options.glossary_sha256`` is the sha256 over the *prompt-term set*
  of the exact glossary files the run used (project layer first, then
  home, or the single explicit ``--glossary`` file), as hashed by
  :func:`prompt_term_set_hash` — the non-correction, non-``@`` lines
  after layer merge, deduped case-insensitively (first-seen spelling
  wins), i.e. the canonical deduplicated prompt-term set that is the
  input to ``glossary_prompt`` before its token-budget truncation.
  ``render`` recomputes the same hash over the current layers for the
  drift warning, so an unchanged prompt-term set never warns (correction
  pairs and ``@`` names are render-safe and never trip it). ``null``
  when no glossary file existed at run time.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

__all__ = [
    "build_sidecar",
    "glossary_layer_files",
    "group_durations",
    "group_part_names",
    "group_part_paths",
    "resolve_run_glossary_files",
    "prompt_term_set_hash",
    "prompt_term_list",
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
      no duration was measured for that part. Measuring it costs one extra
      streamed ffmpeg decode per part (~1.6 s per hour of audio — under 1%
      of a typical run, measured), which is why the measurement stays
      optional and fail-open (a decode failure leaves the key absent).
    * ``options``: ``{command, glossary_files, glossary_sha256}``.
      ``glossary_files`` is the list of files actually used (``[]`` when
      none); ``glossary_sha256`` is the sha256 over their prompt-term set
      (see :func:`prompt_term_set_hash`), or ``None`` when
      ``glossary_files`` is empty. ``preprocess`` (issue #135) is
      recorded present-only: added to the options dict only when the run
      set it (``"loudnorm"``), so ``render`` keeps it; absent (or
      ``None``) otherwise, so the flag-less sidecar stays byte-identical
      to before.
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
        "glossary_sha256": prompt_term_set_hash(files) if files else None,
    }
    # Issue #135: the opt-in preprocess is recorded present-only, so
    # ``render`` keeps it. The value comes from the result dict (the
    # pipeline stashes it there when the flag is set); absent or None
    # when the flag was not set, so the flag-less sidecar is unchanged.
    preprocess = result.get("preprocess")
    if preprocess:
        result["options"]["preprocess"] = preprocess
        result.pop("preprocess", None)

    result.setdefault("speaker_names", {})

    # M6 (issue #107): drop an explicit None duration_s so a failed
    # duration measurement (ffmpeg fail-open) leaves no duration_s key on
    # the sidecar — format_json mirrors duration_s present-only (is not
    # None), so a null value would otherwise land in the JSON and the
    # sidecar → render → md round-trip would render a "Kesto: [00:00:00]"
    # header line that the original run's md did not have. glossary_source
    # is only ever set by the preset seam when a glossary was present, so
    # no None-drop is needed there.
    if result.get("duration_s") is None and "duration_s" in result:
        result.pop("duration_s")

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


def group_part_names(label: Path | str) -> list[str]:
    """The group label's part names, in label order.

    A single-part group's label is that part's path (its name is the
    label's basename); a multi-part label (``a.m4a+b.m4a``) splits on
    ``+``. One seam for the ``a+b`` label grammar: :func:`group_durations`
    and :func:`group_part_paths` derive their part list from here.
    """
    if isinstance(label, Path):
        return [label.name]
    return label.split("+")


def group_part_paths(label: Path | str, files: list[Path]) -> list[Path]:
    """The group's per-part source paths, in label order.

    A single-part group's label *is* that part's path, so the label is
    returned as-is (its directory is part of the identity — never a
    basename lookup). A multi-part label (``a.m4a+b.m4a``) resolves each
    part's filename against the original *files* list (the group is built
    over ``natural_sort(files)``, so each part is a member of *files*),
    falling back to the bare name when no member matches. The sidecar's
    ``source[].path`` carries these real on-disk paths (not the decorative
    marker label) so ``render`` can re-apply per-part offsets to the
    actual files.
    """
    if isinstance(label, Path):
        return [label]
    parts: list[Path] = []
    for name in group_part_names(label):
        for f in files:
            if f.name == name:
                parts.append(f)
                break
        else:
            parts.append(Path(name))
    return parts


def group_durations(paths: list[Path], *, preprocess: str | None = None) -> list[float]:
    """Per-part decoded-PCM durations for a group's sidecar ``source``.

    *paths* are the group's real part paths (:func:`group_part_paths`),
    so each measurement sees the file the run actually decoded (never a
    bare name resolved against the process CWD) and the caller need not
    resolve the label a second time. The same measurement the sidecar's
    ``part_offset_s`` (derived from ``part_markers``) is based on, so
    the two stay consistent. ``preprocess`` (issue #135) is threaded
    through so the duration matches the processed decode (the loudnorm
    gain is linear, so the sample count — and thus the duration — is
    identical either way, but the argv must stay in lock-step).
    """
    from vemoizer.ingest import pcm_duration_seconds

    return [pcm_duration_seconds(p, preprocess=preprocess) for p in paths]


def resolve_run_glossary_files(
    command: str, options_glossary_path: str | None
) -> list[str] | None:
    """The real glossary files a preset run actually read from disk.

    The M5a sidecar's ``options.glossary_files`` must point at files that
    survive the run: ``run_preset`` composes the layered glossary into a
    temp file (deleted in its ``finally``), so the temp path must never be
    stored — only the layer files that composed it (project first, then
    home, via :func:`glossary_layer_files`) or the explicit ``--glossary``
    file (``options_glossary_path``). ``None`` when the run had no glossary
    file at all (so ``options.glossary_sha256`` is ``null``). The layered
    list is also what :func:`vemoizer.render_cli` recomputes the hash
    over, so an untouched glossary renders with no drift warning.
    """
    if options_glossary_path is not None:
        return [str(options_glossary_path)]
    files = glossary_layer_files()
    return [str(p) for p in files] or None


def prompt_term_list(files: list[str] | list[Path]) -> list[str]:
    """The glossary *prompt-term set* for *files* (list order = priority).

    Reads each glossary file in list order (project layer first) and keeps
    the non-correction (no ``=>``), non-``@`` lines — the prompt terms —
    deduped case-insensitively with the first-seen spelling winning
    (project over home, mirroring ``glossary_layers.merge``). This is the
    single derivation that :func:`prompt_term_set_hash` hashes and that
    the run seam (``load_layers`` + ``merge``) produces, so the two
    cannot drift silently.

    Non-regular paths (directories, special files) and missing/unreadable
    files are skipped silently (fail-open, like a missing glossary).
    """
    from vemoizer.glossary import load_glossary

    seen: set[str] = set()
    terms: list[str] = []
    for f in files:
        path = Path(f)
        # Skip non-regular paths up front: a FIFO or special file must
        # not block render's drift check (a read could hang on a FIFO).
        if not path.is_file():
            continue
        try:
            lines = load_glossary(path)
        except (OSError, ValueError):
            # Fail-open: an unreadable file contributes no terms. The render
            # command's _load_corrections prints the user-visible warning for
            # the non-UTF-8 case; the run path has no such warning (the
            # file was already read successfully by load_layers/merge).
            continue
        for line in lines:
            if line.startswith("@"):
                continue
            key = line.lower()
            if key not in seen:
                seen.add(key)
                terms.append(line)
    return terms


def prompt_term_set_hash(files: list[str] | list[Path]) -> str | None:
    """sha256 over the newline-joined :func:`prompt_term_list` of *files*.

    The hash is over the prompt-term set only, so two glossaries that feed
    the same whisper prompt hash equal, regardless of correction pairs,
    ``@`` names, or comment lines.

    The same function hashes both the run's glossary (stored in the
    sidecar's ``options.glossary_sha256``) and the current glossary at
    render time, so a mismatch means the prompt-term set actually changed.

    ``None`` when *files* is empty (fail-open); a non-empty list of
    files yields the hash of the (possibly empty) term set.
    """
    if not files:
        return None
    return hashlib.sha256(
        "\n".join(prompt_term_list(files)).encode("utf-8")
    ).hexdigest()
