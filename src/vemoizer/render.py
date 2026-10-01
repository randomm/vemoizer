"""Model-free re-render of a stored meeting/memo sidecar (issue #89, M5a).

``render_markdown`` takes the ``.json`` a meeting/memo run wrote (a
"sidecar": transcript keys plus the ``notes``/``source``/``options``/
``speaker_names`` M5a keys) and re-applies glossary correction pairs and
speaker names to it, then hands the result to the existing
``format_md`` — so renaming a speaker or adding a correction pair never
requires a re-transcribe. This module is PURE: it must never pull in a
model dependency (mlx, pyannote, torch), so it works on a machine without
them.

Speaker names are applied by REWRITING the ``speaker`` fields of
paragraphs/segments and the flat ``OWNER: item`` action-item strings
BEFORE ``format_md`` runs — never by post-render substitution, so the
rendered ``[LABEL]`` prefixes and ``- [ ] OWNER: item`` bullets carry the
names. When several labels are mapped to the same name, the EARLIEST
label in time order (first paragraph/segment carrying that speaker)
becomes canonical and every other label is rewritten to it.
"""

from __future__ import annotations

import copy
import re
from typing import Any

from vemoizer.glossary import apply_corrections, apply_corrections_to_notes

__all__ = ["render_markdown"]


def _owner_prefix(label: str) -> re.Pattern[str]:
    """Whole-word match on ``LABEL: `` at the start of an action item.

    ``\\b`` after the label is what keeps ``SPEAKER_1`` from matching the
    ``SPEAKER_12`` prefix of ``SPEAKER_12: item`` (``_`` is a word
    character, so ``_12`` extends the word and the boundary fails).
    """
    return re.compile(rf"^{re.escape(label)}\s*:\s")


def _apply_names_to_items(items: list[str], rename: dict[str, str]) -> list[str]:
    """Rewrite the ``OWNER: `` prefix of flat action-item strings.

    The owner prefix is a string, not a structured field (see
    ``notes._ground_action_items``), so this is prefix surgery: the label
    is matched whole-word at the item start, followed by the ``:``
    separator. Items without a matching owner are left untouched.
    """
    renamed: list[str] = []
    for item in items:
        out = item
        for old, new in rename.items():
            out = _owner_prefix(old).sub(f"{new}: ", out)
        renamed.append(out)
    return renamed


def _render_dict(
    sidecar: dict[str, Any],
    *,
    corrections: dict[str, str],
    speaker_names: dict[str, str],
) -> dict[str, Any]:
    """Copy *sidecar* with corrections and speaker names applied."""
    out = copy.deepcopy(sidecar)

    # Corrections first: names applied to owner prefixes must match the
    # post-correction labels the notes actually carry.
    notes = out.get("notes")
    if isinstance(notes, dict) and notes:
        out["notes"] = apply_corrections_to_notes(notes, corrections)

    paragraphs = out.get("paragraphs")
    if isinstance(paragraphs, list) and paragraphs:
        # ``apply_corrections`` maps over dict entries (``para.get``) and
        # ``format_md`` does the same — a corrupted sidecar can carry
        # non-dict entries, so drop them here (consistent with ``_blocks``)
        # rather than raising ``AttributeError`` downstream.
        out["paragraphs"] = [p for p in paragraphs if isinstance(p, dict)] or None
        if out["paragraphs"]:
            out["paragraphs"] = apply_corrections(out["paragraphs"], corrections)

    # Speaker names: ``--name`` rewrites a label to a display name; the
    # stored sidecar ``speaker_names`` (persisted by a previous render) is
    # merged in so a plain re-render stays stable. The two maps compose
    # (merge rule over all named labels), then the effective rename is
    # applied to the structured ``speaker`` fields.
    stored = out.get("speaker_names")
    sidecar_names = stored if isinstance(stored, dict) else {}
    effective_names = {**sidecar_names, **speaker_names}
    rename = _resolve_rename(out, effective_names)
    for key in ("paragraphs", "segments"):
        blocks = out.get(key)
        if not isinstance(blocks, list):
            continue
        for block in blocks:
            if not isinstance(block, dict):
                continue
            label = block.get("speaker")
            if label is not None and str(label) in rename:
                block["speaker"] = rename[str(label)]

    # Names in the notes: owner prefixes on the flat action items.
    if isinstance(out.get("notes"), dict) and rename:
        out["notes"] = _apply_names_to_notes(out["notes"], rename)
    return out


def _resolve_rename(result: dict[str, Any], names: dict[str, str]) -> dict[str, str]:
    """Effective ``old label -> new value`` map, with the merge rule.

    ``names`` maps speaker labels to display names (``--name`` values plus
    the sidecar's stored ``speaker_names``). Every label mapped to the same
    target name is rewritten to the EARLIEST label (by first paragraph/
    segment start) that carries that name, so two labels named the same
    person become one speaker in the rendered output. The canonical label
    of a merge group carries the given name (it IS that person); every
    other label in the group is rewritten to the canonical LABEL, so the
    rendered output shows one speaker rather than two names for the same
    person. Single-label names map label → name directly.
    """
    if not names:
        return {}
    first_seen: dict[str, float] = {}
    for para in _blocks(result):
        label = para.get("speaker")
        if label is None:
            continue
        label = str(label)
        name = names.get(label)
        if name is None:
            continue
        start = _block_start(para)
        if label not in first_seen or start < first_seen[label]:
            first_seen[label] = start

    # Merge rule: labels sharing a target name collapse to the earliest
    # label in time order; that canonical label keeps the given name, and
    # every other label in the group is rewritten to the canonical LABEL
    # (so one speaker remains in the rendered output).
    rename: dict[str, str] = {}
    for label, name in names.items():
        siblings = [lb for lb in names if names[lb] == name]
        if len(siblings) == 1:
            rename[label] = name
            continue
        canonical = min(siblings, key=lambda lb: first_seen.get(lb, float("inf")))
        rename[label] = name if label == canonical else canonical
    return rename


def _blocks(result: dict[str, Any]) -> list[dict[str, Any]]:
    """Paragraphs, else segments — the structures carrying ``speaker``."""
    for key in ("paragraphs", "segments"):
        blocks = result.get(key)
        if isinstance(blocks, list) and blocks:
            return [b for b in blocks if isinstance(b, dict)]
    return []


def _block_start(block: dict[str, Any]) -> float:
    try:
        return float(block.get("start", 0.0))
    except (TypeError, ValueError):
        return 0.0


def _apply_names_to_notes(
    notes: dict[str, Any], rename: dict[str, str]
) -> dict[str, Any]:
    """Copy *notes* with owner prefixes rewritten on the action items."""
    out = dict(notes)
    items = out.get("action_items")
    if isinstance(items, list):
        out["action_items"] = _apply_names_to_items(
            [str(item) for item in items], rename
        )
    return out


def render_markdown(
    sidecar: dict,
    *,
    corrections: dict[str, str],
    speaker_names: dict[str, str],
) -> str:
    """Re-render a stored sidecar as Markdown (pure, model-free).

    The pure core; the ``--name`` CLI values and the sidecar's stored
    ``speaker_names`` are merged inside :func:`_render_dict` (explicit
    values win), so a plain re-render of an already-named sidecar stays
    stable.

    Re-applies *corrections* (whole-word ``wrong => right`` pairs, via
    ``apply_corrections`` / ``apply_corrections_to_notes``) and the
    *speaker_names* label rewrites to the sidecar's paragraphs, segments
    and notes (including flat action-item owner prefixes), then calls
    ``format_md``. Notes are re-emitted from the sidecar verbatim apart
    from those rewrites; the LLM is never invoked. Old sidecars missing
    the M5a keys render unchanged.
    """
    from vemoizer.output.markdown import format_md

    return format_md(
        _render_dict(sidecar, corrections=corrections, speaker_names=speaker_names)
    )
