"""Island detection and rejection for speaker clip selection (issue #280).

An *island* is a labelled item squeezed between two items of one other
label of the SAME track within :data:`ISLAND_GAP_S` on both sides — almost
always a diarization mislabel that played someone else's voice. Neighbours
are sought only among items of the item's own ``track`` (a missing track is
its own group), so on a two-track call sidecar a genuine reply of the remote
side between two of the owner's segments is never an island.

:func:`islands` finds the island spans per label in a guard list;
:func:`reject_islands` drops a label's candidates that overlap one of its
own islands from the candidate pools of :mod:`vemoizer.speaker_clip_select`,
with a last-resort exemption for a label that has nothing else (issue #202).

Nothing here is allowed to load a model, read a file, or play audio. The
``_Candidate`` import is type-checking only, so this module has no runtime
dependency on :mod:`vemoizer.speaker_clip_select` and no import cycle.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .speaker_clip_select import _Candidate

__all__ = ["ISLAND_GAP_S", "islands", "reject_islands"]

#: How far from a same-labelled same-track neighbour an item may sit and
#: still count as an island of the neighbouring label (issue #280). 3.0 s
#: catches the measured real case: a 1.56 s gap the first 1.5 s draft
#: missed. A genuine short reply wedged between one speaker's turns on the
#: same track is skipped too, which is acceptable because it only affects
#: clip choice, and the last-resort fallback still keeps it when the label
#: has nothing else.
ISLAND_GAP_S = 3.0


def _labelled(p: dict[str, Any]) -> str | None:
    """The speaker label of *p*, or ``None`` when unlabelled.

    A paragraph is labelled iff ``speaker`` is a non-empty string; a
    missing key, ``None``, or ``""`` all count as unlabelled.
    """
    label = p.get("speaker")
    return label if isinstance(label, str) and label else None


def islands(items: list[dict[str, Any]]) -> dict[str, list[tuple[float, float]]]:
    """Label -> half-open intervals of that label's *islands* among *items*.

    An item is an island of its own label (issue #280) when the nearest
    labelled item before it and the nearest labelled item after it OF THE
    SAME TRACK carry the SAME label, that label differs from the item's, and
    both gaps are <= :data:`ISLAND_GAP_S`. Such an item is nearly always a
    diarization mislabel: a single short segment inside another speaker's
    monologue, so the clip would play that other person's voice. Neighbours
    are sought only within the item's own ``track`` group (a missing track
    is its own group), so on a two-track call sidecar a genuine reply of
    the remote side between two of the owner's segments is never an island.
    The first and last items are never islands (no neighbour on one side);
    unlabelled items are skipped — neither neighbours nor islands.
    """
    groups: dict[Any, list[tuple[str, float, float]]] = {}
    for p in sorted(items, key=lambda p: float(p.get("start", 0.0))):
        label = _labelled(p)
        if label is not None:
            group = groups.setdefault(p.get("track"), [])
            group.append((label, float(p.get("start", 0.0)), float(p.get("end", 0.0))))
    found: dict[str, list[tuple[float, float]]] = {}
    for labelled in groups.values():
        for prev, item, nxt in zip(labelled, labelled[1:], labelled[2:], strict=False):
            prev_label, _, prev_end = prev
            label, start, end = item
            next_label, next_start, _ = nxt
            if prev_label != next_label or label == prev_label:
                continue
            if start - prev_end <= ISLAND_GAP_S and next_start - end <= ISLAND_GAP_S:
                found.setdefault(label, []).append((start, end))
    return found


def reject_islands(
    pools: list[dict[str, list[_Candidate]]],
    islands: dict[str, list[tuple[float, float]]],
) -> None:
    """Drop, from every pool in *pools*, a label's candidates overlapping an
    island of that same label (issue #280).

    Pools are filtered together so a label's paragraph candidate is not
    resurrected by its segment pool or the other way round: a label with at
    least one non-island candidate anywhere loses all of its islands. A label
    whose candidates are all islands keeps them, so every label with speech
    still gets a clip (issue #202). *pools* is mutated in place.
    """

    def overlaps_island(candidate: _Candidate, label: str) -> bool:
        return any(
            candidate.start < e and s < candidate.end for s, e in islands.get(label, [])
        )

    survivors: set[str] = set()
    for pool in pools:
        for label, candidates in pool.items():
            if any(not overlaps_island(c, label) for c in candidates):
                survivors.add(label)
    for pool in pools:
        for label in pool:
            if label in survivors:
                pool[label] = [c for c in pool[label] if not overlaps_island(c, label)]
