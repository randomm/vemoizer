"""Owner-quiz probe metrics for the accept bench (issue #243, epic #220).

The owner answers short multiple-choice questions on 3-5 s clips ("who is
speaking?", "which term was said?"). Each answer is a probe: sparse ground
truth at one time window of the excerpt. These are pure functions of plain
segment dicts (the same shape as :mod:`vemoizer.accept_metrics`) and replace
the transcript-based speaker and glossary keys when a reference has probes.

Only the owner can say who spoke, so these metrics need no reference text.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from .accept_bundle import OTHER, OVERLAP, UNCLEAR
from .accept_metrics import (
    count_token_run,
    greedy_mapping,
    overlap,
    speaker_label,
)
from .textnorm import textnorm

Segment = dict[str, Any]

# Answers that say something other than "this name spoke": no name to score.
UNSCORED = frozenset({OTHER, OVERLAP, UNCLEAR})


def _first(name: str) -> str:
    # Names compare case-insensitively on the first word ("Eero Virtanen" ~ "eero").
    words = name.split()
    return words[0].casefold() if words else ""


def _hyp_label_at(hyp: list[Segment], start: float, end: float) -> str | None:
    """Casefolded label of the hypothesis segment with the largest overlap with
    [start, end), or ``None`` when nothing overlaps or the winner is unlabelled.

    WHY largest overlap: a probe clip can straddle a turn change, and the
    speaker who holds most of the clip is the one the owner heard.
    """
    window = {"start": start, "end": end}
    best: Segment | None = None
    best_seconds = 0.0
    for seg in hyp:
        seconds = overlap(seg, window)
        if seconds > best_seconds:
            best, best_seconds = seg, seconds
    if best is None:
        return None
    return speaker_label(best) or None


def speaker_probe_metrics(
    hyp: list[Segment], probes: list[dict[str, Any]]
) -> dict[str, Any]:
    """Name accuracy, mapped accuracy, split and merge counts from speaker probes.

    A probe whose answer is ``other``, ``overlap`` or ``unclear`` is unscored.
    A scored probe with no overlapping or labelled hypothesis segment is a
    wrong name. Accuracies are ``None`` when no probe is scorable: no signal,
    never a vacuous 1.0 that would hide a missing quiz.
    """
    scored = [
        (_first(p["answer"]), _hyp_label_at(hyp, p["start"], p["end"]))
        for p in probes
        if p["answer"] not in UNSCORED
    ]
    total = len(scored)
    if total == 0:
        return {
            "speaker_name_acc": None,
            "speaker_mapped_acc": None,
            "split_count": 0,
            "merge_count": 0,
        }
    cells = Counter(
        (name, _first(label)) for name, label in scored if label is not None
    )
    mapping = greedy_mapping({cell: float(n) for cell, n in cells.items()})
    name_hits = sum(1 for n, lb in scored if lb is not None and _first(lb) == n)
    mapped_hits = sum(
        1 for n, lb in scored if lb is not None and mapping.get(_first(lb)) == n
    )
    held: dict[str, set[str]] = {}  # answer name -> hypothesis labels heard
    cover: dict[str, set[str]] = {}  # hypothesis label -> answer names heard
    for name, label in scored:
        if label is None:
            continue
        held.setdefault(name, set()).add(label)
        cover.setdefault(label, set()).add(name)
    return {
        "speaker_name_acc": name_hits / total,
        # Backstop: the mapping is a convenience, never a penalty.
        "speaker_mapped_acc": max(mapped_hits, name_hits) / total,
        "split_count": sum(len(labels) - 1 for labels in held.values()),
        "merge_count": sum(len(names) - 1 for names in cover.values()),
    }


#: Slack around a term probe: hypothesis words are timed by spreading them
#: evenly over their segment, so a word said inside the window can land up to
#: about a second outside it.
WINDOW_PAD_S = 1.0


def _window_tokens(hyp: list[Segment], start: float, end: float) -> list[str]:
    """Normalised hypothesis words timed inside ``[start, end]`` (padded).

    The hypothesis is paragraph-level (a same-speaker run can span a
    minute), so a whole overlapping paragraph is NOT the window: each word
    gets an evenly spread time over its segment and only words landing in
    the padded window count.
    """
    lo, hi = start - WINDOW_PAD_S, end + WINDOW_PAD_S
    tokens: list[str] = []
    for seg in sorted(hyp, key=lambda s: float(s.get("start", 0.0))):
        s0, s1 = float(seg.get("start", 0.0)), float(seg.get("end", 0.0))
        if s1 < lo or s0 > hi:
            continue
        words = textnorm(str(seg.get("text", ""))).split()
        step = (s1 - s0) / len(words) if words else 0.0
        tokens += [w for k, w in enumerate(words) if lo <= s0 + (k + 0.5) * step <= hi]
    return tokens


def _mask_run(tokens: list[str], run: list[str]) -> list[str]:
    """*tokens* with every occurrence of *run* replaced by a gap marker, so a
    shorter option nested in the heard answer ("kontti" in "IBC-kontti") is
    never counted against it, and no run can join across the gap."""
    if not run:
        return tokens
    out: list[str] = []
    i = 0
    while i < len(tokens):
        if tokens[i : i + len(run)] == run:
            out.append("\x00")
            i += len(run)
        else:
            out.append(tokens[i])
            i += 1
    return out


def glossary_probe_insertions(
    hyp: list[Segment], term_probes: list[dict[str, Any]]
) -> int | None:
    """Listed glossary terms the hypothesis said in a probe window though the
    owner did not hear them, or ``None`` when there are no term probes.

    Every listed option other than the answer counts, so a hypothesis that
    writes the wrong option is penalised once per such token. An ``other``
    answer ("Something else") makes every listed option wrong. ``unclear``
    is unscored.
    """
    if not term_probes:
        return None
    total = 0
    for probe in term_probes:
        if probe["answer"] == UNCLEAR:
            continue
        answer = textnorm(probe["answer"]).split()
        tokens = _mask_run(_window_tokens(hyp, probe["start"], probe["end"]), answer)
        for option in probe["options"]:
            option_tokens = textnorm(option).split()
            if option_tokens and option_tokens != answer:
                total += count_token_run(tokens, option_tokens)
    return total


def probe_metrics(
    hyp: list[Segment],
    probes: list[dict[str, Any]],
    term_probes: list[dict[str, Any]],
) -> dict[str, Any]:
    """The five keys a reference with probes reports in place of the
    transcript-based speaker and glossary keys."""
    out: dict[str, Any] = speaker_probe_metrics(hyp, probes)
    out["glossary_insertions"] = glossary_probe_insertions(hyp, term_probes)
    return out
