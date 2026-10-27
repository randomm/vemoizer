"""Pure two-track helpers for live call capture (issue #170).

A call is captured as two mono tracks: ``me`` (the owner's microphone) and
``them`` (system audio, the remote side). Everything here is geometry or
config resolution: no audio I/O, no models, no CLI.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from vemoizer.audio_contract import SAMPLE_RATE
from vemoizer.speaker_align import Turn

#: Label given to every remote-side turn when the remote track is one speaker.
REMOTE_LABEL = "Others"

#: Gaps shorter than this between one side's segments are the same turn.
OWNER_MERGE_GAP_S = 0.5

#: Allowed length mismatch between the two tracks of one call.
STEM_TOLERANCE_S = 2.0

#: Speaker-count spec, as pyannote takes it: exact count or (min, max).
SpeakerCount = int | tuple[int, int]


@dataclass(frozen=True)
class TrackPaths:
    """The two per-call track files."""

    me: Path
    them: Path


def samples_to_seconds(samples: int) -> float:
    """Sample count at the audio contract rate to seconds."""
    return samples / SAMPLE_RATE


def _relabel_merged(
    segments: Sequence[Mapping[str, Any]],
    label: str,
    merge_gap_s: float,
) -> list[Turn]:
    """Give every segment *label*, then fuse neighbours closer than the gap."""
    turns: list[Turn] = []
    for seg in sorted(segments, key=lambda s: float(s["start"])):
        start, end = float(seg["start"]), float(seg["end"])
        if turns and start - turns[-1][1] < merge_gap_s:
            turns[-1] = (turns[-1][0], max(turns[-1][1], end), label)
        else:
            turns.append((start, end, label))
    return turns


def owner_turns(
    segments: Sequence[Mapping[str, Any]],
    label: str,
    merge_gap_s: float = OWNER_MERGE_GAP_S,
) -> list[Turn]:
    """Owner-track segments as labelled turns, short gaps merged."""
    return _relabel_merged(segments, label, merge_gap_s)


def single_remote_turns(segments: Sequence[Mapping[str, Any]]) -> list[Turn]:
    """Remote-track segments as turns of one undifferentiated speaker."""
    return _relabel_merged(segments, REMOTE_LABEL, OWNER_MERGE_GAP_S)


def remote_speaker_count(spec: SpeakerCount | None) -> SpeakerCount | None:
    """Remote speakers implied by a total count, the owner excluded.

    Input and output use pyannote's shape: an exact int, or a (min, max)
    pair. The owner is one of the total, so the remote count is one less.
    Raises ``ValueError`` when the total leaves no room for a remote speaker.
    """
    if spec is None:
        return None
    lo, hi = (spec, spec) if isinstance(spec, int) else spec
    if lo < 2 or hi < lo:
        raise ValueError(f"speaker count {spec!r} leaves no remote speaker")
    remote_lo, remote_hi = lo - 1, hi - 1
    return remote_lo if remote_lo == remote_hi else (remote_lo, remote_hi)


def merge_track_turns(owner: Sequence[Turn], remote: Sequence[Turn]) -> list[Turn]:
    """Both tracks' turns on one timeline, ordered by start time."""
    return sorted([*owner, *remote], key=lambda t: (t[0], t[1]))


def _clean_name(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    return value.strip() or None


def _meeting_owner(config: Mapping[str, Any] | None) -> str | None:
    if not config:
        return None
    meeting = config.get("meeting")
    if not isinstance(meeting, Mapping):
        return None
    return _clean_name(meeting.get("owner"))


def resolve_owner(
    flag: str | None,
    project_raw: Mapping[str, Any] | None,
    home_raw: Mapping[str, Any] | None,
    language: str | None,
) -> str:
    """Owner label: ``--me`` flag, then project config, then home config.

    The ``[meeting] owner`` key falls through past a project config that
    omits it. Without any name, the language-dependent fallback is used.
    """
    for candidate in (
        _clean_name(flag),
        _meeting_owner(project_raw),
        _meeting_owner(home_raw),
    ):
        if candidate is not None:
            return candidate
    return "Minä" if (language or "").lower().startswith("fi") else "Me"


def check_stem_lengths(
    mix_len: int,
    stem_lens: Sequence[int],
    tolerance_s: float = STEM_TOLERANCE_S,
) -> list[int]:
    """Zero-padding per stem so every track reaches the longest one.

    The mix length must be within *tolerance_s* of the longest stem; a larger
    mismatch means the tracks are not from the same call and raises
    ``ValueError``.
    """
    longest = max(stem_lens)
    mismatch = abs(mix_len - longest)
    if mismatch > tolerance_s * SAMPLE_RATE:
        raise ValueError(
            f"track lengths differ by {samples_to_seconds(mismatch):.2f}s "
            f"(limit {tolerance_s}s)"
        )
    return [longest - n for n in stem_lens]
