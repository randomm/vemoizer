"""Pure two-track helpers for live call capture (issue #170).

A call is captured as two mono tracks: ``me`` (the owner's microphone) and
``them`` (system audio, the remote side). Everything here is geometry or
config resolution: no audio I/O, no models, no CLI.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from vemoizer.audio_contract import SAMPLE_RATE
from vemoizer.ingest import ingest_audio
from vemoizer.speaker_align import Turn

#: Label given to every remote-side turn when the remote track is one speaker.
REMOTE_LABEL = "Others"

#: Owner label when the stage is called without one (the CLI always resolves it).
DEFAULT_OWNER = "Me"

#: Gaps shorter than this between one side's segments are the same turn.
OWNER_MERGE_GAP_S = 0.5

#: Allowed length mismatch between the two tracks of one call.
STEM_TOLERANCE_S = 2.0

#: Owner speech on the mic stem sits near -23 to -26 dBFS in a headphone
#: call; below this the mic holds leakage or room noise, whatever VAD fires on.
MIC_FLOOR_DBFS = -40.0

#: Mic level more than this far under the system level over the same span is
#: the remote audio leaking into the mic, not the owner speaking.
LEAKAGE_MARGIN_DB = 12.0

#: Frame length of the owner gate. Short enough that a leakage stretch is cut
#: at its boundary, long enough that one frame's RMS is stable.
FRAME_S = 0.1
FRAME_SAMPLES = round(FRAME_S * SAMPLE_RATE)

#: Level reported for silence (and empty spans) instead of -inf.
_SILENCE_DBFS = -200.0

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


def level_dbfs(samples: np.ndarray) -> float:
    """RMS level of a span in dBFS; silence and empty spans are a finite floor."""
    if samples.size == 0:
        return _SILENCE_DBFS
    rms = float(np.sqrt(np.mean(np.square(samples, dtype=np.float64))))
    return 20.0 * math.log10(max(rms, 10.0 ** (_SILENCE_DBFS / 20.0)))


def owner_speech_segments(
    segments: Sequence[Mapping[str, Any]],
    mic: np.ndarray,
    them: np.ndarray,
) -> list[Mapping[str, Any]]:
    """Keep the parts of the mic VAD segments whose level says the owner spoke.

    VAD on the mic stem also fires on headphone leakage of the remote side,
    which then wins equal-overlap ties against the remote turn. The gate runs
    per frame, not per segment: a loud owner part must not carry leakage that
    VAD stretched over a remote reply. Passing frames are joined into runs, so
    a leakage stretch splits the segment; the owner turn merge (0.5 s) runs on
    what is left.
    """
    kept: list[Mapping[str, Any]] = []
    for seg in segments:
        start, end = _to_sample(seg["start"]), _to_sample(seg["end"])
        for lo, hi in _passing_runs(start, end, mic, them):
            kept.append({"start": lo / SAMPLE_RATE, "end": hi / SAMPLE_RATE})
    return kept


def _passing_runs(
    start: int,
    end: int,
    mic: np.ndarray,
    them: np.ndarray,
) -> list[tuple[int, int]]:
    """Contiguous sample ranges in ``[start, end)`` whose frames pass the gate."""
    runs: list[list[int]] = []
    for lo in range(start, end, FRAME_SAMPLES):
        hi = min(lo + FRAME_SAMPLES, end)
        if not _frame_is_owner(mic[lo:hi], them[lo:hi]):
            continue
        if runs and runs[-1][1] == lo:
            runs[-1][1] = hi
        else:
            runs.append([lo, hi])
    return [(lo, hi) for lo, hi in runs]


def _frame_is_owner(mic_frame: np.ndarray, them_frame: np.ndarray) -> bool:
    """Mic above the floor and not far under the system stem over the frame.

    Double talk and a silent remote both pass; mic leakage under remote
    speech fails the margin.
    """
    mic_db = level_dbfs(mic_frame)
    them_db = level_dbfs(them_frame)
    return mic_db >= MIC_FLOOR_DBFS and mic_db - them_db >= -LEAKAGE_MARGIN_DB


def _to_sample(seconds: Any) -> int:
    return round(float(seconds) * SAMPLE_RATE)


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


def configured_owner(
    flag: str | None,
    project_raw: Mapping[str, Any] | None,
    home_raw: Mapping[str, Any] | None,
) -> str | None:
    """The owner the user named, or ``None`` when nothing is configured.

    Order: ``--me`` flag, then project config, then home config. The
    ``[meeting] owner`` key falls through past a project config that omits it.
    """
    for candidate in (
        _clean_name(flag),
        _meeting_owner(project_raw),
        _meeting_owner(home_raw),
    ):
        if candidate is not None:
            return candidate
    return None


def fallback_owner(language: str | None) -> str:
    """The unnamed owner's label: "Minä" for Finnish, "Me" otherwise."""
    return "Minä" if (language or "").lower().startswith("fi") else "Me"


def resolve_owner(
    flag: str | None,
    project_raw: Mapping[str, Any] | None,
    home_raw: Mapping[str, Any] | None,
    language: str | None,
) -> str:
    """Owner label: the configured name, else the language-dependent fallback."""
    named = configured_owner(flag, project_raw, home_raw)
    return named if named is not None else fallback_owner(language)


def ingest_stems(
    tracks: TrackPaths,
    mix_len: int,
    ingest_fn: Callable[..., np.ndarray] = ingest_audio,
) -> tuple[np.ndarray, np.ndarray]:
    """Decode both stems once, zero-padded to one length (the longest).

    Stems are decoded without preprocessing: loudnorm applies to the mix
    only. Raises ``IngestError`` / ``ValueError`` for the caller to fail
    open (see :func:`check_stem_lengths` for the length rule).
    """
    mic = ingest_fn(tracks.me, preprocess=None)
    them = ingest_fn(tracks.them, preprocess=None)
    pads = check_stem_lengths(mix_len, [len(mic), len(them)])
    return _zero_pad(mic, pads[0]), _zero_pad(them, pads[1])


def _zero_pad(samples: np.ndarray, count: int) -> np.ndarray:
    if count == 0:
        return samples
    return np.concatenate([samples, np.zeros(count, dtype=samples.dtype)])


def tracks_error(
    command: str,
    files: Sequence[object],
    speakers: SpeakerCount | None,
) -> str | None:
    """Why a two-track request cannot run, or ``None`` when it can.

    Tracks are a single ``meeting`` recording: the stems pair with exactly
    one mixed file, and a call has at least two participants.
    """
    if command != "meeting":
        return "--mic-track/--system-track apply to the meeting command only"
    if len(files) != 1:
        return "two-track capture takes exactly one mixed recording"
    try:
        remote_speaker_count(speakers)
    except ValueError:
        return "a call has at least two participants (--speakers 2 or more)"
    return None


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
