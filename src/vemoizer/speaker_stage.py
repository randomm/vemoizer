"""Speaker stage: optional diarization with its progress row and warnings.

Extracted from ``pipeline.py`` (issue #169) so the orchestrator only chains
stages. The stage owns the ``diarize`` progress row, the timing log, and the
CC-BY-4.0 attribution that the gated pyannote weights require whenever they
ran. Fail-open: a failure yields ``turns=None`` and a warning, never an abort.

Two-track calls (issue #171): with ``tracks`` the labels come from the two
stems rather than the mix. The owner's turns are VAD on the mic stem; the
remote turns are pyannote on the system stem with the remote count (total
minus the owner), or one undifferentiated remote label when that count is 1.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Mapping
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

import numpy as np

from .diarization import ATTRIBUTION, SpeakerCount, run_diarization_stage
from .ingest import ingest_audio
from .progress import ProgressDisplay, format_duration
from .tracks import (
    DEFAULT_OWNER,
    TrackPaths,
    ingest_stems,
    merge_track_turns,
    owner_speech_segments,
    owner_turns,
    remote_speaker_count,
    samples_to_seconds,
    single_remote_turns,
)
from .vad import load_model as load_vad_model
from .vad import vad_segments

if TYPE_CHECKING:
    from .preset_interrupt import InterruptTracker

logger = logging.getLogger(__name__)

#: Shown when diarization was requested but produced no turns (issue #78).
FAILED_WARNING = "diarization failed; continuing without speaker labels"

#: Shown when a two-track call falls back to labelling the mixed recording.
STEM_WARNING = "two-track stems unusable; labelling from the mixed recording"


@dataclass(frozen=True)
class SpeakerStage:
    """Outcome of the speaker stage.

    ``turns`` is ``None`` when diarization was skipped or failed, and ``[]``
    when it ran but found no speakers; the two are distinct to callers.
    ``trusted`` names the labels whose attribution is never smoothed away
    (the owner's mic track); it is empty outside two-track calls.
    """

    turns: list[tuple[float, float, str]] | None
    ran: bool
    warnings: list[str]
    trusted: frozenset[str] = frozenset()


def run_speaker_stage(
    audio: np.ndarray,
    *,
    diarize: bool,
    speakers: SpeakerCount | None,
    display: ProgressDisplay | None,
    tracker: InterruptTracker | None,
    tracks: TrackPaths | None = None,
    owner: str | None = None,
) -> SpeakerStage:
    """Run diarization when requested and report what the result should carry."""
    if tracks is not None:
        return _run_two_track(
            audio,
            diarize=diarize,
            speakers=speakers,
            display=display,
            tracker=tracker,
            tracks=tracks,
            owner=owner or DEFAULT_OWNER,
        )
    if not diarize:
        return SpeakerStage(turns=None, ran=False, warnings=[])
    turns = _diarize(audio, speakers, display, tracker)
    if turns is None:
        return SpeakerStage(turns=None, ran=False, warnings=[FAILED_WARNING])
    return SpeakerStage(turns=turns, ran=True, warnings=[ATTRIBUTION])


def _diarize(
    audio: np.ndarray,
    speakers: SpeakerCount | None,
    display: ProgressDisplay | None,
    tracker: InterruptTracker | None,
) -> list[tuple[float, float, str]] | None:
    """One pyannote run under the ``diarize`` progress row; ``None`` on failure."""
    logger.info("diarization: starting")
    started = time.monotonic()
    if tracker is not None:
        tracker.set_stage("diarize")
    task = display.add_stage("diarize") if display is not None else None
    turns = run_diarization_stage(audio, speakers)
    if task is not None and display is not None:
        display.finish(task)
    logger.info(
        "diarization: %s speaker segments in %s",
        len(turns) if turns is not None else "no",
        format_duration(time.monotonic() - started),
    )
    return turns


def _run_two_track(
    audio: np.ndarray,
    *,
    diarize: bool,
    speakers: SpeakerCount | None,
    display: ProgressDisplay | None,
    tracker: InterruptTracker | None,
    tracks: TrackPaths,
    owner: str,
) -> SpeakerStage:
    """Labels from the two stems; fail-open to the mix on any stem problem."""
    try:
        mic, them = ingest_stems(tracks, len(audio), ingest_fn=ingest_audio)
        vad_model = load_vad_model()
        mic_speech = owner_speech_segments(_speech_seconds(mic, vad_model), mic, them)
        them_speech = _speech_seconds(them, vad_model)
        remote_spec = remote_speaker_count(speakers) if diarize else 1
    except Exception as e:  # noqa: BLE001 - fail-open stage boundary
        logger.warning("two-track stems unusable, labelling the mix: %s", e)
        plain = run_speaker_stage(
            audio,
            diarize=diarize,
            speakers=speakers,
            display=display,
            tracker=tracker,
        )
        return replace(plain, warnings=[*plain.warnings, STEM_WARNING])

    warnings: list[str] = []
    if remote_spec == 1:
        remote = single_remote_turns(them_speech)
    else:
        remote_turns = _diarize(them, remote_spec, display, tracker)
        # [] means pyannote found no speakers; keeping the owner-only turns
        # would label every remote word with the owner, so treat it as failure.
        if not remote_turns:
            remote = single_remote_turns(them_speech)
            warnings.append(FAILED_WARNING)
        else:
            remote = remote_turns
            warnings.append(ATTRIBUTION)
    turns = merge_track_turns(owner_turns(mic_speech, owner), remote)
    return SpeakerStage(
        turns=turns,
        ran=True,
        warnings=warnings,
        trusted=frozenset({owner}),
    )


def _speech_seconds(stem: np.ndarray, model: Any) -> list[Mapping[str, float]]:
    """VAD speech spans of one stem as ``{"start", "end"}`` seconds."""
    return [
        {
            "start": samples_to_seconds(seg.start),
            "end": samples_to_seconds(seg.end),
        }
        for seg in vad_segments(stem, model)
    ]
