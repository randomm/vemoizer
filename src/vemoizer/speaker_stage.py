"""Speaker stage: optional diarization with its progress row and warnings.

Extracted from ``pipeline.py`` (issue #169) so the orchestrator only chains
stages. The stage owns the ``diarize`` progress row, the timing log, and the
CC-BY-4.0 attribution that the gated pyannote weights require whenever they
ran. Fail-open: a failure yields ``turns=None`` and a warning, never an abort.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from .diarization import ATTRIBUTION, SpeakerCount, run_diarization_stage
from .progress import ProgressDisplay, format_duration

if TYPE_CHECKING:
    from .preset_interrupt import InterruptTracker

logger = logging.getLogger(__name__)

#: Shown when diarization was requested but produced no turns (issue #78).
FAILED_WARNING = "diarization failed; continuing without speaker labels"


@dataclass(frozen=True)
class SpeakerStage:
    """Outcome of the speaker stage.

    ``turns`` is ``None`` when diarization was skipped or failed, and ``[]``
    when it ran but found no speakers; the two are distinct to callers.
    """

    turns: list[tuple[float, float, str]] | None
    ran: bool
    warnings: list[str]


def run_speaker_stage(
    audio: np.ndarray,
    *,
    diarize: bool,
    speakers: SpeakerCount | None,
    display: ProgressDisplay | None,
    tracker: InterruptTracker | None,
) -> SpeakerStage:
    """Run diarization when requested and report what the result should carry."""
    if not diarize:
        return SpeakerStage(turns=None, ran=False, warnings=[])
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
    if turns is None:
        return SpeakerStage(turns=None, ran=False, warnings=[FAILED_WARNING])
    return SpeakerStage(turns=turns, ran=True, warnings=[ATTRIBUTION])
