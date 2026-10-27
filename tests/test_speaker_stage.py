"""Speaker stage: diarization call, progress rows, fail-open warnings (#169)."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from vemoizer import speaker_stage
from vemoizer.diarization import ATTRIBUTION
from vemoizer.speaker_stage import FAILED_WARNING, run_speaker_stage

TURNS = [(0.0, 1.5, "SPEAKER_00"), (1.5, 3.0, "SPEAKER_01")]


@pytest.fixture
def audio() -> np.ndarray:
    return np.zeros(16000, dtype=np.float32)


@pytest.fixture
def diarizer(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    fake = MagicMock(return_value=TURNS)
    monkeypatch.setattr(speaker_stage, "run_diarization_stage", fake)
    return fake


def test_stage_calls_run_diarization_stage_with_mix_audio_exactly(
    diarizer: MagicMock, audio: np.ndarray
) -> None:
    run_speaker_stage(audio, diarize=True, speakers=(2, 4), display=None, tracker=None)

    diarizer.assert_called_once()
    (passed_audio, passed_speakers), kwargs = diarizer.call_args
    assert kwargs == {}
    assert passed_audio is audio
    assert passed_speakers == (2, 4)


def test_attribution_warning_only_when_ran(
    diarizer: MagicMock, audio: np.ndarray
) -> None:
    ran = run_speaker_stage(
        audio, diarize=True, speakers=None, display=None, tracker=None
    )
    assert ran.ran is True
    assert ran.turns == TURNS
    assert ran.warnings == [ATTRIBUTION]

    diarizer.return_value = None
    failed = run_speaker_stage(
        audio, diarize=True, speakers=None, display=None, tracker=None
    )
    assert failed.ran is False
    assert ATTRIBUTION not in failed.warnings

    skipped = run_speaker_stage(
        audio, diarize=False, speakers=None, display=None, tracker=None
    )
    assert skipped.ran is False
    assert skipped.turns is None
    assert skipped.warnings == []
    assert diarizer.call_count == 2


def test_failed_warning_when_diarize_requested_but_failed(
    diarizer: MagicMock, audio: np.ndarray
) -> None:
    diarizer.return_value = None

    stage = run_speaker_stage(
        audio, diarize=True, speakers=None, display=None, tracker=None
    )

    assert stage.ran is False
    assert stage.turns is None
    assert stage.warnings == [FAILED_WARNING]


def test_empty_turns_count_as_ran(diarizer: MagicMock, audio: np.ndarray) -> None:
    diarizer.return_value = []

    stage = run_speaker_stage(
        audio, diarize=True, speakers=None, display=None, tracker=None
    )

    assert stage.ran is True
    assert stage.turns == []
    assert stage.warnings == [ATTRIBUTION]


def test_owns_diarize_progress_row_and_tracker_stage(
    diarizer: MagicMock, audio: np.ndarray
) -> None:
    display = MagicMock()
    tracker = MagicMock()

    run_speaker_stage(
        audio, diarize=True, speakers=None, display=display, tracker=tracker
    )

    tracker.set_stage.assert_called_once_with("diarize")
    display.add_stage.assert_called_once_with("diarize")
    display.finish.assert_called_once_with(display.add_stage.return_value)


def test_skipped_stage_touches_no_progress(
    diarizer: MagicMock, audio: np.ndarray
) -> None:
    display = MagicMock()
    tracker = MagicMock()

    run_speaker_stage(
        audio, diarize=False, speakers=None, display=display, tracker=tracker
    )

    tracker.set_stage.assert_not_called()
    display.add_stage.assert_not_called()
    display.finish.assert_not_called()
