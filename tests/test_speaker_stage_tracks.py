"""Two-track speaker stage: owner turns from the mic stem, remote from the
system stem (issue #171). Stems, VAD and pyannote are faked; no models."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from vemoizer import speaker_stage
from vemoizer.ingest import IngestError
from vemoizer.speaker_stage import FAILED_WARNING, run_speaker_stage
from vemoizer.tracks import REMOTE_LABEL, TrackPaths
from vemoizer.vad import SpeechSegment

SR = 16_000
MIC, THEM = 1.0, 2.0
PYANNOTE_TURNS = [(0.0, 1.0, "SPEAKER_00"), (1.0, 2.0, "SPEAKER_01")]
TRACKS = TrackPaths(me=Path("mic.wav"), them=Path("them.wav"))
OWNER_TURN = (0.0, 1.0, "Janni")
REMOTE_TURN = (2.0, 3.0, REMOTE_LABEL)


def _ingest(path: Path, *, preprocess: str | None = None) -> np.ndarray:
    value = MIC if path.name == "mic.wav" else THEM
    return np.full(SR * 4, value, dtype=np.float32)


@pytest.fixture
def stems(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(speaker_stage, "ingest_audio", _ingest)
    monkeypatch.setattr(speaker_stage, "load_vad_model", lambda: None)

    def fake_vad(audio, model, **kwargs):
        if float(audio[0]) == MIC:
            return [SpeechSegment(start=0, end=SR)]  # owner speaks 0-1 s
        return [SpeechSegment(start=SR * 2, end=SR * 3)]  # remote 2-3 s

    monkeypatch.setattr(speaker_stage, "vad_segments", fake_vad)


@pytest.fixture
def diarizer(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    fake = MagicMock(return_value=PYANNOTE_TURNS)
    monkeypatch.setattr(speaker_stage, "run_diarization_stage", fake)
    return fake


@pytest.fixture
def mix() -> np.ndarray:
    return np.zeros(SR * 4, dtype=np.float32)


def _run(mix: np.ndarray, **kwargs):
    defaults = {"diarize": True, "speakers": (2, 6), "display": None}
    defaults.update(kwargs)
    return run_speaker_stage(
        mix, tracker=None, tracks=TRACKS, owner="Janni", **defaults
    )


def test_pyannote_gets_the_them_stem_and_remote_count(
    stems, diarizer: MagicMock, mix: np.ndarray
) -> None:
    _run(mix, speakers=(2, 6))

    diarizer.assert_called_once()
    (passed_audio, passed_speakers), _ = diarizer.call_args
    assert float(passed_audio[0]) == THEM
    assert passed_speakers == (1, 5)


def test_one_to_one_call_never_runs_pyannote(
    stems, diarizer: MagicMock, mix: np.ndarray
) -> None:
    stage = _run(mix, speakers=2)

    diarizer.assert_not_called()
    assert stage.turns == [OWNER_TURN, REMOTE_TURN]


def test_pyannote_failure_keeps_owner_and_one_remote_label(
    stems, diarizer: MagicMock, mix: np.ndarray
) -> None:
    diarizer.return_value = None

    stage = _run(mix)

    assert stage.turns == [OWNER_TURN, REMOTE_TURN]
    assert stage.warnings == [FAILED_WARNING]


def test_pyannote_remote_turns_replace_the_single_label(
    stems, diarizer: MagicMock, mix: np.ndarray
) -> None:
    stage = _run(mix)

    assert stage.turns == [
        (0.0, 1.0, "Janni"),
        (0.0, 1.0, "SPEAKER_00"),
        (1.0, 2.0, "SPEAKER_01"),
    ]
    assert REMOTE_LABEL not in {t[2] for t in stage.turns}


def test_no_diarize_with_tracks_still_splits_by_vad(
    stems, diarizer: MagicMock, mix: np.ndarray
) -> None:
    stage = _run(mix, diarize=False)

    diarizer.assert_not_called()
    assert stage.turns == [OWNER_TURN, REMOTE_TURN]
    assert stage.warnings == []


def test_trusted_set_is_the_owner_label(
    stems, diarizer: MagicMock, mix: np.ndarray
) -> None:
    stage = _run(mix)

    assert stage.trusted == frozenset({"Janni"})
    assert stage.ran is True


def test_stem_ingest_failure_warns_and_diarizes_the_mix(
    monkeypatch: pytest.MonkeyPatch, diarizer: MagicMock, mix: np.ndarray
) -> None:
    def boom(path, *, preprocess=None):
        raise IngestError("ffmpeg failed")

    monkeypatch.setattr(speaker_stage, "ingest_audio", boom)

    stage = _run(mix, speakers=(2, 6))

    diarizer.assert_called_once()
    (passed_audio, passed_speakers), _ = diarizer.call_args
    assert passed_audio is mix
    assert passed_speakers == (2, 6)
    assert stage.trusted == frozenset()
    assert any("two-track" in w for w in stage.warnings)


def test_length_mismatch_falls_back_to_the_mix(
    monkeypatch: pytest.MonkeyPatch, diarizer: MagicMock, mix: np.ndarray
) -> None:
    def long_stem(path, *, preprocess=None):
        return np.zeros(SR * 30, dtype=np.float32)

    monkeypatch.setattr(speaker_stage, "ingest_audio", long_stem)

    stage = _run(mix, speakers=(2, 6))

    (passed_audio, _), _ = diarizer.call_args
    assert passed_audio is mix
    assert any("two-track" in w for w in stage.warnings)


def test_pyannote_no_speakers_keeps_remote_speech_remote(
    stems, diarizer: MagicMock, mix: np.ndarray
) -> None:
    diarizer.return_value = []

    stage = _run(mix)

    assert stage.turns == [OWNER_TURN, REMOTE_TURN]
    assert stage.warnings == [FAILED_WARNING]


def test_mic_leakage_under_remote_speech_is_not_owner_speech(
    monkeypatch: pytest.MonkeyPatch, mix: np.ndarray
) -> None:
    """Headphone leakage makes mic VAD fire over remote speech (Teams, 16 s)."""

    def ingest(path: Path, *, preprocess: str | None = None) -> np.ndarray:
        level = 0.0035 if path.name == "mic.wav" else 0.09  # -49 / -21 dBFS
        return np.full(SR * 4, level, dtype=np.float32)

    monkeypatch.setattr(speaker_stage, "ingest_audio", ingest)
    monkeypatch.setattr(speaker_stage, "load_vad_model", lambda: None)
    monkeypatch.setattr(
        speaker_stage,
        "vad_segments",
        lambda audio, model, **kwargs: [SpeechSegment(start=SR * 2, end=SR * 3)],
    )

    stage = _run(mix, speakers=2)

    assert stage.turns == [REMOTE_TURN]


def test_leakage_stretching_a_vad_segment_over_remote_speech_stays_remote(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Owner answers at 0-2 s, then mic leakage over the remote reply 2-12 s
    with no VAD pause: the owner turn must stop at 2 s (no 'Janni' on remote)."""

    def ingest(path: Path, *, preprocess: str | None = None) -> np.ndarray:
        if path.name == "mic.wav":
            owner = np.full(SR * 2, 10 ** (-24 / 20), dtype=np.float32)
            leak = np.full(SR * 10, 10 ** (-45 / 20), dtype=np.float32)
            return np.concatenate([owner, leak])
        silent = np.zeros(SR * 2, dtype=np.float32)
        remote = np.full(SR * 10, 10 ** (-21 / 20), dtype=np.float32)
        return np.concatenate([silent, remote])

    def fake_vad(audio, model, **kwargs):
        if float(audio[0]) != 0.0:  # mic: one VAD run 0-12 s, leakage included
            return [SpeechSegment(start=0, end=SR * 12)]
        return [SpeechSegment(start=SR * 2, end=SR * 12)]  # remote 2-12 s

    monkeypatch.setattr(speaker_stage, "ingest_audio", ingest)
    monkeypatch.setattr(speaker_stage, "load_vad_model", lambda: None)
    monkeypatch.setattr(speaker_stage, "vad_segments", fake_vad)

    stage = _run(np.zeros(SR * 12, dtype=np.float32), diarize=False)

    assert stage.turns == [(0.0, 2.0, "Janni"), (2.0, 12.0, REMOTE_LABEL)]
