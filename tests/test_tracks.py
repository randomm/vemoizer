"""Pure two-track helpers for live call capture (issue #170)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from vemoizer.ingest import IngestError
from vemoizer.tracks import (
    MIC_FLOOR_DBFS,
    REMOTE_LABEL,
    TrackPaths,
    check_stem_lengths,
    configured_owner,
    ingest_stems,
    level_dbfs,
    merge_track_turns,
    owner_speech_segments,
    owner_turns,
    remote_speaker_count,
    resolve_owner,
    samples_to_seconds,
    single_remote_turns,
    tracks_error,
)


def _seg(start: float, end: float) -> dict[str, float]:
    return {"start": start, "end": end}


def test_samples_to_seconds_uses_contract_rate() -> None:
    assert samples_to_seconds(16_000) == 1.0
    assert samples_to_seconds(8_000) == 0.5


def test_owner_turns_merge_gaps_shorter_than_half_a_second() -> None:
    segments = [_seg(0.0, 1.0), _seg(1.3, 2.0), _seg(2.6, 3.0)]
    assert owner_turns(segments, "Janni") == [
        (0.0, 2.0, "Janni"),
        (2.6, 3.0, "Janni"),
    ]


def test_owner_turns_keep_gap_of_exactly_half_a_second_apart() -> None:
    assert owner_turns([_seg(0.0, 1.0), _seg(1.5, 2.0)], "Me") == [
        (0.0, 1.0, "Me"),
        (1.5, 2.0, "Me"),
    ]


def test_single_remote_turns_label_everything_remote() -> None:
    turns = single_remote_turns([_seg(4.0, 5.0), _seg(0.0, 1.0)])
    assert turns == [(0.0, 1.0, REMOTE_LABEL), (4.0, 5.0, REMOTE_LABEL)]


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        ((2, 6), (1, 5)),
        (4, 3),
        (2, 1),
        ((2, 2), 1),
        (None, None),
    ],
)
def test_remote_speaker_count(spec, expected) -> None:
    assert remote_speaker_count(spec) == expected


@pytest.mark.parametrize("spec", [(1, 3), 1, (5, 3)])
def test_remote_speaker_count_rejects_impossible_totals(spec) -> None:
    with pytest.raises(ValueError):
        remote_speaker_count(spec)


def test_merge_track_turns_sorts_by_start() -> None:
    owner = [(5.0, 6.0, "Janni")]
    remote = [(0.0, 1.0, REMOTE_LABEL), (2.0, 3.0, REMOTE_LABEL)]
    assert merge_track_turns(owner, remote) == [
        (0.0, 1.0, REMOTE_LABEL),
        (2.0, 3.0, REMOTE_LABEL),
        (5.0, 6.0, "Janni"),
    ]


def test_resolve_owner_flag_beats_both_configs() -> None:
    project = {"meeting": {"owner": "Project"}}
    home = {"meeting": {"owner": "Home"}}
    assert resolve_owner("Flag", project, home, "en") == "Flag"


def test_resolve_owner_project_beats_home() -> None:
    project = {"meeting": {"owner": "Project"}}
    home = {"meeting": {"owner": "Home"}}
    assert resolve_owner(None, project, home, "en") == "Project"


def test_resolve_owner_project_without_owner_falls_through_to_home() -> None:
    project = {"meeting": {"other": 1}}
    home = {"meeting": {"owner": "Home"}}
    assert resolve_owner(None, project, home, "en") == "Home"


def test_resolve_owner_fallback_follows_language() -> None:
    assert resolve_owner(None, None, None, "fi") == "Minä"
    assert resolve_owner(None, None, None, "en") == "Me"
    assert resolve_owner(None, None, None, None) == "Me"


def test_resolve_owner_ignores_blank_and_non_string_values() -> None:
    project = {"meeting": {"owner": "   "}}
    home = {"meeting": {"owner": 42}}
    assert resolve_owner("  ", project, home, "fi") == "Minä"
    assert resolve_owner(None, {"meeting": None}, {"meeting": ["x"]}, "en") == "Me"


def test_resolve_owner_strips_surrounding_whitespace() -> None:
    assert resolve_owner("  Janni ", None, None, "en") == "Janni"


def test_check_stem_lengths_pads_shorter_stems_to_longest() -> None:
    mix = 16_000 * 10
    assert check_stem_lengths(mix, [16_000 * 10, 16_000 * 9]) == [0, 16_000]


def test_check_stem_lengths_tolerates_up_to_two_seconds_against_longest() -> None:
    assert check_stem_lengths(16_000 * 10, [16_000 * 12, 16_000 * 11]) == [0, 16_000]


def test_check_stem_lengths_rejects_mismatch_over_two_seconds() -> None:
    with pytest.raises(ValueError):
        check_stem_lengths(16_000 * 10, [16_000 * 13, 16_000 * 12])


def test_configured_owner_is_none_without_any_name() -> None:
    assert configured_owner(None, None, {"meeting": {"owner": " "}}) is None


def test_configured_owner_returns_flag_then_project_then_home() -> None:
    project = {"meeting": {"owner": "Project"}}
    home = {"meeting": {"owner": "Home"}}
    assert configured_owner("Flag", project, home) == "Flag"
    assert configured_owner(None, project, home) == "Project"
    assert configured_owner(None, None, home) == "Home"


def test_ingest_stems_pads_shorter_stem_and_skips_preprocess() -> None:
    calls: list[tuple[str, object]] = []

    def fake_ingest(path, *, preprocess=None):
        calls.append((str(path), preprocess))
        n = 16_000 * 9 if "them" in str(path) else 16_000 * 10
        return np.full(n, 0.5, dtype=np.float32)

    tracks = TrackPaths(me=Path("mic.wav"), them=Path("them.wav"))
    mic, them = ingest_stems(tracks, 16_000 * 10, ingest_fn=fake_ingest)

    assert len(mic) == len(them) == 16_000 * 10
    assert them[-1] == 0.0 and them[0] == 0.5
    assert calls == [("mic.wav", None), ("them.wav", None)]


def test_ingest_stems_propagates_ingest_error() -> None:
    def boom(path, *, preprocess=None):
        raise IngestError("ffmpeg failed")

    tracks = TrackPaths(me=Path("mic.wav"), them=Path("them.wav"))
    with pytest.raises(IngestError):
        ingest_stems(tracks, 16_000, ingest_fn=boom)


def test_ingest_stems_rejects_length_mismatch() -> None:
    def fake_ingest(path, *, preprocess=None):
        return np.zeros(16_000 * 20, dtype=np.float32)

    tracks = TrackPaths(me=Path("mic.wav"), them=Path("them.wav"))
    with pytest.raises(ValueError):
        ingest_stems(tracks, 16_000 * 10, ingest_fn=fake_ingest)


@pytest.mark.parametrize(
    ("command", "n_files", "speakers", "fragment"),
    [
        ("memo", 1, None, "meeting"),
        ("meeting", 2, None, "exactly one"),
        ("meeting", 1, 1, "two participants"),
        ("meeting", 1, (1, 3), "two participants"),
    ],
)
def test_tracks_error_names_each_unsupported_combination(
    command, n_files, speakers, fragment
) -> None:
    error = tracks_error(command, ["a.m4a"] * n_files, speakers)
    assert error is not None
    assert fragment in error


def test_tracks_error_accepts_one_meeting_file_with_two_participants() -> None:
    assert tracks_error("meeting", ["a.m4a"], 2) is None
    assert tracks_error("meeting", ["a.m4a"], None) is None


def _level_span(level: float, seconds: float = 1.0) -> np.ndarray:
    """A constant stem at linear *level* (dBFS = 20*log10(level))."""
    return np.full(int(16_000 * seconds), level, dtype=np.float32)


def _gate(mic_level: float, them_level: float) -> list:
    """Gate one 1 s owner segment (0-1 s) against a 1 s remote span."""
    mic = _level_span(mic_level)
    them = _level_span(them_level)
    return owner_speech_segments([_seg(0.0, 1.0)], mic, them)


def test_level_dbfs_of_a_constant_stem() -> None:
    assert level_dbfs(_level_span(0.01)) == pytest.approx(-40.0, abs=0.01)


def test_level_dbfs_of_silence_is_a_finite_floor() -> None:
    assert level_dbfs(np.zeros(16_000, dtype=np.float32)) < MIC_FLOOR_DBFS
    assert level_dbfs(np.zeros(0, dtype=np.float32)) < MIC_FLOOR_DBFS


def test_leakage_below_the_mic_floor_is_not_owner_speech() -> None:
    # Teams call: mic leakage sat at -49 to -61 dBFS while the remote was -21.
    assert _gate(mic_level=0.0035, them_level=0.09) == []


def test_leakage_under_a_loud_remote_is_not_owner_speech() -> None:
    # Above the floor, but more than the margin under the system level.
    assert _gate(mic_level=0.05, them_level=0.5) == []


def test_owner_speech_over_a_quiet_remote_is_kept() -> None:
    assert _gate(mic_level=0.05, them_level=0.0) == [_seg(0.0, 1.0)]


def test_owner_speech_over_remote_speech_is_kept() -> None:
    # Double talk: the owner speaks at the same level the system plays.
    assert _gate(mic_level=0.1, them_level=0.1) == [_seg(0.0, 1.0)]


def test_owner_speech_just_above_the_floor_is_kept() -> None:
    assert _gate(mic_level=0.0105, them_level=0.0) == [_seg(0.0, 1.0)]


def test_gate_keeps_only_the_owner_segments_that_are_speech() -> None:
    mic = np.concatenate([_level_span(0.05), _level_span(0.0035)])
    them = np.concatenate([_level_span(0.0), _level_span(0.09)])
    kept = owner_speech_segments([_seg(0.0, 1.0), _seg(1.0, 2.0)], mic, them)
    assert kept == [_seg(0.0, 1.0)]


def _db_span(db: float, seconds: float) -> np.ndarray:
    """A constant stem at *db* dBFS; -200 dBFS is digital silence."""
    if db <= -200.0:
        return np.zeros(int(16_000 * seconds), dtype=np.float32)
    return _level_span(10.0 ** (db / 20.0), seconds)


def test_leakage_after_owner_speech_in_one_vad_segment_is_trimmed() -> None:
    # One VAD segment covers 0-12 s: owner 0-2 s at -24 dBFS, then mic leakage
    # (-45 dBFS) over remote speech (-21 dBFS) 2-12 s. The loud owner part must
    # not carry the leakage through a whole-segment level check.
    mic = np.concatenate([_db_span(-24.0, 2.0), _db_span(-45.0, 10.0)])
    them = np.concatenate([_db_span(-200.0, 2.0), _db_span(-21.0, 10.0)])
    kept = owner_speech_segments([_seg(0.0, 12.0)], mic, them)
    assert kept == [_seg(0.0, 2.0)]


def test_owner_speech_split_by_a_leakage_stretch_is_two_runs() -> None:
    # Owner, leakage over remote, owner again: two kept runs, not one.
    mic = np.concatenate(
        [_db_span(-24.0, 1.0), _db_span(-45.0, 1.0), _db_span(-24.0, 1.0)]
    )
    them = np.concatenate(
        [_db_span(-200.0, 1.0), _db_span(-21.0, 1.0), _db_span(-200.0, 1.0)]
    )
    kept = owner_speech_segments([_seg(0.0, 3.0)], mic, them)
    assert kept == [_seg(0.0, 1.0), _seg(2.0, 3.0)]
