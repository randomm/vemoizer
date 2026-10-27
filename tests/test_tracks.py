"""Pure two-track helpers for live call capture (issue #170)."""

from __future__ import annotations

import pytest

from vemoizer.tracks import (
    REMOTE_LABEL,
    check_stem_lengths,
    merge_track_turns,
    owner_turns,
    remote_speaker_count,
    resolve_owner,
    samples_to_seconds,
    single_remote_turns,
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
