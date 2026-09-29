"""Contract tests for the grouping fixtures under ``tests/fixtures/grouping/``.

The checked-in fixtures (issue #77, WS5) pin two things:

- a **synthesized 2-part m4a pair** (``pair_a.m4a`` 2.5 s, ``pair_b.m4a``
  3.0 s — sine tones, never a real memo) that exercises ``concat_groups``
  (one real ffmpeg concat → 5.5 s output) and ``part_offsets`` (PCM
  offsets 0.0 / 2.5).
- **Finnish boundary-pair transcript texts** (``tails.txt`` /
  ``heads.txt``, ``<label>: <text>`` lines) that drive the pure
  ``propose_groups`` heuristic: a trailing closing cue forces a break,
  a mid-sentence tail continues, and a silent boundary degrades to "no
  evidence" (break, no crash).

Real ffmpeg is required for the audio goldens (``skipif`` when absent);
the transcript texts are exercised with pure Python only.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from vemoizer.grouping import (
    EN_CLOSING_CUES,
    FI_CLOSING_CUES,
    concat_groups,
    part_offsets,
    propose_groups,
)
from vemoizer.ingest import duration_seconds, ingest_audio

GROUPING_DIR = Path(__file__).resolve().parent / "fixtures" / "grouping"

FFMPEG = shutil.which("ffmpeg") is not None
FFPROBE = shutil.which("ffprobe") is not None
requires_ffmpeg = pytest.mark.skipif(
    not (FFMPEG and FFPROBE), reason="ffmpeg/ffprobe not on PATH"
)

#: Exact decoded-PCM durations (seconds) of the checked-in pair. The
#: files are synthesized from exact-length sine tones and re-encoded with
#: ffmpeg; AAC frame padding is absorbed by the decode, so the PCM
#: duration must be exact (part offsets must be exact too).
DURATION_A = 2.5
DURATION_B = 3.0


def _read_labeled(path: Path) -> dict[str, str]:
    """``<label>: <text>`` per line -> ``{label: text}`` (blank value OK)."""
    out: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        label, _, text = line.partition(":")
        out[label.strip()] = text.strip()
    return out


TAILS = (
    _read_labeled(GROUPING_DIR / "tails.txt")
    if (GROUPING_DIR / "tails.txt").is_file()
    else {}
)
HEADS = (
    _read_labeled(GROUPING_DIR / "heads.txt")
    if (GROUPING_DIR / "heads.txt").is_file()
    else {}
)


def _two_files() -> list[Path]:
    return [GROUPING_DIR / "pair_a.m4a", GROUPING_DIR / "pair_b.m4a"]


# ---------------------------------------------------------------------------
# Fixture presence
# ---------------------------------------------------------------------------


def test_grouping_fixtures_present() -> None:
    assert (GROUPING_DIR / "pair_a.m4a").is_file(), "missing pair_a.m4a"
    assert (GROUPING_DIR / "pair_b.m4a").is_file(), "missing pair_b.m4a"
    assert (GROUPING_DIR / "tails.txt").is_file(), "missing tails.txt"
    assert (GROUPING_DIR / "heads.txt").is_file(), "missing heads.txt"
    # The transcript texts must cover with-cue, without-cue, and silent
    # boundaries, else the heuristic goldens lose their fixture.
    for label in (
        "fi_no_cue",
        "fi_cue_kiitos",
        "en_cue_bye",
        "mid_cue",
        "silence",
    ):
        assert label in TAILS, f"tails.txt missing label {label!r}"
    for label in ("fi_head_no_cue", "silence"):
        assert label in HEADS, f"heads.txt missing label {label!r}"


# ---------------------------------------------------------------------------
# Audio pair goldens (real ffmpeg)
# ---------------------------------------------------------------------------


@requires_ffmpeg
def test_pair_durations_match_goldens() -> None:
    a = duration_seconds(ingest_audio(GROUPING_DIR / "pair_a.m4a"))
    b = duration_seconds(ingest_audio(GROUPING_DIR / "pair_b.m4a"))
    assert a == pytest.approx(DURATION_A, abs=0.01)
    assert b == pytest.approx(DURATION_B, abs=0.01)


@requires_ffmpeg
def test_pair_concat_golden() -> None:
    """concat_groups over the pair: one file, decoded duration a+b."""
    out = concat_groups(_two_files())
    try:
        assert out.is_file()
        total = duration_seconds(ingest_audio(out))
        expected = DURATION_A + DURATION_B
        assert total == pytest.approx(expected, abs=0.1)
    finally:
        out.unlink(missing_ok=True)


@requires_ffmpeg
def test_pair_part_offsets_golden() -> None:
    """part_offsets: 0.0 / 2.5, from decoded PCM (never ffprobe)."""
    offsets = part_offsets(_two_files())
    assert [p.part_number for p in offsets] == [1, 2]
    assert [p.source_filename for p in offsets] == ["pair_a.m4a", "pair_b.m4a"]
    assert offsets[0].start_offset == pytest.approx(0.0)
    assert offsets[1].start_offset == pytest.approx(DURATION_A, abs=0.01)


# ---------------------------------------------------------------------------
# Boundary transcript text goldens (pure heuristic, no I/O)
# ---------------------------------------------------------------------------


def test_boundary_pair_no_cue_is_continuation() -> None:
    files = _two_files()
    proposals = propose_groups(files, [TAILS["fi_no_cue"]], [HEADS["fi_head_no_cue"]])
    assert len(proposals) == 1
    assert proposals[0].is_continuation is True


def test_boundary_pair_fi_cues_force_break() -> None:
    files = _two_files()
    for label in ("fi_cue_kiitos", "fi_cue_moi", "fi_cue_naahdaan"):
        proposals = propose_groups(files, [TAILS[label]], [HEADS["fi_head_no_cue"]])
        assert proposals[0].is_continuation is False, f"{label} did not force a break"


def test_boundary_pair_en_cues_force_break() -> None:
    files = _two_files()
    for label in ("en_cue_thanks", "en_cue_bye"):
        proposals = propose_groups(files, [TAILS[label]], [HEADS["en_head_no_cue"]])
        assert proposals[0].is_continuation is False, f"{label} did not force a break"


def test_boundary_pair_mid_cue_is_continuation() -> None:
    """A closing cue mid-sentence (outside the tail window) does not break."""
    files = _two_files()
    proposals = propose_groups(files, [TAILS["mid_cue"]], [HEADS["fi_head_no_cue"]])
    assert proposals[0].is_continuation is True


def test_boundary_pair_silent_boundary_degrades_to_no_evidence() -> None:
    files = _two_files()
    proposals = propose_groups(files, [TAILS["silence"]], [HEADS["silence"]])
    assert proposals[0].is_continuation is False
    assert proposals[0].evidence == ("(no audio)", "(no audio)")


def test_boundary_pair_head_cue_also_breaks() -> None:
    """A closing cue in the head snippet (e.g. an early "bye") also breaks."""
    files = _two_files()
    proposals = propose_groups(files, [TAILS["fi_no_cue"]], ["and then bye for today"])
    assert proposals[0].is_continuation is False


def test_fixture_cues_are_a_subset_of_seeded_lists() -> None:
    """The with-cue fixture labels' cues must actually live in the seeded
    lists — otherwise the goldens pass vacuously."""
    assert _cue_in_text(TAILS["fi_cue_kiitos"], FI_CLOSING_CUES)
    assert _cue_in_text(TAILS["fi_cue_moi"], FI_CLOSING_CUES)
    assert _cue_in_text(TAILS["fi_cue_naahdaan"], FI_CLOSING_CUES)
    assert _cue_in_text(TAILS["en_cue_thanks"], EN_CLOSING_CUES)
    assert _cue_in_text(TAILS["en_cue_bye"], EN_CLOSING_CUES)


def _cue_in_text(text: str, cues: tuple[str, ...]) -> bool:
    from vemoizer.textnorm import textnorm

    norm = textnorm(text)
    return any(cue in norm for cue in cues)
