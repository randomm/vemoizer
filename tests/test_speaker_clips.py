"""Tests for M5b speaker clip selection and playback helpers (issue #90).

Pure selector + talk_share tests, window-to-part mapping and extraction
(ffmpeg mocked for argv assertions, real ffmpeg behind skipif), play()
fail-open matrix, and clip_session cleanup on every exit path.
"""

from __future__ import annotations

import shutil
import struct
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from vemoizer.speaker_clip_select import MAX_TURN_S
from vemoizer.speaker_clips import (
    ClipWindow,
    clip_session,
    extract_clips,
    play,
    select_clips,
    talk_share,
)

pytestmark = pytest.mark.real_system_calls

FFMPEG = pytest.mark.skipif(
    shutil.which("ffmpeg") is None, reason="ffmpeg not available"
)


def _para(
    start: float,
    end: float,
    text: str = "lorem ipsum dolor sit amet consectetur",
    speaker: str | None = "A",
    suspect: str | None = None,
) -> dict:
    p: dict = {"start": start, "end": end, "text": text, "speaker": speaker}
    if suspect is not None:
        p["suspect"] = suspect
    return p


def _seg(
    start: float,
    end: float,
    text: str = "segment own words here",
    speaker: str | None = "A",
) -> dict:
    return {"start": start, "end": end, "text": text, "speaker": speaker}


def _make_wav(path: Path, seconds: float) -> Path:
    """A tiny mono 16 kHz 16-bit PCM WAV."""
    rate = 16000
    n = int(seconds * rate)
    samples = b"".join(struct.pack("<h", 3000) for _ in range(n))
    filesize = 36 + len(samples)
    hdr = (
        b"RIFF"
        + struct.pack("<I", filesize - 8)
        + b"WAVE"
        + b"fmt "
        + struct.pack("<I", 16)
        + struct.pack("<H", 1)
        + struct.pack("<H", 1)
        + struct.pack("<I", rate)
        + struct.pack("<I", rate * 2)
        + struct.pack("<H", 2)
        + struct.pack("<H", 16)
        + b"data"
        + struct.pack("<I", len(samples))
    )
    path.write_bytes(hdr + samples)
    return path


# ---------------------------------------------------------------------------
# talk_share
# ---------------------------------------------------------------------------


class TestTalkShare:
    def test_fractions_sum_to_one(self):
        paras = [_para(0, 20, speaker="A"), _para(20, 60, speaker="B")]
        shares = talk_share(paras)
        assert set(shares) == {"A", "B"}
        assert shares["A"] == pytest.approx(1 / 3)
        assert shares["B"] == pytest.approx(2 / 3)
        assert abs(sum(shares.values()) - 1.0) < 1e-6

    def test_unlabelled_excluded(self):
        bare = {"start": 10, "end": 20, "text": "no speaker key"}
        paras = [_para(0, 10, speaker="A"), _para(10, 20, speaker=None), bare]
        assert talk_share(paras) == {"A": 1.0}

    def test_empty_gives_empty_dict(self):
        assert talk_share([]) == {}
        assert talk_share([_para(0, 10, speaker=None)]) == {}


# ---------------------------------------------------------------------------
# select_clips — exclusions
# ---------------------------------------------------------------------------


class TestSelectClipsExclusions:
    def test_suspect_only_label_still_gets_a_clip(self):
        # Issue #202: every label with >= 2 s of speech gets a quote and a clip.
        clips = select_clips([_para(0, 10, suspect="garble")], [])
        assert len(clips["A"]) == 1

    def test_clean_turn_preferred_over_suspect(self):
        paras = [
            _para(0, 10, text="epäselvä osuus tässä", suspect="garble"),
            _para(20, 25, text="selkeä puhe tässä lauseessa"),
        ]
        (clip,) = select_clips(paras, [])["A"]
        assert clip.quote == "selkeä puhe tässä lauseessa"

    def test_unlabelled_excluded(self):
        assert select_clips([_para(0, 10, speaker=None)], []) == {}

    def test_short_turn_excluded(self):
        assert select_clips([_para(0, 1.9)], []) == {}

    def test_backchannel_only_label_still_gets_a_clip(self):
        assert len(select_clips([_para(0, 5, text="joo")], [])["A"]) == 1

    def test_backchannel_phrase_only_label_still_gets_a_clip(self):
        assert len(select_clips([_para(0, 5, text="hyvä kysymys")], [])["A"]) == 1

    def test_clean_turn_preferred_over_backchannel(self):
        paras = [
            _para(0, 5, text="joo"),
            _para(10, 15, text="tämä on oikea puheenvuoro tässä"),
        ]
        (clip,) = select_clips(paras, [])["A"]
        assert clip.quote == "tämä on oikea puheenvuoro tässä"

    def test_textless_turn_excluded(self):
        assert select_clips([_para(0, 10, text="")], []) == {}

    def test_backchannel_word_inside_longer_turn_not_excluded(self):
        paras = [_para(0, 10, text="joo mutta tämä on pidempi lause jatkossa")]
        assert len(select_clips(paras, [])["A"]) == 1

    def test_paragraph_without_speaker_key_tolerated(self):
        paras = [
            {"start": 0, "end": 5, "text": "hello there friend, test"},
            _para(6, 16),
        ]
        assert len(select_clips(paras, [])["A"]) == 1


# ---------------------------------------------------------------------------
# select_clips — window geometry
# ---------------------------------------------------------------------------


class TestSelectClipsWindow:
    def test_window_starts_at_paragraph_start(self):
        # No segments -> paragraph fallback: the window starts at the turn
        # start, not its midpoint (issue #253).
        clips = select_clips([_para(10, 40)], [])["A"]  # 30 s turn
        assert clips[0].start_s == pytest.approx(10.0)
        assert clips[0].end_s == pytest.approx(15.0)

    def test_window_shrinks_for_short_turns(self):
        # 4 s turn -> 4 s window (whole turn).
        clips = select_clips([_para(0, 4)], [])["A"]
        assert clips[0].start_s == pytest.approx(0.0)
        assert clips[0].end_s == pytest.approx(4.0)

    def test_window_clamped_to_audio_duration(self):
        # Turn [25,30] but audio ends at 28: clamp to [25, 28] (3 s).
        clips = select_clips([_para(25, 30)], [], total_duration=28.0)["A"]
        w = clips[0]
        assert w.end_s == pytest.approx(28.0)
        assert w.end_s - w.start_s == pytest.approx(3.0)

    def test_quote_bounded_to_120_chars(self):
        long_text = " ".join(f"w{i}" for i in range(60))
        clips = select_clips([_para(0, 20, text=long_text)], [])["A"]
        assert len(clips[0].quote) <= 121
        assert clips[0].quote.endswith("…")

    def test_quote_is_paragraph_start_text_for_long_turn(self):
        # The quote starts where the clip starts: the first words heard are
        # the first words printed (issue #253).
        words = [f"w{i}" for i in range(40)]
        text = " ".join(words)
        w = select_clips([_para(10.0, 40.0, text=text)], [])["A"][0]
        assert (w.start_s, w.end_s) == (pytest.approx(10.0), pytest.approx(15.0))
        assert w.quote.startswith("w0 ")
        assert text.startswith(w.quote.rstrip("…"))

    def test_short_quote_whitespace_collapsed(self):
        clips = select_clips([_para(0, 5, text="  hello   world  ")], [])["A"]
        assert clips[0].quote == "hello world"


# ---------------------------------------------------------------------------
# select_clips — spread across thirds
# ---------------------------------------------------------------------------


class TestSelectClipsSpread:
    def test_one_clip_from_each_third(self):
        paras = [_para(0, 20), _para(30, 50), _para(70, 90)]
        clips = select_clips(paras, [])["A"]
        assert [w.start_s for w in clips] == [
            pytest.approx(0.0),
            pytest.approx(30.0),
            pytest.approx(70.0),
        ]

    def test_longest_candidate_wins_within_third(self):
        # Three turns in the first third of 40 s of audio. Ranked by the
        # turn's own duration, so the 7 s one wins: under the old window-span
        # ranking the 6 s and 7 s turns both spanned max_s and the earliest
        # (the 6 s one at 4 s) silently won.
        paras = [_para(0.0, 40.0)]
        segs = [
            _seg(0.0, 3.0, "three second turn"),
            _seg(4.0, 10.0, "six second turn here"),
            _seg(11.0, 18.0, "seven second turn wins the third"),
        ]
        (w,) = select_clips(paras, segs, per_speaker=1)["A"]
        assert (w.start_s, w.end_s) == (pytest.approx(11.0), pytest.approx(18.0))
        assert w.quote == "seven second turn wins the third"

    def test_longest_paragraph_candidate_wins_within_third(self):
        # Same ranking rule on the paragraph fallback (no segments at all).
        paras = [
            _para(0.0, 6.0, text="six second turn here"),
            _para(9.0, 20.0, text="eleven second turn wins the third"),
        ]
        (w,) = select_clips(paras, None, per_speaker=1, total_duration=40.0)["A"]
        assert (w.start_s, w.end_s) == (pytest.approx(9.0), pytest.approx(14.0))
        assert w.quote.startswith("eleven second")

    def test_two_disjoint_candidates_in_one_third_are_both_picked(self):
        # Only overlapping candidates are deduped; two disjoint turns of the
        # same third are both kept (spread rule tops up from the third).
        clips = select_clips([_para(0, 8), _para(10, 14)], [], total_duration=40.0)["A"]
        assert {(round(w.start_s, 1), round(w.end_s, 1)) for w in clips} == {
            (0.0, 5.0),
            (10.0, 14.0),
        }

    def test_no_two_windows_of_a_label_overlap(self):
        # De-dup is by TIME OVERLAP, not equal start: these two turns share
        # no start but their windows do overlap, so only one is chosen.
        paras = [
            _para(0.0, 10.0, text="first turn words here today"),
            _para(3.0, 7.0, text="second overlapping turn words"),
        ]
        clips = select_clips(paras, [])["A"]
        for i, a in enumerate(clips):
            for b in clips[i + 1 :]:
                assert not (a.start_s < b.end_s and b.start_s < a.end_s)
        assert len(clips) == 1
        assert clips[0].start_s == pytest.approx(0.0)

    def test_empty_third_falls_back_to_longest_remaining(self):
        paras = [_para(0, 4), _para(5, 15), _para(16, 21)]
        clips = select_clips(paras, [])["A"]
        assert [w.start_s for w in clips] == [
            pytest.approx(0.0),
            pytest.approx(5.0),
            pytest.approx(16.0),
        ]

    def test_fewer_than_three_returns_fewer(self):
        clips = select_clips([_para(0, 5), _para(10, 12)], [])["A"]
        assert len(clips) == 2

    def test_per_speaker_limit(self):
        paras = [_para(i * 10, i * 10 + 8) for i in range(6)]
        assert len(select_clips(paras, [], per_speaker=2)["A"]) == 2

    def test_speaker_boundary_never_crossed(self):
        paras = [_para(0, 10, speaker="A"), _para(10.5, 20, speaker="B")]
        r = select_clips(paras, [])
        assert r["A"][0].end_s <= 10.0
        assert r["B"][0].start_s >= 10.5


# ---------------------------------------------------------------------------
# select_clips — segment-anchored windows and quote purity (issue #253)
# ---------------------------------------------------------------------------


class TestSelectClipsSegments:
    def test_window_and_quote_come_from_own_segment(self):
        paras = [_para(0.0, 90.0, text="FULL PARAGRAPH spans everything")]
        segs = [
            _seg(20.0, 26.0, "toinen puheenvuoro tässä"),
            _seg(21.0, 22.9, "lyhyt pätkä"),  # < 2 s -> never a candidate
        ]
        clips = select_clips(paras, segs)["A"]
        w = next(c for c in clips if c.start_s == pytest.approx(20.0))
        assert w.end_s == pytest.approx(26.0)  # the WHOLE segment, not max_s
        assert w.quote == "toinen puheenvuoro tässä"
        assert "lyhyt" not in w.quote

    def test_window_is_the_whole_clean_segment(self):
        # The window is the segment start-to-end, so a longer clean segment
        # plays (and quotes) more words instead of being cut at max_s.
        paras = [_para(0.0, 90.0, text="P0")]
        segs = [_seg(10.0, 17.0, "seitsen sekunnin puhesuora tältä puhujalta")]
        w = next(c for c in select_clips(paras, segs)["A"] if c.start_s == 10.0)
        assert (w.start_s, w.end_s) == (pytest.approx(10.0), pytest.approx(17.0))
        assert w.quote == "seitsen sekunnin puhesuora tältä puhujalta"

    def test_segment_at_max_turn_s_is_still_a_candidate(self):
        paras = [_para(0.0, 90.0, text="P0")]
        segs = [_seg(0.0, MAX_TURN_S, "exact max turn segment words here")]
        w = select_clips(paras, segs)["A"][0]
        assert (w.start_s, w.end_s) == (pytest.approx(0.0), pytest.approx(MAX_TURN_S))

    def test_segment_over_max_turn_s_is_never_quoted(self):
        # A 48 s turn must not become a 5 s clip with a 121-char quote: the
        # over-long segment is dropped and the paragraph fallback is used.
        long_seg_text = " ".join(f"s{i}" for i in range(200))
        paras = [_para(0.0, 48.0, text="own paragraph spoken words")]
        segs = [_seg(0.0, 48.0, long_seg_text)]
        clips = select_clips(paras, segs)["A"]
        assert clips
        for w in clips:
            assert "s0 s1 s2" not in w.quote
            assert w.end_s - w.start_s <= MAX_TURN_S
        assert any(w.quote.startswith("own") for w in clips)

    def test_long_segment_quote_starts_at_window_start(self):
        # A clean (<= MAX_TURN_S) segment with more text than the quote can
        # carry: the quote stays a prefix of what the clip plays.
        text = " ".join(f"s{i}" for i in range(80))
        paras = [_para(0.0, 90.0, text="P0")]
        segs = [_seg(0.0, MAX_TURN_S - 0.5, text)]
        clips = select_clips(paras, segs)["A"]
        w = next(
            c
            for c in clips
            if c.start_s == pytest.approx(0.0) and c.quote.startswith("s0 ")
        )
        assert w.end_s == pytest.approx(MAX_TURN_S - 0.5)
        assert len(w.quote) <= 121 and w.quote.endswith("…")
        assert text.startswith(w.quote.rstrip("…"))

    def test_no_clean_segment_falls_back_to_paragraph_start(self):
        paras = [_para(0.0, 12.0, text="own paragraph words here now")]
        segs = [
            _seg(5.0, 6.5, "lyhyt"),  # too short
            _seg(10.0, 15.0, "overlapped"),  # overlaps the B segment below
            _seg(12.0, 13.0, "bleeding in", speaker="B"),
        ]
        clips = select_clips(paras, segs)
        (w,) = clips["A"]
        assert w.start_s == pytest.approx(0.0)  # paragraph START, not midpoint
        assert w.end_s == pytest.approx(5.0)  # min(max_s, duration)
        # Only the played 5/12 of the text is quoted, cut at a word boundary.
        assert w.quote == "own paragraph…"
        for foreign in ("lyhyt", "overlapped", "bleeding in"):
            assert foreign not in w.quote

    def test_fallback_quote_is_only_the_played_share(self):
        # 20 words over 20 s, the window plays the first 5 s -> 5 words.
        text = " ".join(f"w{i}" for i in range(20))
        (w,) = select_clips([_para(0.0, 20.0, text=text)], [])["A"]
        assert (w.start_s, w.end_s) == (pytest.approx(0.0), pytest.approx(5.0))
        assert w.quote == "w0 w1 w2 w3 w4…"
        assert "w19" not in w.quote

    def test_fallback_quote_keeps_at_least_one_word(self):
        (w,) = select_clips([_para(0.0, 120.0, text="ensin toinen kolmas")], [])["A"]
        assert w.quote == "ensin…"

    def test_fallback_rejects_window_overlapping_other_label_segment(self):
        # The paragraph fallback is guarded by the segments too: the early
        # turn's window bleeds into B's segment, so that third stays empty.
        paras = [
            _para(0.0, 10.0, text="alpha early words"),
            _para(20.0, 25.0, text="alpha late words"),
        ]
        segs = [_seg(3.0, 6.0, "bravo bleed", speaker="B")]
        clips = select_clips(paras, segs)["A"]
        assert [w.start_s for w in clips] == [pytest.approx(20.0)]

    def test_fallback_rejects_paragraph_window_overlapping_other_paragraph(self):
        # An old sidecar has no segments at all, so other labels' PARAGRAPHS
        # are the guard.
        paras = [
            _para(0.0, 10.0, text="alpha early words"),
            _para(3.0, 6.0, text="bravo bleed", speaker="B"),
            _para(20.0, 25.0, text="alpha late words"),
        ]
        clips = select_clips(paras, None)
        assert [w.start_s for w in clips["A"]] == [pytest.approx(20.0)]

    def test_one_clean_segment_per_third_in_order(self):
        paras = [_para(0.0, 90.0, text="P0")]
        segs = [
            _seg(60.0, 66.0, "late words"),
            _seg(0.0, 6.0, "early words"),
            _seg(31.0, 38.0, "mid words"),
        ]
        clips = select_clips(paras, segs)["A"]
        assert [w.start_s for w in clips] == [
            pytest.approx(0.0),
            pytest.approx(31.0),
            pytest.approx(60.0),
        ]
        assert [w.quote for w in clips] == ["early words", "mid words", "late words"]

    def test_quote_never_contains_other_labels_text(self):
        paras = [
            _para(0.0, 30.0, text="alpha own speech"),
            _para(31.0, 60.0, text="bravo own speech", speaker="B"),
        ]
        segs = [
            _seg(0.0, 7.0, "alpha segment words"),
            _seg(31.0, 38.0, "bravo segment words", speaker="B"),
            _seg(36.0, 38.0, "ALPHA BLEED", speaker="A"),
            _seg(12.0, 14.0, "bravo early words", speaker="B"),
        ]
        clips = select_clips(paras, segs)
        foreign = {
            "A": [s["text"] for s in segs if s["speaker"] == "B"]
            + [p["text"] for p in paras if p["speaker"] == "B"],
            "B": [s["text"] for s in segs if s["speaker"] == "A"]
            + [p["text"] for p in paras if p["speaker"] == "A"],
        }
        assert clips  # sanity: something was selected
        for label, ws in clips.items():
            for w in ws:
                assert w.quote  # every clip has a quote
                for text in foreign[label]:
                    assert text not in w.quote
        # The A segment at 36-38 s is the label's OWN text but overlaps B's
        # segment, so it is foreign to the clip and must never be quoted.
        assert all("ALPHA BLEED" not in w.quote for w in clips["A"])

    def test_unlabelled_segment_contaminates_candidate(self):
        # The unlabelled 4-6 s segment poisons both the own-segment candidate
        # (0-6 s overlaps it) and the paragraph window (0-5 s overlaps it),
        # so the first third yields no clip at all — and its words are never
        # quoted (issue #253).
        paras = [
            _para(0.0, 20.0, text="own paragraph"),
            _para(50.0, 60.0, text="second region words"),
        ]
        segs = [
            _seg(0.0, 6.0, "own segment"),
            {"start": 4.0, "end": 6.0, "text": "mystery voice"},
        ]
        clips = select_clips(paras, segs)["A"]
        assert [w.start_s for w in clips] == [pytest.approx(50.0)]
        for w in clips:
            assert not (w.start_s < 6.0 and w.end_s > 4.0)
            assert "mystery voice" not in w.quote

    def test_old_sidecar_without_segments_uses_paragraph_fallback(self):
        # segments=None (old sidecars carry no segments key at all).
        paras = [_para(0.0, 10.0, text="vanha teksti"), _para(12.0, 20.0)]
        clips = select_clips(paras, None)
        assert [w.start_s for w in clips["A"]] == [
            pytest.approx(0.0),
            pytest.approx(12.0),
        ]
        # 5 of the 10 s played -> the leading half of the two words.
        assert clips["A"][0].quote == "vanha…"


# ---------------------------------------------------------------------------
# extract_clips — part mapping
# ---------------------------------------------------------------------------


class TestExtractClipsMapping:
    def test_window_maps_to_second_part(self, tmp_path, monkeypatch):
        """Window [12,15] maps to [2.0, 5.0] of part b (offset 10)."""
        src_a, src_b = tmp_path / "a.wav", tmp_path / "b.wav"
        _make_wav(src_a, 1.0)
        _make_wav(src_b, 2.0)
        source = [
            {"path": str(src_a), "part_offset_s": 0.0, "duration_s": 10.0},
            {"path": str(src_b), "part_offset_s": 10.0, "duration_s": 10.0},
        ]
        recorded: dict[str, list] = {}

        def fake_run(argv, *a, **kw):
            recorded["argv"] = argv
            Path(argv[-1]).write_bytes(b"RIFF" + b"\x00" * 74)
            return subprocess.CompletedProcess(argv, 0)

        monkeypatch.setattr("vemoizer.speaker_clips.subprocess.run", fake_run)
        window = ClipWindow(12.0, 15.0)
        with clip_session() as tmp:
            result = extract_clips(source, [window], tmp)
        # 2 s fixture can't satisfy -ss 2.0 -t 3.0 -> empty -> None.
        assert result[window] is None
        argv = recorded["argv"]
        assert argv[argv.index("-i") + 1] == str(src_b)
        assert argv[argv.index("-ss") + 1] == "2.000"
        assert argv[argv.index("-t") + 1] == "3.000"
        assert argv.index("-ss") < argv.index("-i")
        assert argv.index("-t") < argv.index("-i")

    def test_straddling_window_dropped(self, tmp_path):
        src_a, src_b = tmp_path / "a.wav", tmp_path / "b.wav"
        _make_wav(src_a, 1.0)
        _make_wav(src_b, 1.0)
        source = [
            {"path": str(src_a), "part_offset_s": 0.0, "duration_s": 10.0},
            {"path": str(src_b), "part_offset_s": 10.0, "duration_s": 10.0},
        ]
        with clip_session() as tmp:
            result = extract_clips(source, [ClipWindow(9.0, 11.0)], tmp)
        assert result[ClipWindow(9.0, 11.0)] is None

    def test_window_beyond_part_duration_dropped(self, tmp_path):
        src_a, src_b = tmp_path / "a.wav", tmp_path / "b.wav"
        _make_wav(src_a, 1.0)
        _make_wav(src_b, 1.0)
        source = [
            {"path": str(src_a), "part_offset_s": 0.0, "duration_s": 5.0},
            {"path": str(src_b), "part_offset_s": 5.0, "duration_s": 10.0},
        ]
        with clip_session() as tmp:
            result = extract_clips(source, [ClipWindow(3.0, 8.0)], tmp)
        assert result[ClipWindow(3.0, 8.0)] is None

    def test_missing_source_returns_none(self, tmp_path):
        source = [{"path": str(tmp_path / "gone.wav"), "part_offset_s": 0.0}]
        with clip_session() as tmp:
            result = extract_clips(source, [ClipWindow(1.0, 3.0)], tmp)
        assert result[ClipWindow(1.0, 3.0)] is None

    def test_no_source_returns_none(self, tmp_path):
        with clip_session() as tmp:
            result = extract_clips([], [ClipWindow(1.0, 3.0)], tmp)
        assert result[ClipWindow(1.0, 3.0)] is None

    def test_last_part_without_duration_unbounded(self, tmp_path):
        src_a = tmp_path / "a.wav"
        _make_wav(src_a, 1.0)
        source = [{"path": str(src_a), "part_offset_s": 0.0}]
        with clip_session() as tmp:
            result = extract_clips(source, [ClipWindow(50.0, 55.0)], tmp)
        assert result[ClipWindow(50.0, 55.0)] is None


@FFMPEG
class TestExtractClipsFfmpeg:
    def test_real_ffmpeg_decode(self, tmp_path):
        src = _make_wav(tmp_path / "src.wav", 3.0)
        source = [{"path": str(src), "part_offset_s": 0.0, "duration_s": 3.0}]
        window = ClipWindow(0.5, 2.5)
        with clip_session() as tmp:
            result = extract_clips(source, [window], tmp)
            out = result[window]
            assert out is not None and out.is_file()
            data = out.read_bytes()
            assert data[:4] == b"RIFF"
            # Find the data chunk (WAV may have LIST etc.).
            pos = 12
            while pos + 8 <= len(data):
                cid = data[pos : pos + 4]
                csize = struct.unpack("<I", data[pos + 4 : pos + 8])[0]
                if cid == b"data":
                    assert 63000 < csize < 65000
                    break
                pos += 8 + csize

    def test_ffmpeg_failure_returns_none(self, tmp_path):
        src = tmp_path / "bad.wav"
        src.write_bytes(b"not a real wav")
        source = [{"path": str(src), "part_offset_s": 0.0, "duration_s": 3.0}]
        with clip_session() as tmp:
            result = extract_clips(source, [ClipWindow(0.5, 2.5)], tmp)
        assert result[ClipWindow(0.5, 2.5)] is None


# ---------------------------------------------------------------------------
# play() — fail-open matrix
# ---------------------------------------------------------------------------


class TestPlay:
    @pytest.fixture(autouse=True)
    def _darwin(self, monkeypatch):
        monkeypatch.setattr("vemoizer.speaker_clips.sys.platform", "darwin")

    def test_success(self, tmp_path):
        f = tmp_path / "c.wav"
        f.write_bytes(b"RIFFxxxx")
        with patch(
            "subprocess.run",
            return_value=subprocess.CompletedProcess(args=[], returncode=0),
        ):
            assert play(f) is True

    def test_non_darwin(self, tmp_path, monkeypatch):
        monkeypatch.setattr("vemoizer.speaker_clips.sys.platform", "linux")
        assert play(tmp_path / "c.wav") is False

    def test_missing_file(self, tmp_path):
        assert play(tmp_path / "gone.wav") is False

    def test_missing_afplay(self, tmp_path):
        f = tmp_path / "c.wav"
        f.write_bytes(b"RIFFxxxx")
        with patch("subprocess.run", side_effect=FileNotFoundError("afplay")):
            assert play(f) is False

    def test_nonzero_exit(self, tmp_path):
        f = tmp_path / "c.wav"
        f.write_bytes(b"RIFFxxxx")
        with patch(
            "subprocess.run",
            return_value=subprocess.CompletedProcess(args=[], returncode=1),
        ):
            assert play(f) is False

    def test_timeout(self, tmp_path):
        f = tmp_path / "c.wav"
        f.write_bytes(b"RIFFxxxx")
        with patch(
            "subprocess.run", side_effect=subprocess.TimeoutExpired("afplay", 30)
        ):
            assert play(f) is False

    def test_os_error(self, tmp_path):
        f = tmp_path / "c.wav"
        f.write_bytes(b"RIFFxxxx")
        with patch("subprocess.run", side_effect=OSError("boom")):
            assert play(f) is False


# ---------------------------------------------------------------------------
# clip_session — cleanup on every exit path
# ---------------------------------------------------------------------------


class TestClipSession:
    @pytest.fixture(autouse=True)
    def _capture(self, monkeypatch):
        import vemoizer.speaker_clips as sc

        captured: list[Path] = []
        real = sc.tempfile.mkdtemp

        def fake(*a, **kw):
            p = real(*a, **kw)
            captured.append(Path(p))
            return p

        monkeypatch.setattr(sc.tempfile, "mkdtemp", fake)
        self._captured = captured

    def test_yields_0o700_and_removes(self):
        with clip_session() as d:
            assert d.is_dir()
            assert d.stat().st_mode & 0o777 == 0o700
            (d / "clip.wav").write_bytes(b"x")
        assert not d.exists()

    def test_removes_on_exception(self):
        with pytest.raises(RuntimeError), clip_session():
            raise RuntimeError("crash")
        assert not self._captured[0].exists()

    def test_removes_on_keyboard_interrupt(self):
        with pytest.raises(KeyboardInterrupt), clip_session():
            raise KeyboardInterrupt
        assert not self._captured[0].exists()

    def test_no_clips_remain(self):
        with clip_session() as d:
            (d / "clip_00.wav").write_bytes(b"RIFFxxxx")
        assert not any(p.exists() for p in self._captured)


# ---------------------------------------------------------------------------
# ClipWindow identity
# ---------------------------------------------------------------------------


class TestClipWindow:
    def test_equality_and_hash(self):
        a = ClipWindow(1.0, 2.0, "q")
        assert a == (1.0, 2.0, "q")
        assert hash(a) == hash((1.0, 2.0, "q"))

    def test_named_fields(self):
        w = ClipWindow(3.5, 5.5, "quote")
        assert w.start_s == 3.5 and w.end_s == 5.5 and w.quote == "quote"
