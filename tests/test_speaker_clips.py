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
    def test_suspect_excluded(self):
        assert select_clips([_para(0, 10, suspect="garble")], []) == {}

    def test_unlabelled_excluded(self):
        assert select_clips([_para(0, 10, speaker=None)], []) == {}

    def test_short_turn_excluded(self):
        assert select_clips([_para(0, 1.9)], []) == {}

    def test_backchannel_single_word_excluded(self):
        assert select_clips([_para(0, 5, text="joo")], []) == {}

    def test_backchannel_phrase_excluded(self):
        assert select_clips([_para(0, 5, text="hyvä kysymys")], []) == {}

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
    def test_window_is_middle_five_seconds_for_long_turn(self):
        clips = select_clips([_para(10, 40)], [])["A"]  # 30 s turn
        assert clips[0].start_s == pytest.approx(22.5)
        assert clips[0].end_s == pytest.approx(27.5)

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
            pytest.approx(7.5),
            pytest.approx(37.5),
            pytest.approx(77.5),
        ]

    def test_longest_candidate_wins_within_third(self):
        # Two turns in first third of 14 s audio: both picked.
        clips = select_clips([_para(0, 8), _para(2, 14)], [])["A"]
        assert len(clips) == 2
        ws = {(round(w.start_s, 1), round(w.end_s, 1)) for w in clips}
        assert (1.5, 6.5) in ws and (5.5, 10.5) in ws

    def test_empty_third_falls_back_to_longest_remaining(self):
        paras = [_para(0, 4), _para(5, 15), _para(16, 21)]
        clips = select_clips(paras, [])["A"]
        assert len(clips) == 3
        assert any(w.start_s == pytest.approx(7.5) for w in clips)

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
