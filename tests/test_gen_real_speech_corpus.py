"""Tests for ``scripts/gen_real_speech_corpus.py`` WAV parsing helpers.

The script is imported via ``importlib`` from ``scripts/`` (it is not a
package module). The WAV parsing functions are pure and need no models or
network.
"""

from __future__ import annotations

import importlib.util
import struct
import sys
from pathlib import Path

import pytest

_SCRIPT = (
    Path(__file__).resolve().parent.parent / "scripts" / "gen_real_speech_corpus.py"
)


def _load_script():
    spec = importlib.util.spec_from_file_location("gen_real_speech_corpus", _SCRIPT)
    if spec is None:
        raise RuntimeError(f"could not load {_SCRIPT}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["gen_real_speech_corpus"] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


gen = _load_script()
wav_duration_seconds = gen.wav_duration_seconds


def _make_wav(chunks: list[tuple[bytes, bytes]]) -> bytes:
    """Build a RIFF/WAVE file from ``(chunk_id, body)`` pairs.

    Applies the RIFF spec's word-alignment pad byte (one ``\\x00`` after
    any odd-sized chunk body) so the result is a spec-correct file.
    """
    body = b""
    for cid, payload in chunks:
        body += cid + struct.pack("<I", len(payload)) + payload
        if len(payload) % 2:
            body += b"\x00"
    return b"RIFF" + struct.pack("<I", 4 + len(body)) + b"WAVE" + body


def _fmt32(rate: int = 16000) -> bytes:
    """Standard 16-byte fmt chunk: IEEE float (3), mono, *rate*, 32-bit."""
    return struct.pack("<HHIIHH", 3, 1, rate, 64, 4, 32)


def _make_fleurs_row(cid: int, duration_s: float, n_words: int) -> dict[str, object]:
    """Build a FLEURS-style row dict with a synthetic float32 WAV payload.

    The payload's length is chosen so that
    ``wav_duration_seconds`` returns approximately *duration_s*.
    """
    n_bytes = int(duration_s * 16000 * 4)
    wav = _make_wav([(b"fmt ", _fmt32()), (b"data", b"\x00" * n_bytes)])
    return {
        "id": cid,
        "transcription": " ".join(["sana"] * n_words),
        "audio": {"bytes": wav, "path": f"row-{cid}"},
    }


def _fmt16(rate: int = 16000) -> bytes:
    """Standard 16-byte fmt chunk: PCM (1), mono, *rate*, 16-bit."""
    return struct.pack("<HHIIHH", 1, 1, rate, rate, 2, 16)


def _data(n_frames: int, sample_width: int = 2) -> bytes:
    return b"\x00\x01" * (n_frames * sample_width)


class TestWavDurationSeconds:
    """``wav_duration_seconds`` on synthetic WAV payloads."""

    def test_simple_pcm(self) -> None:
        wav = _make_wav([(b"fmt ", _fmt16()), (b"data", _data(800))])
        # _data(800) -> 800*2 pairs = 3200 bytes = 1600 frames -> 0.1 s
        assert wav_duration_seconds(wav) == pytest.approx(0.1)

    def test_float32(self) -> None:
        # 32-bit float (format 3), 16 kHz mono
        fmt = struct.pack("<HHIIHH", 3, 1, 16000, 64, 4, 32)
        # _data(400, 4) -> 400*4 pairs = 3200 bytes = 800 frames -> 0.05 s
        wav = _make_wav([(b"fmt ", fmt), (b"data", _data(400, 4))])
        assert wav_duration_seconds(wav) == pytest.approx(0.05)

    def test_odd_chunk_before_fmt_pad_byte(self) -> None:
        """Spec-correct: odd-sized unknown chunk followed by its pad byte.

        The RIFF spec says a chunk with an odd size is followed by one pad
        byte that is NOT counted in the chunk's size field. The parser's
        ``off += 8 + size + (size & 1)`` advance is correct for such files.
        A malformed file missing the pad byte (what the reviewer probably
        built) fails cleanly with ``ValueError``.
        """
        junk = b"junk1"  # 5 bytes — odd
        # _data(800) -> 3200 bytes = 1600 frames -> 0.1 s at 16 kHz
        wav = _make_wav([(b"junk", junk), (b"fmt ", _fmt16()), (b"data", _data(800))])
        assert wav_duration_seconds(wav) == pytest.approx(0.1)

    def test_odd_chunk_no_pad_byte_fails_cleanly(self) -> None:
        """Malformed: odd-sized unknown chunk WITHOUT the pad byte.

        The parser's word-aligned advance skips past the next chunk header
        and finds no ``fmt`` or ``data`` chunk; the documented clean
        ``ValueError`` is raised (not a bare ``IndexError`` or misread).
        """
        body = b""
        body += b"junk" + struct.pack("<I", 5) + b"junk1"  # no pad byte
        body += b"fmt " + struct.pack("<I", 16) + _fmt16()
        body += b"data" + struct.pack("<I", 3200) + _data(1600)
        bad = b"RIFF" + struct.pack("<I", 4 + len(body)) + b"WAVE" + body
        with pytest.raises(ValueError, match="WAV payload has no fmt or data chunk"):
            wav_duration_seconds(bad)

    def test_non_riff_payload(self) -> None:
        """A non-RIFF payload raises the documented clean ``ValueError``."""
        with pytest.raises(ValueError, match="unparseable WAV payload"):
            wav_duration_seconds(b"NOTAWAVFILE000000000000")

    def test_empty_payload(self) -> None:
        with pytest.raises(ValueError, match="unparseable WAV payload"):
            wav_duration_seconds(b"")

    def test_truncated_payload(self) -> None:
        with pytest.raises(ValueError, match="unparseable WAV payload"):
            wav_duration_seconds(b"RIFF")


class TestSelectClipsIdsPath:
    """``select_clips`` with *ids* — the recorded-id path (issue #62).

    FLEURS rows repeat an ``id`` across speakers (different takes of the
    same reference), and the duration/word WINDOW is exactly how the wanted
    take is picked among them — the window applies on the ``--ids`` path
    too. After window-filtering, each id's in-window row count must equal
    its recorded take count (duplicates in the id list); a drift is a
    ``ValueError`` naming the id, the recorded take count and the in-window
    count, so a same-id take drifting out of the window (or a new same-id
    row entering it) can never re-letter a stem relative to
    ``CORPUS_ATTRIBUTION.md``.
    """

    def test_ids_path_window_picks_the_wanted_take(self) -> None:
        """Only the in-window take of a repeated id is selected (no error).

        FLEURS id 24 has three rows across speakers; only one is in the
        duration/word window. The ids path must select exactly that one —
        the window is the take-selection, not a filter that raises.
        """
        rows = [
            _make_fleurs_row(24, 2.0, 10),  # take 1: outside window (too short)
            _make_fleurs_row(24, 4.0, 10),  # take 2: in window  <- the wanted one
            _make_fleurs_row(24, 7.0, 10),  # take 3: outside window (too long)
        ]
        clips = gen.select_clips(rows, seed=gen.SEED, n=28, ids=[24])
        assert len(clips) == 1
        assert clips[0].row_index == 1

    def test_ids_path_two_in_window_takes_select_stable_order(self) -> None:
        """An id recorded twice with two in-window takes: 0036 then 0036b.

        Both takes of id 36 fall in the window (as in the real corpus: the
        two takes of 36 and the two of 748 are the only multi-take ids).
        Selection is in (clip_id, row_index) order, so the stems come out
        0036 (first row) then 0036b (second row) — no re-lettering.
        """
        rows = [
            _make_fleurs_row(36, 4.0, 10),  # take 1: in window -> 0036
            _make_fleurs_row(36, 4.5, 10),  # take 2: in window -> 0036b
        ]
        clips = gen.select_clips(rows, seed=gen.SEED, n=28, ids=[36, 36])
        assert len(clips) == 2
        assert [c.row_index for c in clips] == [0, 1]
        assert all(c.clip_id == 36 for c in clips)

    def test_ids_path_recorded_twice_one_out_of_window_take_ok(self) -> None:
        """Of three same-id rows, two in-window: the recorded-two id is fine.

        Mimics the real id 36 (3 rows in the parquet, 2 in the window): the
        out-of-window take is ignored, the two in-window ones are selected
        in row order, no error.
        """
        rows = [
            _make_fleurs_row(36, 1.0, 10),  # out of window (too short)
            _make_fleurs_row(36, 4.0, 10),  # in window -> 0036
            _make_fleurs_row(36, 5.0, 10),  # in window -> 0036b
        ]
        clips = gen.select_clips(rows, seed=gen.SEED, n=28, ids=[36, 36])
        assert len(clips) == 2
        assert [c.row_index for c in clips] == [1, 2]

    def test_ids_path_zero_in_window_rows_raises(self) -> None:
        """A recorded id with no in-window rows is a clear error."""
        rows = [
            _make_fleurs_row(300, 1.0, 10),  # out of window (too short)
            _make_fleurs_row(300, 9.0, 10),  # out of window (too long)
        ]
        with pytest.raises(
            ValueError,
            match=r"clip id 300: recorded 1 take\(s\) but found 0 in-window row\(s\)",
        ):
            gen.select_clips(rows, seed=gen.SEED, n=28, ids=[300])

    def test_ids_path_extra_in_window_row_raises(self) -> None:
        """A recorded-once id with two in-window rows is a clear error.

        A new same-id row entering the window would re-letter the stem; the
        guard refuses instead of guessing.
        """
        rows = [
            _make_fleurs_row(400, 4.0, 10),
            _make_fleurs_row(400, 4.5, 10),  # extra in-window row
        ]
        with pytest.raises(
            ValueError,
            match=r"clip id 400: recorded 1 take\(s\) but found 2 in-window row\(s\)",
        ):
            gen.select_clips(rows, seed=gen.SEED, n=28, ids=[400])

    def test_ids_path_take_order_deterministic(self) -> None:
        """Selection order is deterministic: (clip_id, row_index), not file order.

        Rows are shuffled relative to row_index; the result must still come
        out sorted by (clip_id, row_index) so stems 0036 / 0036b / 0748 /
        0748b land on the recorded takes (matching the committed corpus).
        """
        rows = [
            _make_fleurs_row(748, 4.5, 10),  # row 0: id 748 take 2
            _make_fleurs_row(36, 4.0, 10),  # row 1: id 36 take 1
            _make_fleurs_row(24, 4.0, 10),  # row 2: id 24 (in window)
            _make_fleurs_row(36, 4.5, 10),  # row 3: id 36 take 2
            _make_fleurs_row(748, 4.0, 10),  # row 4: id 748 take 1
        ]
        clips = gen.select_clips(rows, seed=gen.SEED, n=28, ids=[24, 36, 36, 748, 748])
        assert [(c.clip_id, c.row_index) for c in clips] == [
            (24, 2),
            (36, 1),
            (36, 3),
            (748, 0),
            (748, 4),
        ]

    def test_ids_path_out_of_window_extra_row_does_not_raise(self) -> None:
        """An extra same-id row OUTSIDE the window is ignored (the window picks)."""
        rows = [
            _make_fleurs_row(500, 4.0, 10),  # in window -> selected
            _make_fleurs_row(500, 1.0, 10),  # out of window -> ignored
        ]
        clips = gen.select_clips(rows, seed=gen.SEED, n=28, ids=[500])
        assert len(clips) == 1
        assert clips[0].row_index == 0

    def test_ids_path_unparseable_audio_raises(self) -> None:
        """A recorded id whose audio fails to parse raises a clear error."""
        good_row = _make_fleurs_row(500, 4.0, 10)
        bad_row = {
            "id": 500,
            "transcription": "word " * 10,
            "audio": {"bytes": b"NOTAWAVFILE000000000000", "path": "bad"},
        }
        rows = [good_row, bad_row]
        with pytest.raises(ValueError, match="unparseable WAV payload"):
            gen.select_clips(rows, seed=gen.SEED, n=28, ids=[500])

    def test_ids_path_empty_ids_raises(self) -> None:
        with pytest.raises(ValueError, match="--ids was given but is empty"):
            gen.select_clips([], seed=gen.SEED, n=28, ids=[])

    def test_ids_path_word_window_also_applies(self) -> None:
        """The word-count half of the window also gates take selection."""
        rows = [
            _make_fleurs_row(600, 4.0, 3),  # in duration window, too few words
            _make_fleurs_row(600, 4.0, 10),  # fully in window -> selected
        ]
        clips = gen.select_clips(rows, seed=gen.SEED, n=28, ids=[600])
        assert len(clips) == 1
        assert clips[0].row_index == 1


class TestResolveCorpusDir:
    """``resolve_corpus_dir`` — the path-guard tests (issue #62, LENS LOW)."""

    def test_path_equal_to_project_root_is_refused(self, tmp_path: Path) -> None:
        """A path that resolves to the project root itself is refused.

        Previously the guard accepted ``resolved == project_root``, which
        would have allowed a symlink to the project root to pass.
        """
        project_root = Path(gen.__file__).resolve().parent.parent
        with pytest.raises(RuntimeError, match="refusing to write corpus"):
            gen.resolve_corpus_dir(project_root)

    def test_symlink_to_project_root_is_refused(self, tmp_path: Path) -> None:
        """A symlink whose target is the project root is refused.

        The old ``resolved == project_root`` check accepted this case;
        the new strict-descendant check rejects it.
        """
        project_root = Path(gen.__file__).resolve().parent.parent
        symlink = tmp_path / "link_to_root"
        symlink.symlink_to(project_root)
        # resolve() follows the symlink and yields the project root, which
        # is not a strict descendant of itself.
        with pytest.raises(RuntimeError, match="refusing to write corpus"):
            gen.resolve_corpus_dir(symlink)

    def test_strict_descendant_is_accepted(self, tmp_path: Path) -> None:
        """A directory under tests/fixtures/corpus (the default) is accepted."""
        # The default corpus dir is under the project root.
        default_dir = (
            Path(gen.__file__).resolve().parent.parent / "tests" / "fixtures" / "corpus"
        )
        resolved = gen.resolve_corpus_dir(default_dir)
        assert resolved.is_dir()
