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
    """``select_clips`` with *ids* — the recorded-id (authoritative) path.

    Bug scenario (issue #62, LENS MEDIUM): the recorded ``--ids`` list is the
    authoritative selection.  If a committed same-id take ever falls outside
    the duration/word window, the OLD code would silently apply the window
    filter and re-letter the stems, assigning them to different rows than
    ``CORPUS_ATTRIBUTION.md`` records.  The fix: on the ``--ids`` path, do
    NOT apply the window; instead require that each id's number of rows in
    the dataset matches the number of times it appears in the id list.
    """

    def test_duplicate_id_out_of_window_take_does_not_shift_stems(self) -> None:
        """Reproduction: id 36 has two takes; the first is outside the window.

        Under the OLD code, the window filter excluded row 1, so only one
        36-take row survived, and the second take (in-window) got stem
        ``fleurs_fi_0036`` instead of ``fleurs_fi_0036b``.  Under the fixed
        code, both takes are preserved regardless of the window, and the
        recorded take count (2) matches, so no ``ValueError`` is raised.
        """
        rows = [
            _make_fleurs_row(36, 1.0, 10),  # take 1: outside window (too short)
            _make_fleurs_row(36, 4.0, 10),  # take 2: in window
            _make_fleurs_row(748, 4.0, 10),  # take 1: in window
            _make_fleurs_row(748, 1.0, 10),  # take 2: outside window (too short)
        ]
        ids = [36, 36, 748, 748]  # 2 takes each
        clips = gen.select_clips(rows, seed=gen.SEED, n=28, ids=ids)
        # All 4 rows must be selected (window not applied on the ids path).
        assert len(clips) == 4
        # Verify stem assignment: 36 has 2 takes → "0036" and "0036b".
        by_id: dict[int, list[int]] = {}
        for c in clips:
            by_id.setdefault(c.clip_id, []).append(c.row_index)
        assert by_id[36] == [0, 1]  # both rows, in order
        assert by_id[748] == [2, 3]

    def test_ids_path_no_window_filter_applied(self) -> None:
        """A row outside the window is still selected when its id is recorded.

        The OLD code applied the window filter on the ids path, which would
        have silently excluded the out-of-window take and then raised a
        ``ValueError`` (missing take).  The fixed code skips the window
        entirely and selects all rows for the recorded ids.
        """
        rows = [
            _make_fleurs_row(100, 1.0, 10),  # in-window, not a recorded take
            _make_fleurs_row(200, 4.0, 10),  # recorded, in window
            _make_fleurs_row(200, 2.0, 10),  # recorded, outside window (too short)
        ]
        ids = [200, 200]  # 2 takes of id 200
        clips = gen.select_clips(rows, seed=gen.SEED, n=28, ids=ids)
        # Both recorded takes are selected (window not applied).
        assert len(clips) == 2
        assert all(c.clip_id == 200 for c in clips)

    def test_ids_path_missing_take_raises_value_error(self) -> None:
        """If the dataset has fewer rows for an id than recorded, raise ValueError.

        This is the guard that prevents a silent corpus swap when a same-id
        take is dropped from a revised parquet.
        """
        rows = [
            _make_fleurs_row(300, 4.0, 10),  # only one row for id 300
        ]
        ids = [300, 300]  # records 2 takes, but only 1 row exists
        with pytest.raises(ValueError, match="clip id 300"):
            gen.select_clips(rows, seed=gen.SEED, n=28, ids=ids)

    def test_ids_path_extra_row_raises_value_error(self) -> None:
        """If the dataset has more rows for an id than recorded, raise ValueError.

        Guards against a dataset that inserted a new same-id row, which would
        shift the stem letters relative to the committed attribution.
        """
        rows = [
            _make_fleurs_row(400, 4.0, 10),
            _make_fleurs_row(400, 4.5, 10),
            _make_fleurs_row(400, 5.0, 10),  # extra row not in the recorded list
        ]
        ids = [400, 400]  # records 2 takes, but 3 rows exist
        with pytest.raises(ValueError, match="clip id 400"):
            gen.select_clips(rows, seed=gen.SEED, n=28, ids=ids)

    def test_ids_path_unparseable_audio_raises(self) -> None:
        """A recorded id whose audio fails to parse raises a clear error."""
        good_row = _make_fleurs_row(500, 4.0, 10)
        bad_row = {
            "id": 500,
            "transcription": "word " * 10,
            "audio": {"bytes": b"NOTAWAVFILE000000000000", "path": "bad"},
        }
        rows = [good_row, bad_row]
        ids = [500, 500]
        with pytest.raises(ValueError, match="unparseable WAV payload"):
            gen.select_clips(rows, seed=gen.SEED, n=28, ids=ids)


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
