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
