"""WAV header parsing for the FLEURS corpus generator.

Extracted from ``scripts/gen_real_speech_corpus.py`` (which stayed over the
500-line source cap) into its own single-responsibility module. Importable
both as a ``scripts`` module (``import wav_header``) and via
``importlib.util.spec_from_file_location`` from the tests.

The FLEURS parquet ships 16 kHz mono **32-bit float** WAV payloads, so the
duration must come from the real sample width read out of the header — a
hard-coded ``len(raw) / 4`` would silently misread a future 16-bit revision.
"""

from __future__ import annotations

import struct


def wav_duration_seconds(wav_bytes: bytes) -> float:
    """Duration of a WAV payload in seconds, from its real header.

    Reads the ``fmt `` and ``data`` chunks (skipping unknown ones), so the
    result is the true length regardless of the source's sample width. A
    non-PCM header or a truncated header raises ``ValueError`` rather than
    guessing a byte width.
    """
    try:
        header, payload = _parse_wav_chunks(wav_bytes)
    except (struct.error, IndexError, ValueError) as e:
        # _parse_wav_chunks raises ValueError for a non-RIFF/WAVE payload or
        # a struct.error for a truncated header; both map to the clean error.
        raise ValueError(f"unparseable WAV payload: {e}") from e
    if header is None or payload is None:
        raise ValueError("WAV payload has no fmt or data chunk")
    audio_format, channels, rate, _, sample_width = header
    # WAVE_FORMAT_PCM (1) and WAVE_FORMAT_IEEE_FLOAT (3) — the FLEURS parquet
    # ships float32; both carry an honest sample width in the header, so the
    # duration is exact either way.
    if audio_format not in (1, 3) or rate == 0:
        raise ValueError(
            f"unsupported WAV format {audio_format} at {rate} Hz; "
            f"expected PCM (1) or IEEE float (3) at a non-zero rate"
        )
    return len(payload) / (sample_width * max(channels, 1)) / rate


def _parse_wav_chunks(
    wav_bytes: bytes,
) -> tuple[tuple[int, int, int, int, int] | None, bytes | None]:
    """Parse a WAV payload into ``(fmt fields, data payload)``.

    ``fmt fields`` is ``(audio_format, channels, rate, block_align,
    sample_width_bytes)``; either side is ``None`` when the chunk is absent.
    Unknown chunks are skipped (robust to the FLEURS layout); chunk offsets
    account for the RIFF odd-size pad byte.
    """
    if wav_bytes[:4] != b"RIFF" or wav_bytes[8:12] != b"WAVE":
        raise ValueError("not a RIFF/WAVE payload")
    header: tuple[int, int, int, int, int] | None = None
    payload: bytes | None = None
    off = 12
    while off + 8 <= len(wav_bytes):
        chunk_id = wav_bytes[off : off + 4]
        size = struct.unpack("<I", wav_bytes[off + 4 : off + 8])[0]
        body = wav_bytes[off + 8 : off + 8 + size]
        if chunk_id == b"fmt " and len(body) >= 14:
            # FLEURS payloads carry an 18-byte fmt whose 4th field is the
            # float32 sample width (not the standard block align); take it as
            # the width — exact for the float32 source and any 16-bit revision.
            audio_format, channels, rate, _block_align, sample_width = struct.unpack(
                "<HHIIH", body[:14]
            )
            header = (audio_format, channels, rate, _block_align, sample_width)
        elif chunk_id == b"data" and payload is None:
            payload = body
        # An odd-sized chunk is followed by one pad byte not counted in size;
        # (size & 1) accounts for it.
        off += 8 + size + (size & 1)
    return header, payload
