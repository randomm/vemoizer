"""Loudnorm measurement: OSError handling in path resolution (issue #135, fix pass 2).

The spec says: `_measurement_for`: `path.resolve()` is outside the
try/except that guards `stat()`; resolve can raise OSError: wrap it too
(skip the cache and fail open to a direct measurement).

This test verifies that an injected OSError from `path.resolve()` is
caught and the measurement falls back to a direct `_measure_loudnorm`
call (fail open).
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from vemoizer import loudnorm as ln
from vemoizer.loudnorm import LoudnormMeasurement


@pytest.fixture(autouse=True)
def _clean_cache():
    ln._MEASURE_CACHE.clear()
    yield
    ln._MEASURE_CACHE.clear()


def _make_wav(path: Path, seconds: float = 0.2) -> Path:
    """A tiny mono 16 kHz 16-bit PCM WAV (a constant tone)."""
    import struct

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


def test_resolve_oserror_fails_open(tmp_path: Path, monkeypatch) -> None:
    """An OSError from path.resolve() is caught and the measurement
    falls back to a direct _measure_loudnorm call (fail open)."""
    fixture = _make_wav(tmp_path / "x.wav", 0.2)
    m = LoudnormMeasurement(-16.0, -3.0, 0.5, -26.0, 0.0)
    calls: list[str] = []

    def fake_measure(p: Path):
        calls.append("measure")
        return m

    monkeypatch.setattr(ln, "_measure_loudnorm", fake_measure)

    # Patch Path.resolve to raise OSError
    original_resolve = Path.resolve

    def fake_resolve(self):
        if self == fixture:
            raise OSError("resolve failed")
        return original_resolve(self)

    monkeypatch.setattr(Path, "resolve", fake_resolve)

    # The measurement should still work (fail open to direct measurement)
    result = ln.measurement_for(fixture)

    assert result == m  # the measurement succeeded
    assert calls == ["measure"]  # _measure_loudnorm was called directly
    assert len(ln._MEASURE_CACHE) == 0  # no cache entry (resolve failed)
