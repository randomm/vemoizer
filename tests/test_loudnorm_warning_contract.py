"""Loudnorm warning contract: exactly ONE warning per failing file
(issue #135, fix pass 2).

The spec says: "ONE warning per failing file" — not one per call site.
Today, `preprocess_audio` logs a warning on EVERY call that gets a cached
None while `decode_argv` is silent. This test verifies the corrected
contract:

- A failing file seen by `pcm_duration_seconds` x3 and `ingest_audio` x2
  logs exactly ONE warning in total.
- A changed file (size/mtime) that fails again warns again.
- A successful measurement logs none.

The warning fires inside the memoized function on a cache MISS that
returned None (file name only, no full path).
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from vemoizer import loudnorm as ln
from vemoizer.loudnorm import LoudnormMeasurement

_M = LoudnormMeasurement(-16.0, -3.0, 0.5, -26.0, 0.0)


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


def test_failing_file_logs_one_warning_total(
    tmp_path: Path, monkeypatch, caplog
) -> None:
    """A failing file seen by pcm_duration_seconds x3 and ingest_audio x2
    logs exactly ONE warning in total (the warning fires on the first
    cache MISS that returns None, not on every call site)."""
    fixture = _make_wav(tmp_path / "x.wav", 0.2)

    def fake_measure(p: Path):
        return None  # the measurement fails

    monkeypatch.setattr(ln, "_measure_loudnorm", fake_measure)

    class _Pipe:
        def __init__(self, data: bytes) -> None:
            self._data = data
            self._pos = 0

        def read(self, size: int | None = None) -> bytes:
            chunk = (
                self._data[self._pos :]
                if size is None
                else self._data[self._pos : self._pos + size]
            )
            self._pos += len(chunk)
            return chunk

    class _Proc:
        stdout = _Pipe(b"")
        returncode = 0

        def poll(self) -> int | None:
            return self.returncode

        def wait(self, timeout: float | None = None) -> int:
            return self.returncode

        def kill(self) -> None:
            pass

    def fake_popen(argv, **kw):
        return _Proc()

    def fake_run(argv, **kw):
        return subprocess.CompletedProcess(
            argv, 0, stdout=np.zeros(3200, dtype=np.float32).tobytes(), stderr=b""
        )

    with (
        patch("vemoizer.ingest.subprocess.Popen", side_effect=fake_popen),
        patch("vemoizer.ingest.subprocess.run", side_effect=fake_run),
        caplog.at_level("WARNING"),
    ):
        from vemoizer.ingest import ingest_audio, pcm_duration_seconds

        # 3x pcm_duration_seconds + 2x ingest_audio = 5 calls total
        pcm_duration_seconds(fixture, preprocess="loudnorm")
        pcm_duration_seconds(fixture, preprocess="loudnorm")
        pcm_duration_seconds(fixture, preprocess="loudnorm")
        ingest_audio(fixture, preprocess="loudnorm")
        ingest_audio(fixture, preprocess="loudnorm")

    # Exactly ONE warning in total (file name only, no full path)
    warnings = [r for r in caplog.records if r.levelname == "WARNING"]
    assert len(warnings) == 1, (
        f"expected 1 warning, got {len(warnings)}: {[r.message for r in warnings]}"
    )
    assert "x.wav" in warnings[0].message  # file name only
    assert str(fixture) not in warnings[0].message  # no full path


def test_changed_failing_file_warns_again(tmp_path: Path, monkeypatch, caplog) -> None:
    """A changed file (size/mtime) that fails again warns again (the cache
    key changes, so it's a fresh cache MISS)."""
    fixture = _make_wav(tmp_path / "x.wav", 0.2)
    calls: list[str] = []

    def fake_measure(p: Path):
        calls.append("measure")
        return None  # the measurement fails

    monkeypatch.setattr(ln, "_measure_loudnorm", fake_measure)

    class _Pipe:
        def __init__(self, data: bytes) -> None:
            self._data = data
            self._pos = 0

        def read(self, size: int | None = None) -> bytes:
            chunk = (
                self._data[self._pos :]
                if size is None
                else self._data[self._pos : self._pos + size]
            )
            self._pos += len(chunk)
            return chunk

    class _Proc:
        stdout = _Pipe(b"")
        returncode = 0

        def poll(self) -> int | None:
            return self.returncode

        def wait(self, timeout: float | None = None) -> int:
            return self.returncode

        def kill(self) -> None:
            pass

    def fake_popen(argv, **kw):
        return _Proc()

    def fake_run(argv, **kw):
        return subprocess.CompletedProcess(
            argv, 0, stdout=np.zeros(3200, dtype=np.float32).tobytes(), stderr=b""
        )

    with (
        patch("vemoizer.ingest.subprocess.Popen", side_effect=fake_popen),
        patch("vemoizer.ingest.subprocess.run", side_effect=fake_run),
        caplog.at_level("WARNING"),
    ):
        from vemoizer.ingest import ingest_audio

        # First call: cache MISS, measurement fails → warning #1
        ingest_audio(fixture, preprocess="loudnorm")

        # Change the file (append a byte) → new cache key → cache MISS
        with fixture.open("ab") as f:
            f.write(b"0")

        # Second call: cache MISS (new key), measurement fails → warning #2
        ingest_audio(fixture, preprocess="loudnorm")

    # TWO warnings total (one per cache MISS that failed)
    warnings = [r for r in caplog.records if r.levelname == "WARNING"]
    assert len(warnings) == 2, (
        f"expected 2 warnings (one per changed file), got {len(warnings)}: "
        f"{[r.message for r in warnings]}"
    )
    assert len(calls) == 2  # measured twice (once per change)


def test_successful_measurement_logs_no_warning(
    tmp_path: Path, monkeypatch, caplog
) -> None:
    """A successful measurement logs NO warning."""
    fixture = _make_wav(tmp_path / "x.wav", 0.2)

    def fake_measure(p: Path):
        return _M  # the measurement succeeds

    monkeypatch.setattr(ln, "_measure_loudnorm", fake_measure)

    class _Pipe:
        def __init__(self, data: bytes) -> None:
            self._data = data
            self._pos = 0

        def read(self, size: int | None = None) -> bytes:
            chunk = (
                self._data[self._pos :]
                if size is None
                else self._data[self._pos : self._pos + size]
            )
            self._pos += len(chunk)
            return chunk

    class _Proc:
        stdout = _Pipe(b"")
        returncode = 0

        def poll(self) -> int | None:
            return self.returncode

        def wait(self, timeout: float | None = None) -> int:
            return self.returncode

        def kill(self) -> None:
            pass

    def fake_popen(argv, **kw):
        return _Proc()

    def fake_run(argv, **kw):
        return subprocess.CompletedProcess(
            argv, 0, stdout=np.zeros(3200, dtype=np.float32).tobytes(), stderr=b""
        )

    with (
        patch("vemoizer.ingest.subprocess.Popen", side_effect=fake_popen),
        patch("vemoizer.ingest.subprocess.run", side_effect=fake_run),
        caplog.at_level("WARNING"),
    ):
        from vemoizer.ingest import ingest_audio, pcm_duration_seconds

        pcm_duration_seconds(fixture, preprocess="loudnorm")
        ingest_audio(fixture, preprocess="loudnorm")

    # NO warnings (the measurement succeeded)
    warnings = [r for r in caplog.records if r.levelname == "WARNING"]
    assert len(warnings) == 0, (
        f"expected 0 warnings, got {len(warnings)}: {[r.message for r in warnings]}"
    )
