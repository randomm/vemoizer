"""The loudnorm measurement memo and the single pass-2 argv builder
(issue #135, lens fixes).

Covers:

- The measurement cache: an unchanged file is measured ONCE (pass-1
  call count), a changed file (size or mtime) is re-measured, and a
  cached failure (``None``) is not re-measured (one warning per call
  site, no spam).
- The single pass-2 argv builder: ``decode_argv`` (the duration decode)
  and ``ingest_audio``'s pass 2 (the transcript decode) use the SAME
  argv list for the same measurement (a test asserts equality with
  fakes — no ffmpeg needed).
- ``OSError`` handling in ``_measure_loudnorm``: a ``FileNotFoundError``
  (ffmpeg vanished) and an ``OSError`` from the stderr temp-file read
  both fail open to the plain decode (the docstring's "ANY failure"
  contract).

The measurement seam is ``_measure_loudnorm`` (the raw pass-1 call);
the cache sits in ``_measurement_for``. All fakes — no models, no
network, no real ffmpeg.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from vemoizer import loudnorm as ln
from vemoizer.loudnorm import (
    LoudnormMeasurement,
    decode_argv,
    loudnorm_pass2_filter,
)

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


def test_unchanged_file_measured_once(tmp_path: Path, monkeypatch) -> None:
    """(2) Two ``pcm_duration_seconds`` calls on the same unchanged file
    run pass 1 ONCE (the measurement is memoized)."""
    fixture = _make_wav(tmp_path / "x.wav", 0.2)
    calls: list[str] = []

    def fake_measure(p: Path):
        calls.append("measure")
        return _M

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

    with patch("vemoizer.ingest.subprocess.Popen", side_effect=fake_popen):
        from vemoizer.ingest import pcm_duration_seconds

        pcm_duration_seconds(fixture, preprocess="loudnorm")
        pcm_duration_seconds(fixture, preprocess="loudnorm")

    assert calls == ["measure"]  # pass 1 ran ONCE for two duration calls


def test_changed_file_remeasured(tmp_path: Path, monkeypatch) -> None:
    """(2) Modifying the file (size) forces a re-measurement (the cache
    key includes size and mtime)."""
    fixture = _make_wav(tmp_path / "x.wav", 0.2)
    calls: list[str] = []

    def fake_measure(p: Path):
        calls.append("measure")
        return _M

    monkeypatch.setattr(ln, "_measure_loudnorm", fake_measure)

    ln._measurement_for(fixture)
    # Change the size (append a byte) → the key changes → re-measured.
    with fixture.open("ab") as f:
        f.write(b"0")
    ln._measurement_for(fixture)

    assert calls == ["measure", "measure"]


def test_unchanged_mtime_only_change_remeasured(tmp_path: Path, monkeypatch) -> None:
    """(2) An mtime change (same size) also forces a re-measurement."""
    fixture = _make_wav(tmp_path / "x.wav", 0.2)
    calls: list[str] = []

    def fake_measure(p: Path):
        calls.append("measure")
        return _M

    monkeypatch.setattr(ln, "_measure_loudnorm", fake_measure)

    ln._measurement_for(fixture)
    import os

    old = fixture.stat()
    os.utime(fixture, (old.st_atime, old.st_mtime_ns // 10**9 + 10))
    ln._measurement_for(fixture)

    assert calls == ["measure", "measure"]


def test_cached_failure_not_remeasured_or_rewarned(
    tmp_path: Path, monkeypatch, caplog
) -> None:
    """(2) A failing measurement is cached (``None``): the second call
    does NOT re-measure (the cached ``None`` is used), so a failing file
    is measured exactly once even under repeated duration decodes."""
    fixture = _make_wav(tmp_path / "x.wav", 0.2)
    calls: list[str] = []

    def fake_measure(p: Path):
        calls.append("measure")
        return None  # the measurement failed

    monkeypatch.setattr(ln, "_measure_loudnorm", fake_measure)

    def fake_popen(argv, **kw):
        return _Popen()

    with patch("vemoizer.ingest.subprocess.Popen", side_effect=fake_popen):
        from vemoizer.ingest import pcm_duration_seconds

        pcm_duration_seconds(fixture, preprocess="loudnorm")
        pcm_duration_seconds(fixture, preprocess="loudnorm")

    # The measurement ran exactly once (the cached ``None`` was used the
    # second time); the failure is in the cache.
    assert calls == ["measure"]
    assert ln._MEASURE_CACHE  # the failure was cached (a ``None`` entry)


class _Popen:
    """A minimal fake Popen for the plain decode (no measurement)."""

    def __init__(self) -> None:
        self.returncode = 0
        self.stdout = _FakePipe(b"")
        self.stderr = b""

    def poll(self) -> int | None:
        return self.returncode

    def wait(self, timeout: float | None = None) -> int:
        return self.returncode

    def kill(self) -> None:
        pass


class _FakePipe:
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


def test_decode_argv_uses_single_builder(tmp_path: Path, monkeypatch) -> None:
    """(3) The duration decode's argv (``decode_argv``) equals the
    transcript's pass-2 argv (``ingest_audio``'s ``_decode_args``) for
    the same measurement — exactly ONE pass-2 argv builder."""
    fixture = _make_wav(tmp_path / "x.wav", 0.2)
    monkeypatch.setattr(ln, "_measurement_for", lambda p: _M)

    # The transcript's pass 2 (what ingest_audio builds via _decode_args):
    pass2_argv = ln._decode_args(("-af", loudnorm_pass2_filter(_M))) + [str(fixture)]
    # The duration decode's argv (decode_argv, the pcm_duration path):
    duration_argv = decode_argv(fixture, "loudnorm")

    assert duration_argv == pass2_argv


def test_decode_argv_no_flag_matches_plain(tmp_path: Path) -> None:
    """(3) No-flag ``decode_argv`` stays byte-identical to the plain
    decode (the existing contract the other tests pin)."""
    from vemoizer.ingest import _FFMPEG_AUDIO_ARGS

    fixture = _make_wav(tmp_path / "x.wav", 0.2)
    argv = decode_argv(fixture, None)
    assert argv == ["ffmpeg", *_FFMPEG_AUDIO_ARGS, "-i", str(fixture)]


def test_decode_argv_loudnorm_shape(tmp_path: Path, monkeypatch) -> None:
    """(3) The loudnorm ``decode_argv`` has the pass-2 filter in the
    correct position (after resample/downmix, before ``-i``) — the same
    shape the transcript's pass 2 uses."""
    fixture = _make_wav(tmp_path / "x.wav", 0.2)
    monkeypatch.setattr(ln, "_measurement_for", lambda p: _M)
    argv = decode_argv(fixture, "loudnorm")
    filt = loudnorm_pass2_filter(_M)
    assert "-af" in argv
    assert argv[argv.index("-af") + 1] == filt
    assert argv.index("-af") > argv.index("16000")  # after resample
    assert argv.index("-af") > argv.index("pcm_f32le")  # after codec
    assert argv.index("-af") < argv.index("-i")  # before the input
    assert argv[-1] == str(fixture)


def test_measure_loudnorm_file_not_found_fails_open(
    tmp_path: Path, monkeypatch
) -> None:
    """(4) A ``FileNotFoundError`` (ffmpeg vanished) in ``_measure_loudnorm``
    fails open (returns ``None``) — the docstring's "ANY failure"
    contract (previously only ``TimeoutExpired`` was caught)."""
    fixture = _make_wav(tmp_path / "x.wav", 0.2)

    def fake_popen(argv, **kw):
        raise FileNotFoundError("ffmpeg vanished")

    with patch("subprocess.Popen", side_effect=fake_popen):
        result = ln._measure_loudnorm(fixture)

    assert result is None  # fail open, no exception


def test_measure_loudnorm_temp_file_oserror_fails_open(
    tmp_path: Path, monkeypatch
) -> None:
    """(4) An ``OSError`` from the stderr temp-file read in
    ``_measure_loudnorm`` fails open (returns ``None``) — the
    docstring's "ANY failure" contract (previously only
    ``TimeoutExpired`` was caught)."""
    fixture = _make_wav(tmp_path / "x.wav", 0.2)

    class _OSErrorFile:
        """A fake temp file whose seek/read raises ``OSError``."""

        def seek(self, offset: int, whence: int = 0) -> None:
            raise OSError("temp file read failed")

        def read(self, size: int | None = None) -> bytes:
            raise OSError("temp file read failed")

        def close(self) -> None:
            pass

    def fake_tempfile():
        return _OSErrorFile()

    def fake_popen(argv, **kw):
        return _Popen()

    with (
        patch("subprocess.Popen", side_effect=fake_popen),
        patch.object(ln.tempfile, "TemporaryFile", side_effect=fake_tempfile),
    ):
        result = ln._measure_loudnorm(fixture)

    assert result is None  # fail open, no exception


def test_preprocess_audio_unchanged_file_measured_once(
    tmp_path: Path, monkeypatch
) -> None:
    """(2) ``ingest_audio`` (the transcript decode) and
    ``pcm_duration_seconds`` (the duration decode) on the same unchanged
    file run pass 1 ONCE in total (the cache is shared)."""
    fixture = _make_wav(tmp_path / "x.wav", 0.2)
    calls: list[str] = []

    def fake_measure(p: Path):
        calls.append("measure")
        return _M

    monkeypatch.setattr(ln, "_measure_loudnorm", fake_measure)

    def fake_popen(argv, **kw):
        return _Popen()

    def fake_run(argv, **kw):
        return subprocess.CompletedProcess(
            argv, 0, stdout=np.zeros(3200, dtype=np.float32).tobytes(), stderr=b""
        )

    with (
        patch("vemoizer.ingest.subprocess.Popen", side_effect=fake_popen),
        patch("vemoizer.ingest.subprocess.run", side_effect=fake_run),
    ):
        from vemoizer.ingest import ingest_audio, pcm_duration_seconds

        pcm_duration_seconds(fixture, preprocess="loudnorm")
        ingest_audio(fixture, preprocess="loudnorm")

    assert calls == ["measure"]  # pass 1 ran ONCE for both decodes
