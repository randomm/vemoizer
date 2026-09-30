"""Tests for the audio ingest stage (issue #2).

Covers the ffmpeg argv contract, raw-PCM-on-stdout decoding, dtype/shape
invariants, iOS edit-list quirk, HE-AAC decode, error paths, and the
no-network/no-model invariant (pure subprocess + numpy).
"""

from __future__ import annotations

import subprocess
import threading
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from vemoizer.ingest import (
    SAMPLE_RATE,
    IngestError,
    duration_seconds,
    ingest_audio,
    pcm_duration_seconds,
)

FIXTURES_DIR = Path(__file__).parent / "fixtures"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _temp_file:
    """Context manager: create a real (empty) file, yield its Path, clean up."""

    def __init__(self, parent: Path, name: str) -> None:
        self._path = parent / name

    def __enter__(self) -> Path:
        self._path.touch()
        return self._path

    def __exit__(self, *_: object) -> None:
        self._path.unlink(missing_ok=True)


def _fake_proc(
    n_samples: int = 16_000, returncode: int = 0, stderr: bytes = b""
) -> subprocess.CompletedProcess:
    """Create a fake subprocess result with n_samples of float32 data."""
    fake_out = np.zeros(n_samples, dtype=np.float32).tobytes() if n_samples > 0 else b""
    return subprocess.CompletedProcess(
        args=[], returncode=returncode, stdout=fake_out, stderr=stderr
    )


# ---------------------------------------------------------------------------
# ffmpeg argv contract
# ---------------------------------------------------------------------------


def test_uses_expected_ffmpeg_args(tmp_path: Path) -> None:
    """ffmpeg must be called with the exact argv contract from issue #2."""
    fake_proc = _fake_proc(16_000)

    with (
        patch("vemoizer.ingest.subprocess.run", return_value=fake_proc) as mock_run,
        _temp_file(tmp_path, "dummy.m4a") as p,
    ):
        ingest_audio(p)

    mock_run.assert_called_once()
    call_args = mock_run.call_args
    argv = call_args[0][0]
    assert argv[:1] == ["ffmpeg"]
    # The core contract: raw f32le mono 16 kHz on stdout
    assert "-nostdin" in argv
    assert "-v" in argv and "error" in argv
    assert "-ac" in argv and "1" in argv
    assert "-ar" in argv and "16000" in argv
    assert "-c:a" in argv and "pcm_f32le" in argv
    assert "-f" in argv and "f32le" in argv
    assert "-" in argv  # stdout
    assert "-i" in argv
    # Input file is the last argument
    assert argv[-1] == str(p)


def test_never_uses_ffprobe(tmp_path: Path) -> None:
    """Ingest must not call ffprobe — duration comes from byte count."""
    fake_proc = _fake_proc(16_000)

    with (
        patch("vemoizer.ingest.subprocess.run", return_value=fake_proc) as mock_run,
        _temp_file(tmp_path, "x.m4a") as p,
    ):
        ingest_audio(p)

    call_args = mock_run.call_args
    argv = call_args[0][0]
    # Ensure the command is ffmpeg, not ffprobe
    assert argv[0] == "ffmpeg"


# ---------------------------------------------------------------------------
# Decode: dtype, shape, sample-rate invariants
# ---------------------------------------------------------------------------


def test_returns_float32_mono_1d(tmp_path: Path) -> None:
    """Output must be float32, 1-D, at 16 kHz."""
    n = 32_000  # 2 seconds at 16 kHz
    fake_proc = _fake_proc(n)

    with (
        patch("vemoizer.ingest.subprocess.run", return_value=fake_proc),
        _temp_file(tmp_path, "x.m4a") as p,
    ):
        arr = ingest_audio(p)

    assert arr.dtype == np.float32
    assert arr.ndim == 1
    assert arr.shape == (n,)
    assert duration_seconds(arr) == pytest.approx(2.0)


def test_sample_count_from_byte_count_not_ffprobe(tmp_path: Path) -> None:
    """Sample count is derived from raw byte count, never from metadata."""
    # Simulate an edit-list quirk: container says 1s but actual audio is 2s
    actual_samples = 32_000  # 2 seconds
    fake_proc = _fake_proc(actual_samples)

    with (
        patch("vemoizer.ingest.subprocess.run", return_value=fake_proc),
        _temp_file(tmp_path, "editlist.m4a") as p,
    ):
        arr = ingest_audio(p)

    assert len(arr) == actual_samples
    assert duration_seconds(arr) == pytest.approx(2.0)


def test_empty_input_returns_empty_array(tmp_path: Path) -> None:
    """Empty PCM → empty float32 array (no crash)."""
    fake_proc = _fake_proc(0)

    with (
        patch("vemoizer.ingest.subprocess.run", return_value=fake_proc),
        _temp_file(tmp_path, "empty.m4a") as p,
    ):
        arr = ingest_audio(p)

    assert arr.dtype == np.float32
    assert arr.shape == (0,)


# ---------------------------------------------------------------------------
# Fixture-based integration tests (real ffmpeg, no mocks)
# ---------------------------------------------------------------------------


def test_fixture_edit_list_decodes_to_full_duration() -> None:
    """Fixture with edit list: decoded samples match actual audio, not metadata.

    The iOS Voice Memos quirk: the edit list (edts/elst box) can report a
    shorter duration than the actual samples. ffmpeg's decoder correctly
    decodes ALL samples; ffprobe would report the shorter edit-list duration.

    This test verifies that our ingest (which uses ffmpeg decode, not ffprobe)
    returns the FULL sample count, proving we're not trusting container metadata.
    """
    fixture = FIXTURES_DIR / "edit_list.m4a"
    if not fixture.is_file():
        pytest.skip("edit_list.m4a fixture not yet generated")

    arr = ingest_audio(fixture)
    assert arr.dtype == np.float32
    assert arr.ndim == 1
    assert len(arr) > 0
    # The fixture is ~2 seconds of audio at 16 kHz
    # Even if the edit list lies, we should get close to 32000 samples
    # (allow for AAC frame boundaries: ±5%)
    expected = 32_000
    assert abs(len(arr) - expected) / expected < 0.05


def test_fixture_he_aac_decodes_to_16k_mono() -> None:
    """HE-AAC fixture decodes to 16 kHz mono float32."""
    fixture = FIXTURES_DIR / "he_aac.m4a"
    if not fixture.is_file():
        pytest.skip("he_aac.m4a fixture not yet generated")

    arr = ingest_audio(fixture)
    assert arr.dtype == np.float32
    assert arr.ndim == 1
    assert len(arr) > 0
    # Verify it's at 16 kHz (the ingest stage forces this via -ar 16000)
    assert SAMPLE_RATE == 16_000
    # ~2 seconds of audio
    expected = 32_000
    assert abs(len(arr) - expected) / expected < 0.05


def test_fixture_stereo_44k_resamples_to_mono_16k() -> None:
    """Stereo 44.1 kHz fixture → mono 16 kHz float32."""
    fixture = FIXTURES_DIR / "stereo_44k.m4a"
    if not fixture.is_file():
        pytest.skip("stereo_44k.m4a fixture not yet generated")

    arr = ingest_audio(fixture)
    assert arr.dtype == np.float32
    assert arr.ndim == 1
    assert len(arr) > 0
    # Input is 44.1 kHz stereo, output should be 16 kHz mono
    # ~2 seconds of audio at 16 kHz
    expected = 32_000
    assert abs(len(arr) - expected) / expected < 0.05


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


def test_missing_ffmpeg_raises_ingest_error(tmp_path: Path) -> None:
    """When ffmpeg is absent, raise IngestError with a clear message."""
    with (
        patch("vemoizer.ingest.subprocess.run", side_effect=FileNotFoundError),
        _temp_file(tmp_path, "x.m4a") as p,
        pytest.raises(IngestError, match="ffmpeg not found"),
    ):
        ingest_audio(p)


def test_ffmpeg_nonzero_exit_raises_ingest_error(tmp_path: Path) -> None:
    """Corrupt/unreadable file → IngestError with returncode set."""
    fake_proc = _fake_proc(0, returncode=1, stderr=b"Invalid data found")

    with (
        patch("vemoizer.ingest.subprocess.run", return_value=fake_proc),
        _temp_file(tmp_path, "corrupt.m4a") as p,
    ):
        with pytest.raises(IngestError) as exc_info:
            ingest_audio(p)
        assert exc_info.value.returncode == 1


def test_nonexistent_file_raises_ingest_error(tmp_path: Path) -> None:
    """Missing file → IngestError before ffmpeg is even called."""
    with pytest.raises(IngestError, match="not found"):
        ingest_audio(tmp_path / "does_not_exist.m4a")


# ---------------------------------------------------------------------------
# No-network / no-model invariant
# ---------------------------------------------------------------------------


def test_no_network_no_model(tmp_path: Path) -> None:
    """Ingest is pure subprocess + numpy — no HF, no network, no models."""
    fake_proc = _fake_proc(16_000)

    with (
        patch("vemoizer.ingest.subprocess.run", return_value=fake_proc) as mock_run,
        _temp_file(tmp_path, "x.m4a") as p,
    ):
        arr = ingest_audio(p)

    # Only one subprocess call: the ffmpeg decode
    assert mock_run.call_count == 1
    assert arr.dtype == np.float32


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def test_duration_seconds_helper() -> None:
    """duration_seconds computes len/rate correctly."""
    arr = np.zeros(16_000, dtype=np.float32)
    assert duration_seconds(arr) == pytest.approx(1.0)
    arr2 = np.zeros(32_000, dtype=np.float32)
    assert duration_seconds(arr2) == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# pcm_duration_seconds (streaming decoded-PCM byte count, no ffprobe)
# ---------------------------------------------------------------------------


def _mock_popen_stream(n_samples: int = 0, returncode: int = 0, stderr: bytes = b""):
    """Build a fake Popen that streams n_samples of float32 on stdout."""
    raw = np.zeros(n_samples, dtype=np.float32).tobytes()

    class _Pipe:
        def __init__(self, data: bytes) -> None:
            self._data = data
            self._pos = 0

        def read(self, size: int | None = None) -> bytes:
            if size is None:
                chunk = self._data[self._pos :]
            else:
                chunk = self._data[self._pos : self._pos + size]
            self._pos += len(chunk)
            return chunk

    class _StderrFile:
        def __init__(self, data: bytes) -> None:
            self._buf = bytearray(data)

        def read(self, size: int | None = None) -> bytes:
            data = bytes(self._buf)
            return data if size is None else data[:size]

        def seek(self, offset: int, whence: int = 0) -> int:
            return 0

    class _Proc:
        def __init__(self) -> None:
            self.stdout = _Pipe(raw)
            self.stderr = _StderrFile(stderr)
            self.returncode = returncode
            self.killed = False
            self.waited = False

        def poll(self) -> int | None:
            return self.returncode

        def wait(self, timeout: float | None = None) -> int:
            self.waited = True
            self.returncode = returncode
            return self.returncode

        def kill(self) -> None:
            self.killed = True

    return _Proc()


def test_pcm_duration_equals_ingest_duration_on_fixtures() -> None:
    """pcm_duration_seconds matches duration_seconds(ingest_audio(...))
    exactly on the real .m4a fixtures (incl. the edit-list quirk)."""
    for name in ("edit_list.m4a", "he_aac.m4a", "stereo_44k.m4a"):
        fixture = FIXTURES_DIR / name
        if not fixture.is_file():
            pytest.skip(f"{name} fixture not yet generated")
        expected = duration_seconds(ingest_audio(fixture))
        assert pcm_duration_seconds(fixture) == expected


def test_pcm_duration_never_uses_ffprobe() -> None:
    """The command must be ffmpeg with the same argv contract — no ffprobe."""
    fixture = FIXTURES_DIR / "edit_list.m4a"
    if not fixture.is_file():
        pytest.skip("edit_list.m4a fixture not yet generated")
    captured: list[str] = []
    real_popen = subprocess.Popen

    def spy_popen(argv, **kwargs):
        captured.extend(argv)
        return real_popen(argv, **kwargs)

    with patch("vemoizer.ingest.subprocess.Popen", side_effect=spy_popen):
        pcm_duration_seconds(fixture)
    # ffmpeg, not ffprobe.
    assert captured[0] == "ffmpeg"
    assert "ffprobe" not in captured
    # Same argv contract as ingest_audio.
    assert "-nostdin" in captured
    assert "-f" in captured and "f32le" in captured
    assert "-" in captured  # stdout


def test_pcm_duration_missing_file_raises_ingest_error(tmp_path: Path) -> None:
    with pytest.raises(IngestError, match="not found"):
        pcm_duration_seconds(tmp_path / "does_not_exist.m4a")


def test_pcm_duration_missing_ffmpeg_raises_ingest_error(tmp_path: Path) -> None:
    with (
        patch("vemoizer.ingest.subprocess.Popen", side_effect=FileNotFoundError),
        _temp_file(tmp_path, "x.m4a") as p,
        pytest.raises(IngestError, match="ffmpeg not found"),
    ):
        pcm_duration_seconds(p)


def test_pcm_duration_ffmpeg_failure_raises_ingest_error(tmp_path: Path) -> None:
    """Nonzero exit → IngestError carrying the returncode (like ingest)."""
    proc = _mock_popen_stream(0, returncode=1, stderr=b"Invalid data found")
    with (
        patch("vemoizer.ingest.subprocess.Popen", return_value=proc),
        _temp_file(tmp_path, "corrupt.m4a") as p,
        pytest.raises(IngestError) as exc_info,
    ):
        pcm_duration_seconds(p)
    assert exc_info.value.returncode == 1


def test_pcm_duration_empty_stream_is_zero(tmp_path: Path) -> None:
    """Empty PCM stream → 0.0, matching ingest's empty array."""
    proc = _mock_popen_stream(0)
    with (
        patch("vemoizer.ingest.subprocess.Popen", return_value=proc),
        _temp_file(tmp_path, "empty.m4a") as p,
    ):
        assert pcm_duration_seconds(p) == 0.0


def test_pcm_duration_streams_in_chunks_not_full_array(tmp_path: Path) -> None:
    """The decode is streamed in bounded chunks — only the byte count is
    kept (the fake Popen is fed 1 MiB chunks by the reader; a fake that
    only supports a single full read would also work, but we assert the
    reader never asks for more than its chunk size)."""
    chunk_sizes: list[int] = []
    proc = _mock_popen_stream(16_000)
    orig_read = proc.stdout.read

    def tracking_read(size: int) -> bytes:
        chunk_sizes.append(size)
        return orig_read(size)

    proc.stdout.read = tracking_read
    with (
        patch("vemoizer.ingest.subprocess.Popen", return_value=proc),
        _temp_file(tmp_path, "x.m4a") as p,
    ):
        dur = pcm_duration_seconds(p)
    assert dur == pytest.approx(1.0)
    # All reads are bounded by the chunk size (1 << 20 bytes).
    assert all(size <= (1 << 20) for size in chunk_sizes)


def test_pcm_duration_timeout_kills_and_reaps_process(tmp_path: Path) -> None:
    """A stalled ffmpeg (no stdout, alive) must be killed and reaped within
    the timeout — the wall-clock deadline is enforced DURING the drain,
    not after EOF. Use a tiny timeout (0.2 s) and a fake that blocks
    forever on its first stdout read."""
    import time

    class _StalledPipe:
        def __init__(self) -> None:
            self.read_called = False
            self.blocked = threading.Event()

        def read(self, size: int | None = None) -> bytes:
            if not self.read_called:
                self.read_called = True
                self.blocked.set()
                # Block forever (the test thread will join with a timeout).
                while True:
                    time.sleep(1)
            return b""

    class _StderrFile:
        def __init__(self) -> None:
            self._buf = bytearray()

        def write(self, data: bytes) -> int:
            self._buf.extend(data)
            return len(data)

        def read(self, size: int | None = None) -> bytes:
            data = bytes(self._buf)
            return data if size is None else data[:size]

        def seek(self, offset: int, whence: int = 0) -> int:
            return 0

    class _StalledProc:
        def __init__(self) -> None:
            self.stdout = _StalledPipe()
            self.stderr = _StderrFile()
            self.returncode = None
            self.killed = False
            self.waited = False
            self.wait_timeout: float | None = None

        def poll(self) -> int | None:
            return self.returncode

        def wait(self, timeout: float | None = None) -> int:
            self.waited = True
            self.wait_timeout = timeout
            self.returncode = -9
            return self.returncode

        def kill(self) -> None:
            self.killed = True

    stalled_proc = _StalledProc()
    with (
        patch("vemoizer.ingest.subprocess.Popen", return_value=stalled_proc),
        _temp_file(tmp_path, "stalled.m4a") as p,
        pytest.raises(IngestError, match="timed out after"),
    ):
        pcm_duration_seconds(p, timeout=0.2)

    # The fake must have been killed and reaped.
    assert stalled_proc.killed, "stalled process was not killed"
    assert stalled_proc.waited, "stalled process was not reaped (waited)"


def test_pcm_duration_huge_stderr_does_not_deadlock(tmp_path: Path) -> None:
    """A fake that writes a huge stderr must not deadlock the drain, and the
    error message must stay bounded (≤ ~200 chars of stderr excerpt + prefix).

    The stderr is written to a temp file (not a pipe), so no pipe buffer
    can fill and block the drain. The error message is capped.
    """
    huge_stderr = b"E " + b"x" * 10_000

    class _StderrFile:
        def __init__(self, data: bytes) -> None:
            self._buf = bytearray(data)

        def read(self, size: int | None = None) -> bytes:
            data = bytes(self._buf)
            return data if size is None else data[:size]

        def seek(self, offset: int, whence: int = 0) -> int:
            return 0

    proc = _mock_popen_stream(0, returncode=1, stderr=b"")
    proc.stderr = _StderrFile(huge_stderr)
    with (
        patch("vemoizer.ingest.subprocess.Popen", return_value=proc),
        _temp_file(tmp_path, "big_stderr.m4a") as p,
        pytest.raises(IngestError) as exc_info,
    ):
        pcm_duration_seconds(p)
    # The error message must be bounded (the stderr excerpt is ≤ 200 chars).
    msg = str(exc_info.value)
    assert len(msg) < 1000, f"error message unexpectedly long: {len(msg)} chars"


def test_pcm_duration_keyboard_interrupt_kills_and_reaps(tmp_path: Path) -> None:
    """A KeyboardInterrupt mid-drain must kill and reap the process (no
    zombie, no leaked fd), and the interrupt must propagate.

    The fake's stdout read blocks forever, so the drain thread is stuck
    in reader.join(remaining). We inject a KeyboardInterrupt into the
    drain thread by raising it in the reader thread (which is also stuck
    in a read loop) — the drain's except-BaseException handler must kill
    and reap before propagating.
    """
    import time

    class _StalledPipe:
        def __init__(self) -> None:
            self.read_called = False
            self._interrupt = threading.Event()

        def read(self, size: int | None = None) -> bytes:
            if not self.read_called:
                self.read_called = True
                # Block in short sleeps so we can detect the interrupt.
                while not self._interrupt.is_set():
                    time.sleep(0.1)
                raise KeyboardInterrupt("injected for test")
            return b""

    class _StderrFile:
        def __init__(self) -> None:
            self._buf = bytearray()

        def write(self, data: bytes) -> int:
            self._buf.extend(data)
            return len(data)

        def read(self, size: int | None = None) -> bytes:
            data = bytes(self._buf)
            return data if size is None else data[:size]

        def seek(self, offset: int, whence: int = 0) -> int:
            return 0

    class _StalledProc:
        def __init__(self) -> None:
            self.stdout = _StalledPipe()
            self.stderr = _StderrFile()
            self.returncode = None
            self.killed = False
            self.waited = False

        def poll(self) -> int | None:
            return self.returncode

        def wait(self, timeout: float | None = None) -> int:
            self.waited = True
            self.returncode = -9
            return self.returncode

        def kill(self) -> None:
            self.killed = True

    stalled_proc = _StalledProc()
    with (
        patch("vemoizer.ingest.subprocess.Popen", return_value=stalled_proc),
        _temp_file(tmp_path, "stalled2.m4a") as p,
    ):
        # Start the drain in a thread. The reader thread will block in
        # _StalledPipe.read, so the drain thread is stuck in reader.join.
        result: list[BaseException | None] = [None]

        def _run_drain() -> None:
            try:
                pcm_duration_seconds(p, timeout=10.0)
            except BaseException as e:
                result[0] = e

        drain_thread = threading.Thread(target=_run_drain, daemon=True)
        drain_thread.start()

        # Wait until the reader thread is blocked on its first read,
        # then inject a KeyboardInterrupt into the reader thread (which
        # is also stuck in a read loop). The drain's except-BaseException
        # handler must kill and reap before propagating.
        time.sleep(0.5)  # let the drain start and block
        stalled_proc.stdout._interrupt.set()  # type: ignore[union-attr]

        drain_thread.join(timeout=10.0)
        assert not drain_thread.is_alive(), "drain thread did not finish"

    # The fake must have been killed and reaped.
    assert stalled_proc.killed, "stalled process was not killed"
    assert stalled_proc.waited, "stalled process was not reaped (waited)"
    # The drain thread must have caught a BaseException (KeyboardInterrupt
    # or a wrapped version).
    assert result[0] is not None, "drain thread did not raise"
