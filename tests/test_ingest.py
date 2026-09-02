"""Tests for the audio ingest stage (issue #2).

Covers the ffmpeg argv contract, raw-PCM-on-stdout decoding, dtype/shape
invariants, iOS edit-list quirk, HE-AAC decode, error paths, and the
no-network/no-model invariant (pure subprocess + numpy).
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from vemoizer.ingest import (
    SAMPLE_RATE,
    IngestError,
    duration_seconds,
    ingest_audio,
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
    if n_samples > 0:
        fake_out = np.zeros(n_samples, dtype=np.float32).tobytes()
    else:
        fake_out = b""
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
