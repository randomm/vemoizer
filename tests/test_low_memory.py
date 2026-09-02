"""Tests for the low-memory mode helper (issue #14, --low-memory).

The contract:
- total_ram_bytes() → int | None via `sysctl -n hw.memsize`
- default_low_memory() → True if RAM <= 16 GiB, False otherwise (including None → False)
- apply_low_memory_mode(enabled: bool) → no-op for now (pipeline wiring deferred)

Pattern: ported from kuiskaus test_silicon_check.py (sysctl subprocess mock).
"""

from __future__ import annotations

import subprocess
from unittest.mock import patch

import pytest

from vemoizer.low_memory import (
    apply_low_memory_mode,
    default_low_memory,
    total_ram_bytes,
)

GI = 1024**3


class TestTotalRamBytes:
    @pytest.fixture(autouse=True)
    def _patch_platform(self):
        with patch("vemoizer.low_memory.sys.platform", "darwin"):
            yield

    def test_16_gb(self):
        with patch(
            "subprocess.run",
            return_value=subprocess.CompletedProcess(
                args=[], returncode=0, stdout=str(16 * GI) + "\n"
            ),
        ):
            assert total_ram_bytes() == 16 * GI

    def test_32_gb(self):
        with patch(
            "subprocess.run",
            return_value=subprocess.CompletedProcess(
                args=[], returncode=0, stdout=str(32 * GI) + "\n"
            ),
        ):
            assert total_ram_bytes() == 32 * GI

    def test_8_gb(self):
        with patch(
            "subprocess.run",
            return_value=subprocess.CompletedProcess(
                args=[], returncode=0, stdout=str(8 * GI) + "\n"
            ),
        ):
            assert total_ram_bytes() == 8 * GI

    def test_nonzero_rc_returns_none(self):
        with patch(
            "subprocess.run",
            return_value=subprocess.CompletedProcess(args=[], returncode=1, stdout=""),
        ):
            assert total_ram_bytes() is None

    def test_file_not_found_returns_none(self):
        with patch("subprocess.run", side_effect=FileNotFoundError("sysctl not found")):
            assert total_ram_bytes() is None

    def test_os_error_returns_none(self):
        with patch("subprocess.run", side_effect=OSError("sysctl failed")):
            assert total_ram_bytes() is None

    def test_calls_sysctl_with_correct_args(self):
        with patch(
            "subprocess.run",
            return_value=subprocess.CompletedProcess(
                args=[], returncode=0, stdout="17179869184\n"
            ),
        ) as mock_run:
            total_ram_bytes()
            mock_run.assert_called_once()
            call_args = mock_run.call_args
            assert call_args[0][0] == ["sysctl", "-n", "hw.memsize"]

    def test_invalid_output_returns_none(self):
        with patch(
            "subprocess.run",
            return_value=subprocess.CompletedProcess(
                args=[], returncode=0, stdout="not-a-number\n"
            ),
        ):
            assert total_ram_bytes() is None


class TestDefaultLowMemory:
    @pytest.fixture(autouse=True)
    def _patch_platform(self):
        with patch("vemoizer.low_memory.sys.platform", "darwin"):
            yield

    def test_8_gb_defaults_on(self):
        with patch("vemoizer.low_memory.total_ram_bytes", return_value=8 * GI):
            assert default_low_memory() is True

    def test_16_gb_defaults_on(self):
        with patch("vemoizer.low_memory.total_ram_bytes", return_value=16 * GI):
            assert default_low_memory() is True

    def test_32_gb_defaults_off(self):
        with patch("vemoizer.low_memory.total_ram_bytes", return_value=32 * GI):
            assert default_low_memory() is False

    def test_none_defaults_off(self):
        with patch("vemoizer.low_memory.total_ram_bytes", return_value=None):
            assert default_low_memory() is False

    def test_exact_boundary_16_gi(self):
        # 16 GiB = 17179869184 bytes — should be on
        with patch("vemoizer.low_memory.total_ram_bytes", return_value=16 * GI):
            assert default_low_memory() is True

    def test_just_above_16_gi(self):
        with patch("vemoizer.low_memory.total_ram_bytes", return_value=16 * GI + 1):
            assert default_low_memory() is False


class TestApplyLowMemoryMode:
    def test_accepts_true(self):
        apply_low_memory_mode(True)  # should not raise

    def test_accepts_false(self):
        apply_low_memory_mode(False)  # should not raise
