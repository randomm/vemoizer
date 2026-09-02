"""Tests for the pmset battery check (issue #14, battery warning).

Pattern: ported from kuiskaus test_silicon_check.py.
- darwin + "Battery Power" → True
- darwin + "AC Power" → False
- darwin + non-zero rc → False (fail-open)
- darwin + FileNotFoundError → False
- darwin + OSError → False
- non-darwin → False, no subprocess call
"""

from __future__ import annotations

import subprocess
from unittest.mock import patch

import pytest

from vemoizer.battery import on_battery


class TestDarwin:
    @pytest.fixture(autouse=True)
    def _patch_platform(self):
        with patch("vemoizer.battery.sys.platform", "darwin"):
            yield

    def test_battery_power_returns_true(self):
        stdout = (
            "Now drawing from 'Battery Power'\n"
            " -InternalBattery-0 (id=23003235)\t87%; discharging; 1:23:45 remaining"
        )
        with patch(
            "subprocess.run",
            return_value=subprocess.CompletedProcess(
                args=[], returncode=0, stdout=stdout
            ),
        ):
            assert on_battery() is True

    def test_ac_power_returns_false(self):
        stdout = (
            "Now drawing from 'AC Power'\n"
            " -InternalBattery-0 (id=23003235)\t100%; charged; 0:00 remaining"
        )
        with patch(
            "subprocess.run",
            return_value=subprocess.CompletedProcess(
                args=[], returncode=0, stdout=stdout
            ),
        ):
            assert on_battery() is False

    def test_nonzero_rc_returns_false(self):
        with patch(
            "subprocess.run",
            return_value=subprocess.CompletedProcess(args=[], returncode=1, stdout=""),
        ):
            assert on_battery() is False

    def test_file_not_found_returns_false(self):
        with patch("subprocess.run", side_effect=FileNotFoundError("pmset not found")):
            assert on_battery() is False

    def test_os_error_returns_false(self):
        with patch("subprocess.run", side_effect=OSError("pmset failed")):
            assert on_battery() is False

    def test_calls_pmset_with_correct_args(self):
        stdout = "Now drawing from 'Battery Power'\n"
        with patch(
            "subprocess.run",
            return_value=subprocess.CompletedProcess(
                args=[], returncode=0, stdout=stdout
            ),
        ) as mock_run:
            on_battery()
            mock_run.assert_called_once()
            call_args = mock_run.call_args
            assert call_args[0][0] == ["pmset", "-g", "batt"]

    def test_mixed_output_with_battery_keyword(self):
        # Some macOS versions include both lines; "Battery Power"
        # anywhere means battery.
        stdout = (
            " -InternalBattery-0 (id=23003235)\t55%; discharging\n"
            "Now drawing from 'Battery Power'\n"
        )
        with patch(
            "subprocess.run",
            return_value=subprocess.CompletedProcess(
                args=[], returncode=0, stdout=stdout
            ),
        ):
            assert on_battery() is True

    def test_empty_output_returns_false(self):
        with patch(
            "subprocess.run",
            return_value=subprocess.CompletedProcess(args=[], returncode=0, stdout=""),
        ):
            assert on_battery() is False


class TestNonDarwin:
    @pytest.fixture(params=["linux", "win32"])
    def _patch_platform(self, request):
        with patch("vemoizer.battery.sys.platform", request.param):
            yield

    def test_returns_false_without_subprocess(self, _patch_platform):
        with patch("subprocess.run") as mock_run:
            assert on_battery() is False
            mock_run.assert_not_called()
