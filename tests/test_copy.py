"""Tests for the pbcopy clipboard helper (issue #14, --copy).

Pattern: ported from kuiskaus test_postprocessor.py fail-open matrix.
- darwin + success → True
- darwin + non-zero rc → False (fail-open, no exception)
- darwin + FileNotFoundError → False
- darwin + TimeoutExpired → False
- darwin + OSError → False
- non-darwin → False, no subprocess call
"""

from __future__ import annotations

import subprocess
from unittest.mock import patch

import pytest

from vemoizer.copy import copy_to_clipboard


class TestDarwin:
    """Tests that patch sys.platform to darwin."""

    @pytest.fixture(autouse=True)
    def _patch_platform(self):
        with patch("vemoizer.copy.sys.platform", "darwin"):
            yield

    def test_success_returns_true(self):
        with patch(
            "subprocess.run",
            return_value=subprocess.CompletedProcess(args=[], returncode=0, stdout=b""),
        ) as mock_run:
            assert copy_to_clipboard("hello") is True
            # pbcopy receives the text as stdin
            mock_run.assert_called_once()
            call_args = mock_run.call_args
            assert call_args[0][0][0] == "pbcopy"

    def test_text_passed_via_stdin(self):
        with patch(
            "subprocess.run",
            return_value=subprocess.CompletedProcess(args=[], returncode=0, stdout=b""),
        ) as mock_run:
            copy_to_clipboard("täyttävä teksti\nlinja 2")
            # The text must be passed as input (bytes)
            call_kwargs = mock_run.call_args[1]
            assert call_kwargs.get("input") == "täyttävä teksti\nlinja 2".encode()

    def test_nonzero_rc_returns_false(self):
        with patch(
            "subprocess.run",
            return_value=subprocess.CompletedProcess(args=[], returncode=1, stdout=b""),
        ):
            assert copy_to_clipboard("hello") is False

    def test_file_not_found_returns_false(self):
        with patch("subprocess.run", side_effect=FileNotFoundError("pbcopy not found")):
            assert copy_to_clipboard("hello") is False

    def test_timeout_returns_false(self):
        with patch(
            "subprocess.run", side_effect=subprocess.TimeoutExpired("pbcopy", 5)
        ):
            assert copy_to_clipboard("hello") is False

    def test_os_error_returns_false(self):
        with patch("subprocess.run", side_effect=OSError("write failed")):
            assert copy_to_clipboard("hello") is False

    def test_empty_string_succeeds(self):
        with patch(
            "subprocess.run",
            return_value=subprocess.CompletedProcess(args=[], returncode=0, stdout=b""),
        ):
            assert copy_to_clipboard("") is True

    def test_long_text_succeeds(self):
        with patch(
            "subprocess.run",
            return_value=subprocess.CompletedProcess(args=[], returncode=0, stdout=b""),
        ):
            assert copy_to_clipboard("x" * 100_000) is True


class TestNonDarwin:
    """Tests that patch sys.platform to a non-macOS value."""

    @pytest.fixture(params=["linux", "win32"])
    def _patch_platform(self, request):
        with patch("vemoizer.copy.sys.platform", request.param):
            yield

    def test_returns_false_without_subprocess(self, _patch_platform):
        with patch("subprocess.run") as mock_run:
            assert copy_to_clipboard("hello") is False
            mock_run.assert_not_called()
