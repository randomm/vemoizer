"""Tests for the caffeinate wake-assertion context manager (issue #14).

Pattern: context manager that spawns ``caffeinate -dims`` as a daemon,
terminates it on exit. Fail-open: if Popen raises, the context is a no-op
(no exception, no assertion held — the run continues without the wake lock).

The context manager is tested by patching subprocess.Popen and inspecting
the Popen mock's lifecycle (start, terminate, wait).
"""

from __future__ import annotations

import subprocess
from unittest.mock import MagicMock, patch

import pytest

from vemoizer.caffeinate import caffeinate_context


class TestDarwin:
    @pytest.fixture(autouse=True)
    def _patch_platform(self):
        with patch("vemoizer.caffeinate.sys.platform", "darwin"):
            yield

    def test_popen_called_with_caffeinate_dims(self):
        with patch("subprocess.Popen") as mock_popen:
            mock_popen.return_value = MagicMock()
            with caffeinate_context():
                pass
            mock_popen.assert_called_once()
            call_args = mock_popen.call_args
            assert call_args[0][0][0] == "caffeinate"
            # -dims: display, idle, disk, network
            assert "-dims" in call_args[0][0]

    def test_process_terminated_on_exit(self):
        with patch("subprocess.Popen") as mock_popen:
            mock_proc = MagicMock()
            mock_popen.return_value = mock_proc
            with caffeinate_context():
                pass
            mock_proc.terminate.assert_called_once()

    def test_process_wait_called_after_terminate(self):
        with patch("subprocess.Popen") as mock_popen:
            mock_proc = MagicMock()
            mock_popen.return_value = mock_proc
            with caffeinate_context():
                pass
            mock_proc.wait.assert_called_once()

    def test_work_runs_inside_context(self):
        executed = []
        with patch("subprocess.Popen") as mock_popen:
            mock_popen.return_value = MagicMock()
            with caffeinate_context():
                executed.append("work")
        assert executed == ["work"]

    def test_popen_file_not_found_is_noop(self):
        # Fail-open: Popen raises → context is a no-op, no exception
        # propagates.
        exc = FileNotFoundError("caffeinate missing")
        with patch("subprocess.Popen", side_effect=exc):
            executed = []
            with caffeinate_context():
                executed.append("work")
            assert executed == ["work"]

    def test_popen_os_error_is_noop(self):
        exc = OSError("caffeinate spawn failed")
        with patch("subprocess.Popen", side_effect=exc):
            executed = []
            with caffeinate_context():
                executed.append("work")
            assert executed == ["work"]

    def test_popen_subprocess_error_is_noop(self):
        exc = subprocess.SubprocessError("spawn failed")
        with patch("subprocess.Popen", side_effect=exc):
            executed = []
            with caffeinate_context():
                executed.append("work")
            assert executed == ["work"]

    def test_nested_context_does_not_double_spawn(self):
        # A nested caffeinate_context should not spawn a second process
        with patch("subprocess.Popen") as mock_popen:
            mock_popen.return_value = MagicMock()
            with caffeinate_context(), caffeinate_context():
                pass
            # Popen called only once (the outer context)
            assert mock_popen.call_count == 1


class TestNonDarwin:
    @pytest.fixture(params=["linux", "win32"])
    def _patch_platform(self, request):
        with patch("vemoizer.caffeinate.sys.platform", request.param):
            yield

    def test_no_popen_on_non_darwin(self, _patch_platform):
        with patch("subprocess.Popen") as mock_popen:
            with caffeinate_context():
                pass
            mock_popen.assert_not_called()

    def test_work_runs_on_non_darwin(self, _patch_platform):
        executed = []
        with caffeinate_context():
            executed.append("work")
        assert executed == ["work"]
