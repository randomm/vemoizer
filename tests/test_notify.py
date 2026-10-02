"""Unit tests for the macOS completion notification (issue #100, M4a).

Covers the ``notify`` / ``escape_apple_script`` surface:
- the ``osascript`` argv is built as a *list* (never ``shell=True``)
- AppleScript escaping of double quotes, backslashes, and newlines
- non-darwin platforms are a no-op (subprocess.run is never called)
- fail-open on every error mode (missing osascript, non-zero exit, timeout,
  OS error) — ``notify`` never raises
- the subprocess ``timeout`` is a short, bounded value
- the fixed ``vemoizer`` title is present in the AppleScript source

All ``subprocess.run`` calls are mocked; no real notification is ever posted.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest

from vemoizer.notify import (
    NOTIFY_TITLE,
    escape_apple_script,
    notify,
    notify_result,
    notify_write,
)

pytestmark = pytest.mark.real_system_calls


class TestDarwin:
    @pytest.fixture(autouse=True)
    def _patch_platform(self):
        with patch("vemoizer.notify.sys.platform", "darwin"):
            yield

    def test_argv_is_a_list_and_never_shell(self):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(
                args=[], returncode=0, stdout=""
            )
            notify("vemoizer", "done")
            mock_run.assert_called_once()
            args = mock_run.call_args[0]
            # First positional arg is the argv, and it must be a list...
            assert isinstance(args[0], list)
            # ...and shell=True must never appear (shell default is False).
            assert mock_run.call_args[1].get("shell") in (None, False)

    def test_argv_shape_osascript_minus_e_script(self):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(
                args=[], returncode=0, stdout=""
            )
            notify("vemoizer", "memo done")
            argv = mock_run.call_args[0][0]
            assert argv[0] == "osascript"
            assert argv[1] == "-e"
            assert "display notification" in argv[2]

    def test_title_vemoizer_is_present(self):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(
                args=[], returncode=0, stdout=""
            )
            notify("vemoizer", "memo done")
            script = mock_run.call_args[0][0][2]
            assert 'with title "vemoizer"' in script

    def test_message_in_script(self):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(
                args=[], returncode=0, stdout=""
            )
            notify("vemoizer", "memo done")
            script = mock_run.call_args[0][0][2]
            assert '"memo done"' in script

    def test_escaping_double_quote(self):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(
                args=[], returncode=0, stdout=""
            )
            notify("vemoizer", 'say "hi"')
            script = mock_run.call_args[0][0][2]
            # Raw double quotes must not survive into the literal.
            assert '"say "hi""' not in script
            # The escape form (\b) must be present.
            assert "say \\bhi\\b" in script

    def test_escaping_backslash(self):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(
                args=[], returncode=0, stdout=""
            )
            notify("vemoizer", "a\\b")
            script = mock_run.call_args[0][0][2]
            assert "a\\\\b" in script

    def test_escaping_newline(self):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(
                args=[], returncode=0, stdout=""
            )
            notify("vemoizer", "line1\nline2")
            script = mock_run.call_args[0][0][2]
            assert "line1\\nline2" in script
            # No literal newline inside the -e argument.
            assert "\n" not in script

    def test_escaping_title(self):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(
                args=[], returncode=0, stdout=""
            )
            notify('vemo"izer', "msg")
            script = mock_run.call_args[0][0][2]
            assert "vemo\\bizer" in script

    def test_nonzero_rc_does_not_raise(self):
        with patch(
            "subprocess.run",
            return_value=subprocess.CompletedProcess(args=[], returncode=1, stdout=""),
        ):
            # Must not raise; nothing to assert beyond that.
            notify("vemoizer", "x")

    def test_file_not_found_does_not_raise(self):
        with patch(
            "subprocess.run", side_effect=FileNotFoundError("osascript missing")
        ):
            notify("vemoizer", "x")

    def test_os_error_does_not_raise(self):
        with patch("subprocess.run", side_effect=OSError("osascript failed")):
            notify("vemoizer", "x")

    def test_timeout_does_not_raise(self):
        with patch(
            "subprocess.run", side_effect=subprocess.TimeoutExpired("osascript", 5)
        ):
            notify("vemoizer", "x")

    def test_timeout_is_short_bounded(self):
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(
                args=[], returncode=0, stdout=""
            )
            notify("vemoizer", "x")
            timeout = mock_run.call_args[1].get("timeout")
            assert timeout is not None
            # The notification post is bounded to a short window.
            assert 0 < timeout <= 10


class TestEscapeAppleScript:
    def test_escape_double_quote(self):
        assert escape_apple_script('a"b') == "a\\bb"

    def test_escape_backslash(self):
        assert escape_apple_script("a\\b") == "a\\\\b"

    def test_escape_newline(self):
        assert escape_apple_script("a\nb") == "a\\nb"

    def test_escape_combined_ordering(self):
        # Backslashes must be escaped before quotes/newlines introduce new
        # backslashes; the result must not double-escape those.
        assert escape_apple_script('a"b\\c\nd') == "a\\bb\\\\c\\nd"

    def test_plain_string_unchanged(self):
        assert escape_apple_script("hello world") == "hello world"


class TestNonDarwin:
    @pytest.fixture(params=["linux", "win32"])
    def _patch_platform(self, request):
        with patch("vemoizer.notify.sys.platform", request.param):
            yield

    def test_no_subprocess_call_on_non_darwin(self, _patch_platform):
        with patch("subprocess.run") as mock_run:
            notify("vemoizer", "msg")
            mock_run.assert_not_called()


class TestNotifyResult:
    """The M4a seam helpers (notify_result / notify_write) build the
    one-line message: NFC stem + outcome (+ the reason, stripped of the
    leading ``error: `` and capped), and never raise."""

    def _calls(self, mock_notify) -> list[str]:
        return [c.args[1] for c in mock_notify.call_args_list]

    def test_done_message_names_nfc_stem(self):
        nfd = Path("caf\u0065\u0301.m4a")  # "café" decomposed
        with patch("vemoizer.notify.notify") as mock_notify:
            notify_result(nfd, "done")
        assert mock_notify.call_count == 1
        assert mock_notify.call_args[0][0] == NOTIFY_TITLE
        assert self._calls(mock_notify)[0] == "caf\u00e9: done"

    def test_failed_without_reason(self):
        with patch("vemoizer.notify.notify") as mock_notify:
            notify_result(Path("a.m4a"), "failed")
        assert self._calls(mock_notify)[0] == "a: failed"

    def test_failed_reason_strips_error_prefix_and_keeps_one_line(self):
        with patch("vemoizer.notify.notify") as mock_notify:
            notify_result(Path("a.m4a"), "failed", "error: could not write x.txt: boom")
        assert self._calls(mock_notify)[0] == "a: failed - could not write x.txt: boom"

    def test_failed_reason_multiline_keeps_first_line(self):
        with patch("vemoizer.notify.notify") as mock_notify:
            notify_result(Path("a.m4a"), "failed", "error: one\ntwo")
        assert self._calls(mock_notify)[0] == "a: failed - one"

    def test_failed_reason_capped(self):
        long_reason = "error: " + "x" * 300
        with patch("vemoizer.notify.notify") as mock_notify:
            notify_result(Path("a.m4a"), "failed", long_reason)
        message = self._calls(mock_notify)[0]
        assert len(message) <= 200 + len("a: failed - ")
        assert message.endswith("…")

    def test_group_label_uses_first_part_stem(self):
        with patch("vemoizer.notify.notify") as mock_notify:
            notify_result("a.m4a+b.m4a", "done")
            notify_result("a.m4a+b.m4a", "failed", "error: could not write a.md")
        calls = self._calls(mock_notify)
        assert calls[0] == "a: done"
        assert calls[1] == "a: failed - could not write a.md"
        assert "a+b" not in calls[0]

    def test_never_raises(self):
        with patch(
            "vemoizer.notify.notify", side_effect=RuntimeError("osascript dead")
        ):
            notify_result(Path("a.m4a"), "done")  # must not raise

    def test_notify_write_partial_is_failure(self):
        with patch("vemoizer.notify.notify") as mock_notify:
            notify_write(Path("a.m4a"), 1, 2, reason="error: could not write a.json")
            notify_write(Path("a.m4a"), 2, 2)
        calls = self._calls(mock_notify)
        assert calls[0] == "a: failed - could not write a.json"
        assert calls[1] == "a: done"
