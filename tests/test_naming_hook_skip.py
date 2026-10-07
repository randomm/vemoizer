"""Skip-reason narration tests for the end-of-meeting naming hook (issue #148).

Moved out of ``test_meeting_naming_hook.py`` (800-line cap). Every skip
fires exactly one ``skipping speaker naming: <reason>`` line unless
*quiet* is set, and the <2-label sidecar stays silent (a legitimate
no-op, not a mystery).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from _cli_helpers import isolate_home
from _naming_hook_helpers import (
    _capture_echo,
    _no_label_paragraphs,
    _one_label_paragraphs,
    _two_label_paragraphs,
    _write_sidecar,
)

import vemoizer.naming_hook as naming_hook


class TestSkipNarration:
    """Issue #148: every skip fires exactly one ``skipping speaker
    naming: <reason>`` line unless *quiet* is set, and the <2-label
    sidecar stays silent (a legitimate no-op, not a mystery)."""

    def test_yes_narrates_not_requested(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        isolate_home(monkeypatch, tmp_path, tmp_path)
        _write_sidecar(tmp_path, "a.json", _two_label_paragraphs())
        rec = _capture_echo(monkeypatch)

        rc = naming_hook.ask_naming_hook(
            ["a.json"], yes=True, input_fn=lambda p: "y", tty_isatty=lambda: True
        )
        assert rc == 0
        assert [m["msg"] for m in rec["stdout"]] == [
            "skipping speaker naming: not requested"
        ]

    def test_yes_quiet_prints_nothing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        isolate_home(monkeypatch, tmp_path, tmp_path)
        _write_sidecar(tmp_path, "a.json", _two_label_paragraphs())
        rec = _capture_echo(monkeypatch)

        rc = naming_hook.ask_naming_hook(
            ["a.json"],
            yes=True,
            quiet=True,
            input_fn=lambda p: "y",
            tty_isatty=lambda: True,
        )
        assert rc == 0
        assert rec["stdout"] == []

    def test_non_tty_stdin_narrates(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        isolate_home(monkeypatch, tmp_path, tmp_path)
        _write_sidecar(tmp_path, "a.json", _two_label_paragraphs())
        rec = _capture_echo(monkeypatch)

        rc = naming_hook.ask_naming_hook(
            ["a.json"], yes=False, input_fn=lambda p: "y", tty_isatty=lambda: False
        )
        assert rc == 0
        assert [m["msg"] for m in rec["stdout"]] == [
            "skipping speaker naming: not an interactive terminal"
        ]

    def test_non_tty_stdout_narrates(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        isolate_home(monkeypatch, tmp_path, tmp_path)
        monkeypatch.setattr(naming_hook, "_stdout_isatty", lambda: False)
        _write_sidecar(tmp_path, "a.json", _two_label_paragraphs())
        rec = _capture_echo(monkeypatch)

        rc = naming_hook.ask_naming_hook(
            ["a.json"], yes=False, input_fn=lambda p: "y", tty_isatty=lambda: True
        )
        assert rc == 0
        assert [m["msg"] for m in rec["stdout"]] == [
            "skipping speaker naming: not an interactive terminal"
        ]

    def test_quiet_non_tty_prints_nothing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        isolate_home(monkeypatch, tmp_path, tmp_path)
        _write_sidecar(tmp_path, "a.json", _two_label_paragraphs())
        rec = _capture_echo(monkeypatch)

        rc = naming_hook.ask_naming_hook(
            ["a.json"],
            yes=False,
            quiet=True,
            input_fn=lambda p: "y",
            tty_isatty=lambda: False,
        )
        assert rc == 0
        assert rec["stdout"] == []

    def test_no_sidecar_at_all_narrates_no_labels(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """TTY run, only a .md written (partial pair): narrate."""
        isolate_home(monkeypatch, tmp_path, tmp_path)
        monkeypatch.setattr(naming_hook, "_stdout_isatty", lambda: True)
        rec = _capture_echo(monkeypatch)

        rc = naming_hook.ask_naming_hook(
            ["2025-01-01 T.md"],
            yes=False,
            input_fn=lambda p: "y",
            tty_isatty=lambda: True,
        )
        assert rc == 0
        assert [m["msg"] for m in rec["stdout"]] == [
            "skipping speaker naming: no speaker labels in this run"
        ]

    def test_no_sidecar_missing_file_stays_silent(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """TTY run, .json listed but absent on disk: legitimate no-op,
        stays silent (a sidecar name was in the written list — not a
        partial-pair mystery)."""
        isolate_home(monkeypatch, tmp_path, tmp_path)
        monkeypatch.setattr(naming_hook, "_stdout_isatty", lambda: True)
        rec = _capture_echo(monkeypatch)

        rc = naming_hook.ask_naming_hook(
            ["2025-01-01 Missing.json"],
            yes=False,
            input_fn=lambda p: "y",
            tty_isatty=lambda: True,
        )
        assert rc == 0
        assert rec["stdout"] == []

    def test_no_sidecar_quiet_prints_nothing(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        isolate_home(monkeypatch, tmp_path, tmp_path)
        monkeypatch.setattr(naming_hook, "_stdout_isatty", lambda: True)
        rec = _capture_echo(monkeypatch)

        rc = naming_hook.ask_naming_hook(
            ["2025-01-01 T.md"],
            yes=False,
            quiet=True,
            input_fn=lambda p: "y",
            tty_isatty=lambda: True,
        )
        assert rc == 0
        assert rec["stdout"] == []

    def test_one_label_sidecar_stays_silent(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """A sidecar with <2 labels is a legitimate no-op: no line."""
        isolate_home(monkeypatch, tmp_path, tmp_path)
        monkeypatch.setattr(naming_hook, "_stdout_isatty", lambda: True)
        _write_sidecar(tmp_path, "one.json", _one_label_paragraphs())
        rec = _capture_echo(monkeypatch)

        rc = naming_hook.ask_naming_hook(
            ["one.json"], yes=False, input_fn=lambda p: "y", tty_isatty=lambda: True
        )
        assert rc == 0
        assert rec["stdout"] == []

    def test_zero_label_sidecar_stays_silent(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        isolate_home(monkeypatch, tmp_path, tmp_path)
        monkeypatch.setattr(naming_hook, "_stdout_isatty", lambda: True)
        _write_sidecar(tmp_path, "nolabels.json", _no_label_paragraphs())
        rec = _capture_echo(monkeypatch)

        rc = naming_hook.ask_naming_hook(
            ["nolabels.json"],
            yes=False,
            input_fn=lambda p: "y",
            tty_isatty=lambda: True,
        )
        assert rc == 0
        assert rec["stdout"] == []
