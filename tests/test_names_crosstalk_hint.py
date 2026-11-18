"""Crosstalk hint for tiny-share labels in ``vemoizer names`` (issue #281).

A label whose speech is only cross-talk fragments pyannote collected shows
0% of talk and its clips sound like several voices at once. Below
``CROSSTALK_SHARE`` the prompt header gets a hint line; the two-track owner
line never does. Prompts are scripted; no audio, no TTY.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from _cli_helpers import isolate_home

import vemoizer.names_cli as names_cli
from vemoizer.names_cli import CROSSTALK_SHARE, run_names

OWNER = "Janni"
HINT = "  (very little speech, often several voices mixed — leave empty to skip)"


def _sidecar(tmp_path: Path, *, owner: str | None = None) -> Path:
    options: dict[str, Any] = {
        "command": "meeting",
        "glossary_files": [],
        "glossary_sha256": None,
    }
    speaker_names: dict[str, str] = {}
    if owner is not None:
        options["owner"] = owner
        options["owner_track"] = "calls/t/me.flac"
        speaker_names[owner] = owner
    data = {
        "text": "Yksi kaksi kolme.",
        "paragraphs": [
            {"start": 0.0, "end": 1.0, "text": "Yksi.", "speaker": "Others"},
            {"start": 1.5, "end": 2.0, "text": "Kaksi.", "speaker": "Bot"},
        ],
        "source": [],
        "options": options,
        "speaker_names": speaker_names,
    }
    path = tmp_path / "2026-11-04 Crosstalk.json"
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return path


def _run(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    shares: dict[str, float],
    *,
    owner: str | None = None,
) -> None:
    """Run the naming loop with a scripted ``talk_share``; return stdout."""
    isolate_home(monkeypatch, tmp_path, tmp_path)
    monkeypatch.setattr(names_cli, "talk_share", lambda paragraphs: shares)
    answers = iter([""] * sum(1 for label in shares if label != owner))
    rc = run_names(
        _sidecar(tmp_path, owner=owner),
        no_play=True,
        input_fn=lambda prompt: next(answers),
        tty_isatty=lambda: True,
        discard_input=lambda: None,
        config_path=tmp_path / "people.toml",
    )
    assert rc == 0


class TestCrosstalkHint:
    def test_below_threshold_shows_hint(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        _run(monkeypatch, tmp_path, {"Others": CROSSTALK_SHARE / 2}, owner=None)
        out = capsys.readouterr().out
        assert "Others  (0% of talk)" in out
        assert HINT in out

    def test_at_or_above_threshold_shows_no_hint(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        _run(monkeypatch, tmp_path, {"Others": CROSSTALK_SHARE, "Bot": 0.5})
        out = capsys.readouterr().out
        assert "Others  (1% of talk)" in out
        assert HINT not in out

    def test_owner_never_gets_hint(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        _run(
            monkeypatch,
            tmp_path,
            {OWNER: CROSSTALK_SHARE / 4, "Others": 0.5},
            owner=OWNER,
        )
        out = capsys.readouterr().out
        assert f"{OWNER}  (0% of talk) — you (own mic track)" in out
        assert HINT not in out
