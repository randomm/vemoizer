"""Shared helpers for the meeting naming-hook test modules.

The sidecar builders and the ``typer.echo`` recorder live here (not in
``conftest.py``) so they are an explicit import in both
``test_meeting_naming_hook.py`` and ``test_naming_hook_skip.py``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

import vemoizer.naming_hook as naming_hook


def _two_label_paragraphs() -> list[dict[str, Any]]:
    return [
        {"start": 0.0, "end": 10.0, "text": "Moikka.", "speaker": "SPEAKER_1"},
        {"start": 12.0, "end": 20.0, "text": "Kyll\u00e4.", "speaker": "SPEAKER_2"},
    ]


def _one_label_paragraphs() -> list[dict[str, Any]]:
    return [
        {"start": 0.0, "end": 10.0, "text": "Moikka.", "speaker": "SPEAKER_1"},
    ]


def _no_label_paragraphs() -> list[dict[str, Any]]:
    return [
        {"start": 0.0, "end": 10.0, "text": "Moikka."},
    ]


def _write_sidecar(tmp_path: Path, name: str, paragraphs: list[dict]) -> str:
    """Write a minimal sidecar with *paragraphs*; return the file name."""
    data = {
        "text": "moikka",
        "paragraphs": paragraphs,
        "notes": {"title": name.rstrip(".json")},
        "options": {
            "command": "meeting",
            "glossary_files": [],
            "glossary_sha256": None,
        },
        "speaker_names": {},
    }
    path = tmp_path / name
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return name


def _capture_echo(
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, list[dict[str, Any]]]:
    """Patch ``naming_hook.typer.echo`` into a recorder; return the record.

    ``stdout`` = calls without ``err=True``; ``stderr`` = calls with it.
    Used to assert on the hook's own printed lines (skip-reason narration,
    warnings, "naming cancelled") independent of the CLI-level runner.
    """
    record: dict[str, list[dict[str, Any]]] = {"stdout": [], "stderr": []}

    def fake_echo(msg, err: bool = False, **kw: Any) -> None:
        record["stderr" if err else "stdout"].append({"msg": msg, "kw": kw})

    monkeypatch.setattr(naming_hook.typer, "echo", fake_echo)
    return record
