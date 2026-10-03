"""Shared helpers for the CLI test modules (transcribe / meeting / memo).

Kept in its own module rather than conftest so the helpers are an
explicit import (the test modules use them directly) and the conftest
stays a pure test-suite configuration file.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

import vemoizer.llm_config as llm_module
import vemoizer.pipeline as pipeline_module


def touch_files(names: list[str], tmp_path: Path) -> list[Path]:
    """Create zero-byte files in *tmp_path* and return their paths."""
    files = [tmp_path / n for n in names]
    for f in files:
        f.touch()
    return files


def fake_transcribe(
    monkeypatch: pytest.MonkeyPatch,
    record: list[str],
    title: str | None = None,
) -> None:
    """Patch ``pipeline.transcribe_file`` with a fake that records the
    transcribed files; the result carries a per-file or fixed title."""

    def fake_transcribe(path, **kwargs):
        record.append(Path(path).name)
        # A vemoizer.* INFO record per transcribed file: the real transcribe
        # logs its own pipeline activity, and a log with no records at all
        # would be indistinguishable from a failed seam (the coverage matrix
        # asserts non-empty log content).
        logging.getLogger("vemoizer.pipeline").info("transcribe: %s", Path(path).name)
        return {
            "text": "moikka maailma",
            "segments": [],
            "notes": {"title": title if title is not None else Path(path).stem.upper()},
        }

    monkeypatch.setattr(pipeline_module, "transcribe_file", fake_transcribe)


def isolate_home(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    chdir_to: Path | None = None,
) -> None:
    """Point HOME at ``tmp_path/home`` and chdir into the tmp dir.

    The dev machine has a live ``~/.config/vemoizer/config.toml`` (and
    possibly ``~/.vemoizer``); the layered config search and glossary
    layer load must never see them. A fresh, empty ``tmp_path`` has no
    ``.vemoizer`` on its walk-up, so the home layer (fake) and any
    project layer written into the test are the only candidates.

    ``Path.home`` honours ``$HOME`` on POSIX, so the env patch is
    sufficient for the ``~/.vemoizer`` layer. The legacy layer is NOT
    covered by the env patch: ``llm_config._LEGACY_CONFIG_PATHS`` is a
    module-level tuple computed at import time from the real
    ``Path.home()`` (issue #93, R1), so it is monkeypatched to the fake
    home as well.
    """
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setattr(
        llm_module,
        "_LEGACY_CONFIG_PATHS",
        (
            home / ".config" / "vemoizer" / "config.toml",
            home / ".vemoizer.toml",
        ),
    )
    monkeypatch.chdir(chdir_to if chdir_to is not None else tmp_path)
