"""Shared helpers for the CLI test modules (transcribe / meeting / memo).

Kept in its own module rather than conftest so the helpers are an
explicit import (the test modules use them directly) and the conftest
stays a pure test-suite configuration file.
"""

from __future__ import annotations

from pathlib import Path

import pytest


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
    sufficient — no monkeypatching of ``Path.home`` needed.
    """
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.chdir(chdir_to if chdir_to is not None else tmp_path)
