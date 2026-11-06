"""Issue #268 rework: saving confirmed terms to the project glossary.

HOME and CWD are pointed at a tmp dir (the real dev machine has a live
``~/.vemoizer``); the atomicity check spies on ``os.replace``.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from _cli_helpers import isolate_home

from vemoizer.glossary_write import (
    HomeGlossaryWriteError,
    append_to_project_glossary,
    project_glossary_path,
)


@pytest.fixture
def cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    isolate_home(monkeypatch, tmp_path, tmp_path)
    return tmp_path


def test_creates_the_folder_glossary(cwd: Path) -> None:
    target = append_to_project_glossary(["AVS", "CDCI => CDI"])
    assert target == cwd / ".vemoizer" / "glossary.txt"
    assert target.is_file()
    assert target.read_text(encoding="utf-8") == "AVS\nCDCI => CDI\n"


def test_appends_to_the_nearest_project_glossary(cwd: Path) -> None:
    project = cwd / ".vemoizer" / "glossary.txt"
    project.parent.mkdir()
    project.write_text("Yki\n", encoding="utf-8")
    deep = cwd / "a" / "b"
    deep.mkdir(parents=True)
    target = append_to_project_glossary(["AVS"], start=deep)
    assert target == project
    assert project.read_text(encoding="utf-8") == "Yki\nAVS\n"


def test_missing_trailing_newline_is_repaired(cwd: Path) -> None:
    project = cwd / ".vemoizer" / "glossary.txt"
    project.parent.mkdir()
    project.write_text("Yki", encoding="utf-8")
    append_to_project_glossary(["AVS"], start=project.parent)
    assert project.read_text(encoding="utf-8") == "Yki\nAVS\n"


def test_dedupes_bare_terms_case_insensitively(cwd: Path) -> None:
    target = append_to_project_glossary(["AVS", "avs", "Avs"])
    assert target.read_text(encoding="utf-8") == "AVS\n"


def test_dedupes_pairs_by_the_wrong_side_case_insensitively(cwd: Path) -> None:
    target = append_to_project_glossary(["cdcI => CDI", "CDCI => CDII", "CDCI"])
    text = target.read_text(encoding="utf-8")
    assert text == "cdcI => CDI\n"
    # A bare term whose word is already a pair's wrong side is a duplicate.
    append_to_project_glossary(["cdci"], start=cwd)
    assert target.read_text(encoding="utf-8") == "cdcI => CDI\n"


def test_never_writes_wildcard_pairs(cwd: Path) -> None:
    target = append_to_project_glossary(["epit* => ebit", "AVS"])
    assert target.read_text(encoding="utf-8") == "AVS\n"


def test_empty_lines_are_dropped(cwd: Path) -> None:
    target = append_to_project_glossary(["", "  ", "a => ", " => b", "AVS"])
    assert target.read_text(encoding="utf-8") == "AVS\n"


def test_write_is_atomic_via_os_replace(
    cwd: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    project = cwd / ".vemoizer" / "glossary.txt"
    project.parent.mkdir()
    project.write_text("Yki\n", encoding="utf-8")
    calls: list[tuple[str, str]] = []
    real = os.replace

    def spy(src: str, dst: str) -> None:
        calls.append((src, dst))
        real(src, dst)

    monkeypatch.setattr(os, "replace", spy)
    target = append_to_project_glossary(["AVS"])
    assert calls and calls[0][1] == str(target)
    assert Path(calls[0][0]).parent == target.parent  # same-dir temp
    assert project.read_text(encoding="utf-8") == "Yki\nAVS\n"


def test_no_new_lines_does_not_touch_the_file(cwd: Path) -> None:
    project = cwd / ".vemoizer" / "glossary.txt"
    project.parent.mkdir()
    project.write_text("AVS\n", encoding="utf-8")
    before = project.stat().st_mtime_ns
    target = append_to_project_glossary(["avs"])
    assert target == project
    assert project.stat().st_mtime_ns == before


def test_project_glossary_path_never_returns_home(cwd: Path) -> None:
    home = Path.home() / ".vemoizer" / "glossary.txt"
    home.parent.mkdir(parents=True)
    home.write_text("Yki\n", encoding="utf-8")
    assert project_glossary_path() == cwd / ".vemoizer" / "glossary.txt"


def test_refuses_the_home_glossary_when_cwd_is_home(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """HOME == cwd and no project glossary: the fallback IS the home layer."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    with pytest.raises(HomeGlossaryWriteError):
        append_to_project_glossary(["AVS"])
    assert not (tmp_path / ".vemoizer").exists()


def test_refuses_the_home_glossary_even_when_it_already_exists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    home = tmp_path / ".vemoizer" / "glossary.txt"
    home.parent.mkdir(parents=True)
    home.write_text("Yki\n", encoding="utf-8")
    with pytest.raises(HomeGlossaryWriteError):
        append_to_project_glossary(["AVS", "Yki"])
    assert home.read_text(encoding="utf-8") == "Yki\n"


def test_nothing_to_add_creates_no_folder(cwd: Path) -> None:
    """All lines rejected/duplicate: no ``.vemoizer/``, no file touched."""
    target = append_to_project_glossary(["", "  ", " => b", "a => ", "epit* => ebit"])
    assert target == cwd / ".vemoizer" / "glossary.txt"
    assert not (cwd / ".vemoizer").exists()


def test_all_duplicates_do_not_touch_the_file(cwd: Path) -> None:
    project = cwd / ".vemoizer" / "glossary.txt"
    project.parent.mkdir(parents=True)
    project.write_text("AVS\nkala => kana\n", encoding="utf-8")
    before = project.stat().st_mtime_ns
    assert append_to_project_glossary(["avs", "KALA => kana"]) == project
    assert project.stat().st_mtime_ns == before
