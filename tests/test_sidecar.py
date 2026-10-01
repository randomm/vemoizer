"""Tests for the M5a JSON sidecar assembly (issue #89, workstream A).

Pure-stdlib: no model imports, no network, no ffmpeg. The sidecar is
assembled by :func:`vemoizer.sidecar.build_sidecar` and the pure helpers
``glossary_layer_files`` / ``sha256_over_files``.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, cast

import pytest

from vemoizer.sidecar import build_sidecar, glossary_layer_files, sha256_over_files

# ---------------------------------------------------------------------------
# build_sidecar — single-file runs
# ---------------------------------------------------------------------------


def test_build_sidecar_single_file_no_glossary(tmp_path: Path) -> None:
    """Single-file run with no glossary: source has path + part_offset_s 0.0,
    options has command + empty glossary_files + null sha256, speaker_names
    {}."""
    result: dict[str, Any] = {
        "text": "hei",
        "segments": [],
        "notes": {"title": "Test", "summary": "S"},
        "source_path": str(tmp_path / "a.m4a"),
        "_source_durations": [3.5],
    }
    build_sidecar(result, command="meeting", glossary_files=None)

    assert result["notes"] == {"title": "Test", "summary": "S"}
    src = cast(list[dict[str, Any]], result["source"])
    assert src == [
        {"path": str(tmp_path / "a.m4a"), "part_offset_s": 0.0, "duration_s": 3.5}
    ]
    opts = cast(dict[str, Any], result["options"])
    assert opts["command"] == "meeting"
    assert opts["glossary_files"] == []
    assert opts["glossary_sha256"] is None
    assert result["speaker_names"] == {}
    # _source_durations is consumed (popped) — must not leak into output.
    assert "_source_durations" not in result
    # No clips key.
    assert "clips" not in result


def test_build_sidecar_single_file_no_durations(tmp_path: Path) -> None:
    """Single-file run with no measured duration: source entry omits
    duration_s."""
    result: dict[str, Any] = {
        "text": "hei",
        "segments": [],
        "source_path": str(tmp_path / "a.m4a"),
    }
    build_sidecar(result, command="memo", glossary_files=None)

    src = cast(list[dict[str, Any]], result["source"])
    assert src == [{"path": str(tmp_path / "a.m4a"), "part_offset_s": 0.0}]
    assert "duration_s" not in src[0]


def test_build_sidecar_single_file_no_source_path() -> None:
    """Single-file run with no source_path: source key is absent."""
    result: dict[str, Any] = {"text": "hei", "segments": []}
    build_sidecar(result, command="meeting", glossary_files=None)

    assert "source" not in result
    opts = cast(dict[str, Any], result["options"])
    assert opts["command"] == "meeting"
    assert result["speaker_names"] == {}


def test_build_sidecar_no_notes() -> None:
    """Result without notes: notes key is absent (not fabricated)."""
    result: dict[str, Any] = {"text": "hei", "segments": [], "source_path": "/a.m4a"}
    build_sidecar(result, command="memo", glossary_files=None)

    assert "notes" not in result
    assert result["speaker_names"] == {}


def test_build_sidecar_none_notes() -> None:
    """Result with notes=None: notes key is absent (None → omit)."""
    result: dict[str, Any] = {
        "text": "hei",
        "segments": [],
        "notes": None,
        "source_path": "/a.m4a",
    }
    build_sidecar(result, command="memo", glossary_files=None)

    assert "notes" not in result


def test_build_sidecar_with_glossary_files(tmp_path: Path) -> None:
    """Glossary files present: options has the file list and a sha256 hash."""
    gfile = tmp_path / "glossary.txt"
    gfile.write_text("Nordea => Nordea\nBlacksit => Flagship\n", encoding="utf-8")
    result: dict[str, Any] = {
        "text": "hei",
        "segments": [],
        "source_path": str(tmp_path / "a.m4a"),
    }
    build_sidecar(result, command="meeting", glossary_files=[str(gfile)])

    opts = cast(dict[str, Any], result["options"])
    assert opts["glossary_files"] == [str(gfile)]
    assert isinstance(opts["glossary_sha256"], str)
    assert len(opts["glossary_sha256"]) == 64


def test_build_sidecar_glossary_files_empty_list(tmp_path: Path) -> None:
    """glossary_files=[] (explicit empty): options has empty list + null
    sha256."""
    result: dict[str, Any] = {
        "text": "hei",
        "segments": [],
        "source_path": str(tmp_path / "a.m4a"),
    }
    build_sidecar(result, command="memo", glossary_files=[])

    opts = cast(dict[str, Any], result["options"])
    assert opts["glossary_files"] == []
    assert opts["glossary_sha256"] is None


def test_build_sidecar_no_clips_key() -> None:
    """The sidecar must never contain a 'clips' key."""
    result: dict[str, Any] = {"text": "hei", "segments": [], "source_path": "/a.m4a"}
    build_sidecar(result, command="meeting", glossary_files=None)
    assert "clips" not in result


# ---------------------------------------------------------------------------
# build_sidecar — grouped runs (part_markers)
# ---------------------------------------------------------------------------


def test_build_sidecar_grouped_run(tmp_path: Path) -> None:
    """Grouped run: source has one entry per part_markers part, with
    path from source_paths (the real part paths), part_offset_s from the
    marker and duration_s from _source_durations."""
    result: dict[str, Any] = {
        "text": "hei",
        "segments": [],
        "part_markers": [
            {"offset": 0.0, "label": "— osa 1 (äänitys a.m4a)"},
            {"offset": 12.5, "label": "— osa 2 (äänitys b.m4a)"},
        ],
        "_source_durations": [12.5, 8.0],
    }
    build_sidecar(
        result,
        command="meeting",
        glossary_files=None,
        source_paths=[tmp_path / "a.m4a", tmp_path / "b.m4a"],
    )

    src = cast(list[dict[str, Any]], result["source"])
    assert len(src) == 2
    assert src[0]["path"] == str(tmp_path / "a.m4a")
    assert src[0]["part_offset_s"] == 0.0
    assert src[0]["duration_s"] == 12.5
    assert src[1]["path"] == str(tmp_path / "b.m4a")
    assert src[1]["part_offset_s"] == 12.5
    assert src[1]["duration_s"] == 8.0
    assert "_source_durations" not in result


def test_build_sidecar_grouped_run_fewer_source_paths() -> None:
    """Grouped run with fewer source_paths than markers: the missing part
    falls back to the marker label for its path."""
    result: dict[str, Any] = {
        "text": "hei",
        "segments": [],
        "part_markers": [
            {"offset": 0.0, "label": "— osa 1 (äänitys a.m4a)"},
            {"offset": 10.0, "label": "— osa 2 (äänitys b.m4a)"},
        ],
    }
    build_sidecar(
        result,
        command="meeting",
        glossary_files=None,
        source_paths=[Path("a.m4a")],
    )

    src = cast(list[dict[str, Any]], result["source"])
    assert src[0]["path"] == "a.m4a"
    assert src[1]["path"] == "— osa 2 (äänitys b.m4a)"


def test_build_sidecar_grouped_run_fewer_durations() -> None:
    """Grouped run with fewer durations than parts: missing durations
    omitted."""
    result: dict[str, Any] = {
        "text": "hei",
        "segments": [],
        "part_markers": [
            {"offset": 0.0, "label": "— osa 1 (äänitys a.m4a)"},
            {"offset": 10.0, "label": "— osa 2 (äänitys b.m4a)"},
        ],
        "_source_durations": [10.0],
    }
    build_sidecar(result, command="meeting", glossary_files=None)

    src = cast(list[dict[str, Any]], result["source"])
    assert "duration_s" in src[0]
    assert "duration_s" not in src[1]


def test_build_sidecar_grouped_run_no_durations() -> None:
    """Grouped run with no durations: all source entries omit duration_s."""
    result: dict[str, Any] = {
        "text": "hei",
        "segments": [],
        "part_markers": [
            {"offset": 0.0, "label": "— osa 1 (äänitys a.m4a)"},
            {"offset": 10.0, "label": "— osa 2 (äänitys b.m4a)"},
        ],
    }
    build_sidecar(result, command="meeting", glossary_files=None)

    src = cast(list[dict[str, Any]], result["source"])
    assert len(src) == 2
    for entry in src:
        assert "duration_s" not in entry
        assert entry["part_offset_s"] >= 0.0


def test_build_sidecar_grouped_run_no_source_path() -> None:
    """Grouped run: source is derived from part_markers, not source_path."""
    result: dict[str, Any] = {
        "text": "hei",
        "segments": [],
        "part_markers": [
            {"offset": 0.0, "label": "— osa 1 (äänitys a.m4a)"},
        ],
    }
    build_sidecar(result, command="meeting", glossary_files=None)

    # source_path is not set, but source is still built from part_markers.
    assert "source" in result
    src = cast(list[dict[str, Any]], result["source"])
    assert len(src) == 1


def test_build_sidecar_part_markers_non_dict_entries() -> None:
    """Non-dict entries in part_markers are dropped (no crash)."""
    result: dict[str, Any] = {
        "text": "hei",
        "segments": [],
        "part_markers": [
            {"offset": 0.0, "label": "— osa 1 (äänitys a.m4a)"},
            "not a dict",
            None,
        ],
        "_source_durations": [5.0, 5.0, 5.0],
    }
    build_sidecar(result, command="meeting", glossary_files=None)

    # Only the one valid dict entry survives.
    src = cast(list[dict[str, Any]], result["source"])
    assert len(src) == 1


# ---------------------------------------------------------------------------
# sha256_over_files
# ---------------------------------------------------------------------------


def test_sha256_over_files_single(tmp_path: Path) -> None:
    """sha256 over a single file's raw bytes."""
    f = tmp_path / "g.txt"
    f.write_bytes(b"hello")
    expected = hashlib.sha256(b"hello").hexdigest()
    assert sha256_over_files([str(f)]) == expected


def test_sha256_over_files_multiple(tmp_path: Path) -> None:
    """sha256 over concatenated raw bytes of multiple files, in order."""
    f1 = tmp_path / "a.txt"
    f2 = tmp_path / "b.txt"
    f1.write_bytes(b"aaa")
    f2.write_bytes(b"bbb")
    expected = hashlib.sha256(b"aaabbb").hexdigest()
    assert sha256_over_files([str(f1), str(f2)]) == expected


def test_sha256_over_files_empty_list() -> None:
    """Empty list: None (no files to hash)."""
    assert sha256_over_files([]) is None


def test_sha256_over_files_missing_file(tmp_path: Path) -> None:
    """Missing file: None (fail-open)."""
    f = tmp_path / "nonexistent.txt"
    assert sha256_over_files([str(f)]) is None


def test_sha256_over_files_non_utf8_file_hashes_bytes(tmp_path: Path) -> None:
    """A non-UTF-8 glossary still hashes its raw bytes (never raises)."""
    f = tmp_path / "g.txt"
    f.write_bytes(b"\xff\xfe\x00bad")
    expected = hashlib.sha256(b"\xff\xfe\x00bad").hexdigest()
    assert sha256_over_files([str(f)]) == expected


def test_sha256_over_files_deterministic(tmp_path: Path) -> None:
    """Same files in same order produce the same hash."""
    f = tmp_path / "g.txt"
    f.write_bytes(b"test")
    h1 = sha256_over_files([str(f)])
    h2 = sha256_over_files([str(f)])
    assert h1 == h2


def test_sha256_over_files_order_matters(tmp_path: Path) -> None:
    """Different file order produces a different hash."""
    f1 = tmp_path / "a.txt"
    f2 = tmp_path / "b.txt"
    f1.write_bytes(b"111")
    f2.write_bytes(b"222")
    h_ab = sha256_over_files([str(f1), str(f2)])
    h_ba = sha256_over_files([str(f2), str(f1)])
    assert h_ab != h_ba


# ---------------------------------------------------------------------------
# glossary_layer_files
# ---------------------------------------------------------------------------


def test_glossary_layer_files_none(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No glossary layers: empty list."""
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.chdir(tmp_path)
    # No .vemoizer/glossary.txt in tmp_path or home.
    assert glossary_layer_files() == []


def test_glossary_layer_files_home_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only home layer exists: one entry."""
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    (home / ".vemoizer").mkdir()
    (home / ".vemoizer" / "glossary.txt").write_text("Nordea\n", encoding="utf-8")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.chdir(tmp_path)
    files = glossary_layer_files()
    assert len(files) == 1
    assert files[0] == home / ".vemoizer" / "glossary.txt"


def test_glossary_layer_files_project_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only project layer exists: one entry."""
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".vemoizer").mkdir()
    (tmp_path / ".vemoizer" / "glossary.txt").write_text("Flagship\n", encoding="utf-8")
    files = glossary_layer_files()
    assert len(files) == 1
    assert files[0] == tmp_path / ".vemoizer" / "glossary.txt"


def test_glossary_layer_files_both(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Both layers exist: two entries, project first, then home."""
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    (home / ".vemoizer").mkdir()
    (home / ".vemoizer" / "glossary.txt").write_text("Nordea\n", encoding="utf-8")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".vemoizer").mkdir()
    (tmp_path / ".vemoizer" / "glossary.txt").write_text("Flagship\n", encoding="utf-8")
    files = glossary_layer_files()
    assert len(files) == 2
    assert files[0] == tmp_path / ".vemoizer" / "glossary.txt"
    assert files[1] == home / ".vemoizer" / "glossary.txt"


def test_glossary_layer_files_project_parent_walkup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Project layer in a parent directory is found by walk-up."""
    home = tmp_path / "home"
    home.mkdir(exist_ok=True)
    monkeypatch.setenv("HOME", str(home))
    # Project glossary is in tmp_path, CWD is tmp_path/sub.
    sub = tmp_path / "sub"
    sub.mkdir()
    (tmp_path / ".vemoizer").mkdir()
    (tmp_path / ".vemoizer" / "glossary.txt").write_text("x\n", encoding="utf-8")
    monkeypatch.chdir(sub)
    files = glossary_layer_files()
    assert len(files) == 1
    assert files[0] == tmp_path / ".vemoizer" / "glossary.txt"
