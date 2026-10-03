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

from vemoizer.sidecar import (
    build_sidecar,
    glossary_layer_files,
    group_durations,
    group_part_paths,
    prompt_term_set_hash,
    sha256_over_files,
)

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
# group_part_paths / group_durations — real part paths (issue #89)
# ---------------------------------------------------------------------------


def test_group_part_paths_single_part_path_label_returns_label(tmp_path: Path):
    """A single-part group's label *is* the part's path: returned as-is,
    never basename-looked-up against files (even when a same-basename
    file is in the list)."""
    a = tmp_path / "sub" / "a.m4a"
    decoy = tmp_path / "decoy" / "a.m4a"
    assert group_part_paths(a, [decoy]) == [a]
    # A multi-part str label still resolves names against files.
    assert group_part_paths("a.m4a", [decoy]) == [decoy]


def test_group_durations_measures_each_resolved_path(tmp_path: Path, monkeypatch):
    """``group_durations`` measures exactly the resolved paths it is given
    (never re-resolves against the CWD or a files list)."""
    a = tmp_path / "sub" / "a.m4a"
    a.parent.mkdir()

    def fake_pcm(path, **kwargs):
        if str(path) == str(a):
            return 7.5
        raise AssertionError(f"unexpected path {path!r}")

    import vemoizer.ingest as ingest_module

    monkeypatch.setattr(ingest_module, "pcm_duration_seconds", fake_pcm)
    assert group_durations([a]) == [7.5]


def test_group_durations_multi_part_paths(tmp_path, monkeypatch):
    """Each resolved part path is measured at its real on-disk path
    (never a CWD-relative name)."""
    a = tmp_path / "a.m4a"
    b = tmp_path / "b.m4a"

    def fake_pcm(path, **kwargs):
        if str(path) == str(a):
            return 1.0
        if str(path) == str(b):
            return 2.0
        raise AssertionError(f"unexpected path {path!r}")

    import vemoizer.ingest as ingest_module

    monkeypatch.setattr(ingest_module, "pcm_duration_seconds", fake_pcm)
    assert group_durations([a, b]) == [1.0, 2.0]


# ---------------------------------------------------------------------------
# prompt_term_set_hash
# ---------------------------------------------------------------------------


def test_prompt_term_set_hash_prompt_terms_only(tmp_path: Path) -> None:
    """Only non-correction, non-@ lines are hashed; pairs and @-names don't count."""
    f = tmp_path / "g.txt"
    f.write_text(
        "Blacksit => Flagship\nFlagship\nNordea\n@Howard\n",
        encoding="utf-8",
    )
    expected = hashlib.sha256(b"Flagship\nNordea").hexdigest()
    assert prompt_term_set_hash([str(f)]) == expected


def test_prompt_term_set_hash_ignores_pairs_and_at_names(tmp_path: Path) -> None:
    """Adding only a correction pair or an @-name does not change the hash."""
    f = tmp_path / "g.txt"
    f.write_text("Flagship\n", encoding="utf-8")
    h1 = prompt_term_set_hash([str(f)])
    f.write_text("Flagship\nBlacksit => Flagship\n@Howard\n", encoding="utf-8")
    assert prompt_term_set_hash([str(f)]) == h1


def test_prompt_term_set_hash_dedup_case_insensitive_first_wins(tmp_path: Path) -> None:
    """Dedup is case-insensitive, first-seen (project = first file) wins."""
    proj = tmp_path / "project.txt"
    home = tmp_path / "home.txt"
    proj.write_text("Flagship\n", encoding="utf-8")
    home.write_text("flagship\nNordea\n", encoding="utf-8")
    expected = hashlib.sha256(b"Flagship\nNordea").hexdigest()
    assert prompt_term_set_hash([str(proj), str(home)]) == expected


def test_prompt_term_set_hash_empty_list_is_none() -> None:
    assert prompt_term_set_hash([]) is None


def test_prompt_term_set_hash_missing_file_fail_open(tmp_path: Path) -> None:
    """A missing file contributes no terms; the remaining file still hashes."""
    f = tmp_path / "g.txt"
    f.write_text("Flagship\n", encoding="utf-8")
    missing = tmp_path / "absent.txt"
    expected = hashlib.sha256(b"Flagship").hexdigest()
    assert prompt_term_set_hash([str(missing), str(f)]) == expected


def test_prompt_term_set_hash_empty_prompt_set(tmp_path: Path) -> None:
    """A glossary with only correction pairs hashes the empty set."""
    f = tmp_path / "g.txt"
    f.write_text("Blacksit => Flagship\n", encoding="utf-8")
    assert prompt_term_set_hash([str(f)]) == hashlib.sha256(b"").hexdigest()


def test_build_sidecar_stores_prompt_term_set_hash(tmp_path: Path) -> None:
    """build_sidecar stores the prompt-term-set hash, so an untouched
    glossary at render time hashes equal (no drift warning)."""
    f = tmp_path / "g.txt"
    f.write_text("Blacksit => Flagship\nFlagship\n", encoding="utf-8")
    result: dict[str, Any] = {"text": "hei", "segments": []}
    build_sidecar(result, command="meeting", glossary_files=[str(f)])
    opts = cast(dict[str, Any], result["options"])
    assert opts["glossary_sha256"] == hashlib.sha256(b"Flagship").hexdigest()


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


# ---------------------------------------------------------------------------
# build_sidecar — header persistence (issue #107, finding 2)
# ---------------------------------------------------------------------------


def test_build_sidecar_persists_duration_s(tmp_path: Path) -> None:
    """A result with duration_s (from pipeline.py) keeps it in the sidecar."""
    result: dict[str, Any] = {
        "text": "hei",
        "segments": [],
        "source_path": str(tmp_path / "a.m4a"),
        "duration_s": 9.15,
        "_source_durations": [9.15],
    }
    build_sidecar(result, command="meeting", glossary_files=None)
    assert result["duration_s"] == 9.15
    assert "_source_durations" not in result


def test_build_sidecar_omits_duration_s_when_absent(tmp_path: Path) -> None:
    """A result without duration_s (ffmpeg fail-open) has no duration_s key."""
    result: dict[str, Any] = {
        "text": "hei",
        "segments": [],
        "source_path": str(tmp_path / "a.m4a"),
    }
    build_sidecar(result, command="meeting", glossary_files=None)
    assert "duration_s" not in result


def test_build_sidecar_drops_explicit_null_duration_s(tmp_path: Path) -> None:
    """An explicit None duration_s (duration measurement failed) is dropped.

    build_sidecar drops the key so format_json's present-only mirror does
    not serialize ``null`` — a null would render a "Kesto: [00:00:00]"
    header line the original run's md did not have (issue #107, finding 2).
    """
    result: dict[str, Any] = {
        "text": "hei",
        "segments": [],
        "source_path": str(tmp_path / "a.m4a"),
        "duration_s": None,
    }
    build_sidecar(result, command="meeting", glossary_files=None)
    assert "duration_s" not in result


def test_build_sidecar_persists_glossary_source(tmp_path: Path) -> None:
    """A result with glossary_source (stashed by the preset seam) keeps it."""
    gfile = tmp_path / "glossary.txt"
    gfile.write_text("Blacksit => Flagship\n", encoding="utf-8")
    result: dict[str, Any] = {
        "text": "hei",
        "segments": [],
        "source_path": str(tmp_path / "a.m4a"),
        "duration_s": 10.0,
        "glossary_source": f"{gfile} (1 terms)",
        "_source_durations": [10.0],
    }
    build_sidecar(result, command="meeting", glossary_files=[str(gfile)])
    assert result["glossary_source"] == f"{gfile} (1 terms)"


def test_build_sidecar_omits_glossary_source_when_absent(tmp_path: Path) -> None:
    """A result without glossary_source (no glossary) has no such key."""
    result: dict[str, Any] = {
        "text": "hei",
        "segments": [],
        "source_path": str(tmp_path / "a.m4a"),
        "duration_s": 5.0,
        "_source_durations": [5.0],
    }
    build_sidecar(result, command="meeting", glossary_files=None)
    assert "glossary_source" not in result


def test_sidecar_round_trip_md_includes_header_lines() -> None:
    """A direct md render equals a sidecar → render → md render (issue #107, finding 2).

    The sidecar carries duration_s and glossary_source; the rendered md
    must include the Kesto and Sanasto header lines.
    """
    from vemoizer.output.markdown import format_md
    from vemoizer.render import render_markdown

    run_result: dict[str, Any] = {
        "text": "Puhuttiin Blacksit-hankkeesta.",
        "duration_s": 9.15,
        "glossary_source": "/path/to/.vemoizer/glossary.txt (1 terms)",
        "paragraphs": [
            {
                "start": 0.0,
                "end": 5.0,
                "text": "Puhuttiin Blacksit-hankkeesta.",
                "speaker": "SPEAKER_1",
            },
        ],
        "notes": {"title": "Alustus"},
    }
    sidecar = build_sidecar(dict(run_result), command="meeting", glossary_files=None)

    # The sidecar carries both header keys.
    assert sidecar["duration_s"] == 9.15
    assert sidecar["glossary_source"] == "/path/to/.vemoizer/glossary.txt (1 terms)"

    # Direct md render (what the original run wrote).
    run_md = format_md(run_result)

    # Sidecar → render → md render.
    render_md = render_markdown(sidecar, corrections={}, speaker_names={})

    # Both header lines present in both renders.
    for md in (run_md, render_md):
        assert "Kesto: [00:00:09]" in md
        assert "Sanasto: /path/to/.vemoizer/glossary.txt (1 terms)" in md

    # Round-trip identity.
    assert render_md == run_md
