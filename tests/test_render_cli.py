"""CLI tests for the ``vemoizer render`` command (issue #89, M5a workstream).

All tests use synthetic sidecar JSON fixtures and ``tmp_path``; every test
chdirs to ``tmp_path`` with ``HOME`` monkeypatched (``isolate_home``).
No models, no network, no ffmpeg.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from _cli_helpers import isolate_home
from typer.testing import CliRunner

from vemoizer.cli import app

runner = CliRunner()


def _sidecar(**extra: Any) -> dict[str, Any]:
    """A minimal M5a sidecar with notes and paragraphs."""
    base: dict[str, Any] = {
        "text": "Puhuttiin Blacksit-hankkeesta.",
        "paragraphs": [
            {
                "start": 0.0,
                "end": 5.0,
                "text": "Puhuttiin Blacksit-hankkeesta.",
                "speaker": "SPEAKER_1",
            },
            {
                "start": 8.0,
                "end": 12.0,
                "text": "epittä selvitetään myöhemmin.",
                "speaker": "SPEAKER_2",
            },
        ],
        "notes": {
            "title": "Alustus",
            "summary": "Puhuttiin Blacksit-hankkeesta.",
            "key_points": ["Blacksit käynnistyy ensi kuussa"],
            "action_items": ["SPEAKER_1: Kirjaa epittä"],
        },
        "options": {
            "command": "meeting",
            "glossary_files": [],
            "glossary_sha256": None,
        },
        "speaker_names": {},
        **extra,
    }
    return base


def _write_sidecar(
    tmp_path: Path, data: dict[str, Any], name: str = "sidecar.json"
) -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# Basic render: no glossary, no names
# ---------------------------------------------------------------------------


def test_render_basic_writes_md(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """render of a sidecar with no glossary writes a .md next to the .json."""
    isolate_home(monkeypatch, tmp_path, tmp_path)
    sc = _write_sidecar(tmp_path, _sidecar())
    result = runner.invoke(app, ["render", str(sc)])
    assert result.exit_code == 0
    md_files = list(tmp_path.glob("*.md"))
    assert len(md_files) == 1
    content = md_files[0].read_text(encoding="utf-8")
    assert "# Alustus" in content
    assert "Puhuttiin" in content
    assert "wrote" in result.stdout


def test_render_old_sidecar_without_m5a_keys(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A pre-M5 .json (no source/options/speaker_names) still renders."""
    isolate_home(monkeypatch, tmp_path, tmp_path)
    old = {
        "text": "Vanha äänitys.",
        "paragraphs": [{"start": 0.0, "text": "Vanha äänitys."}],
        "notes": {"title": "Vanha", "action_items": []},
    }
    sc = _write_sidecar(tmp_path, old)
    result = runner.invoke(app, ["render", str(sc)])
    assert result.exit_code == 0
    md_files = list(tmp_path.glob("*.md"))
    assert len(md_files) == 1
    assert "# Vanha" in md_files[0].read_text(encoding="utf-8")


def test_render_sidecar_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A non-existent sidecar exits 1 with a clean error."""
    isolate_home(monkeypatch, tmp_path, tmp_path)
    result = runner.invoke(app, ["render", str(tmp_path / "nope.json")])
    assert result.exit_code == 1
    assert "not found" in result.stderr


def test_render_malformed_json(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A non-JSON file exits 1 with a clean error."""
    isolate_home(monkeypatch, tmp_path, tmp_path)
    bad = tmp_path / "bad.json"
    bad.write_text("not json {", encoding="utf-8")
    result = runner.invoke(app, ["render", str(bad)])
    assert result.exit_code == 1
    assert "malformed" in result.stderr


def test_render_non_object_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A JSON array (not an object) exits 1."""
    isolate_home(monkeypatch, tmp_path, tmp_path)
    bad = tmp_path / "arr.json"
    bad.write_text("[1, 2, 3]", encoding="utf-8")
    result = runner.invoke(app, ["render", str(bad)])
    assert result.exit_code == 1
    assert "malformed" in result.stderr


def test_render_missing_text_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A JSON object without a 'text' key exits 1."""
    isolate_home(monkeypatch, tmp_path, tmp_path)
    bad = tmp_path / "no_text.json"
    bad.write_text('{"paragraphs": []}', encoding="utf-8")
    result = runner.invoke(app, ["render", str(bad)])
    assert result.exit_code == 1
    assert "text" in result.stderr


# ---------------------------------------------------------------------------
# --out flag
# ---------------------------------------------------------------------------


def test_render_out_writes_to_explicit_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """--out writes the .md to the specified path."""
    isolate_home(monkeypatch, tmp_path, tmp_path)
    sc = _write_sidecar(tmp_path, _sidecar())
    out = tmp_path / "custom" / "out.md"
    result = runner.invoke(app, ["render", str(sc), "--out", str(out)])
    assert result.exit_code == 0
    assert out.is_file()
    assert "# Alustus" in out.read_text(encoding="utf-8")
    # No .md file next to the .json (the default path was not used).
    md_files = [f for f in tmp_path.glob("*.md") if f.name != "out.md"]
    assert len(md_files) == 0


# ---------------------------------------------------------------------------
# --glossary: correction re-application
# ---------------------------------------------------------------------------


def test_render_glossary_applies_corrections(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """--glossary with correction pairs: the .md shows the corrected text."""
    isolate_home(monkeypatch, tmp_path, tmp_path)
    g = tmp_path / "glossary.txt"
    g.write_text("Blacksit => Flagship\nepit* => EBITDA\n", encoding="utf-8")
    sc = _write_sidecar(tmp_path, _sidecar())
    result = runner.invoke(app, ["render", str(sc), "--glossary", str(g)])
    assert result.exit_code == 0
    md = list(tmp_path.glob("*.md"))
    assert len(md) == 1
    content = md[0].read_text(encoding="utf-8")
    assert "Flagship-hankkeesta" in content
    assert "Blacksit" not in content
    assert "EBITDA selvitetään" in content


def test_render_missing_glossary_warns_and_proceeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A missing --glossary file warns to stderr and proceeds (exit 0)."""
    isolate_home(monkeypatch, tmp_path, tmp_path)
    sc = _write_sidecar(tmp_path, _sidecar())
    missing = tmp_path / "nope.txt"
    result = runner.invoke(app, ["render", str(sc), "--glossary", str(missing)])
    assert result.exit_code == 0
    assert "not found" in result.stderr
    md_files = list(tmp_path.glob("*.md"))
    assert len(md_files) == 1


def test_render_non_utf8_glossary_warns_and_proceeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A non-UTF-8 --glossary warns to stderr and proceeds (exit 0)."""
    isolate_home(monkeypatch, tmp_path, tmp_path)
    g = tmp_path / "glossary.txt"
    g.write_bytes(b"\xff\xfe\x00bad")
    sc = _write_sidecar(tmp_path, _sidecar())
    result = runner.invoke(app, ["render", str(sc), "--glossary", str(g)])
    assert result.exit_code == 0
    assert "warning: could not read glossary" in result.stderr
    md_files = list(tmp_path.glob("*.md"))
    assert len(md_files) == 1


def test_render_layered_glossary_missing_file_warns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A missing layered glossary file warns and proceeds (exit 0)."""
    isolate_home(monkeypatch, tmp_path, tmp_path)
    sc = _write_sidecar(tmp_path, _sidecar())
    # No glossary files exist in tmp_path, so glossary_layer_files returns [].
    # This is the normal path — no warning expected.
    result = runner.invoke(app, ["render", str(sc)])
    assert result.exit_code == 0


# ---------------------------------------------------------------------------
# Hash mismatch warning
# ---------------------------------------------------------------------------


def test_render_hash_mismatch_warns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Stored glossary_sha256 differs from current hash: one stderr line."""
    isolate_home(monkeypatch, tmp_path, tmp_path)
    g = tmp_path / "glossary.txt"
    g.write_text("Blacksit => Flagship\n", encoding="utf-8")
    sc_data = _sidecar()
    # Store a hash that does NOT match the current glossary file.
    sc_data["options"]["glossary_sha256"] = "deadbeef" * 8
    sc = _write_sidecar(tmp_path, sc_data)
    result = runner.invoke(app, ["render", str(sc), "--glossary", str(g)])
    assert result.exit_code == 0
    assert "re-transcribe" in result.stderr
    # Corrections were still applied.
    md = list(tmp_path.glob("*.md"))
    assert len(md) == 1
    assert "Flagship" in md[0].read_text(encoding="utf-8")


def test_render_hash_match_no_warning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Matching hash: no warning on stderr."""
    isolate_home(monkeypatch, tmp_path, tmp_path)
    import hashlib

    g = tmp_path / "glossary.txt"
    g.write_text("Blacksit => Flagship\n", encoding="utf-8")
    correct_hash = hashlib.sha256(g.read_bytes()).hexdigest()
    sc_data = _sidecar()
    sc_data["options"]["glossary_sha256"] = correct_hash
    sc = _write_sidecar(tmp_path, sc_data)
    result = runner.invoke(app, ["render", str(sc), "--glossary", str(g)])
    assert result.exit_code == 0
    assert "re-transcribe" not in result.stderr


# ---------------------------------------------------------------------------
# --name: persist into sidecar atomically
# ---------------------------------------------------------------------------


def test_render_name_persists_to_sidecar(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """--name LABEL=NAME is persisted into the sidecar's speaker_names."""
    isolate_home(monkeypatch, tmp_path, tmp_path)
    sc = _write_sidecar(tmp_path, _sidecar())
    result = runner.invoke(app, ["render", str(sc), "--name", "SPEAKER_1=Mats"])
    assert result.exit_code == 0
    # The .json on disk now has the persisted name.
    updated = json.loads(sc.read_text(encoding="utf-8"))
    assert updated["speaker_names"] == {"SPEAKER_1": "Mats"}
    # The .md shows the name.
    md = list(tmp_path.glob("*.md"))
    assert len(md) == 1
    assert "[Mats]" in md[0].read_text(encoding="utf-8")


def test_render_name_atomic_rewrite(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The sidecar .json is valid JSON after --name (no partial write)."""
    isolate_home(monkeypatch, tmp_path, tmp_path)
    sc = _write_sidecar(tmp_path, _sidecar())
    result = runner.invoke(app, ["render", str(sc), "--name", "SPEAKER_1=Mats"])
    assert result.exit_code == 0
    # The file is valid JSON (atomic os.replace guarantees this).
    data = json.loads(sc.read_text(encoding="utf-8"))
    assert data["speaker_names"] == {"SPEAKER_1": "Mats"}
    # All other keys preserved.
    assert data["text"] == _sidecar()["text"]
    assert "notes" in data


def test_render_name_multiple(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Multiple --name flags are all persisted."""
    isolate_home(monkeypatch, tmp_path, tmp_path)
    sc = _write_sidecar(tmp_path, _sidecar())
    result = runner.invoke(
        app,
        [
            "render",
            str(sc),
            "--name",
            "SPEAKER_1=Mats",
            "--name",
            "SPEAKER_2=Sanna",
        ],
    )
    assert result.exit_code == 0
    data = json.loads(sc.read_text(encoding="utf-8"))
    assert data["speaker_names"] == {"SPEAKER_1": "Mats", "SPEAKER_2": "Sanna"}


def test_render_name_bad_format_exits_2(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """--name without = exits 2."""
    isolate_home(monkeypatch, tmp_path, tmp_path)
    sc = _write_sidecar(tmp_path, _sidecar())
    result = runner.invoke(app, ["render", str(sc), "--name", "badformat"])
    assert result.exit_code == 2
    assert "LABEL=NAME" in result.stderr


# ---------------------------------------------------------------------------
# Collision-free output naming
# ---------------------------------------------------------------------------


def test_render_collision_suffix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When the .md already exists, a collision suffix is added."""
    isolate_home(monkeypatch, tmp_path, tmp_path)
    sc = _write_sidecar(tmp_path, _sidecar())
    # Pre-create the .md that render would write (same stem as the .json).
    existing_md = tmp_path / "sidecar.md"
    existing_md.write_text("existing", encoding="utf-8")
    result = runner.invoke(app, ["render", str(sc)])
    assert result.exit_code == 0
    # A " (2)" suffix was added.
    md_files = list(tmp_path.glob("*.md"))
    assert len(md_files) == 2, f"expected 2 .md files, got {md_files}"
    assert any(" (2)" in f.name for f in md_files)


# ---------------------------------------------------------------------------
# Round-trip: render of its own JSON with same glossary
# ---------------------------------------------------------------------------


def test_render_round_trip_byte_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """render of a sidecar with the same glossary produces the same .md
    as format_md applied to the same result with corrections applied."""
    from vemoizer.glossary import apply_corrections, apply_corrections_to_notes
    from vemoizer.output.markdown import format_md

    isolate_home(monkeypatch, tmp_path, tmp_path)

    corrections = {"Blacksit": "Flagship", "epit*": "EBITDA"}
    # Write the glossary file.
    g = tmp_path / "glossary.txt"
    g.write_text("Blacksit => Flagship\nepit* => EBITDA\n", encoding="utf-8")

    import hashlib

    correct_hash = hashlib.sha256(g.read_bytes()).hexdigest()
    sc_data = _sidecar()
    sc_data["options"]["glossary_sha256"] = correct_hash
    sc = _write_sidecar(tmp_path, sc_data)

    # What the run wrote: corrections already applied to the in-memory result.
    run_result = _sidecar()
    run_result["notes"] = apply_corrections_to_notes(run_result["notes"], corrections)
    run_result["paragraphs"] = apply_corrections(run_result["paragraphs"], corrections)
    run_md = format_md(run_result)

    result = runner.invoke(app, ["render", str(sc), "--glossary", str(g)])
    assert result.exit_code == 0
    md_files = list(tmp_path.glob("*.md"))
    assert len(md_files) == 1
    render_md = md_files[0].read_text(encoding="utf-8")
    assert render_md == run_md
