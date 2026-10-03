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


# ---------------------------------------------------------------------------
# End-to-end: preset run with project glossary → sidecar glossary_files
# point at real files; render emits no drift warning until the glossary
# is edited (issue #89, finding B).
# ---------------------------------------------------------------------------


def test_preset_run_sidecar_glossary_files_exist_and_render_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A meeting run with a project ``.vemoizer/glossary.txt`` stores the
    real layer file in ``options.glossary_files`` (not the deleted temp
    file). After the run, every stored path exists. ``vemoizer render``
    emits no drift warning on the untouched glossary (both sides hash the
    same prompt-term set), and exactly one when a prompt term is added.
    """
    import vemoizer.pipeline as pipeline_module

    # Project glossary with a correction pair.
    vemoizer_dir = tmp_path / ".vemoizer"
    vemoizer_dir.mkdir()
    glossary = vemoizer_dir / "glossary.txt"
    glossary.write_text("Blacksit => Flagship\n", encoding="utf-8")

    def fake_transcribe(path, **kwargs):
        return {
            "text": "Puhuttiin Blacksit-hankkeesta.",
            "paragraphs": [
                {
                    "start": 0.0,
                    "end": 5.0,
                    "text": "Puhuttiin Blacksit.",
                    "speaker": "SPEAKER_1",
                }
            ],
            "notes": {"title": "Alustus"},
        }

    monkeypatch.setattr(pipeline_module, "transcribe_file", fake_transcribe)
    isolate_home(monkeypatch, tmp_path, tmp_path)

    # Run the meeting preset.
    result = runner.invoke(app, ["meeting", "a.m4a"])
    assert result.exit_code == 0

    # Find the sidecar .json.
    json_files = list(tmp_path.glob("*.json"))
    assert len(json_files) == 1, f"expected 1 .json, got {json_files}"
    sidecar = json.loads(json_files[0].read_text(encoding="utf-8"))

    opts = sidecar["options"]
    stored_files = opts["glossary_files"]
    # The stored glossary files must be the real layer file, not a temp path.
    assert len(stored_files) == 1
    assert stored_files[0] == str(glossary)
    # Every stored path must exist after the run (the temp file is deleted).
    for p in stored_files:
        assert Path(p).is_file(), f"stored glossary path does not exist: {p}"

    # Both the run (build_sidecar) and render hash the glossary's
    # prompt-term set (via prompt_term_set_hash), so an untouched
    # glossary must NOT warn. Adding only correction pairs does not
    # change the prompt-term set (no warning); adding a prompt term
    # does (exactly one warning).
    assert opts["glossary_sha256"] is not None

    out_md = tmp_path / "rendered.md"
    result2 = runner.invoke(app, ["render", str(json_files[0]), "--out", str(out_md)])
    assert result2.exit_code == 0
    assert "re-transcribe" not in result2.stderr

    # Add only a correction pair (no new prompt term): no warning.
    glossary.write_text("Blacksit => Flagship\nOldpair => Newpair\n", encoding="utf-8")
    result3 = runner.invoke(app, ["render", str(json_files[0]), "--out", str(out_md)])
    assert result3.exit_code == 0
    assert "re-transcribe" not in result3.stderr

    # Add a prompt term: the prompt-term set changes, so exactly one warning.
    glossary.write_text(
        "Blacksit => Flagship\nOldpair => Newpair\nNewterm\n", encoding="utf-8"
    )
    result4 = runner.invoke(app, ["render", str(json_files[0]), "--out", str(out_md)])
    assert result4.exit_code == 0
    assert result4.stderr.count("re-transcribe") == 1


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


def test_render_non_utf8_sidecar(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A non-UTF-8 (binary) sidecar exits 1 with a clean error line, not a
    traceback."""
    isolate_home(monkeypatch, tmp_path, tmp_path)
    bad = tmp_path / "binary.json"
    bad.write_bytes(b"\xff\xfe\x00bad")
    result = runner.invoke(app, ["render", str(bad)])
    assert result.exit_code == 1
    assert "could not read" in result.stderr
    assert "Traceback" not in result.stderr


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
    # The warning names the exception class, not raw exception text.
    assert "warning: could not read glossary" in result.stderr
    assert "ValueError" in result.stderr
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
# Hash mismatch warning (prompt-term-set based)
# ---------------------------------------------------------------------------


def test_render_prompt_term_hash_mismatch_warns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Stored prompt-term-set hash differs from current: one stderr line."""
    isolate_home(monkeypatch, tmp_path, tmp_path)
    g = tmp_path / "glossary.txt"
    g.write_text("Blacksit => Flagship\n", encoding="utf-8")
    sc_data = _sidecar()
    # Store a hash that does NOT match the current prompt-term set.
    # The glossary has no prompt terms, so the prompt-term hash is
    # sha256("") — use a different value to guarantee a mismatch.
    sc_data["options"]["glossary_sha256"] = "deadbeef" * 8
    sc = _write_sidecar(tmp_path, sc_data)
    result = runner.invoke(app, ["render", str(sc), "--glossary", str(g)])
    assert result.exit_code == 0
    assert "re-transcribe" in result.stderr
    # Corrections were still applied.
    md = list(tmp_path.glob("*.md"))
    assert len(md) == 1
    assert "Flagship" in md[0].read_text(encoding="utf-8")


def test_render_prompt_term_hash_match_no_warning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Matching prompt-term-set hash: no warning on stderr."""
    isolate_home(monkeypatch, tmp_path, tmp_path)
    import hashlib

    g = tmp_path / "glossary.txt"
    # Glossary with prompt terms AND a correction pair.
    g.write_text("Blacksit => Flagship\nFlagship\nNordea\n", encoding="utf-8")
    # Compute the prompt-term-set hash: terms are {"Flagship", "Nordea"}
    # (correction line excluded). The hash is over the newline-joined list.
    expected_hash = hashlib.sha256(b"Flagship\nNordea").hexdigest()
    sc_data = _sidecar()
    sc_data["options"]["glossary_sha256"] = expected_hash
    sc = _write_sidecar(tmp_path, sc_data)
    result = runner.invoke(app, ["render", str(sc), "--glossary", str(g)])
    assert result.exit_code == 0
    assert "re-transcribe" not in result.stderr


def test_render_adding_correction_pair_no_warning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Adding only a correction pair (``a => b``) does not trigger a warning
    because the prompt-term set is unchanged."""
    isolate_home(monkeypatch, tmp_path, tmp_path)
    import hashlib

    g = tmp_path / "glossary.txt"
    # Start with one prompt term.
    g.write_text("Flagship\n", encoding="utf-8")
    expected_hash = hashlib.sha256(b"Flagship").hexdigest()
    sc_data = _sidecar()
    sc_data["options"]["glossary_sha256"] = expected_hash
    sc = _write_sidecar(tmp_path, sc_data)

    # Render: no warning (hash matches).
    result = runner.invoke(app, ["render", str(sc), "--glossary", str(g)])
    assert result.exit_code == 0
    assert "re-transcribe" not in result.stderr

    # Add a correction pair only — no new prompt term.
    g.write_text("Flagship\nBlacksit => Flagship\n", encoding="utf-8")
    result2 = runner.invoke(app, ["render", str(sc), "--glossary", str(g)])
    assert result2.exit_code == 0
    assert "re-transcribe" not in result2.stderr


def test_render_adding_at_name_no_warning(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Adding only an ``@``-prefixed LLM-only name does not trigger a warning
    because the prompt-term set is unchanged (``@`` terms are not prompt
    terms)."""
    isolate_home(monkeypatch, tmp_path, tmp_path)
    import hashlib

    g = tmp_path / "glossary.txt"
    g.write_text("Flagship\n", encoding="utf-8")
    expected_hash = hashlib.sha256(b"Flagship").hexdigest()
    sc_data = _sidecar()
    sc_data["options"]["glossary_sha256"] = expected_hash
    sc = _write_sidecar(tmp_path, sc_data)

    result = runner.invoke(app, ["render", str(sc), "--glossary", str(g)])
    assert result.exit_code == 0
    assert "re-transcribe" not in result.stderr

    # Add an @-name only — no new prompt term.
    g.write_text("Flagship\n@Howard\n", encoding="utf-8")
    result2 = runner.invoke(app, ["render", str(sc), "--glossary", str(g)])
    assert result2.exit_code == 0
    assert "re-transcribe" not in result2.stderr


def test_render_adding_prompt_term_warns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Adding a new prompt term (non-``=>``, non-``@``) triggers exactly
    one warning."""
    isolate_home(monkeypatch, tmp_path, tmp_path)
    import hashlib

    g = tmp_path / "glossary.txt"
    g.write_text("Flagship\n", encoding="utf-8")
    expected_hash = hashlib.sha256(b"Flagship").hexdigest()
    sc_data = _sidecar()
    sc_data["options"]["glossary_sha256"] = expected_hash
    sc = _write_sidecar(tmp_path, sc_data)

    result = runner.invoke(app, ["render", str(sc), "--glossary", str(g)])
    assert result.exit_code == 0
    assert "re-transcribe" not in result.stderr

    # Add a prompt term — the prompt-term set changes.
    g.write_text("Flagship\nNewterm\n", encoding="utf-8")
    result2 = runner.invoke(app, ["render", str(sc), "--glossary", str(g)])
    assert result2.exit_code == 0
    assert result2.stderr.count("re-transcribe") == 1


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
# Render overwrites existing .md (issue #107 finding 1)
# ---------------------------------------------------------------------------


def test_render_overwrites_existing_md(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Re-rendering a sidecar whose .md already exists overwrites it in place;
    no `` (2)`` suffix is created (issue #107 finding 1)."""
    isolate_home(monkeypatch, tmp_path, tmp_path)
    sc = _write_sidecar(tmp_path, _sidecar())
    # Pre-create the .md that render would write (same stem as the .json).
    existing_md = tmp_path / "sidecar.md"
    existing_md.write_text("stale content", encoding="utf-8")
    result = runner.invoke(app, ["render", str(sc)])
    assert result.exit_code == 0
    # Exactly one .md file exists — the original was overwritten, not
    # duplicated with a " (2)" suffix.
    md_files = list(tmp_path.glob("*.md"))
    assert len(md_files) == 1, f"expected 1 .md file, got {md_files}"
    assert not any(" (2)" in f.name for f in md_files)
    # The file was overwritten with fresh content.
    content = existing_md.read_text(encoding="utf-8")
    assert "stale content" not in content
    assert "# Alustus" in content


# ---------------------------------------------------------------------------
# Atomic write: interrupted write, symlink safety (issue #107 finding 4)
# ---------------------------------------------------------------------------


def test_render_atomic_write_interrupted_leaves_previous_intact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An interrupted write (OSError during the temp file write) leaves the
    previous file intact and leaves no temp file behind."""
    isolate_home(monkeypatch, tmp_path, tmp_path)
    sc = _write_sidecar(tmp_path, _sidecar())
    existing_md = tmp_path / "sidecar.md"
    existing_md.write_text("original content", encoding="utf-8")

    # Monkeypatch Path.write_text to raise OSError when writing the temp file.
    from pathlib import Path as _Path

    original_write_text = _Path.write_text

    def _failing_write_text(self, data, **kwargs):
        # The temp file name contains ".tmp-"; the final target does not.
        if ".tmp-" in self.name:
            raise OSError("simulated disk full")
        return original_write_text(self, data, **kwargs)

    monkeypatch.setattr(_Path, "write_text", _failing_write_text)

    result = runner.invoke(app, ["render", str(sc)])
    assert result.exit_code == 1
    assert "could not write" in result.stderr

    # The previous file is intact.
    assert existing_md.read_text(encoding="utf-8") == "original content"

    # No temp file left behind.
    tmp_files = [f for f in tmp_path.glob("*.tmp-*")]
    assert len(tmp_files) == 0, f"temp file left behind: {tmp_files}"


def test_render_atomic_write_symlink_target_not_modified(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When the .md path is a symlink to another file, the pointed-to file
    is NOT modified; the symlink path now holds the new render (os.replace
    replaces the symlink, not the target)."""
    isolate_home(monkeypatch, tmp_path, tmp_path)
    sc = _write_sidecar(tmp_path, _sidecar())

    # Create a real file that the symlink will point to.
    real_file = tmp_path / "real_target.md"
    real_file.write_text("original target content", encoding="utf-8")

    # Create a symlink at the .md path that points to the real file.
    md_path = tmp_path / "sidecar.md"
    md_path.symlink_to(real_file)
    assert md_path.is_symlink()

    result = runner.invoke(app, ["render", str(sc)])
    assert result.exit_code == 0

    # The pointed-to file is NOT modified.
    assert real_file.read_text(encoding="utf-8") == "original target content"

    # The symlink path is no longer a symlink (os.replace replaced it
    # with a regular file).
    assert not md_path.is_symlink()
    assert md_path.is_file()
    # The symlink path now holds the new render.
    content = md_path.read_text(encoding="utf-8")
    assert "# Alustus" in content
    assert "original target content" not in content


def test_render_atomic_write_out_flag_symlink_target_not_modified(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Same symlink safety for the --out path."""
    isolate_home(monkeypatch, tmp_path, tmp_path)
    sc = _write_sidecar(tmp_path, _sidecar())

    real_file = tmp_path / "real.md"
    real_file.write_text("original", encoding="utf-8")

    out_path = tmp_path / "custom" / "out.md"
    out_path.parent.mkdir(exist_ok=True)
    out_path.symlink_to(real_file)
    assert out_path.is_symlink()

    result = runner.invoke(app, ["render", str(sc), "--out", str(out_path)])
    assert result.exit_code == 0

    # The pointed-to file is NOT modified.
    assert real_file.read_text(encoding="utf-8") == "original"
    # The symlink path is now a regular file with the new render.
    assert not out_path.is_symlink()
    assert out_path.is_file()
    assert "# Alustus" in out_path.read_text(encoding="utf-8")


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

    correct_hash = hashlib.sha256(b"").hexdigest()
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
