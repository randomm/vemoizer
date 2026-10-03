"""Atomic-write tests for the ``vemoizer render`` output writer (issue #107).

Covers the mode-preservation and special-file behaviour of
``render_cli._atomic_write_text`` plus the prompt-term-set hash cross-check
against the run's real glossary seam. All tests use synthetic sidecar JSON
and ``tmp_path``; no models, no network, no ffmpeg.
"""

from __future__ import annotations

import json
import os
import stat as statmod
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
        ],
        "notes": {"title": "Alustus"},
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


def _imode(path: Path) -> int:
    return statmod.S_IMODE(os.stat(path).st_mode)


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
# Mode preservation across the atomic replace (issue #107 fix pass 2, #1)
# ---------------------------------------------------------------------------


def test_render_existing_md_mode_600_preserved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A re-render of an existing 0600 ``.md`` keeps it 0600 — a user's
    ``chmod 600`` on a transcript must survive (regression: the fresh temp
    file made it 0644)."""
    isolate_home(monkeypatch, tmp_path, tmp_path)
    sc = _write_sidecar(tmp_path, _sidecar())
    md = tmp_path / "sidecar.md"
    md.write_text("stale", encoding="utf-8")
    md.chmod(0o600)

    result = runner.invoke(app, ["render", str(sc)])
    assert result.exit_code == 0
    assert _imode(md) == 0o600
    assert "# Alustus" in md.read_text(encoding="utf-8")


def test_render_existing_md_mode_640_preserved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A re-render of an existing 0640 ``.md`` keeps it 0640 (arbitrary
    modes survive, not just 0600)."""
    isolate_home(monkeypatch, tmp_path, tmp_path)
    sc = _write_sidecar(tmp_path, _sidecar())
    md = tmp_path / "sidecar.md"
    md.write_text("stale", encoding="utf-8")
    md.chmod(0o640)

    result = runner.invoke(app, ["render", str(sc)])
    assert result.exit_code == 0
    assert _imode(md) == 0o640


def test_render_new_md_mode_matches_run_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A brand-new ``.md`` gets the run path's mode (``Path.write_text``'
    umask default 0644) — render output is not a secret file, so no
    0600 tightening on creation."""
    isolate_home(monkeypatch, tmp_path, tmp_path)
    sc = _write_sidecar(tmp_path, _sidecar())
    md = tmp_path / "sidecar.md"
    assert not md.exists()

    result = runner.invoke(app, ["render", str(sc)])
    assert result.exit_code == 0
    assert _imode(md) == 0o644


def test_render_out_flag_mode_preserved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``--out`` into an existing 0600 file keeps 0600 (same policy as the
    default ``<stem>.md`` path)."""
    isolate_home(monkeypatch, tmp_path, tmp_path)
    sc = _write_sidecar(tmp_path, _sidecar())
    out = tmp_path / "elsewhere" / "out.md"
    out.parent.mkdir()
    out.write_text("stale", encoding="utf-8")
    out.chmod(0o600)

    result = runner.invoke(app, ["render", str(sc), "--out", str(out)])
    assert result.exit_code == 0
    assert _imode(out) == 0o600
    assert "# Alustus" in out.read_text(encoding="utf-8")


def test_render_name_persist_keeps_sidecar_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``--name`` persistence rewrites the sidecar JSON through the same
    atomic helper; the sidecar's own mode must survive (it was not
    0600-hardened by the pre-fix inline temp+replace either)."""
    isolate_home(monkeypatch, tmp_path, tmp_path)
    sc = _write_sidecar(tmp_path, _sidecar())
    sc.chmod(0o600)

    result = runner.invoke(app, ["render", str(sc), "--name", "SPEAKER_1=Aila"])
    assert result.exit_code == 0
    assert _imode(sc) == 0o600
    data = json.loads(sc.read_text(encoding="utf-8"))
    assert data["speaker_names"]["SPEAKER_1"] == "Aila"


# ---------------------------------------------------------------------------
# Special-file --out targets (issue #107 fix pass 2, #2)
# ---------------------------------------------------------------------------


def test_render_out_dev_null_succeeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``--out /dev/null`` succeeds: a character-device target is not a
    regular file, so the atomic temp+replace path would fail to create the
    temp in ``/dev``; the helper falls back to an in-place write (the
    pre-atomic behaviour for non-regular targets)."""
    if not os.path.exists("/dev/null"):
        pytest.skip("no /dev/null on this platform")
    isolate_home(monkeypatch, tmp_path, tmp_path)
    sc = _write_sidecar(tmp_path, _sidecar())

    result = runner.invoke(app, ["render", str(sc), "--out", "/dev/null"])
    assert result.exit_code == 0
    assert "wrote /dev/null" in result.stdout


def test_render_out_file_named_dash_is_plain_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``--out <path>-`` where the name is ``-`` is a PLAIN file named
    ``-`` (render has no stdout convention; the run path's ``-``=stdout
    lives in ``_write_output`` only). Behaviour is unchanged: it is
    written like any other regular file."""
    isolate_home(monkeypatch, tmp_path, tmp_path)
    sc = _write_sidecar(tmp_path, _sidecar())
    dash = tmp_path / "-"
    dash.write_text("stale", encoding="utf-8")
    dash.chmod(0o600)

    result = runner.invoke(app, ["render", str(sc), "--out", str(dash)])
    assert result.exit_code == 0
    assert _imode(dash) == 0o600
    assert "# Alustus" in dash.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Prompt-term-set hash: render's derivation vs the run's seam (lens medium)
# ---------------------------------------------------------------------------


def test_prompt_term_hash_equals_run_seam_derivation(tmp_path: Path) -> None:
    """The term list ``prompt_term_set_hash`` hashes equals the term list
    derived through the run's real seam (``load_layers`` + ``merge`` with a
    ``None`` tokenizer, ``@`` terms filtered) across layered fixtures —
    project + home layers, case-differing duplicates, ``@`` names,
    ``=>`` pairs, comments. If the two derivations drift, this fails."""
    import hashlib

    from vemoizer.glossary_layers import load_layers, merge
    from vemoizer.sidecar import prompt_term_list, prompt_term_set_hash

    home_g = tmp_path / "home" / "glossary.txt"
    home_g.parent.mkdir()
    proj_g = tmp_path / "proj" / "glossary.txt"
    proj_g.parent.mkdir()
    proj_g.write_text(
        "Flagship\nflagship\n@LLMonly\nNewport => Nyborg\n# a comment\nUusi äänitys\n",
        encoding="utf-8",
    )
    home_g.write_text(
        "Hometerm\nFlagship\n@HomeLLM\nOldterm => Neoteric\ndup\n",
        encoding="utf-8",
    )
    # Project layer first, then home (render's layered order).
    files = [proj_g, home_g]
    hashed = prompt_term_list(files)

    home_terms, home_corr, proj_terms, proj_corr = load_layers(
        home_path=home_g, project_path=proj_g
    )
    merged, _, _ = merge(proj_terms, proj_corr, home_terms, home_corr)
    run_terms = [t for t in merged if not t.startswith("@")]

    assert hashed == run_terms
    assert hashlib.sha256("\n".join(hashed).encode("utf-8")).hexdigest() == (
        prompt_term_set_hash(files)
    )


def test_prompt_term_set_hash_skips_non_regular_paths(tmp_path: Path) -> None:
    """A non-regular path (a directory) in the file list is skipped
    before any read attempt, so a special file cannot block or crash
    ``render``'s drift check (fail-open, like a missing file)."""
    import hashlib

    from vemoizer.sidecar import prompt_term_set_hash

    good = tmp_path / "good.txt"
    good.write_text("Flagship\n", encoding="utf-8")
    adir = tmp_path / "adir"
    adir.mkdir()

    # Must not raise, and the directory contributes nothing.
    assert prompt_term_set_hash([adir, good]) == hashlib.sha256(b"Flagship").hexdigest()
    # A non-regular path alone: empty term set (still a hash, not None —
    # the list was non-empty).
    assert prompt_term_set_hash([adir]) == hashlib.sha256(b"").hexdigest()
