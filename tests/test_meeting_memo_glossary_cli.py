"""CLI tests for the layered glossary temp-file seam (issue #82): corrections-only
for memo, merged terms + ``@`` lines + pairs for meeting.

Every test monkeypatches ``HOME`` and chdirs into a tmp dir (``isolate_home``).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from _cli_helpers import isolate_home
from typer.testing import CliRunner

from vemoizer.cli import app

runner = CliRunner()


# -- layered glossary composition (issue #82, DESIGN DECISION) -----------
# run_preset calls glossary_layers.load_layers + merge, writes the
# composed glossary to a temp file (corrections-only for memo, merged
# terms+@+pairs for meeting), passes it through the existing glossary_path
# argument, and deletes the temp file after the run. The fake transcribe
# seam captures the glossary_path it actually received.


def test_meeting_writes_merged_glossary_to_temp_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """meeting: the temp glossary carries merged terms (@ lines) + pairs."""
    import vemoizer.pipeline as pipeline_module

    seen: dict = {}
    glossary_content: dict = {}
    tmp_path.mkdir(exist_ok=True)
    # .vemoizer layer in tmp_path — the nearest project layer on the walk-up.
    vemoizer_dir = tmp_path / ".vemoizer"
    vemoizer_dir.mkdir()
    (vemoizer_dir / "glossary.txt").write_text(
        "Flagship-hanke\n@Nordea\nBlacksit => Flagship\n", encoding="utf-8"
    )

    def fake_transcribe(path, **kwargs):
        seen.update(kwargs)
        gp = kwargs.get("glossary_path")
        if gp is not None:
            glossary_content["text"] = Path(gp).read_text(encoding="utf-8")
        return {
            "text": "moikka",
            "segments": [],
            "notes": {"title": "T"},
        }

    monkeypatch.setattr(pipeline_module, "transcribe_file", fake_transcribe)
    isolate_home(monkeypatch, tmp_path, tmp_path)
    result = runner.invoke(app, ["meeting", "a.m4a"])
    assert result.exit_code == 0
    glossary_path = seen["glossary_path"]
    # A temp file was passed (not None, not the --glossary path).
    assert glossary_path is not None
    contents = glossary_content["text"]
    assert "Flagship-hanke" in contents
    assert "@Nordea" in contents
    assert "Blacksit => Flagship" in contents
    # The temp file is deleted after the run.
    assert not Path(glossary_path).exists()


def test_memo_writes_corrections_only_to_temp_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """memo: the temp glossary contains ONLY correction pairs — no terms, no @ lines."""
    import vemoizer.pipeline as pipeline_module

    seen: dict = {}
    glossary_content: dict = {}
    vemoizer_dir = tmp_path / ".vemoizer"
    vemoizer_dir.mkdir()
    (vemoizer_dir / "glossary.txt").write_text(
        "Flagship-hanke\n@Nordea\nBlacksit => Flagship\n", encoding="utf-8"
    )

    def fake_transcribe(path, **kwargs):
        seen.update(kwargs)
        gp = kwargs.get("glossary_path")
        if gp is not None:
            glossary_content["text"] = Path(gp).read_text(encoding="utf-8")
        return {
            "text": "moikka",
            "segments": [],
            "notes": {"title": "T"},
        }

    monkeypatch.setattr(pipeline_module, "transcribe_file", fake_transcribe)
    isolate_home(monkeypatch, tmp_path, tmp_path)
    result = runner.invoke(app, ["memo", "a.m4a"])
    assert result.exit_code == 0
    glossary_path = seen["glossary_path"]
    assert glossary_path is not None
    contents = glossary_content["text"]
    # Corrections-only: the prompt terms must NOT be in the temp file.
    assert "Blacksit => Flagship" in contents
    assert "Flagship-hanke" not in contents
    assert "@Nordea" not in contents
    # The temp file is deleted after the run.
    assert not Path(glossary_path).exists()


def test_meeting_no_glossary_layers_no_temp_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No .vemoizer layers and no --glossary: glossary_path is None."""
    import vemoizer.pipeline as pipeline_module

    seen: dict = {}
    # Clean tmp dir with no .vemoizer subdirectory and HOME pointed away.
    clean = tmp_path / "clean"
    clean.mkdir()
    isolate_home(monkeypatch, clean, clean)

    def fake_transcribe(path, **kwargs):
        seen.update(kwargs)
        return {"text": "moikka", "segments": [], "notes": {"title": "T"}}

    monkeypatch.setattr(pipeline_module, "transcribe_file", fake_transcribe)
    result = runner.invoke(app, ["meeting", "a.m4a"])
    assert result.exit_code == 0
    assert seen["glossary_path"] is None


def test_memo_explicit_glossary_filters_to_corrections(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """memo --glossary: the whisper prompt stays empty (issue #82).

    The explicit file replaces both layers, but prompt terms (bare and
    @-lines) must not seed the whisper initial_prompt — only that file's
    correction pairs pass through, via a temp file deleted after the run.
    """
    import vemoizer.pipeline as pipeline_module
    from vemoizer.glossary import glossary_prompt, load_glossary

    seen: dict = {}
    g = tmp_path / "override.txt"
    g.write_text("Nordea\n@Jukka\nBlacksit => Flagship\n", encoding="utf-8")

    def fake_transcribe(path, **kwargs):
        # The real transcribe_file reads the glossary file; the fake
        # captures it too, while the temp file still exists (it is
        # deleted only after the run, in the batch runner's finally).
        gp = kwargs.get("glossary_path")
        seen["terms"] = load_glossary(gp)
        seen["prompt"] = glossary_prompt(seen["terms"])
        seen["glossary_text"] = Path(gp).read_text(encoding="utf-8") if gp else None
        seen.update(kwargs)
        return {
            "text": "moikka",
            "segments": [],
            "notes": {"title": "T"},
        }

    monkeypatch.setattr(pipeline_module, "transcribe_file", fake_transcribe)
    isolate_home(monkeypatch, tmp_path, tmp_path)
    result = runner.invoke(app, ["memo", "a.m4a", "--glossary", str(g)])
    assert result.exit_code == 0
    gp = seen["glossary_path"]
    # A temp file (not the explicit path itself) carried the corrections.
    assert gp is not None
    assert gp != str(g)
    # The temp file held only the correction pairs: load_glossary sees
    # nothing (=> lines are excluded from prompt terms) and the whisper
    # prompt is therefore None (the issue's test-surface note).
    assert seen["terms"] == []
    assert seen["prompt"] is None
    # Corrections survive: the pair the file defined is in the temp file.
    assert seen["glossary_text"] == "Blacksit => Flagship\n"


def test_meeting_explicit_glossary_passes_path_through(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """meeting --glossary: the file replaces both layers and is passed as-is."""
    import vemoizer.pipeline as pipeline_module

    seen: dict = {}
    g = tmp_path / "override.txt"
    g.write_text("Nordea\n@Jukka\nBlacksit => Flagship\n", encoding="utf-8")

    def fake_transcribe(path, **kwargs):
        seen.update(kwargs)
        return {
            "text": "moikka",
            "segments": [],
            "notes": {"title": "T"},
        }

    monkeypatch.setattr(pipeline_module, "transcribe_file", fake_transcribe)
    isolate_home(monkeypatch, tmp_path, tmp_path)
    result = runner.invoke(app, ["meeting", "a.m4a", "--glossary", str(g)])
    assert result.exit_code == 0
    assert seen["glossary_path"] == str(g)


def test_memo_explicit_glossary_without_pairs_yields_none_glossary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """memo --glossary with only prompt terms: glossary_path is None."""
    import vemoizer.pipeline as pipeline_module

    seen: dict = {}
    g = tmp_path / "terms-only.txt"
    g.write_text("Nordea\n@Jukka\n", encoding="utf-8")

    def fake_transcribe(path, **kwargs):
        seen.update(kwargs)
        return {"text": "moikka", "segments": [], "notes": {"title": "T"}}

    monkeypatch.setattr(pipeline_module, "transcribe_file", fake_transcribe)
    isolate_home(monkeypatch, tmp_path, tmp_path)
    result = runner.invoke(app, ["memo", "a.m4a", "--glossary", str(g)])
    assert result.exit_code == 0
    assert seen["glossary_path"] is None
