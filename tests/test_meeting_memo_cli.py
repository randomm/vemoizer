"""CLI tests for the ``meeting`` and ``memo`` presets: flag forwarding,
profile/diarization defaults, speaker and title handling, output naming and
--quiet. Layered config and glossary tests live in sibling modules (issue #82).

Every test monkeypatches ``HOME`` and chdirs into a tmp dir (``isolate_home``).
"""

from __future__ import annotations

from _cli_helpers import isolate_home
from typer.testing import CliRunner

from vemoizer.cli import app
from vemoizer.output.naming import dated_basename

runner = CliRunner()


# -- flag forwarding and preset defaults ---------------------------------


def test_meeting_help_lists_flags() -> None:
    result = runner.invoke(app, ["meeting", "--help"])
    assert result.exit_code == 0
    for flag in (
        "--config",
        "--glossary",
        "--repair",
        "--no-repair",
        "--speakers",
        "--no-diarize",
        "--yes",
        "--no-group",
    ):
        assert flag in result.stdout
    assert "files" in result.stdout


def test_memo_help_lists_flags() -> None:
    result = runner.invoke(app, ["memo", "--help"])
    assert result.exit_code == 0
    for flag in ("--config", "--glossary", "--repair", "--no-repair"):
        assert flag in result.stdout
    assert "files" in result.stdout


def test_meeting_forwards_profile_meeting_and_diarize(tmp_path, monkeypatch) -> None:
    """meeting uses profile=meeting and diarize=True by default."""
    import vemoizer.model_cache as _mc

    _mc.clear_memo()  # isolate probe memo from prior tests

    import vemoizer.pipeline as pipeline_module

    seen: dict = {}

    def fake_transcribe(path, **kwargs):
        seen.update(kwargs)
        return {
            "text": "moikka maailma",
            "segments": [],
            "notes": {"title": "Team Sync"},
        }

    monkeypatch.setattr(pipeline_module, "transcribe_file", fake_transcribe)
    isolate_home(monkeypatch, tmp_path, tmp_path)
    result = runner.invoke(app, ["meeting", "a.m4a"])
    assert result.exit_code == 0
    assert seen["profile"] == "meeting"
    assert seen["diarize"] is True
    assert seen["repair"] is True
    assert seen["speakers"] == (2, 6)


def test_meeting_no_diarize_flag(tmp_path, monkeypatch) -> None:
    """--no-diarize disables diarization in the meeting preset."""
    import vemoizer.pipeline as pipeline_module

    seen: dict = {}

    def fake_transcribe(path, **kwargs):
        seen.update(kwargs)
        return {
            "text": "moikka",
            "segments": [],
            "notes": {"title": "Test"},
        }

    monkeypatch.setattr(pipeline_module, "transcribe_file", fake_transcribe)
    isolate_home(monkeypatch, tmp_path, tmp_path)
    result = runner.invoke(app, ["meeting", "a.m4a", "--no-diarize"])
    assert result.exit_code == 0
    assert seen["diarize"] is False


def test_meeting_speakers_override(tmp_path, monkeypatch) -> None:
    """--speakers overrides the default (2, 6) range."""
    import vemoizer.pipeline as pipeline_module

    seen: dict = {}

    def fake_transcribe(path, **kwargs):
        seen.update(kwargs)
        return {
            "text": "moikka",
            "segments": [],
            "notes": {"title": "Test"},
        }

    monkeypatch.setattr(pipeline_module, "transcribe_file", fake_transcribe)
    isolate_home(monkeypatch, tmp_path, tmp_path)
    result = runner.invoke(app, ["meeting", "a.m4a", "--speakers", "3-5"])
    assert result.exit_code == 0
    assert seen["speakers"] == (3, 5)


def test_meeting_writes_md_and_json_to_cwd(tmp_path, monkeypatch) -> None:
    """meeting writes .md and .json to the CWD with a dated title."""
    import vemoizer.pipeline as pipeline_module

    def fake_transcribe(path, **kwargs):
        return {
            "text": "moikka maailma",
            "segments": [],
            "notes": {"title": "Team Sync"},
        }

    monkeypatch.setattr(pipeline_module, "transcribe_file", fake_transcribe)
    isolate_home(monkeypatch, tmp_path, tmp_path)
    result = runner.invoke(app, ["meeting", "a.m4a"])
    assert result.exit_code == 0
    # The "wrote" lines reference the dated title base
    assert "wrote" in result.stdout
    # Both .md and .json files exist
    md_files = list(tmp_path.glob("*.md"))
    json_files = list(tmp_path.glob("*.json"))
    assert len(md_files) == 1, f"expected 1 .md, got {md_files}"
    assert len(json_files) == 1, f"expected 1 .json, got {json_files}"
    # The title appears in the filename
    assert "Team Sync" in md_files[0].name
    assert "Team Sync" in json_files[0].name


def test_meeting_title_fallback_to_first_stem(tmp_path, monkeypatch) -> None:
    """When notes is absent or title is empty, fall back to the first file's stem."""
    import vemoizer.pipeline as pipeline_module

    def fake_transcribe(path, **kwargs):
        return {
            "text": "moikka maailma",
            "segments": [],
            # No notes key — LLM not configured
        }

    monkeypatch.setattr(pipeline_module, "transcribe_file", fake_transcribe)
    isolate_home(monkeypatch, tmp_path, tmp_path)
    result = runner.invoke(app, ["meeting", "my-memo.m4a"])
    assert result.exit_code == 0
    md_files = list(tmp_path.glob("*.md"))
    assert len(md_files) == 1
    assert "my-memo" in md_files[0].name


def test_memo_forwards_profile_meeting_no_diarize(tmp_path, monkeypatch) -> None:
    """memo uses profile=meeting but diarize=False."""
    import vemoizer.pipeline as pipeline_module

    seen: dict = {}

    def fake_transcribe(path, **kwargs):
        seen.update(kwargs)
        return {
            "text": "moikka",
            "segments": [],
            "notes": {"title": "Quick Note"},
        }

    monkeypatch.setattr(pipeline_module, "transcribe_file", fake_transcribe)
    isolate_home(monkeypatch, tmp_path, tmp_path)
    result = runner.invoke(app, ["memo", "a.m4a"])
    assert result.exit_code == 0
    assert seen["profile"] == "meeting"
    assert seen["diarize"] is False
    assert seen["repair"] is True


def test_memo_writes_md_and_json_to_cwd(tmp_path, monkeypatch) -> None:
    """memo writes .md and .json to the CWD."""
    import vemoizer.pipeline as pipeline_module

    def fake_transcribe(path, **kwargs):
        return {
            "text": "moikka maailma",
            "segments": [],
            "notes": {"title": "Quick Note"},
        }

    monkeypatch.setattr(pipeline_module, "transcribe_file", fake_transcribe)
    isolate_home(monkeypatch, tmp_path, tmp_path)
    result = runner.invoke(app, ["memo", "a.m4a"])
    assert result.exit_code == 0
    md_files = list(tmp_path.glob("*.md"))
    json_files = list(tmp_path.glob("*.json"))
    assert len(md_files) == 1
    assert len(json_files) == 1


def test_meeting_glossary_forwarded(tmp_path, monkeypatch) -> None:
    """--glossary is forwarded to transcribe_file."""
    import vemoizer.pipeline as pipeline_module

    seen: dict = {}

    def fake_transcribe(path, **kwargs):
        seen.update(kwargs)
        return {
            "text": "moikka",
            "segments": [],
            "notes": {"title": "T"},
        }

    monkeypatch.setattr(pipeline_module, "transcribe_file", fake_transcribe)
    isolate_home(monkeypatch, tmp_path, tmp_path)
    result = runner.invoke(app, ["meeting", "a.m4a", "--glossary", "/tmp/g.txt"])
    assert result.exit_code == 0
    assert seen["glossary_path"] == "/tmp/g.txt"


def test_meeting_config_forwarded(tmp_path, monkeypatch) -> None:
    """--config is forwarded to transcribe_file."""
    import vemoizer.pipeline as pipeline_module

    seen: dict = {}

    def fake_transcribe(path, **kwargs):
        seen.update(kwargs)
        return {
            "text": "moikka",
            "segments": [],
            "notes": {"title": "T"},
        }

    monkeypatch.setattr(pipeline_module, "transcribe_file", fake_transcribe)
    isolate_home(monkeypatch, tmp_path, tmp_path)
    result = runner.invoke(app, ["meeting", "a.m4a", "--config", "/tmp/c.toml"])
    assert result.exit_code == 0
    assert seen["config_path"] == "/tmp/c.toml"


def test_meeting_empty_transcript_exits_nonzero(tmp_path, monkeypatch) -> None:
    """Empty transcript (no text, no segments) is a failure."""
    import vemoizer.pipeline as pipeline_module

    def fake_transcribe(path, **kwargs):
        return {"text": "", "segments": []}

    monkeypatch.setattr(pipeline_module, "transcribe_file", fake_transcribe)
    isolate_home(monkeypatch, tmp_path, tmp_path)
    result = runner.invoke(app, ["meeting", "a.m4a"])
    assert result.exit_code == 1
    assert "no transcript" in result.stderr


def test_meeting_diarize_without_labels_exits_nonzero(tmp_path, monkeypatch) -> None:
    """meeting with diarize=True but no speaker labels exits non-zero."""
    import vemoizer.pipeline as pipeline_module

    def fake_transcribe(path, **kwargs):
        return {
            "text": "moikka",
            "segments": [{"start": 0.0, "end": 1.0, "text": "moikka"}],
            "notes": {"title": "T"},
        }

    monkeypatch.setattr(pipeline_module, "transcribe_file", fake_transcribe)
    isolate_home(monkeypatch, tmp_path, tmp_path)
    result = runner.invoke(app, ["meeting", "a.m4a"])
    assert result.exit_code == 1
    assert "no speaker labels" in result.stderr


def test_meeting_collision_suffix(tmp_path, monkeypatch) -> None:
    """When the output file already exists, a collision suffix is added."""
    import vemoizer.pipeline as pipeline_module

    def fake_transcribe(path, **kwargs):
        return {
            "text": "moikka",
            "segments": [],
            "notes": {"title": "Team Sync"},
        }

    monkeypatch.setattr(pipeline_module, "transcribe_file", fake_transcribe)
    isolate_home(monkeypatch, tmp_path, tmp_path)
    # Pre-create the .md file that the first run would write
    base = dated_basename("Team Sync", fallback_stem="a")
    (tmp_path / f"{base}.md").write_text("existing")
    (tmp_path / f"{base}.json").write_text("{}")
    result = runner.invoke(app, ["meeting", "a.m4a"])
    assert result.exit_code == 0
    # A " (2)" suffix was added
    md_files = list(tmp_path.glob("*.md"))
    assert len(md_files) == 2, f"expected 2 .md files, got {md_files}"
    assert any(" (2)" in f.name for f in md_files)


def test_memo_no_diarize_even_with_segments(tmp_path, monkeypatch) -> None:
    """memo never diarizes, even when segments have no speaker labels."""
    import vemoizer.pipeline as pipeline_module

    def fake_transcribe(path, **kwargs):
        assert kwargs["diarize"] is False
        return {
            "text": "moikka",
            "segments": [{"start": 0.0, "end": 1.0, "text": "moikka"}],
            "notes": {"title": "T"},
        }

    monkeypatch.setattr(pipeline_module, "transcribe_file", fake_transcribe)
    isolate_home(monkeypatch, tmp_path, tmp_path)
    result = runner.invoke(app, ["memo", "a.m4a"])
    assert result.exit_code == 0


def test_meeting_quiet_suppresses_wrote_lines(tmp_path, monkeypatch) -> None:
    """--quiet suppresses the `wrote <path>` summary lines (issue #82)."""
    import vemoizer.pipeline as pipeline_module

    def fake_transcribe(path, **kwargs):
        return {
            "text": "moikka",
            "segments": [],
            "notes": {"title": "T"},
        }

    monkeypatch.setattr(pipeline_module, "transcribe_file", fake_transcribe)
    isolate_home(monkeypatch, tmp_path, tmp_path)
    result = runner.invoke(app, ["meeting", "a.m4a", "--quiet"])
    assert result.exit_code == 0
    assert "wrote" not in result.stdout


def test_memo_quiet_suppresses_wrote_lines(tmp_path, monkeypatch) -> None:
    """memo --quiet suppresses the `wrote <path>` summary lines."""
    import vemoizer.pipeline as pipeline_module

    def fake_transcribe(path, **kwargs):
        return {
            "text": "moikka",
            "segments": [],
            "notes": {"title": "T"},
        }

    monkeypatch.setattr(pipeline_module, "transcribe_file", fake_transcribe)
    isolate_home(monkeypatch, tmp_path, tmp_path)
    result = runner.invoke(app, ["memo", "a.m4a", "--quiet"])
    assert result.exit_code == 0
    assert "wrote" not in result.stdout
