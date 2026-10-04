"""CLI validation tests for the opt-in ``--preprocess loudnorm`` flag (issue #135).

Covers:

- Exit 2 on an unknown ``--preprocess`` value for ``meeting``, ``memo``,
  and ``transcribe`` (the validation contract mirrors ``--language``).
- Case-insensitive acceptance (``LOUDNORM``, ``LoudNorm`` are valid).
- The flag reaches ``transcribe_file`` (via fakes) for all three commands:
  ``meeting`` and ``memo`` through ``run_preset`` → ``resolve_options`` →
  ``RunOptions.preprocess`` → ``transcribe_file``; ``transcribe`` through
  ``transcribe_batch`` → ``transcribe_file``.
- Default (no flag) means ``preprocess=None`` in the ``transcribe_file``
  kwargs — the no-flag decode path is unchanged.
- Sidecar ``options.preprocess`` is present-only: the key appears in the
  sidecar JSON only when the flag is set.
"""

from __future__ import annotations

import json
from pathlib import Path

from _cli_helpers import isolate_home
from typer.testing import CliRunner

from vemoizer.cli import app

runner = CliRunner()


# ---------------------------------------------------------------------------
# Help output: --preprocess flag is listed
# ---------------------------------------------------------------------------


def test_transcribe_help_lists_preprocess() -> None:
    result = runner.invoke(app, ["transcribe", "--help"])
    assert result.exit_code == 0
    assert "--preprocess" in result.stdout


def test_meeting_help_lists_preprocess() -> None:
    result = runner.invoke(app, ["meeting", "--help"])
    assert result.exit_code == 0
    assert "--preprocess" in result.stdout


def test_memo_help_lists_preprocess() -> None:
    result = runner.invoke(app, ["memo", "--help"])
    assert result.exit_code == 0
    assert "--preprocess" in result.stdout


# ---------------------------------------------------------------------------
# Validation: unknown value → exit 2, one clean line, no traceback
# ---------------------------------------------------------------------------


def test_transcribe_unknown_preprocess_exits_2(tmp_path: Path, monkeypatch) -> None:
    isolate_home(monkeypatch, tmp_path, tmp_path)
    result = runner.invoke(app, ["transcribe", "a.m4a", "--preprocess", "denoise"])
    assert result.exit_code == 2
    assert "unknown preprocess" in result.stderr
    assert "denoise" in result.stderr
    assert "Traceback" not in result.stderr


def test_meeting_unknown_preprocess_exits_2(tmp_path: Path, monkeypatch) -> None:
    isolate_home(monkeypatch, tmp_path, tmp_path)
    result = runner.invoke(app, ["meeting", "a.m4a", "--preprocess", "normalize"])
    assert result.exit_code == 2
    assert "unknown preprocess" in result.stderr
    assert "normalize" in result.stderr
    assert "Traceback" not in result.stderr


def test_memo_unknown_preprocess_exits_2(tmp_path: Path, monkeypatch) -> None:
    isolate_home(monkeypatch, tmp_path, tmp_path)
    result = runner.invoke(app, ["memo", "a.m4a", "--preprocess", "denoise"])
    assert result.exit_code == 2
    assert "unknown preprocess" in result.stderr
    assert "denoise" in result.stderr
    assert "Traceback" not in result.stderr


def test_transcribe_empty_preprocess_string_exits_2(
    tmp_path: Path, monkeypatch
) -> None:
    """An empty-string value (e.g. ``--preprocess ''``) is not None and
    not 'loudnorm', so it exits 2."""
    isolate_home(monkeypatch, tmp_path, tmp_path)
    result = runner.invoke(app, ["transcribe", "a.m4a", "--preprocess", ""])
    assert result.exit_code == 2
    assert "unknown preprocess" in result.stderr


# ---------------------------------------------------------------------------
# Case-insensitive: LOUDNORM, LoudNorm are accepted
# ---------------------------------------------------------------------------


def test_transcribe_preprocess_case_insensitive(tmp_path: Path, monkeypatch) -> None:
    import vemoizer.pipeline as pipeline_module

    seen: dict = {}

    def fake_transcribe(path, **kwargs):
        seen.update(kwargs)
        return {"text": "moikka", "segments": []}

    monkeypatch.setattr(pipeline_module, "transcribe_file", fake_transcribe)
    isolate_home(monkeypatch, tmp_path, tmp_path)
    result = runner.invoke(app, ["transcribe", "a.m4a", "--preprocess", "LOUDNORM"])
    assert result.exit_code == 0
    assert seen["preprocess"] == "loudnorm"


def test_meeting_preprocess_case_insensitive(tmp_path: Path, monkeypatch) -> None:
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
    result = runner.invoke(app, ["meeting", "a.m4a", "--preprocess", "LoudNorm"])
    assert result.exit_code == 0
    assert seen["preprocess"] == "loudnorm"


# ---------------------------------------------------------------------------
# Flag reaches transcribe_file for all three commands
# ---------------------------------------------------------------------------


def test_transcribe_preprocess_reaches_transcribe_file(
    tmp_path: Path, monkeypatch
) -> None:
    """The --preprocess loudnorm flag on transcribe reaches transcribe_file
    via transcribe_batch."""
    import vemoizer.pipeline as pipeline_module

    seen: dict = {}

    def fake_transcribe(path, **kwargs):
        seen.update(kwargs)
        return {"text": "moikka maailma", "segments": []}

    monkeypatch.setattr(pipeline_module, "transcribe_file", fake_transcribe)
    isolate_home(monkeypatch, tmp_path, tmp_path)
    result = runner.invoke(app, ["transcribe", "a.m4a", "--preprocess", "loudnorm"])
    assert result.exit_code == 0
    assert seen["preprocess"] == "loudnorm"


def test_meeting_preprocess_reaches_transcribe_file(
    tmp_path: Path, monkeypatch
) -> None:
    """The --preprocess loudnorm flag on meeting reaches transcribe_file
    via run_preset → resolve_options → RunOptions → _transcribe_preset_file."""
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
    result = runner.invoke(app, ["meeting", "a.m4a", "--preprocess", "loudnorm"])
    assert result.exit_code == 0
    assert seen["preprocess"] == "loudnorm"


def test_memo_preprocess_reaches_transcribe_file(tmp_path: Path, monkeypatch) -> None:
    """The --preprocess loudnorm flag on memo reaches transcribe_file
    via run_preset → resolve_options → RunOptions → _transcribe_preset_file."""
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
    result = runner.invoke(app, ["memo", "a.m4a", "--preprocess", "loudnorm"])
    assert result.exit_code == 0
    assert seen["preprocess"] == "loudnorm"


# ---------------------------------------------------------------------------
# Default (no flag): preprocess=None
# ---------------------------------------------------------------------------


def test_transcribe_default_preprocess_is_none(tmp_path: Path, monkeypatch) -> None:
    import vemoizer.pipeline as pipeline_module

    seen: dict = {}

    def fake_transcribe(path, **kwargs):
        seen.update(kwargs)
        return {"text": "moikka", "segments": []}

    monkeypatch.setattr(pipeline_module, "transcribe_file", fake_transcribe)
    isolate_home(monkeypatch, tmp_path, tmp_path)
    result = runner.invoke(app, ["transcribe", "a.m4a"])
    assert result.exit_code == 0
    assert seen["preprocess"] is None


def test_meeting_default_preprocess_is_none(tmp_path: Path, monkeypatch) -> None:
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
    result = runner.invoke(app, ["meeting", "a.m4a"])
    assert result.exit_code == 0
    assert seen["preprocess"] is None


def test_memo_default_preprocess_is_none(tmp_path: Path, monkeypatch) -> None:
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
    result = runner.invoke(app, ["memo", "a.m4a"])
    assert result.exit_code == 0
    assert seen["preprocess"] is None


# ---------------------------------------------------------------------------
# Sidecar options.preprocess: present-only
# ---------------------------------------------------------------------------


def test_meeting_sidecar_preprocess_present_when_set(
    tmp_path: Path, monkeypatch
) -> None:
    """When --preprocess loudnorm is set, the sidecar JSON has
    options.preprocess == 'loudnorm'."""
    import vemoizer.pipeline as pipeline_module

    def fake_transcribe(path, **kwargs):
        return {
            "text": "moikka",
            "segments": [],
            "notes": {"title": "Test"},
        }

    monkeypatch.setattr(pipeline_module, "transcribe_file", fake_transcribe)
    isolate_home(monkeypatch, tmp_path, tmp_path)
    result = runner.invoke(app, ["meeting", "a.m4a", "--preprocess", "loudnorm"])
    assert result.exit_code == 0
    json_files = list(tmp_path.glob("*.json"))
    assert len(json_files) == 1
    sidecar = json.loads(json_files[0].read_text(encoding="utf-8"))
    assert sidecar["options"]["preprocess"] == "loudnorm"


def test_meeting_sidecar_preprocess_absent_when_not_set(
    tmp_path: Path, monkeypatch
) -> None:
    """When no --preprocess flag is given, the sidecar JSON's options dict
    does NOT contain a 'preprocess' key (present-only rule)."""
    import vemoizer.pipeline as pipeline_module

    def fake_transcribe(path, **kwargs):
        return {
            "text": "moikka",
            "segments": [],
            "notes": {"title": "Test"},
        }

    monkeypatch.setattr(pipeline_module, "transcribe_file", fake_transcribe)
    isolate_home(monkeypatch, tmp_path, tmp_path)
    result = runner.invoke(app, ["meeting", "a.m4a"])
    assert result.exit_code == 0
    json_files = list(tmp_path.glob("*.json"))
    assert len(json_files) == 1
    sidecar = json.loads(json_files[0].read_text(encoding="utf-8"))
    assert "preprocess" not in sidecar["options"]


def test_memo_sidecar_preprocess_present_when_set(tmp_path: Path, monkeypatch) -> None:
    """When --preprocess loudnorm is set on memo, the sidecar has it."""
    import vemoizer.pipeline as pipeline_module

    def fake_transcribe(path, **kwargs):
        return {
            "text": "moikka",
            "segments": [],
            "notes": {"title": "Test"},
        }

    monkeypatch.setattr(pipeline_module, "transcribe_file", fake_transcribe)
    isolate_home(monkeypatch, tmp_path, tmp_path)
    result = runner.invoke(app, ["memo", "a.m4a", "--preprocess", "loudnorm"])
    assert result.exit_code == 0
    json_files = list(tmp_path.glob("*.json"))
    assert len(json_files) == 1
    sidecar = json.loads(json_files[0].read_text(encoding="utf-8"))
    assert sidecar["options"]["preprocess"] == "loudnorm"


def test_memo_sidecar_preprocess_absent_when_not_set(
    tmp_path: Path, monkeypatch
) -> None:
    """When no --preprocess flag is given on memo, the sidecar has no
    preprocess key."""
    import vemoizer.pipeline as pipeline_module

    def fake_transcribe(path, **kwargs):
        return {
            "text": "moikka",
            "segments": [],
            "notes": {"title": "Test"},
        }

    monkeypatch.setattr(pipeline_module, "transcribe_file", fake_transcribe)
    isolate_home(monkeypatch, tmp_path, tmp_path)
    result = runner.invoke(app, ["memo", "a.m4a"])
    assert result.exit_code == 0
    json_files = list(tmp_path.glob("*.json"))
    assert len(json_files) == 1
    sidecar = json.loads(json_files[0].read_text(encoding="utf-8"))
    assert "preprocess" not in sidecar["options"]
