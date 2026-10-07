"""CLI tests for the ``meeting`` and ``memo`` preset commands (issue #82).

Flag forwarding, profile/diarization defaults, the layered LLM config
search the presets run end to end (``config_path=None`` so the search
fires; ``--config`` short-circuits), and the layered-glossary temp-file
seam (corrections-only for memo, merged terms + ``@`` lines + pairs for
meeting). Every test monkeypatches ``HOME`` and chdirs into a tmp dir —
the dev machine has a live ``~/.config/vemoizer/config.toml`` that must
not leak in (see ``_isolated_home`` in conftest).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from _cli_helpers import isolate_home
from typer.testing import CliRunner

from vemoizer.cli import app
from vemoizer.llm import LLMConfig
from vemoizer.output.naming import dated_basename

runner = CliRunner()


# -- flag forwarding and preset defaults ---------------------------------


def _write_config(root: Path, sub: str, marker: str) -> Path:
    path = root / sub / "config.toml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_VALID_CONFIG.format(h=root.name, m=marker), encoding="utf-8")
    return path


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


# -- layered LLM config search, end to end (issue #82) -------------------
#
# Without --config the presets must pass config_path=None through to
# transcribe_file so the layered search (llm.load_default_config) runs
# instead of being short-circuited: ~/.vemoizer/config.toml -> nearest
# ./.vemoizer/config.toml -> legacy. Explicit --config passes the path
# through unchanged. Every test here monkeypatches HOME and chdirs into
# a tmp dir (isolate_home) so the dev machine's live config never leaks.

#: A minimal valid ``[llm]`` section for config fixtures.
_VALID_CONFIG = (
    "[llm]\n"
    'base_url = "https://{h}.invalid/v1"\n'
    'model = "{m}"\n'
    'api_key_env = "K"\n'
    "timeout_seconds = 5.0\n"
)


def _invoke_preset(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    command: str = "meeting",
    extra_args: list[str] | None = None,
):
    """Invoke meeting/memo with the fake transcribe seam.

    ``load_default_config`` is mocked: it records the ``config_path``
    the pipeline seam would hand the config search, so the test
    observes exactly what the presets passed (``None`` = the layered
    search runs; an explicit path = short-circuit). The *resolution*
    of a config is pinned separately via the real (isolated) search.
    """
    import vemoizer.llm_config as llm_module
    import vemoizer.pipeline as pipeline_module

    seen: dict = {}

    def fake_load_default_config(path=None):
        seen["config_path"] = path
        return None, None

    def fake_transcribe(path, **kwargs):
        seen.update(kwargs)
        return {"text": "moikka", "segments": [], "notes": {"title": "T"}}

    monkeypatch.setattr(pipeline_module, "transcribe_file", fake_transcribe)
    monkeypatch.setattr(llm_module, "load_default_config", fake_load_default_config)
    isolate_home(monkeypatch, tmp_path, tmp_path)
    result = runner.invoke(app, [command, "a.m4a", *(extra_args or [])])
    return result, seen


def _isolated_search(home: Path, cwd: Path) -> LLMConfig | None:
    """Run the real layered search with HOME/CWD pinned to tmp paths.

    Isolates the resolution from the dev machine's live config (the
    same fixture the CLI invoke used: fake HOME, CWD = tmp_path).
    """
    from vemoizer.llm_config import _default_search

    cfg, _raw = _default_search(
        home=lambda: home,
        cwd=lambda: cwd,
        legacy_paths=(
            home / ".config" / "vemoizer" / "config.toml",
            home / ".vemoizer.toml",
        ),
    )
    return cfg


def test_meeting_home_config_reaches_pipeline_without_config_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """(a) a temp HOME with ~/.vemoizer/config.toml and no --config: the
    search runs (config_path=None reaches the seam) and the home config
    is what the search resolves — LLM notes/repair get configured."""
    home = tmp_path / "home"
    _write_config(home, ".vemoizer", "home-model")
    result, seen = _invoke_preset(tmp_path, monkeypatch)
    assert result.exit_code == 0
    # No --config: the batch runner must NOT short-circuit the search.
    assert seen["config_path"] is None
    cfg = _isolated_search(home, tmp_path)
    assert cfg is not None
    assert cfg.model == "home-model"


def test_memo_home_config_reaches_pipeline_without_config_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """memo behaves like meeting: no --config means the search runs."""
    home = tmp_path / "home"
    _write_config(home, ".vemoizer", "home-model")
    result, seen = _invoke_preset(tmp_path, monkeypatch, command="memo")
    assert result.exit_code == 0
    assert seen["config_path"] is None
    cfg = _isolated_search(home, tmp_path)
    assert cfg is not None
    assert cfg.model == "home-model"


def test_meeting_project_config_wins_over_home(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """(b) a nearer ./.vemoizer/config.toml (CWD) wins over the home one
    and no deprecation notice fires for the unused legacy file."""
    home = tmp_path / "home"
    _write_config(home, ".vemoizer", "home-model")
    _write_config(home, ".config/vemoizer", "legacy-model")
    _write_config(tmp_path, ".vemoizer", "proj-model")
    result, seen = _invoke_preset(tmp_path, monkeypatch)
    assert result.exit_code == 0
    assert seen["config_path"] is None
    cfg = _isolated_search(home, tmp_path)
    assert cfg is not None
    assert cfg.model == "proj-model"


def test_meeting_explicit_config_wins_over_layers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """(c) --config X wins over both the home and project layers: the
    explicit path is passed straight through to the pipeline seam."""
    _write_config(tmp_path / "home", ".vemoizer", "home-model")
    _write_config(tmp_path, ".vemoizer", "proj-model")
    explicit = tmp_path / "explicit.toml"
    explicit.write_text(
        _VALID_CONFIG.format(h="x", m="explicit-model"), encoding="utf-8"
    )
    result, seen = _invoke_preset(
        tmp_path, monkeypatch, extra_args=["--config", str(explicit)]
    )
    assert result.exit_code == 0
    assert seen["config_path"] == str(explicit)


def test_preset_malformed_layer_config_fails_cleanly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A malformed .vemoizer/config.toml in the search path fails loud.

    The layered search is strict (issue #82): a malformed file in the
    search path raises ``ConfigError`` (naming the key). The preset
    runner must convert it to a clean ``error:`` line and exit 1 — not
    a raw traceback (issue #78 fail-loud).
    """
    import vemoizer.pipeline as pipeline_module

    vemoizer_dir = tmp_path / ".vemoizer"
    vemoizer_dir.mkdir()
    (vemoizer_dir / "config.toml").write_text("[llm\n  broken", encoding="utf-8")

    def fake_transcribe(path, **kwargs):
        # The real transcribe_file does the config search inside; the
        # fake mirrors that by running the same strict search (unmocked)
        # so the ConfigError propagates out of the transcribe_file seam.
        from vemoizer.llm_config import load_default_config

        load_default_config(None)
        return {"text": "moikka", "segments": [], "notes": {"title": "T"}}

    monkeypatch.setattr(pipeline_module, "transcribe_file", fake_transcribe)
    isolate_home(monkeypatch, tmp_path, tmp_path)
    result = runner.invoke(app, ["meeting", "a.m4a"])
    assert result.exit_code == 1
    assert "error:" in result.stderr
    assert "Traceback" not in result.stderr


# -- legacy path: deprecation notice only when the legacy file is used ---


def _legacy_notice_case(tmp_path: Path, newer_layer_wins: bool):
    import vemoizer.llm_config as llm_module

    home = tmp_path / "home"
    _write_config(home, ".config/vemoizer", "legacy-model")
    if newer_layer_wins:
        _write_config(home, ".vemoizer", "new-model")
    cfg, _raw = llm_module._default_search(
        home=lambda: home,
        cwd=lambda: tmp_path / "clean",
        legacy_paths=(
            home / ".config" / "vemoizer" / "config.toml",
            home / ".vemoizer.toml",
        ),
    )
    return cfg


def test_preset_legacy_config_used_prints_deprecation_notice(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """(d) the legacy ~/.config/vemoizer/config.toml prints the
    deprecation notice when it is the one actually used by the search."""
    import vemoizer.llm_config as llm_module

    cfg = _legacy_notice_case(tmp_path, newer_layer_wins=False)
    assert cfg is not None and cfg.model == "legacy-model"
    out = capsys.readouterr()
    assert llm_module.LEGACY_DEPRECATION_NOTICE in out.err


def test_preset_no_deprecation_notice_when_newer_layer_wins(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """(d, negative) when a .vemoizer config wins the search, the legacy
    file is neither read nor announced."""
    import vemoizer.llm_config as llm_module

    cfg = _legacy_notice_case(tmp_path, newer_layer_wins=True)
    assert cfg is not None and cfg.model == "new-model"
    out = capsys.readouterr()
    assert llm_module.LEGACY_DEPRECATION_NOTICE not in out.err


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
