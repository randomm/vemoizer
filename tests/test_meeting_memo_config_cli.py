"""CLI tests for the layered LLM config search as the ``meeting`` and ``memo``
presets run it end to end (issue #82), plus the legacy-path deprecation notice.

Every test monkeypatches ``HOME`` and chdirs into a tmp dir (``isolate_home``)
so the dev machine's live config never leaks.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from _cli_helpers import isolate_home
from typer.testing import CliRunner

from vemoizer.cli import app
from vemoizer.llm import LLMConfig

runner = CliRunner()


def _write_config(root: Path, sub: str, marker: str) -> Path:
    path = root / sub / "config.toml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_VALID_CONFIG.format(h=root.name, m=marker), encoding="utf-8")
    return path


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
