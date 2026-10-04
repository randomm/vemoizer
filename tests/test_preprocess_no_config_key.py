"""No config key for ``--preprocess`` (issue #135 gap disposition).

The issue decided the option is **flag-only**: there is deliberately NO
config-file key. This pins that decision against config drift:

- A ``[meeting] preprocess = "loudnorm"`` key in a project config is
  IGNORED with a stderr warning naming the key (``[meeting]`` is a
  warned-unknown-key section) — and it does NOT enable the filter
  (``ingest_audio`` still receives ``preprocess=None``).
- A TOP-LEVEL ``preprocess = "loudnorm"`` key is rejected by the strict
  top-level key validation: a clean one-line ``ConfigError`` naming the
  key, exit 1, never a traceback, and the filter is never enabled.
- ``RunOptions.preprocess`` is threaded from the CLI override ONLY
  (``resolve_options`` reads it from ``cli_overrides`` alone).

The test exercises the real config loader and ``resolve_options``
(the heavy stages — ``transcribe_file`` — are faked, the same as the
existing preprocess CLI tests). No models, no network, no ffmpeg.
"""

from __future__ import annotations

from pathlib import Path

from _cli_helpers import isolate_home
from typer.testing import CliRunner

import vemoizer.pipeline as pipeline_module
from vemoizer.cli import app
from vemoizer.llm import ConfigError
from vemoizer.llm_config import _strict_load_raw
from vemoizer.presets import resolve_options

runner = CliRunner()

#: A valid ``[meeting]`` section plus the (deliberately unsupported)
#: ``preprocess`` key under it — ``[meeting]`` is a warned-unknown-key
#: section, so the loader prints a warning instead of raising.
_MEETING_PREPROCESS_CONFIG = (
    '[llm]\nbase_url = "https://x"\nmodel = "m"\n'
    'api_key_env = "K"\ntimeout_seconds = 30\n'
    "[meeting]\n"
    'language = "fi"\n'
    'preprocess = "loudnorm"\n'
)

#: A top-level ``preprocess`` key — rejected by the strict top-level
#: key validation (a clean ``ConfigError``, not a traceback).
_TOPLEVEL_PREPROCESS_CONFIG = (
    '[llm]\nbase_url = "https://x"\nmodel = "m"\n'
    'api_key_env = "K"\ntimeout_seconds = 30\n'
    'preprocess = "loudnorm"\n'
)


def _write_config(tmp_path: Path, text: str) -> None:
    cfg = tmp_path / ".vemoizer" / "config.toml"
    cfg.parent.mkdir(exist_ok=True)
    cfg.write_text(text, encoding="utf-8")


def _fake_transcribe(records: dict, monkeypatch) -> None:
    """Patch ``pipeline.transcribe_file``; record its kwargs."""

    def fake_transcribe(path, **kwargs):
        records.update(kwargs)
        return {
            "text": "moikka maailma",
            "segments": [],
            "notes": {"title": Path(path).stem.upper()},
        }

    monkeypatch.setattr(pipeline_module, "transcribe_file", fake_transcribe)


def test_meeting_section_preprocess_is_ignored_with_warning(
    tmp_path: Path, monkeypatch
) -> None:
    """(a) ``[meeting] preprocess`` in the config: the loader WARNS
    (naming the key) and ignores it; the run's option stays ``None`` —
    ``ingest_audio`` (via ``transcribe_file``) receives no preprocess.
    The config is loadable (``[meeting]`` keys are warned, not fatal)."""
    _write_config(tmp_path, _MEETING_PREPROCESS_CONFIG)
    (tmp_path / "a.m4a").touch()
    seen: dict = {}
    _fake_transcribe(seen, monkeypatch)
    isolate_home(monkeypatch, tmp_path, tmp_path)
    result = runner.invoke(app, ["meeting", "a.m4a"])
    assert result.exit_code == 0, result.stderr
    # The warning names the key (stderr goes to the CLI runner's stderr).
    assert "unknown key meeting.preprocess" in result.stderr
    assert "ignored" in result.stderr
    # The key did NOT enable the filter: transcribe_file (and hence
    # ingest_audio) received preprocess=None.
    assert seen["preprocess"] is None
    # The config loader accepts the file (warned section) — it is a
    # config that parses, not one that errors.
    cfg = tmp_path / ".vemoizer" / "config.toml"
    config, _raw = _strict_load_raw(cfg)
    assert config is not None


def test_toplevel_preprocess_key_is_config_error(tmp_path: Path, monkeypatch) -> None:
    """(b) A TOP-LEVEL ``preprocess`` key is rejected by the strict
    top-level key validation: a clean one-line ``ConfigError`` naming
    the key, exit 1, no traceback — and the filter is never enabled."""
    _write_config(tmp_path, _TOPLEVEL_PREPROCESS_CONFIG)
    (tmp_path / "a.m4a").touch()
    seen: dict = {}
    _fake_transcribe(seen, monkeypatch)
    isolate_home(monkeypatch, tmp_path, tmp_path)

    # The loader itself raises the documented ConfigError naming the key.
    cfg = tmp_path / ".vemoizer" / "config.toml"
    try:
        _strict_load_raw(cfg)
    except ConfigError as e:
        assert "preprocess" in str(e)
    else:
        raise AssertionError("top-level preprocess key must raise ConfigError")

    # The meeting command turns it into a clean exit 1 (no traceback),
    # and the filter is never enabled.
    result = runner.invoke(app, ["meeting", "a.m4a"])
    assert result.exit_code == 1
    assert "preprocess" in result.stderr
    assert "error:" in result.stderr
    assert "Traceback" not in result.stderr
    assert seen == {}  # transcribe_file never ran


def test_run_options_preprocess_comes_from_cli_only(
    tmp_path: Path, monkeypatch
) -> None:
    """(c) ``RunOptions.preprocess`` is threaded from the CLI override
    ONLY. A config with a ``[meeting] preprocess`` key does not feed it:
    with no flag the option is ``None``; with the flag it is
    ``"loudnorm"`` — regardless of what the config says."""
    _write_config(tmp_path, _MEETING_PREPROCESS_CONFIG)
    (tmp_path / "a.m4a").touch()

    # No flag: the config's (ignored) key does not set the option.
    opts = resolve_options(
        "meeting",
        layers=None,
        cli_overrides={"config": None, "preprocess": None},
    )
    assert opts.preprocess is None

    # Flag set: the CLI override is the sole source.
    opts_flag = resolve_options(
        "meeting",
        layers=None,
        cli_overrides={"config": None, "preprocess": "loudnorm"},
    )
    assert opts_flag.preprocess == "loudnorm"


def test_meeting_preprocess_no_flag_with_meeting_config(
    tmp_path: Path, monkeypatch
) -> None:
    """Integration: a project config with the (ignored) ``[meeting]
    preprocess`` key does NOT enable the filter — ``transcribe_file``
    receives ``preprocess=None`` for a no-flag ``meeting`` run."""
    _write_config(tmp_path, _MEETING_PREPROCESS_CONFIG)
    (tmp_path / "a.m4a").touch()
    seen: dict = {}
    _fake_transcribe(seen, monkeypatch)
    isolate_home(monkeypatch, tmp_path, tmp_path)
    result = runner.invoke(app, ["meeting", "a.m4a"])
    assert result.exit_code == 0, result.stderr
    assert seen["preprocess"] is None
