"""ConfigError fail-loud contract for the meeting preset (PR #88, issue #87).

A malformed layered config (an unknown key under ``[llm]`` in a project
``.vemoizer/config.toml``) must produce a CLEAN one-line ``error:`` naming
the offending key — never a raw ``Traceback`` — with exit 1, no output
files, the temp glossary cleaned up, and ``transcribe_file`` never called,
in ALL THREE meeting shapes:

- grouped (2 files + ``--yes``): run_batch -> _transcribe_one
- plain loop (2 files + ``--no-group``)
- single file

Pure stdlib: fakes patch ``pipeline.transcribe_file`` and the grouping
boundaries seam — no models, no network, no ffmpeg. Every test isolates
HOME and chdirs into its own tmp dir (``isolate_home``).
"""

from __future__ import annotations

from pathlib import Path

import pytest
from _cli_helpers import isolate_home
from typer.testing import CliRunner

import vemoizer.grouping as grouping
import vemoizer.pipeline as pipeline_module
from vemoizer.cli import app

runner = CliRunner()

#: A valid ``[llm]`` section plus an unknown key that makes the strict
#: layered search raise ``ConfigError`` naming the key (the same shape
#: tests/test_config_search.py pins).
_MALFORMED_CONFIG = (
    '[llm]\nbase_url = "https://x"\nmodel = "m"\n'
    'api_key_env = "K"\ntimeout_seconds = 30\nextra = "oops"\n'
)


def _write_bad_config(tmp_path: Path) -> None:
    cfg = tmp_path / ".vemoizer" / "config.toml"
    cfg.parent.mkdir(exist_ok=True)
    cfg.write_text(_MALFORMED_CONFIG, encoding="utf-8")


def _fake_continuation_seams(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch the grouping seams so 2 files become ONE multi-part group."""

    def fake_decode_boundaries(files, transcribe_fn=None, **kw):
        return ["ja tässä ollaan nyt siinä vaiheessa missä"], ["tässä jatketaan"]

    monkeypatch.setattr(grouping, "decode_boundaries", fake_decode_boundaries)
    monkeypatch.setattr(grouping, "probe_duration_seconds", lambda path: 30.0)
    monkeypatch.setattr(grouping, "concat_groups", lambda files, **kw: files[0])
    monkeypatch.setattr(grouping, "part_offsets", lambda files, **kw: [])
    import vemoizer.batch as batch

    monkeypatch.setattr(batch, "concat_groups", lambda files, **kw: files[0])
    monkeypatch.setattr(batch, "part_offsets", lambda files, **kw: [])


def _fake_transcribe_never_called(
    monkeypatch: pytest.MonkeyPatch, record: list[str]
) -> None:
    def fake_transcribe(path, **kwargs):
        record.append(Path(path).name)

    monkeypatch.setattr(pipeline_module, "transcribe_file", fake_transcribe)


def _assert_clean_config_failure(
    tmp_path: Path, result, *, transcribe_record: list[str]
) -> None:
    """The shared contract: clean error, no traceback, no output."""
    assert result.exit_code == 1
    assert "unknown key llm.extra" in result.stderr
    assert "error:" in result.stderr
    assert "Traceback" not in result.stderr
    assert result.exception is None or isinstance(result.exception, SystemExit)
    # No output files were written (only the input m4as + config remain).
    assert list(tmp_path.glob("*.md")) == []
    assert list(tmp_path.glob("*.json")) == []
    # The temp glossary file is gone (created and cleaned up, not leaked).
    assert list(tmp_path.glob("vemoizer-glossary-*")) == []
    # transcribe_file was never reached.
    assert transcribe_record == []


def test_grouped_meeting_malformed_config_is_clean(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """2 files + --yes (grouped): the malformed project config must not
    escape as a traceback — clean one-line error, exit 1."""
    _write_bad_config(tmp_path)
    (tmp_path / "a.m4a").touch()
    (tmp_path / "b.m4a").touch()
    record: list[str] = []
    _fake_transcribe_never_called(monkeypatch, record)
    _fake_continuation_seams(monkeypatch)
    isolate_home(monkeypatch, tmp_path, tmp_path)
    result = runner.invoke(app, ["meeting", "a.m4a", "b.m4a", "--yes"])
    _assert_clean_config_failure(tmp_path, result, transcribe_record=record)


def test_no_group_meeting_malformed_config_is_clean(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """2 files + --no-group (plain loop): same clean error contract."""
    _write_bad_config(tmp_path)
    (tmp_path / "a.m4a").touch()
    (tmp_path / "b.m4a").touch()
    record: list[str] = []
    _fake_transcribe_never_called(monkeypatch, record)
    isolate_home(monkeypatch, tmp_path, tmp_path)
    result = runner.invoke(app, ["meeting", "a.m4a", "b.m4a", "--no-group"])
    _assert_clean_config_failure(tmp_path, result, transcribe_record=record)


def test_single_file_meeting_malformed_config_is_clean(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Single file: the plain per-file loop must fail the same way."""
    _write_bad_config(tmp_path)
    (tmp_path / "a.m4a").touch()
    record: list[str] = []
    _fake_transcribe_never_called(monkeypatch, record)
    isolate_home(monkeypatch, tmp_path, tmp_path)
    result = runner.invoke(app, ["meeting", "a.m4a"])
    _assert_clean_config_failure(tmp_path, result, transcribe_record=record)
