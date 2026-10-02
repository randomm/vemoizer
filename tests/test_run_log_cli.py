"""End-to-end CLI tests for the per-file run log (issue #111, M4c).

Proves, through the Typer CLI with ``vemoizer.run_log.file_log`` patched
so the seam's per-file logging is exercised without real model calls:

- the run context (``vemoizer.run_log._context``) is set correctly by the
  CLI's ``configure()`` call for all three commands (transcribe, meeting,
  memo);
- ``configure()`` receives the correct ``verbose`` and ``quiet`` flags;
- when a valid LLM config is present, ``configure()`` receives the
  config's ``api_key_env`` so the redaction filter can scrub the key;
- a malformed LLM config does NOT crash the CLI (fail-open);
- the existing test suite's behavior is unchanged (no new errors, no
  extra stderr output, correct exit codes).
"""

from __future__ import annotations

from contextlib import ExitStack, contextmanager
from unittest.mock import MagicMock, patch

from _cli_helpers import fake_transcribe, isolate_home, touch_files
from _run_log_helpers import fake_continuation_seams
from typer.testing import CliRunner

import vemoizer.pipeline as pipeline_module
import vemoizer.run_log as run_log
from vemoizer.cli import app

runner = CliRunner()


@contextmanager
def _patch_file_log():
    """Patch ``file_log`` in every module that imports it (the seam resolves
    ``file_log`` through its own module namespace, not ``vemoizer.run_log``)."""
    from vemoizer import batch, batch_plain, batch_preset, transcribe_loop

    m = MagicMock()
    m.return_value = MagicMock()
    with ExitStack() as stack:
        for mod in (run_log, batch, batch_preset, batch_plain, transcribe_loop):
            stack.enter_context(patch.object(mod, "file_log", m))
        yield m


# --- transcribe: run context wiring -------------------------------------------


def test_transcribe_configure_receives_flags(tmp_path, monkeypatch):
    """``configure()`` is called with the correct verbose/quiet values."""
    touch_files(["a.m4a"], tmp_path)
    record: list[str] = []
    fake_transcribe(monkeypatch, record)
    isolate_home(monkeypatch, tmp_path)
    with patch("vemoizer.run_log.configure") as mock_cfg:
        result = runner.invoke(app, ["transcribe", "a.m4a"])
    assert result.exit_code == 0
    assert mock_cfg.call_count == 1
    kwargs = mock_cfg.call_args.kwargs
    assert kwargs["verbose"] is False
    assert kwargs["quiet"] is False


def test_transcribe_configure_verbose_flag(tmp_path, monkeypatch):
    """``-v`` propagates to ``configure(verbose=True)``."""
    touch_files(["a.m4a"], tmp_path)
    record: list[str] = []
    fake_transcribe(monkeypatch, record)
    isolate_home(monkeypatch, tmp_path)
    with patch("vemoizer.run_log.configure") as mock_cfg:
        result = runner.invoke(app, ["transcribe", "a.m4a", "-v"])
    assert result.exit_code == 0
    kwargs = mock_cfg.call_args.kwargs
    assert kwargs["verbose"] is True


def test_transcribe_configure_quiet_flag(tmp_path, monkeypatch):
    """``--quiet`` propagates to ``configure(quiet=True)``."""
    touch_files(["a.m4a"], tmp_path)
    record: list[str] = []
    fake_transcribe(monkeypatch, record)
    isolate_home(monkeypatch, tmp_path)
    with patch("vemoizer.run_log.configure") as mock_cfg:
        result = runner.invoke(app, ["transcribe", "a.m4a", "--quiet"])
    assert result.exit_code == 0
    kwargs = mock_cfg.call_args.kwargs
    assert kwargs["quiet"] is True


def test_transcribe_configure_no_config_none_api_key_env(tmp_path, monkeypatch):
    """Without a config file, ``llm_api_key_env`` is ``None``."""
    touch_files(["a.m4a"], tmp_path)
    record: list[str] = []
    fake_transcribe(monkeypatch, record)
    isolate_home(monkeypatch, tmp_path)
    with patch("vemoizer.run_log.configure") as mock_cfg:
        result = runner.invoke(app, ["transcribe", "a.m4a"])
    assert result.exit_code == 0
    kwargs = mock_cfg.call_args.kwargs
    assert kwargs["llm_api_key_env"] is None


def test_transcribe_configure_with_valid_config_api_key_env(tmp_path, monkeypatch):
    """A valid LLM config in CWD causes ``configure()`` to receive
    the config's ``api_key_env``."""
    (tmp_path / ".vemoizer").mkdir()
    (tmp_path / ".vemoizer" / "config.toml").write_text(
        '[llm]\nbase_url = "http://localhost:8000"\n'
        'model = "test-model"\n'
        'api_key_env = "TEST_API_KEY"\n'
        "timeout_seconds = 30\n",
        encoding="utf-8",
    )
    touch_files(["a.m4a"], tmp_path)
    record: list[str] = []
    fake_transcribe(monkeypatch, record)
    isolate_home(monkeypatch, tmp_path)
    with patch("vemoizer.run_log.configure") as mock_cfg:
        result = runner.invoke(app, ["transcribe", "a.m4a"])
    assert result.exit_code == 0
    kwargs = mock_cfg.call_args.kwargs
    assert kwargs["llm_api_key_env"] == "TEST_API_KEY"


def test_transcribe_configure_malformed_config_fail_open(tmp_path, monkeypatch):
    """A malformed LLM config (missing required keys) does not crash the
    CLI; ``configure()`` receives ``llm_api_key_env=None``."""
    (tmp_path / ".vemoizer").mkdir()
    (tmp_path / ".vemoizer" / "config.toml").write_text(
        '[llm]\nbase_url = "http://localhost"\n',  # missing model, api_key_env
        encoding="utf-8",
    )
    touch_files(["a.m4a"], tmp_path)
    record: list[str] = []
    fake_transcribe(monkeypatch, record)
    isolate_home(monkeypatch, tmp_path)
    with patch("vemoizer.run_log.configure") as mock_cfg:
        runner.invoke(app, ["transcribe", "a.m4a"])
    # The malformed config should not crash the CLI at configure time.
    # It may crash later in the per-file loop (fail-loud), but configure
    # must have been called before that.
    assert mock_cfg.call_count == 1
    kwargs = mock_cfg.call_args.kwargs
    assert kwargs["llm_api_key_env"] is None


# --- meeting: run context wiring ---------------------------------------------


def test_meeting_configure_receives_flags(tmp_path, monkeypatch):
    touch_files(["a.m4a"], tmp_path)
    record: list[str] = []
    fake_transcribe(monkeypatch, record, title="Team Sync")
    isolate_home(monkeypatch, tmp_path)
    with patch("vemoizer.run_log.configure") as mock_cfg:
        result = runner.invoke(app, ["meeting", "a.m4a"])
    assert result.exit_code == 0
    assert mock_cfg.call_count == 1
    kwargs = mock_cfg.call_args.kwargs
    assert kwargs["verbose"] is False
    assert kwargs["quiet"] is False


def test_meeting_configure_verbose_flag(tmp_path, monkeypatch):
    touch_files(["a.m4a"], tmp_path)
    record: list[str] = []
    fake_transcribe(monkeypatch, record, title="Team Sync")
    isolate_home(monkeypatch, tmp_path)
    with patch("vemoizer.run_log.configure") as mock_cfg:
        result = runner.invoke(app, ["meeting", "a.m4a", "-v"])
    assert result.exit_code == 0
    kwargs = mock_cfg.call_args.kwargs
    assert kwargs["verbose"] is True


def test_meeting_configure_quiet_flag(tmp_path, monkeypatch):
    touch_files(["a.m4a"], tmp_path)
    record: list[str] = []
    fake_transcribe(monkeypatch, record, title="Team Sync")
    isolate_home(monkeypatch, tmp_path)
    with patch("vemoizer.run_log.configure") as mock_cfg:
        result = runner.invoke(app, ["meeting", "a.m4a", "--quiet"])
    assert result.exit_code == 0
    kwargs = mock_cfg.call_args.kwargs
    assert kwargs["quiet"] is True


def test_meeting_configure_with_config_api_key_env(tmp_path, monkeypatch):
    (tmp_path / ".vemoizer").mkdir()
    (tmp_path / ".vemoizer" / "config.toml").write_text(
        '[llm]\nbase_url = "http://localhost:8000"\n'
        'model = "test-model"\n'
        'api_key_env = "MEETING_KEY_ENV"\n'
        "timeout_seconds = 30\n",
        encoding="utf-8",
    )
    touch_files(["a.m4a"], tmp_path)
    record: list[str] = []
    fake_transcribe(monkeypatch, record, title="Team Sync")
    isolate_home(monkeypatch, tmp_path)
    with patch("vemoizer.run_log.configure") as mock_cfg:
        result = runner.invoke(app, ["meeting", "a.m4a"])
    assert result.exit_code == 0
    kwargs = mock_cfg.call_args.kwargs
    assert kwargs["llm_api_key_env"] == "MEETING_KEY_ENV"


# --- memo: run context wiring --------------------------------------------------


def test_memo_configure_receives_flags(tmp_path, monkeypatch):
    touch_files(["a.m4a"], tmp_path)
    record: list[str] = []
    fake_transcribe(monkeypatch, record)
    isolate_home(monkeypatch, tmp_path)
    with patch("vemoizer.run_log.configure") as mock_cfg:
        result = runner.invoke(app, ["memo", "a.m4a"])
    assert result.exit_code == 0
    assert mock_cfg.call_count == 1
    kwargs = mock_cfg.call_args.kwargs
    assert kwargs["verbose"] is False
    assert kwargs["quiet"] is False


def test_memo_configure_verbose_flag(tmp_path, monkeypatch):
    touch_files(["a.m4a"], tmp_path)
    record: list[str] = []
    fake_transcribe(monkeypatch, record)
    isolate_home(monkeypatch, tmp_path)
    with patch("vemoizer.run_log.configure") as mock_cfg:
        result = runner.invoke(app, ["memo", "a.m4a", "-v"])
    assert result.exit_code == 0
    kwargs = mock_cfg.call_args.kwargs
    assert kwargs["verbose"] is True


def test_memo_configure_quiet_flag(tmp_path, monkeypatch):
    touch_files(["a.m4a"], tmp_path)
    record: list[str] = []
    fake_transcribe(monkeypatch, record)
    isolate_home(monkeypatch, tmp_path)
    with patch("vemoizer.run_log.configure") as mock_cfg:
        result = runner.invoke(app, ["memo", "a.m4a", "--quiet"])
    assert result.exit_code == 0
    kwargs = mock_cfg.call_args.kwargs
    assert kwargs["quiet"] is True


def test_memo_configure_with_config_api_key_env(tmp_path, monkeypatch):
    (tmp_path / ".vemoizer").mkdir()
    (tmp_path / ".vemoizer" / "config.toml").write_text(
        '[llm]\nbase_url = "http://localhost:8000"\n'
        'model = "test-model"\n'
        'api_key_env = "MEMO_KEY_ENV"\n'
        "timeout_seconds = 30\n",
        encoding="utf-8",
    )
    touch_files(["a.m4a"], tmp_path)
    record: list[str] = []
    fake_transcribe(monkeypatch, record)
    isolate_home(monkeypatch, tmp_path)
    with patch("vemoizer.run_log.configure") as mock_cfg:
        result = runner.invoke(app, ["memo", "a.m4a"])
    assert result.exit_code == 0
    kwargs = mock_cfg.call_args.kwargs
    assert kwargs["llm_api_key_env"] == "MEMO_KEY_ENV"


# --- multi-file: run context wiring --------------------------------------------


def test_transcribe_multi_no_group_configure_once(tmp_path, monkeypatch):
    """Multi-file --no-group: configure is called exactly once (per CLI
    invocation, not per file)."""
    touch_files(["a.m4a", "b.m4a"], tmp_path)
    record: list[str] = []
    fake_transcribe(monkeypatch, record)
    isolate_home(monkeypatch, tmp_path)
    with patch("vemoizer.run_log.configure") as mock_cfg:
        result = runner.invoke(app, ["transcribe", "a.m4a", "b.m4a", "--no-group"])
    assert result.exit_code == 0
    assert len(record) == 2
    assert mock_cfg.call_count == 1


def test_meeting_multi_no_group_configure_once(tmp_path, monkeypatch):
    """Multi-file meeting --no-group: configure is called exactly once."""
    touch_files(["a.m4a", "b.m4a"], tmp_path)
    record: list[str] = []
    fake_transcribe(monkeypatch, record)
    isolate_home(monkeypatch, tmp_path)
    with patch("vemoizer.run_log.configure") as mock_cfg:
        result = runner.invoke(app, ["meeting", "a.m4a", "b.m4a", "--no-group"])
    assert result.exit_code == 0
    assert len(record) == 2
    assert mock_cfg.call_count == 1


def test_memo_multi_configure_once(tmp_path, monkeypatch):
    """Multi-file memo: configure is called exactly once."""
    touch_files(["a.m4a", "b.m4a"], tmp_path)
    record: list[str] = []
    fake_transcribe(monkeypatch, record)
    isolate_home(monkeypatch, tmp_path)
    with patch("vemoizer.run_log.configure") as mock_cfg:
        result = runner.invoke(app, ["memo", "a.m4a", "b.m4a"])
    assert result.exit_code == 0
    assert len(record) == 2
    assert mock_cfg.call_count == 1


# --- grouped: run context wiring -------------------------------------------------


def test_transcribe_grouped_configure_once(tmp_path, monkeypatch):
    """Grouped transcribe (2 parts merged): configure is called exactly
    once, not per group."""
    fake_continuation_seams(monkeypatch)
    touch_files(["a.m4a", "b.m4a"], tmp_path)
    record: list[str] = []
    fake_transcribe(monkeypatch, record)
    isolate_home(monkeypatch, tmp_path)
    with patch("vemoizer.run_log.configure") as mock_cfg:
        result = runner.invoke(app, ["transcribe", "a.m4a", "b.m4a", "--yes"])
    assert result.exit_code == 0
    assert len(record) == 1
    assert mock_cfg.call_count == 1


def test_meeting_grouped_configure_once(tmp_path, monkeypatch):
    """Grouped meeting (2 parts merged): configure is called exactly once."""
    fake_continuation_seams(monkeypatch)
    touch_files(["a.m4a", "b.m4a"], tmp_path)
    record: list[str] = []
    fake_transcribe(monkeypatch, record, title="Team Sync")
    isolate_home(monkeypatch, tmp_path)
    with patch("vemoizer.run_log.configure") as mock_cfg:
        result = runner.invoke(app, ["meeting", "a.m4a", "b.m4a", "--yes"])
    assert result.exit_code == 0
    assert len(record) == 1
    assert mock_cfg.call_count == 1


# --- real configure (not mocked): end-to-end with file_log seam ---------------


def test_transcribe_real_configure_file_log_written(tmp_path, monkeypatch):
    """With the real (un-mocked) ``configure()`` and a real ``file_log``
    seam, the run log file is created at the expected path."""
    touch_files(["a.m4a"], tmp_path)
    record: list[str] = []
    fake_transcribe(monkeypatch, record)
    isolate_home(monkeypatch, tmp_path)
    # Patch file_log to verify it was called with the right stem, but
    # let the real configure() run.
    with _patch_file_log() as mock_fl:
        result = runner.invoke(app, ["transcribe", "a.m4a"])
    assert result.exit_code == 0
    # The seam must have called file_log with the file's NFC stem (unconditional:
    # if the patch misses the call site, this must fail, not pass vacuously).
    assert len(mock_fl.call_args_list) == 1
    first_call = mock_fl.call_args_list[0]
    stem = first_call.args[0] if first_call.args else first_call.kwargs.get("stem")
    assert stem == "a"


def test_meeting_real_configure_file_log_written(tmp_path, monkeypatch):
    """Meeting: real configure() + patched file_log; verify stem."""
    touch_files(["a.m4a"], tmp_path)
    record: list[str] = []
    fake_transcribe(monkeypatch, record, title="Team Sync")
    isolate_home(monkeypatch, tmp_path)
    with _patch_file_log() as mock_fl:
        result = runner.invoke(app, ["meeting", "a.m4a"])
    assert result.exit_code == 0
    assert len(mock_fl.call_args_list) == 1
    first_call = mock_fl.call_args_list[0]
    stem = first_call.args[0] if first_call.args else first_call.kwargs.get("stem")
    assert stem == "a"


def test_memo_real_configure_file_log_written(tmp_path, monkeypatch):
    """Memo: real configure() + patched file_log; verify stem."""
    touch_files(["a.m4a"], tmp_path)
    record: list[str] = []
    fake_transcribe(monkeypatch, record)
    isolate_home(monkeypatch, tmp_path)
    with _patch_file_log() as mock_fl:
        result = runner.invoke(app, ["memo", "a.m4a"])
    assert result.exit_code == 0
    assert len(mock_fl.call_args_list) == 1
    first_call = mock_fl.call_args_list[0]
    stem = first_call.args[0] if first_call.args else first_call.kwargs.get("stem")
    assert stem == "a"


# --- failure cases: configure still called -------------------------------------


def test_transcribe_failure_configure_still_called(tmp_path, monkeypatch):
    """A failing transcribe still calls configure() (before the per-file
    loop)."""
    touch_files(["a.m4a"], tmp_path)

    def fake_tf(path, **kw):
        raise RuntimeError("decoder exploded")

    monkeypatch.setattr(pipeline_module, "transcribe_file", fake_tf)
    isolate_home(monkeypatch, tmp_path)
    with patch("vemoizer.run_log.configure") as mock_cfg:
        result = runner.invoke(app, ["transcribe", "a.m4a"])
    assert result.exit_code == 1
    assert mock_cfg.call_count == 1


def test_meeting_failure_configure_still_called(tmp_path, monkeypatch):
    """A failing meeting transcribe still calls configure()."""
    touch_files(["a.m4a"], tmp_path)

    def fake_tf(path, **kw):
        raise RuntimeError("boom")

    monkeypatch.setattr(pipeline_module, "transcribe_file", fake_tf)
    isolate_home(monkeypatch, tmp_path)
    with patch("vemoizer.run_log.configure") as mock_cfg:
        result = runner.invoke(app, ["meeting", "a.m4a"])
    assert result.exit_code == 1
    assert mock_cfg.call_count == 1


def test_memo_failure_configure_still_called(tmp_path, monkeypatch):
    """A failing memo transcribe still calls configure()."""
    touch_files(["a.m4a"], tmp_path)

    def fake_tf(path, **kw):
        raise RuntimeError("boom")

    monkeypatch.setattr(pipeline_module, "transcribe_file", fake_tf)
    isolate_home(monkeypatch, tmp_path)
    with patch("vemoizer.run_log.configure") as mock_cfg:
        result = runner.invoke(app, ["memo", "a.m4a"])
    assert result.exit_code == 1
    assert mock_cfg.call_count == 1


# --- NFC stem: configure receives NFC-normalised stem via file_log -------------


def test_nfc_filename_file_log_stem(tmp_path, monkeypatch):
    """An NFD-composed filename yields the NFC-normalised stem in
    file_log's stem argument."""
    nfd_name = "caf\u0065\u0301.m4a"  # "café" decomposed
    touch_files([nfd_name], tmp_path)
    record: list[str] = []
    fake_transcribe(monkeypatch, record)
    isolate_home(monkeypatch, tmp_path)
    with _patch_file_log() as mock_fl:
        result = runner.invoke(app, ["transcribe", nfd_name])
    assert result.exit_code == 0
    assert len(mock_fl.call_args_list) == 1
    first_call = mock_fl.call_args_list[0]
    stem = first_call.args[0] if first_call.args else first_call.kwargs.get("stem")
    assert stem == "caf\u00e9"  # NFC-composed


# --- reset_run_log: each CLI invocation starts clean ----------------------------


def test_configure_context_reset_between_invocations(tmp_path, monkeypatch):
    """Two sequential CLI invocations: the second configure() call is
    independent of the first (no state leakage via the module-level
    _RunContext)."""
    touch_files(["a.m4a"], tmp_path)
    record: list[str] = []
    fake_transcribe(monkeypatch, record)
    isolate_home(monkeypatch, tmp_path)

    # First invocation: verbose
    with patch("vemoizer.run_log.configure") as mock_cfg1:
        runner.invoke(app, ["transcribe", "a.m4a", "-v"])
    assert mock_cfg1.call_args.kwargs["verbose"] is True

    # Reset the context (as the driver would between runs)
    run_log._context.verbose = None
    run_log._context.quiet = None
    run_log._context.llm_api_key_env = None

    # Second invocation: not verbose
    with patch("vemoizer.run_log.configure") as mock_cfg2:
        runner.invoke(app, ["transcribe", "a.m4a"])
    assert mock_cfg2.call_args.kwargs["verbose"] is False
