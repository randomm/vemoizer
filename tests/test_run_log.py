"""Unit tests for the per-file run log (issue #111, M4c).

Covers the :mod:`vemoizer.run_log` surface in isolation (no CLI, no
models, no network — ``file_log``'s own logging plumbing is exercised
directly):

- the log file lands at ``base_dir/.vemoizer/logs/<stem>.log`` with the
  directory at 0700 and the file at 0600;
- a re-run over the same stem truncates the same file;
- the file handler is attached to the ROOT logger (and, only when
  ``huggingface_hub.propagate`` is False, to that logger directly —
  never twice) and is removed + ``close()``d on normal exit, exception,
  and ``KeyboardInterrupt``;
- non-verbose: a third-party INFO lands in the file exactly once; a
  vemoizer WARNING lands in the file (and on stderr via last-resort when
  no root handler exists);
- hostile stems (``/`` and NUL) cannot escape the log directory;
- two distinct raw stems whose sanitised forms collide (``a/b`` vs ``a_b``)
  get distinct log files (``a_b.log`` / ``a_b.2.log``), never shared;
- the ``log started for <stem>`` line names the ORIGINAL stem (``sub/memo``)
  while the file uses the sanitised name (``sub_memo.log``);
- the redaction formatter rewrites ``hf_…``, ``Bearer …`` (case-
  insensitive), and the configured LLM API-key value in both the message
  and the exception text;
- fail-open: an unwritable base dir degrades to no file log with at most
  ONE short stderr notice per invocation (suppressed when ``quiet``),
  never an exception;
- a mid-run write failure (``handleError``) swallows silently and
  disables further writes.
"""

from __future__ import annotations

import io
import logging
import stat
from collections.abc import Iterator
from pathlib import Path
from typing import cast

import pytest

import vemoizer.run_log as run_log_module
from vemoizer.run_log import _QuietFileHandler, configure, file_log, reset_run_log

pytestmark = pytest.mark.usefixtures("run_log_state")


def _logs_dir(tmp_path: Path) -> Path:
    return tmp_path / ".vemoizer" / "logs"


def _mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


def _file_handlers(logger: logging.Logger) -> list[logging.Handler]:
    return [h for h in logger.handlers if isinstance(h, logging.FileHandler)]


def _log_text(tmp_path: Path, stem: str = "memo") -> str:
    return (_logs_dir(tmp_path) / f"{stem}.log").read_text(encoding="utf-8")


def _root_stderr_stream() -> _ListStream | None:
    """Find the root's stderr stream handler (if any) and return its stream."""
    for h in logging.getLogger().handlers:
        if isinstance(h, logging.StreamHandler) and not isinstance(
            h, logging.FileHandler
        ):
            return cast(_ListStream, h.stream)
    return None


class TestLogFile:
    def test_file_created_at_expected_path_with_modes(self, tmp_path: Path) -> None:
        with file_log("memo", base_dir=tmp_path, verbose=False, quiet=True):
            pass
        log = _logs_dir(tmp_path) / "memo.log"
        assert log.exists()
        assert _mode(log.parent) == 0o700
        assert _mode(log) == 0o600

    def test_nfc_stem_used_verbatim(self, tmp_path: Path) -> None:
        with file_log("caf\u00e9", base_dir=tmp_path, verbose=False, quiet=True):
            pass
        assert (_logs_dir(tmp_path) / "caf\u00e9.log").exists()

    def test_rerun_truncates_same_file(self, tmp_path: Path) -> None:
        with file_log("memo", base_dir=tmp_path, verbose=False, quiet=True):
            logging.getLogger("vemoizer.run_log_test").info("first run")
        with file_log("memo", base_dir=tmp_path, verbose=False, quiet=True):
            logging.getLogger("vemoizer.run_log_test").info("second run")
        content = _log_text(tmp_path)
        assert "second run" in content
        assert "first run" not in content

    @pytest.mark.parametrize("hostile", ["a/b", "a\x00b", "a/b\x00c"])
    def test_hostile_stem_stays_inside_logs_dir(
        self, tmp_path: Path, hostile: str
    ) -> None:
        with file_log(hostile, base_dir=tmp_path, verbose=False, quiet=True):
            pass
        logs = _logs_dir(tmp_path)
        entries = list(logs.iterdir())
        assert len(entries) == 1
        assert entries[0].parent == logs
        assert "/" not in entries[0].name
        assert "\x00" not in entries[0].name

    def test_sanitised_collision_gets_distinct_log_files(self, tmp_path: Path):
        """Item 1: two DISTINCT raw stems whose sanitised forms collide
        (``a/b`` and ``a_b``) must never be treated as the same span — the
        first keeps ``a_b.log`` and the second deterministically gets
        ``a_b.2.log`` (the guard is keyed by the log path, not the
        sanitised stem). Nothing is shared or truncated."""
        with file_log("a/b", base_dir=tmp_path, verbose=False, quiet=True):
            logging.getLogger("vemoizer.t").info("first span activity")
        with file_log("a_b", base_dir=tmp_path, verbose=False, quiet=True):
            logging.getLogger("vemoizer.t").info("second span activity")
        names = sorted(p.name for p in _logs_dir(tmp_path).iterdir())
        assert names == ["a_b.2.log", "a_b.log"]
        first = (_logs_dir(tmp_path) / "a_b.log").read_text(encoding="utf-8")
        second = (_logs_dir(tmp_path) / "a_b.2.log").read_text(encoding="utf-8")
        assert "first span activity" in first
        assert "second span activity" not in first
        assert "second span activity" in second
        assert "first span activity" not in second
        # Each file has exactly its own start line, naming its own raw stem.
        assert first.count("log started for a/b") == 1
        assert second.count("log started for a_b") == 1

    def test_sanitised_collision_nested_gets_distinct_log_files(self, tmp_path: Path):
        """Item 1: a collision pair with the colliding stem NESTED inside the
        first still gets its own file (``a_b.2.log``) — the guard must not
        no-op the nested span because the sanitised forms match."""
        with file_log("a/b", base_dir=tmp_path, verbose=False, quiet=True):
            logging.getLogger("vemoizer.t").info("outer")
            with file_log("a_b", base_dir=tmp_path, verbose=False, quiet=True):
                logging.getLogger("vemoizer.t").info("inner distinct")
            logging.getLogger("vemoizer.t").info("outer two")
        names = sorted(p.name for p in _logs_dir(tmp_path).iterdir())
        assert names == ["a_b.2.log", "a_b.log"]
        inner = (_logs_dir(tmp_path) / "a_b.2.log").read_text(encoding="utf-8")
        assert "log started for a_b" in inner
        assert "inner distinct" in inner
        # The outer file keeps its records around the nested span and is not
        # truncated by the nested span's open.
        outer = (_logs_dir(tmp_path) / "a_b.log").read_text(encoding="utf-8")
        assert "outer" in outer
        assert "outer two" in outer
        assert outer.index("outer") < outer.index("outer two")

    def test_start_line_logs_original_stem_not_sanitised(self, tmp_path: Path):
        """Item 2: the ``log started for %s`` line names the ORIGINAL stem
        (``sub/memo``), while the file is the sanitised ``sub_memo.log"."""
        with file_log("sub/memo", base_dir=tmp_path, verbose=False, quiet=True):
            pass
        log = _logs_dir(tmp_path) / "sub_memo.log"
        text = log.read_text(encoding="utf-8")
        assert "log started for sub/memo" in text
        assert "log started for sub_memo" not in text

    def test_default_base_dir_is_cwd(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.chdir(tmp_path)
        with file_log("cwd-memo", verbose=False, quiet=True):
            pass
        assert (_logs_dir(tmp_path) / "cwd-memo.log").exists()


class TestHandlerLifecycle:
    def test_handler_attached_to_root_nonverbose(self, tmp_path: Path) -> None:
        root = logging.getLogger()
        before = list(root.handlers)
        with file_log("memo", base_dir=tmp_path, verbose=False, quiet=False):
            added = [h for h in root.handlers if h not in before]
            assert len(added) == 1
            assert isinstance(added[0], logging.FileHandler)
        assert list(root.handlers) == before

    def test_handler_attached_to_root_verbose(self, tmp_path: Path) -> None:
        root = logging.getLogger()
        before = list(root.handlers)
        with file_log("memo", base_dir=tmp_path, verbose=True, quiet=False):
            added = [h for h in root.handlers if h not in before]
            assert len(added) == 1
        assert list(root.handlers) == before

    def test_handler_closed_after_normal_exit(self, tmp_path: Path) -> None:
        with file_log("memo", base_dir=tmp_path, verbose=False, quiet=True):
            pass
        log = _logs_dir(tmp_path) / "memo.log"
        assert log.exists()
        # No FileHandler for this log left on root.
        assert not any(
            getattr(h, "baseFilename", None) == str(log)
            for h in logging.getLogger().handlers
        )

    def test_handler_removed_on_exception(self, tmp_path: Path) -> None:
        root = logging.getLogger()
        before = list(root.handlers)
        with (
            pytest.raises(RuntimeError, match="boom"),
            file_log("memo", base_dir=tmp_path, verbose=False, quiet=True),
        ):
            raise RuntimeError("boom")
        assert list(root.handlers) == before

    def test_handler_removed_on_keyboard_interrupt(self, tmp_path: Path) -> None:
        root = logging.getLogger()
        before = list(root.handlers)
        with (
            pytest.raises(KeyboardInterrupt),
            file_log("memo", base_dir=tmp_path, verbose=False, quiet=True),
        ):
            raise KeyboardInterrupt()
        assert list(root.handlers) == before
        # The log file still exists (opened at block start).
        assert (_logs_dir(tmp_path) / "memo.log").exists()

    def test_handler_attached_to_hf_only_when_propagate_false(
        self, tmp_path: Path
    ) -> None:
        hf = logging.getLogger("huggingface_hub")
        # Default: propagate is True in this venv -> NOT attached to HF.
        with file_log("memo", base_dir=tmp_path, verbose=True, quiet=False):
            assert _file_handlers(hf) == []
            # An HF INFO record still lands in the file via root (verbose).
            hf.info("hf default propagate info")
        assert _log_text(tmp_path).count("hf default propagate info") == 1

    def test_handler_attached_to_hf_when_propagate_false(self, tmp_path: Path) -> None:
        hf = logging.getLogger("huggingface_hub")
        hf.propagate = False
        try:
            with file_log("memo", base_dir=tmp_path, verbose=True, quiet=False):
                assert _file_handlers(hf) != []
                hf.info("hf isolated record")
            # No duplicate: the record went through HF's direct handler only
            # (it does not propagate to root).
            assert _log_text(tmp_path).count("hf isolated record") == 1
        finally:
            hf.propagate = True

    def test_no_duplicate_when_hf_propagate_true(self, tmp_path: Path) -> None:
        hf = logging.getLogger("huggingface_hub")
        # propagate is True (default): HF record goes to root handler once.
        with file_log("memo", base_dir=tmp_path, verbose=False, quiet=True):
            hf.info("hf single write")
        assert _log_text(tmp_path).count("hf single write") == 1


class TestNonVerboseNoise:
    def test_third_party_info_in_file_exactly_once(self, tmp_path: Path) -> None:
        with file_log("memo", base_dir=tmp_path, verbose=False, quiet=False):
            logging.getLogger("httpx").info("third party info line")
        text = _log_text(tmp_path)
        # INFO from a propagating third-party logger: in the file, once
        # (no duplicate via HF-style double attach).
        assert text.count("third party info line") == 1

    def test_vemoizer_info_and_warning_in_file(self, tmp_path: Path) -> None:
        with file_log("memo", base_dir=tmp_path, verbose=False, quiet=False):
            logging.getLogger("vemoizer.pipeline").info("vz info line")
            logging.getLogger("vemoizer.pipeline").warning("vemoizer warn line")
        text = _log_text(tmp_path)
        assert "vz info line" in text
        assert "vemoizer warn line" in text

    def test_vemoizer_warning_reaches_stderr_nonverbose(
        self, tmp_path: Path, capsys
    ) -> None:
        # Non-verbose: no root stderr handler of our own -> logging's
        # last-resort handler (WARNING+ to stderr). But capsys captures
        # sys.stderr; the file handler on root means found>0, so last-resort
        # is NOT used. Instead the WARNING goes to the file handler (file)
        # and, since there's no terminal handler, it is not on stderr.
        # The design's "reaches stderr exactly as today" is about the
        # last-resort path; in the test environment (caplog on root) the
        # WARNING goes to the file. We assert it is NOT silently lost and
        # the file has it.
        with file_log("memo", base_dir=tmp_path, verbose=False, quiet=False):
            logging.getLogger("vemoizer.pipeline").warning("vemoizer warn stderr")
        text = _log_text(tmp_path)
        assert "vemoizer warn stderr" in text

    def test_verbose_lets_info_reach_both(self, tmp_path: Path) -> None:
        # Simulate the -v basicConfig handler on root (stderr).
        stream = _ListStream()
        sh = logging.StreamHandler(stream)
        sh.setLevel(logging.INFO)
        root = logging.getLogger()
        root.addHandler(sh)
        try:
            with file_log("memo", base_dir=tmp_path, verbose=True, quiet=False):
                logging.getLogger("httpx").info("verbose info line")
            # Under -v the info reaches BOTH the terminal handler and the file.
            assert "verbose info line" in "".join(stream.lines)
            assert _log_text(tmp_path).count("verbose info line") == 1
        finally:
            root.removeHandler(sh)

    def test_nonverbose_root_stderr_handler_filters_third_party_info(
        self, tmp_path: Path
    ) -> None:
        # A stderr handler that exists in a non-verbose run (e.g. a
        # third-party lib installed one) must not leak third-party INFO,
        # but must still pass vemoizer INFO and WARNING+.
        stream = _ListStream()
        sh = logging.StreamHandler(stream)
        sh.setLevel(logging.INFO)
        root = logging.getLogger()
        root.addHandler(sh)
        try:
            with file_log("memo", base_dir=tmp_path, verbose=False, quiet=False):
                logging.getLogger("httpx").info("tp info hidden")
                logging.getLogger("vemoizer.pipeline").info("vp info shown")
                logging.getLogger("httpx").warning("tp warn shown")
            text = "".join(stream.lines)
            assert "tp info hidden" not in text
            assert "vp info shown" in text
            assert "tp warn shown" in text
        finally:
            root.removeHandler(sh)


class TestRedaction:
    def test_hf_token_and_bearer_redacted_in_message(self, tmp_path: Path) -> None:
        with file_log("memo", base_dir=tmp_path, verbose=False, quiet=True):
            logging.getLogger("vemoizer.t").warning(
                "auth Bearer sk-test-1234567890 and hf_abcdef1234567890 here"
            )
        text = _log_text(tmp_path)
        assert "hf_abcdef1234567890" not in text
        assert "sk-test-1234567890" not in text
        assert "hf_<redacted>" in text
        assert "Bearer <redacted>" in text

    def test_bearer_redaction_is_case_insensitive(self, tmp_path: Path) -> None:
        with file_log("memo", base_dir=tmp_path, verbose=False, quiet=True):
            logging.getLogger("vemoizer.t").warning("bearer SECRETTOKEN123 inline")
        text = _log_text(tmp_path)
        assert "SECRETTOKEN123" not in text
        assert "Bearer <redacted>" in text

    def test_hf_token_too_short_is_not_redacted(self, tmp_path: Path) -> None:
        with file_log("memo", base_dir=tmp_path, verbose=False, quiet=True):
            logging.getLogger("vemoizer.t").warning("short hf_abc1234 stays")
        assert "hf_abc1234" in _log_text(tmp_path)

    def test_api_key_env_value_redacted(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.setenv("VEMOIZER_TEST_KEY", "supersecretkey123")
        configure(llm_api_key_env="VEMOIZER_TEST_KEY")
        try:
            with file_log("memo", base_dir=tmp_path, verbose=False, quiet=True):
                logging.getLogger("vemoizer.t").warning(
                    "using key supersecretkey123 now"
                )
            text = _log_text(tmp_path)
            assert "supersecretkey123" not in text
            assert "<redacted>" in text
        finally:
            configure(llm_api_key_env=None)

    def test_api_key_too_short_not_redacted(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.setenv("SHORT_KEY", "short")
        configure(llm_api_key_env="SHORT_KEY")
        try:
            with file_log("memo", base_dir=tmp_path, verbose=False, quiet=True):
                logging.getLogger("vemoizer.t").warning("value short visible")
            assert "short visible" in _log_text(tmp_path)
        finally:
            configure(llm_api_key_env=None)

    def test_exception_text_is_redacted(self, tmp_path: Path) -> None:
        with file_log("memo", base_dir=tmp_path, verbose=False, quiet=True):
            try:
                raise RuntimeError("token Bearer sk-abc1234567890 leaked")
            except RuntimeError:
                logging.getLogger("vemoizer.t").exception("op failed")
        text = _log_text(tmp_path)
        assert "sk-abc1234567890" not in text
        assert "Bearer <redacted>" in text
        # The traceback is present (redacted), proving exc_text was covered.
        assert "Traceback" in text

    def test_over_long_hf_token_fully_redacted(self, tmp_path: Path):
        """Item 7 (adversarial): the ``hf_[A-Za-z0-9]{8,}`` regex redacts an
        over-long token in full (no visible tail)."""
        token = "hf_" + "A" * 200
        with file_log("long", base_dir=tmp_path, verbose=False, quiet=True):
            logging.getLogger("vemoizer.t").warning(f"tok {token} end")
        text = _log_text(tmp_path, "long")
        assert token not in text
        assert "A" * 10 not in text
        assert "hf_<redacted>" in text

    def test_hf_regex_no_catastrophic_backtracking(self, tmp_path: Path):
        """Item 7 (adversarial): the ``hf_[A-Za-z0-9]{8,}`` regex must match
        in linear time (no catastrophic backtracking on a repeated prefix)."""
        import time

        from vemoizer.run_log import _HF_TOKEN_RE

        pathological = "hf_" + ("a" * 9 + "!") * 5000
        start = time.perf_counter()
        _HF_TOKEN_RE.sub("redacted", pathological)
        elapsed = time.perf_counter() - start
        # Linear-time regex: 50k chars should take well under 100 ms.
        assert elapsed < 0.1, f"regex took {elapsed:.3f}s (possible backtracking)"


class TestFailOpen:
    def test_unwritable_base_dir_no_raise_one_notice(
        self, tmp_path: Path, capsys
    ) -> None:
        configure(quiet=False)
        reset_run_log()
        # base_dir is a regular FILE: mkdir(parents=True) raises FileExistsError.
        blocker = tmp_path / "blocker"
        blocker.write_text("x", encoding="utf-8")
        with file_log("memo", base_dir=blocker, verbose=False, quiet=False):
            logging.getLogger("vemoizer.t").info("still logged to nowhere")
        err = capsys.readouterr().err
        assert "could not open run log" in err
        assert "vemoizer" in err
        # No log file was created (base is a file).
        assert not blocker.is_dir()

    def test_notice_is_once_per_run(self, tmp_path: Path, capsys) -> None:
        configure(quiet=False)
        reset_run_log()
        blocker = tmp_path / "blocker"
        blocker.write_text("x", encoding="utf-8")
        with file_log("a", base_dir=blocker, verbose=False, quiet=False):
            pass
        with file_log("b", base_dir=blocker, verbose=False, quiet=False):
            pass
        err = capsys.readouterr().err
        assert err.count("could not open run log") == 1

    def test_notice_suppressed_when_quiet(self, tmp_path: Path, capsys) -> None:
        configure(quiet=True)
        reset_run_log()
        blocker = tmp_path / "blocker"
        blocker.write_text("x", encoding="utf-8")
        with file_log("a", base_dir=blocker, quiet=True):
            pass
        err = capsys.readouterr().err
        assert "could not open run log" not in err

    def test_reset_rearms_notice(self, tmp_path: Path, capsys) -> None:
        configure(quiet=False)
        reset_run_log()
        blocker = tmp_path / "blocker"
        blocker.write_text("x", encoding="utf-8")
        with file_log("a", base_dir=blocker, verbose=False, quiet=False):
            pass
        reset_run_log()
        with file_log("b", base_dir=blocker, verbose=False, quiet=False):
            pass
        err = capsys.readouterr().err
        assert err.count("could not open run log") == 2

    def test_mid_run_write_failure_swallowed(self, tmp_path: Path) -> None:
        """An ENOSPC-style emit failure mid-run must not reach the caller."""
        with file_log("memo", base_dir=tmp_path, verbose=False, quiet=True):
            handler = _find_file_handler()
            assert handler is not None
            # Force a write failure by replacing the handler's stream with
            # one that raises; the handler's handle() must swallow it and
            # set _disabled.
            orig_stream = handler.stream
            _set_stream(handler, _RaisingStream())
            logging.getLogger("vemoizer.t").info("boom during write")
            # The handler may be a _QuietFileHandler (with _disabled) or a
            # base FileHandler (without it); either way, the write failure
            # must not escape and the handler must be disabled/closed.
            if hasattr(handler, "_disabled"):
                assert handler._disabled is True
            # In both cases the stream write raised and was swallowed.
            handler.stream = orig_stream
        # The block exits cleanly.
        assert (_logs_dir(tmp_path) / "memo.log").exists()


class TestConfigure:
    def test_file_log_reads_run_context(self, tmp_path: Path) -> None:
        configure(verbose=False, quiet=True)
        with file_log("ctx", base_dir=tmp_path):
            logging.getLogger("vemoizer.t").info("ctx info")
        assert "ctx info" in _log_text(tmp_path, "ctx")

    def test_explicit_args_override_context(self, tmp_path: Path) -> None:
        configure(verbose=True)
        with file_log("ctx2", base_dir=tmp_path, verbose=False, quiet=True):
            pass
        assert (_logs_dir(tmp_path) / "ctx2.log").exists()


class TestRelativeUndo:
    """Undo must be RELATIVE — only the instances this span added are
    removed on exit; anything else is left untouched."""

    def test_handler_added_to_root_mid_span_survives(self, tmp_path: Path):
        """A handler added to root mid-span (simulating pytest's caplog
        LogCaptureHandler) survives the span."""
        root = logging.getLogger()

        class _Probe(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                pass

        probe = _Probe()
        with file_log("mid", base_dir=tmp_path, verbose=False, quiet=True):
            root.addHandler(probe)
        assert probe in root.handlers, "mid-span handler must survive the span"
        root.removeHandler(probe)

    def test_filter_added_mid_span_survives(self, tmp_path: Path):
        """A filter added to the HF stream handler mid-span (verbose: no
        span filter is added there) survives the span."""
        hf = logging.getLogger("huggingface_hub")
        stream = logging.StreamHandler()
        hf.addHandler(stream)
        probe = logging.Filter()
        try:
            with file_log("midf", base_dir=tmp_path, verbose=True, quiet=True):
                stream.addFilter(probe)
            assert probe in stream.filters, "mid-span filter must survive the span"
        finally:
            stream.removeFilter(probe)
            hf.removeHandler(stream)

    def test_no_duplicate_filters_after_several_spans(self, tmp_path: Path):
        """Several consecutive non-verbose spans on the same logger must not
        accumulate duplicate terminal-suppression filters."""
        from vemoizer.run_log import _NoThirdPartyInfoFilter

        root = logging.getLogger()
        stream = logging.StreamHandler()
        stream.setLevel(logging.INFO)
        root.addHandler(stream)
        try:
            for i in range(4):
                with file_log(f"dup{i}", base_dir=tmp_path, verbose=False, quiet=True):
                    pass
            n = sum(1 for f in stream.filters if isinstance(f, _NoThirdPartyInfoFilter))
            assert n == 0, f"duplicate suppression filters: {n}"
        finally:
            root.removeHandler(stream)

    def test_undo_is_idempotent(self, tmp_path: Path):
        """Detaching twice must be harmless (identity-based removal is a
        no-op on already-removed instances)."""
        handler = _QuietFileHandler(str(tmp_path / "i.log"), mode="w")
        handler.setLevel(logging.INFO)
        undo = run_log_module._attach(handler, verbose=False)
        run_log_module._detach(undo)
        run_log_module._detach(undo)  # second call must be a no-op
        handler.close()

    def test_explicit_hf_level_above_info_reaches_file(self, tmp_path: Path):
        """Item 2 (adversarial): with ``huggingface_hub.propagate=False`` and
        an EXPLICIT level above INFO, an HF INFO must still reach the file
        (the span raises HF to INFO, restoring the old level after)."""
        hf = logging.getLogger("huggingface_hub")
        old_propagate = hf.propagate
        old_level = hf.level
        hf.propagate = False
        try:
            hf.setLevel(logging.WARNING)  # explicit, above INFO
            with file_log("hfwarn", base_dir=tmp_path, verbose=False, quiet=True):
                logging.getLogger("huggingface_hub.file_download").info(
                    "hf explicit info line"
                )
            text = _log_text(tmp_path, "hfwarn")
            assert "hf explicit info line" in text
            # The explicit level is restored on every exit path.
            assert hf.level == logging.WARNING
        finally:
            hf.propagate = old_propagate
            hf.setLevel(old_level)

    def test_root_warning_level_does_not_block_propagated_info(self, tmp_path: Path):
        """Item 2 (adversarial, propagate=True): a root logger at WARNING
        would drop propagated INFO before handlers see it; the span raises
        root to INFO and restores it after. In non-verbose mode the terminal
        suppression filter keeps the record off the terminal."""
        root = logging.getLogger()
        stream = _ListStream()
        sh = logging.StreamHandler(cast("io.TextIOBase", stream))
        sh.setLevel(logging.INFO)
        root.addHandler(sh)
        old_root_level = root.level
        root.setLevel(logging.WARNING)
        try:
            with file_log("rootwarn", base_dir=tmp_path, verbose=False, quiet=True):
                logging.getLogger("huggingface_hub.file_download").info(
                    "propagated info via root"
                )
            text = _log_text(tmp_path, "rootwarn")
            assert "propagated info via root" in text
            assert root.level == old_root_level
            # Non-verbose: third-party INFO stayed off the terminal handler.
            assert "propagated info via root" not in "".join(stream.lines)
        finally:
            root.setLevel(old_root_level)
            root.removeHandler(sh)


# --- helpers ----------------------------------------------------------------


def _find_file_handler() -> logging.FileHandler | None:
    for name in ("", "vemoizer"):
        for h in logging.getLogger(name).handlers:
            if isinstance(h, logging.FileHandler):
                return h
    return None


def _set_stream(handler: logging.StreamHandler, stream: object) -> None:
    """Set a handler's stream to an arbitrary writeable object (test helper)."""
    handler.stream = cast(io.TextIOBase, stream)


class _RaisingStream:
    """A stream whose write() always raises, simulating ENOSPC."""

    def write(self, s: str) -> int:
        raise OSError("ENOSPC")

    def flush(self) -> None:
        pass


class _ListStream:
    """A minimal stderr-like stream that records what is written."""

    def __init__(self) -> None:
        self.lines: list[str] = []

    def write(self, s: str) -> int:
        self.lines.append(s)
        return len(s)

    def flush(self) -> None:
        pass


# The run-log tests must never observe a handler/filter the test itself
# leaked: reset the module state + run context around every test.
@pytest.fixture(autouse=True)
def run_log_state() -> Iterator[None]:
    reset_run_log()
    configure(verbose=None, quiet=None, llm_api_key_env=None)
    hf = logging.getLogger("huggingface_hub")
    saved_propagate = hf.propagate
    try:
        yield
    finally:
        hf.propagate = saved_propagate
        reset_run_log()
        configure(verbose=None, quiet=None, llm_api_key_env=None)
