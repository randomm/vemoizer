"""Issue #268 rework: the post-meeting term-check hook.

``ask_terms_hook`` mirrors ``ask_naming_hook``'s fail-open boundary, and
skips SILENTLY: the naming hook running right before it narrates the
shared skip conditions, so the terms hook prints nothing when it skips.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from _cli_helpers import isolate_home, touch_files
from _naming_hook_helpers import _two_label_paragraphs
from typer.testing import CliRunner

import vemoizer.naming_hook as naming_hook
import vemoizer.pipeline as pipeline_module
import vemoizer.term_check_cli as term_check_cli
import vemoizer.terms_hook as terms_hook
from vemoizer.cli import app

runner = CliRunner()


@pytest.fixture
def echo_capture(
    monkeypatch: pytest.MonkeyPatch,
) -> list[str]:
    """Capture every ``typer.echo`` (naming and terms hooks share typer)."""
    lines: list[str] = []
    monkeypatch.setattr(
        terms_hook.typer,
        "echo",
        lambda msg, **kw: lines.append(str(msg)),
    )
    return lines


@pytest.fixture
def run_terms_calls(monkeypatch: pytest.MonkeyPatch) -> list[tuple[Any, dict]]:
    calls: list[tuple[Any, dict]] = []

    def fake_run_terms(path: Path, **kw: Any) -> int:
        calls.append((path, kw))
        return 0

    monkeypatch.setattr(term_check_cli, "run_terms", fake_run_terms)
    return calls


def _hook(
    written: list[str],
    *,
    yes: bool = False,
    input_fn: Any = None,
    monkeypatch: pytest.MonkeyPatch,
) -> int:
    monkeypatch.setattr(terms_hook, "_stdout_isatty", lambda: True)
    return terms_hook.ask_terms_hook(
        written, yes=yes, input_fn=input_fn, tty_isatty=lambda: True
    )


class TestSkipRules:
    """Every skip is silent (the naming hook narrates) and exits 0."""

    def test_yes_skips_silently(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, echo_capture: list[str]
    ) -> None:
        isolate_home(monkeypatch, tmp_path, tmp_path)
        assert _hook(["a.json"], yes=True, monkeypatch=monkeypatch) == 0
        assert echo_capture == []

    def test_non_tty_skips_silently(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, echo_capture: list[str]
    ) -> None:
        isolate_home(monkeypatch, tmp_path, tmp_path)
        monkeypatch.setattr(terms_hook, "_stdout_isatty", lambda: True)
        rc = terms_hook.ask_terms_hook(
            ["a.json"], yes=False, input_fn=lambda p: "y", tty_isatty=lambda: False
        )
        assert rc == 0
        assert echo_capture == []

    def test_stdout_non_tty_skips_silently(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, echo_capture: list[str]
    ) -> None:
        isolate_home(monkeypatch, tmp_path, tmp_path)
        monkeypatch.setattr(terms_hook, "_stdout_isatty", lambda: False)
        rc = terms_hook.ask_terms_hook(
            ["a.json"], yes=False, input_fn=lambda p: "y", tty_isatty=lambda: True
        )
        assert rc == 0
        assert echo_capture == []

    def test_no_json_skips_silently(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, echo_capture: list[str]
    ) -> None:
        isolate_home(monkeypatch, tmp_path, tmp_path)
        assert _hook(["a.md"], monkeypatch=monkeypatch) == 0
        assert echo_capture == []

    @pytest.mark.parametrize("yes", [True, False])
    def test_naming_narrates_the_shared_skip_once(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        echo_capture: list[str],
        yes: bool,
    ) -> None:
        """The naming hook narrates the shared reason; terms adds nothing."""
        isolate_home(monkeypatch, tmp_path, tmp_path)
        tty = (lambda: True) if yes else (lambda: False)
        written = ["a.json"]  # the batch call sites hand BOTH hooks one list
        naming_hook.ask_naming_hook(
            written, yes=yes, input_fn=lambda p: "y", tty_isatty=tty
        )
        assert (
            terms_hook.ask_terms_hook(
                written, yes=yes, input_fn=lambda p: "y", tty_isatty=tty
            )
            == 0
        )
        expected = (
            "skipping speaker naming: not requested"
            if yes
            else "skipping speaker naming: not an interactive terminal"
        )
        assert echo_capture == [expected]


class TestPromptAndDispatch:
    def test_declined_prompt_runs_nothing(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        run_terms_calls: list[Any],
    ) -> None:
        isolate_home(monkeypatch, tmp_path, tmp_path)
        rc = _hook(["a.json"], input_fn=lambda p: "n", monkeypatch=monkeypatch)
        assert rc == 0 and run_terms_calls == []

    def test_yes_runs_each_sidecar_with_seams(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        run_terms_calls: list[Any],
    ) -> None:
        isolate_home(monkeypatch, tmp_path, tmp_path)
        prompts: list[str] = []

        def input_fn(p: str) -> str:
            prompts.append(p)
            return "y"

        assert (
            _hook(
                ["a.json", "b.md", "b.json"], input_fn=input_fn, monkeypatch=monkeypatch
            )
            == 0
        )
        assert prompts == ["Check the terms in this transcript? [y/N] "]
        assert [p.name for p, _ in run_terms_calls] == ["a.json", "b.json"]
        assert all(p.is_absolute() for p, _ in run_terms_calls)
        _, kw = run_terms_calls[0]
        assert kw["input_fn"] is input_fn and kw["tty_isatty"] is not None

    def test_eof_at_prompt_is_quiet(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        run_terms_calls: list[Any],
        echo_capture: list[str],
    ) -> None:
        isolate_home(monkeypatch, tmp_path, tmp_path)

        def input_fn(p: str) -> str:
            raise EOFError

        assert _hook(["a.json"], input_fn=input_fn, monkeypatch=monkeypatch) == 0
        assert run_terms_calls == [] and echo_capture == []

    def test_failure_warns_and_continues(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        echo_capture: list[str],
    ) -> None:
        isolate_home(monkeypatch, tmp_path, tmp_path)
        attempted: list[str] = []

        def boom(path: Path, **kw: Any) -> int:
            attempted.append(path.name)
            if path.name == "a.json":
                raise RuntimeError("nope")
            return 0

        monkeypatch.setattr(term_check_cli, "run_terms", boom)
        assert (
            _hook(["a.json", "b.json"], input_fn=lambda p: "y", monkeypatch=monkeypatch)
            == 0
        )
        assert attempted == ["a.json", "b.json"]
        assert "warning: term check failed for a.json: RuntimeError" in echo_capture

    def test_system_exit_is_fail_open(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        echo_capture: list[str],
    ) -> None:
        isolate_home(monkeypatch, tmp_path, tmp_path)

        def bye(path: Path, **kw: Any) -> int:
            raise SystemExit(3)

        monkeypatch.setattr(term_check_cli, "run_terms", bye)
        assert _hook(["a.json"], input_fn=lambda p: "y", monkeypatch=monkeypatch) == 0
        assert "warning: term check failed for a.json: SystemExit" in echo_capture

    def test_interrupt_stops_the_loop(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        echo_capture: list[str],
    ) -> None:
        isolate_home(monkeypatch, tmp_path, tmp_path)
        attempted: list[str] = []

        def cancelled(path: Path, **kw: Any) -> int:
            attempted.append(path.name)
            raise KeyboardInterrupt

        monkeypatch.setattr(term_check_cli, "run_terms", cancelled)
        assert (
            _hook(["a.json", "b.json"], input_fn=lambda p: "y", monkeypatch=monkeypatch)
            == 0
        )
        assert attempted == ["a.json"]
        assert "term check cancelled" in echo_capture


class TestBatchPresetOrdering:
    def _stub_transcribe(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def fake_transcribe(path, **kwargs):
            return {
                "text": "moikka",
                "segments": [],
                "notes": {"title": "T"},
                "paragraphs": _two_label_paragraphs(),
            }

        monkeypatch.setattr(pipeline_module, "transcribe_file", fake_transcribe)

    def _stub_hooks(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, order: list[str]
    ) -> None:
        import vemoizer.batch_preset as batch_preset
        import vemoizer.preset_plain_loop as plain_loop

        self._stub_transcribe(monkeypatch)
        for module in (batch_preset, plain_loop):
            monkeypatch.setattr(
                module,
                "ask_naming_hook",
                lambda *a, **k: order.append("naming") or 0,
            )
            monkeypatch.setattr(
                module,
                "ask_terms_hook",
                lambda *a, **k: order.append("terms") or 0,
            )
        isolate_home(monkeypatch, tmp_path, tmp_path)
        touch_files(["a.m4a"], tmp_path)

    def test_terms_hook_runs_right_after_naming(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        order: list[str] = []
        self._stub_hooks(tmp_path, monkeypatch, order)
        result = runner.invoke(app, ["meeting", "a.m4a"], input="y\n")
        assert result.exit_code == 0, result.output
        assert order == ["naming", "terms"]

    def test_memo_never_hooks(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        order: list[str] = []
        self._stub_hooks(tmp_path, monkeypatch, order)
        result = runner.invoke(app, ["memo", "a.m4a"], input="y\n")
        assert result.exit_code == 0, result.output
        assert order == []

    def test_group_path_terms_run_right_after_naming(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The batch_preset group call site: naming first, terms second."""
        import vemoizer.batch as batch
        import vemoizer.grouping as grouping

        order: list[str] = []
        self._stub_hooks(tmp_path, monkeypatch, order)
        touch_files(["b.m4a"], tmp_path)
        monkeypatch.setattr(
            grouping,
            "decode_boundaries",
            lambda files, transcribe_fn=None, **kw: (
                ["ja tässä ollaan nyt siinä vaiheessa missä"],
                ["jatketaan seuraavaksi kohdassa"],
            ),
        )
        monkeypatch.setattr(grouping, "concat_groups", lambda files, **kw: files[0])
        monkeypatch.setattr(grouping, "part_offsets", lambda files, **kw: [])
        monkeypatch.setattr(batch, "concat_groups", lambda files, **kw: files[0])
        monkeypatch.setattr(batch, "part_offsets", lambda files, **kw: [])
        result = runner.invoke(app, ["meeting", "a.m4a", "b.m4a", "--yes"])
        assert result.exit_code == 0, result.output
        assert order[-2:] == ["naming", "terms"]

    def test_meeting_yes_narrates_once_via_naming_only(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Real hooks, --yes: one naming line; the terms hook prints nothing."""

        def fake_transcribe(path, **kwargs):
            return {
                "text": "moikka",
                "segments": [],
                "notes": {"title": "T"},
                "paragraphs": _two_label_paragraphs(),
            }

        monkeypatch.setattr(pipeline_module, "transcribe_file", fake_transcribe)
        isolate_home(monkeypatch, tmp_path, tmp_path)
        touch_files(["a.m4a"], tmp_path)
        result = runner.invoke(app, ["meeting", "a.m4a", "--yes"], input="y\n")
        assert result.exit_code == 0, result.output
        assert result.stdout.count("skipping speaker naming: not requested") == 1
        assert "skipping term check" not in result.stdout
