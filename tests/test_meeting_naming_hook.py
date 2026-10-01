"""Tests for the end-of-meeting "Name the speakers now?" hook (issue #95, M5c-2).

Two layers of coverage:

1. **Unit tests** for ``ask_naming_hook`` directly — covers all TTY
   branches, yes/no/EOF/Ctrl-C at the prompt, per-sidecar dispatch,
   failure injection, and the exit-code-invariant.

2. **Integration tests** through ``run_preset`` (via the CLI or direct
   call) — covers the command guard (memo vs meeting), the --yes
   no-op, the partial-pair skip, the failed-transcription skip, and
   the exit-code invariant at the call-site level.

All tests isolate HOME + chdir into tmp_path (isolate_home) so the
session guard in conftest.py is never tripped.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from _cli_helpers import isolate_home, touch_files
from typer.testing import CliRunner

import vemoizer.batch as batch
import vemoizer.grouping as grouping
import vemoizer.names_cli as names_cli
import vemoizer.naming_hook as naming_hook
import vemoizer.pipeline as pipeline_module
from vemoizer.cli import app

runner = CliRunner()


# --- Helpers -----------------------------------------------------------------


def _two_label_paragraphs() -> list[dict[str, Any]]:
    return [
        {"start": 0.0, "end": 10.0, "text": "Moikka.", "speaker": "SPEAKER_1"},
        {"start": 12.0, "end": 20.0, "text": "Kyll\u00e4.", "speaker": "SPEAKER_2"},
    ]


def _one_label_paragraphs() -> list[dict[str, Any]]:
    return [
        {"start": 0.0, "end": 10.0, "text": "Moikka.", "speaker": "SPEAKER_1"},
    ]


def _no_label_paragraphs() -> list[dict[str, Any]]:
    return [
        {"start": 0.0, "end": 10.0, "text": "Moikka."},
    ]


def _write_sidecar(tmp_path: Path, name: str, paragraphs: list[dict]) -> str:
    """Write a minimal sidecar with *paragraphs*; return the file name."""
    data = {
        "text": "moikka",
        "paragraphs": paragraphs,
        "notes": {"title": name.rstrip(".json")},
        "options": {
            "command": "meeting",
            "glossary_files": [],
            "glossary_sha256": None,
        },
        "speaker_names": {},
    }
    path = tmp_path / name
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return name


def _stub_run_names(monkeypatch: pytest.MonkeyPatch, return_val: int = 0) -> list:
    """Monkeypatch vemoizer.names_cli.run_names with a recorder.

    Returns a list of (path, kwargs) tuples for each call.
    """
    calls: list = []

    def fake_run_names(
        path, *, no_play=False, input_fn=None, tty_isatty=None, config_path=None
    ):
        calls.append((path, {"input_fn": input_fn, "tty_isatty": tty_isatty}))
        return return_val

    monkeypatch.setattr(names_cli, "run_names", fake_run_names)
    return calls


def _stub_run_names_raise(
    monkeypatch: pytest.MonkeyPatch, exc: type, on_n: int = 1
) -> list:
    """Stub run_names to raise *exc* on its *on_n*-th call; record calls."""
    calls: list = []

    def fake_run_names(
        path, *, no_play=False, input_fn=None, tty_isatty=None, config_path=None
    ):
        calls.append(path)
        if len(calls) == on_n:
            raise exc("test failure")
        return 0

    monkeypatch.setattr(names_cli, "run_names", fake_run_names)
    return calls


def _fake_transcribe_rich(
    monkeypatch: pytest.MonkeyPatch, paragraphs: list[dict]
) -> None:
    """Patch pipeline.transcribe_file to return a result with *paragraphs*."""
    result = {
        "text": "moikka",
        "segments": [],
        "notes": {"title": "T"},
        "paragraphs": paragraphs,
    }
    monkeypatch.setattr(pipeline_module, "transcribe_file", lambda *a, **kw: result)


def _fake_group_seams(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch grouping seams so 2 files become ONE multi-part group."""

    def fake_decode_boundaries(files, transcribe_fn=None):
        return [
            "ja t\u00e4ss\u00e4 ollaan nyt siin\u00e4 vaiheessa miss\u00e4",
            "t\u00e4ss\u00e4 jatketaan",
        ]

    monkeypatch.setattr(grouping, "decode_boundaries", fake_decode_boundaries)
    monkeypatch.setattr(grouping, "concat_groups", lambda files: files[0])
    monkeypatch.setattr(grouping, "part_offsets", lambda files: [])
    monkeypatch.setattr(batch, "concat_groups", lambda files: files[0])
    monkeypatch.setattr(batch, "part_offsets", lambda files: [])


# --- Unit tests for ask_naming_hook ------------------------------------------


class TestHookYesNoOp:
    """--yes: the hook is a complete no-op (no prompt, no run_names)."""

    def test_yes_skips_entirely(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        isolate_home(monkeypatch, tmp_path, tmp_path)
        _write_sidecar(tmp_path, "2025-01-01 T.json", _two_label_paragraphs())
        calls = _stub_run_names(monkeypatch)

        def input_fn(p: str) -> str:
            pytest.fail("input_fn must not be called under --yes")

        rc = naming_hook.ask_naming_hook(
            ["2025-01-01 T.json"], yes=True, input_fn=input_fn, tty_isatty=lambda: True
        )
        assert rc == 0
        assert len(calls) == 0


class TestHookNonTTYStdin:
    def test_non_tty_stdin_skips(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        isolate_home(monkeypatch, tmp_path, tmp_path)
        _write_sidecar(tmp_path, "2025-01-01 T.json", _two_label_paragraphs())
        calls = _stub_run_names(monkeypatch)

        def input_fn(p: str) -> str:
            pytest.fail("input_fn must not be called when stdin is non-TTY")

        rc = naming_hook.ask_naming_hook(
            ["2025-01-01 T.json"],
            yes=False,
            input_fn=input_fn,
            tty_isatty=lambda: False,
        )
        assert rc == 0
        assert len(calls) == 0


class TestHookNonTTYStdout:
    def test_non_tty_stdout_skips(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        isolate_home(monkeypatch, tmp_path, tmp_path)
        _write_sidecar(tmp_path, "2025-01-01 T.json", _two_label_paragraphs())
        calls = _stub_run_names(monkeypatch)
        monkeypatch.setattr(naming_hook, "_stdout_isatty", lambda: False)

        def input_fn(p: str) -> str:
            pytest.fail("input_fn must not be called when stdout is non-TTY")

        rc = naming_hook.ask_naming_hook(
            ["2025-01-01 T.json"], yes=False, input_fn=input_fn, tty_isatty=lambda: True
        )
        assert rc == 0
        assert len(calls) == 0


class TestHookNoEligibleSidecars:
    def test_no_json_in_written(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        """Written list has only a .md (partial pair): no eligible sidecar."""
        isolate_home(monkeypatch, tmp_path, tmp_path)
        calls = _stub_run_names(monkeypatch)

        def input_fn(p: str) -> str:
            pytest.fail("input_fn must not be called when no .json is written")

        rc = naming_hook.ask_naming_hook(
            ["2025-01-01 T.md"], yes=False, input_fn=input_fn, tty_isatty=lambda: True
        )
        assert rc == 0
        assert len(calls) == 0

    def test_missing_json_file(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        """Written list has a .json name but the file doesn't exist: skipped."""
        isolate_home(monkeypatch, tmp_path, tmp_path)
        calls = _stub_run_names(monkeypatch)

        def input_fn(p: str) -> str:
            pytest.fail("input_fn must not be called when the sidecar is missing")

        rc = naming_hook.ask_naming_hook(
            ["2025-01-01 Missing.json"],
            yes=False,
            input_fn=input_fn,
            tty_isatty=lambda: True,
        )
        assert rc == 0
        assert len(calls) == 0

    def test_malformed_json(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        """The .json file exists but is not valid JSON: skipped."""
        isolate_home(monkeypatch, tmp_path, tmp_path)
        (tmp_path / "bad.json").write_text("not json", encoding="utf-8")
        calls = _stub_run_names(monkeypatch)

        def input_fn(p: str) -> str:
            pytest.fail("input_fn must not be called for malformed JSON")

        rc = naming_hook.ask_naming_hook(
            ["bad.json"], yes=False, input_fn=input_fn, tty_isatty=lambda: True
        )
        assert rc == 0
        assert len(calls) == 0

    def test_non_dict_json(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        """The .json file is valid JSON but not a dict: skipped."""
        isolate_home(monkeypatch, tmp_path, tmp_path)
        (tmp_path / "arr.json").write_text("[1, 2, 3]", encoding="utf-8")
        calls = _stub_run_names(monkeypatch)

        def input_fn(p: str) -> str:
            pytest.fail("input_fn must not be called for non-dict JSON")

        rc = naming_hook.ask_naming_hook(
            ["arr.json"], yes=False, input_fn=input_fn, tty_isatty=lambda: True
        )
        assert rc == 0
        assert len(calls) == 0

    def test_paragraphs_not_a_list(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """The sidecar's 'paragraphs' key is not a list: skipped."""
        isolate_home(monkeypatch, tmp_path, tmp_path)
        data = {"text": "x", "paragraphs": "not a list"}
        (tmp_path / "bad_para.json").write_text(json.dumps(data), encoding="utf-8")
        calls = _stub_run_names(monkeypatch)

        def input_fn(p: str) -> str:
            pytest.fail("input_fn must not be called when paragraphs is not a list")

        rc = naming_hook.ask_naming_hook(
            ["bad_para.json"], yes=False, input_fn=input_fn, tty_isatty=lambda: True
        )
        assert rc == 0
        assert len(calls) == 0

    def test_one_label_no_prompt(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        """Sidecar with only one labelled speaker: no prompt."""
        isolate_home(monkeypatch, tmp_path, tmp_path)
        _write_sidecar(tmp_path, "one.json", _one_label_paragraphs())
        calls = _stub_run_names(monkeypatch)

        def input_fn(p: str) -> str:
            pytest.fail("input_fn must not be called for a single-label sidecar")

        rc = naming_hook.ask_naming_hook(
            ["one.json"], yes=False, input_fn=input_fn, tty_isatty=lambda: True
        )
        assert rc == 0
        assert len(calls) == 0

    def test_zero_labels_no_prompt(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """Sidecar with no labelled speakers: no prompt."""
        isolate_home(monkeypatch, tmp_path, tmp_path)
        _write_sidecar(tmp_path, "nolabels.json", _no_label_paragraphs())
        calls = _stub_run_names(monkeypatch)

        def input_fn(p: str) -> str:
            pytest.fail("input_fn must not be called for an unlabelled sidecar")

        rc = naming_hook.ask_naming_hook(
            ["nolabels.json"], yes=False, input_fn=input_fn, tty_isatty=lambda: True
        )
        assert rc == 0
        assert len(calls) == 0


class TestHookPromptAndNaming:
    def test_yes_answer_calls_run_names(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """Two eligible sidecars: both get run_names called."""
        isolate_home(monkeypatch, tmp_path, tmp_path)
        monkeypatch.setattr(naming_hook, "_stdout_isatty", lambda: True)
        _write_sidecar(tmp_path, "A.json", _two_label_paragraphs())
        _write_sidecar(tmp_path, "B.json", _two_label_paragraphs())
        calls = _stub_run_names(monkeypatch)

        input_calls: list[str] = []

        def input_fn(p: str) -> str:
            input_calls.append(p)
            return "y"

        rc = naming_hook.ask_naming_hook(
            ["A.json", "B.json"], yes=False, input_fn=input_fn, tty_isatty=lambda: True
        )
        assert rc == 0
        assert len(calls) == 2
        assert any("Name the speakers now?" in p for p in input_calls)
        # Both paths are CWD-resolved absolute paths.
        for path, _ in calls:
            assert path.is_absolute()
            assert path.name.endswith(".json")

    def test_no_answer_skips(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        """Declining ('n') → no run_names call."""
        isolate_home(monkeypatch, tmp_path, tmp_path)
        monkeypatch.setattr(naming_hook, "_stdout_isatty", lambda: True)
        _write_sidecar(tmp_path, "A.json", _two_label_paragraphs())
        calls = _stub_run_names(monkeypatch)

        def input_fn(p: str) -> str:
            return "n"

        rc = naming_hook.ask_naming_hook(
            ["A.json"], yes=False, input_fn=input_fn, tty_isatty=lambda: True
        )
        assert rc == 0
        assert len(calls) == 0

    def test_empty_answer_skips(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        """Empty answer → declined."""
        isolate_home(monkeypatch, tmp_path, tmp_path)
        monkeypatch.setattr(naming_hook, "_stdout_isatty", lambda: True)
        _write_sidecar(tmp_path, "A.json", _two_label_paragraphs())
        calls = _stub_run_names(monkeypatch)

        def input_fn(p: str) -> str:
            return ""

        rc = naming_hook.ask_naming_hook(
            ["A.json"], yes=False, input_fn=input_fn, tty_isatty=lambda: True
        )
        assert rc == 0
        assert len(calls) == 0

    def test_eof_at_prompt_skips_quietly(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """EOFError at the prompt → no run_names, no output."""
        isolate_home(monkeypatch, tmp_path, tmp_path)
        monkeypatch.setattr(naming_hook, "_stdout_isatty", lambda: True)
        _write_sidecar(tmp_path, "A.json", _two_label_paragraphs())
        calls = _stub_run_names(monkeypatch)

        def input_fn(p: str) -> str:
            raise EOFError()

        rc = naming_hook.ask_naming_hook(
            ["A.json"], yes=False, input_fn=input_fn, tty_isatty=lambda: True
        )
        assert rc == 0
        assert len(calls) == 0

    def test_ctrl_c_at_prompt_skips_quietly(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """KeyboardInterrupt at the prompt → no run_names, no output."""
        isolate_home(monkeypatch, tmp_path, tmp_path)
        monkeypatch.setattr(naming_hook, "_stdout_isatty", lambda: True)
        _write_sidecar(tmp_path, "A.json", _two_label_paragraphs())
        calls = _stub_run_names(monkeypatch)

        def input_fn(p: str) -> str:
            raise KeyboardInterrupt()

        rc = naming_hook.ask_naming_hook(
            ["A.json"], yes=False, input_fn=input_fn, tty_isatty=lambda: True
        )
        assert rc == 0
        assert len(calls) == 0

    def test_input_fn_forwarded_to_run_names(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """The hook's input_fn is forwarded to run_names."""
        isolate_home(monkeypatch, tmp_path, tmp_path)
        monkeypatch.setattr(naming_hook, "_stdout_isatty", lambda: True)
        _write_sidecar(tmp_path, "A.json", _two_label_paragraphs())

        def my_input_fn(p: str) -> str:
            return "y"

        calls = _stub_run_names(monkeypatch)
        naming_hook.ask_naming_hook(
            ["A.json"], yes=False, input_fn=my_input_fn, tty_isatty=lambda: True
        )
        assert len(calls) == 1
        _, kwargs = calls[0]
        assert kwargs["input_fn"] is my_input_fn

    def test_tty_isatty_forwarded_to_run_names(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """The hook's tty_isatty is forwarded to run_names."""
        isolate_home(monkeypatch, tmp_path, tmp_path)
        monkeypatch.setattr(naming_hook, "_stdout_isatty", lambda: True)
        _write_sidecar(tmp_path, "A.json", _two_label_paragraphs())

        def my_tty() -> bool:
            return True

        calls = _stub_run_names(monkeypatch)
        naming_hook.ask_naming_hook(
            ["A.json"], yes=False, input_fn=lambda p: "y", tty_isatty=my_tty
        )
        assert len(calls) == 1
        _, kwargs = calls[0]
        assert kwargs["tty_isatty"] is my_tty


class TestHookFailureInjection:
    def test_oserror_warns_and_continues(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """run_names raises OSError on the first of two sidecars: warning
        line, second sidecar still processed."""
        isolate_home(monkeypatch, tmp_path, tmp_path)
        monkeypatch.setattr(naming_hook, "_stdout_isatty", lambda: True)
        _write_sidecar(tmp_path, "A.json", _two_label_paragraphs())
        _write_sidecar(tmp_path, "B.json", _two_label_paragraphs())
        calls = _stub_run_names_raise(monkeypatch, OSError, on_n=1)

        naming_hook.ask_naming_hook(
            ["A.json", "B.json"],
            yes=False,
            input_fn=lambda p: "y",
            tty_isatty=lambda: True,
        )
        # Both sidecars were attempted.
        assert len(calls) == 2
        assert calls[0].name == "A.json"
        assert calls[1].name == "B.json"

    def test_systemexit_warns_and_continues(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """run_names raises SystemExit(1) on the first of two sidecars:
        warning line with SystemExit, second sidecar still processed,
        and the hook's return code is unchanged (0)."""
        isolate_home(monkeypatch, tmp_path, tmp_path)
        monkeypatch.setattr(naming_hook, "_stdout_isatty", lambda: True)
        _write_sidecar(tmp_path, "A.json", _two_label_paragraphs())
        _write_sidecar(tmp_path, "B.json", _two_label_paragraphs())

        calls: list = []

        def fake_run_names(
            path, *, no_play=False, input_fn=None, tty_isatty=None, config_path=None
        ):
            calls.append(path)
            if len(calls) == 1:
                raise SystemExit(1)
            return 0

        monkeypatch.setattr(names_cli, "run_names", fake_run_names)

        stderr_lines: list[str] = []
        monkeypatch.setattr(
            naming_hook.typer, "echo", lambda *a, **kw: stderr_lines.append(str(a[0]))
        )

        rc = naming_hook.ask_naming_hook(
            ["A.json", "B.json"],
            yes=False,
            input_fn=lambda p: "y",
            tty_isatty=lambda: True,
        )
        assert rc == 0
        assert "warning: naming failed for A.json: SystemExit" in stderr_lines
        # Both sidecars were attempted.
        assert len(calls) == 2
        assert calls[0].name == "A.json"
        assert calls[1].name == "B.json"

    def test_keyboard_interrupt_stops(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """run_names raises KeyboardInterrupt on the first of two sidecars:
        'naming cancelled', second sidecar NOT processed."""
        isolate_home(monkeypatch, tmp_path, tmp_path)
        monkeypatch.setattr(naming_hook, "_stdout_isatty", lambda: True)
        _write_sidecar(tmp_path, "A.json", _two_label_paragraphs())
        _write_sidecar(tmp_path, "B.json", _two_label_paragraphs())
        calls = _stub_run_names_raise(monkeypatch, KeyboardInterrupt, on_n=1)

        naming_hook.ask_naming_hook(
            ["A.json", "B.json"],
            yes=False,
            input_fn=lambda p: "y",
            tty_isatty=lambda: True,
        )
        # Only the first sidecar was attempted.
        assert len(calls) == 1
        assert calls[0].name == "A.json"


# --- Integration tests through run_preset ------------------------------------


class TestIntegrationCommandGuard:
    def test_memo_never_prompts(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        """memo with a two-label result: the hook is never invoked."""
        _fake_transcribe_rich(monkeypatch, _two_label_paragraphs())
        calls = _stub_run_names(monkeypatch)
        isolate_home(monkeypatch, tmp_path, tmp_path)
        touch_files(["a.m4a"], tmp_path)

        result = runner.invoke(app, ["memo", "a.m4a"], input="y\n")
        assert result.exit_code == 0
        assert len(calls) == 0
        assert "Name the speakers now?" not in result.stdout

    def test_meeting_with_yes_no_hook(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """--yes: the hook is a complete no-op."""
        _fake_transcribe_rich(monkeypatch, _two_label_paragraphs())
        calls = _stub_run_names(monkeypatch)
        isolate_home(monkeypatch, tmp_path, tmp_path)
        touch_files(["a.m4a"], tmp_path)

        result = runner.invoke(app, ["meeting", "a.m4a", "--yes"], input="y\n")
        assert result.exit_code == 0
        assert len(calls) == 0
        assert "Name the speakers now?" not in result.stdout


class TestIntegrationTranscriptionFailure:
    def test_failed_transcription_no_prompt(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """A failed transcription: exit 1, no prompt, no run_names."""

        def fake_raise(path, **kwargs):
            raise RuntimeError("boom")

        monkeypatch.setattr(pipeline_module, "transcribe_file", fake_raise)
        calls = _stub_run_names(monkeypatch)
        isolate_home(monkeypatch, tmp_path, tmp_path)
        touch_files(["a.m4a"], tmp_path)

        result = runner.invoke(app, ["meeting", "a.m4a"], input="y\n")
        assert result.exit_code == 1
        assert len(calls) == 0
        assert "Name the speakers now?" not in result.stdout


class TestIntegrationPartialPair:
    def test_partial_pair_no_json_no_prompt(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """A partial pair (only .md written, no .json): no eligible
        sidecar, no prompt. Exit code is 1 (write failure)."""
        import vemoizer.batch_preset as batch_preset
        from vemoizer.batch_output import _write_preset_output as real_write

        def partial_write(result, first_stem, out_dir, *, date_str=None):
            paths = real_write(result, first_stem, out_dir, date_str=date_str)
            return [p for p in paths if p.endswith(".md")]

        monkeypatch.setattr(batch_preset, "_write_preset_output", partial_write)
        _fake_transcribe_rich(monkeypatch, _two_label_paragraphs())
        calls = _stub_run_names(monkeypatch)
        isolate_home(monkeypatch, tmp_path, tmp_path)
        touch_files(["a.m4a"], tmp_path)

        result = runner.invoke(app, ["meeting", "a.m4a"], input="y\n")
        assert result.exit_code == 1
        assert len(calls) == 0
        assert "Name the speakers now?" not in result.stdout
