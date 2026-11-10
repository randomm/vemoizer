"""Issue #268 rework: the interactive hear-and-fix term check.

Offline: HOME and CWD are isolated, extraction and playback are stubbed —
these tests NEVER play audio.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pytest
from _cli_helpers import isolate_home
from typer.testing import CliRunner

import vemoizer.term_check_cli as tcli
from vemoizer.cli import app
from vemoizer.speaker_clips import ClipWindow

runner = CliRunner()


class _Inputs:
    """A scripted stdin; ``None`` in the script raises EOFError."""

    def __init__(self, *values: str | None) -> None:
        self._values: list[str | None] = list(values)
        self.prompts: list[str] = []

    def __call__(self, prompt: str) -> str:
        if not self._values:
            raise EOFError
        value = self._values.pop(0)
        if value is None:
            raise EOFError
        self.prompts.append(prompt)
        return value


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Isolated HOME/CWD, captured echoes, recorded plays and extractions."""
    isolate_home(monkeypatch, tmp_path, tmp_path)
    echoes: list[str] = []

    def echo(msg: Any = "", *, err: bool = False, **kw: Any) -> None:
        echoes.append(str(msg))

    monkeypatch.setattr(tcli.typer, "echo", echo)

    def fake_extract(source, windows, tmp):
        out: dict[ClipWindow, Path | None] = {}
        for i, w in enumerate(windows):
            p = Path(tmp) / f"clip_{i}.wav"
            p.write_bytes(b"fake")
            out[w] = p
        return out

    plays: list[Path] = []
    return {
        "cwd": tmp_path,
        "echoes": echoes,
        "plays": plays,
        "extract_fn": fake_extract,
        "play_fn": plays.append,
    }


def _sidecar(tmp_path: Path, **over: Any) -> Path:
    word = over.get("word", "APS")
    data: dict[str, Any] = {
        "text": f"katsotaan {word} kokous",
        "paragraphs": [
            {
                "start": 0.0,
                "end": 5.0,
                "text": f"katsotaan {word} kokous",
                "speaker": "SPEAKER_1",
            }
        ],
        "words": [
            {"word": "katsotaan", "start": 0.0, "end": 0.4, "p": 0.9},
            {"word": word, "start": 2.0, "end": 2.4, "p": over.get("p", 0.2)},
            {"word": "kokous", "start": 2.6, "end": 3.0, "p": 0.9},
        ],
        "source": [{"path": "call.wav", "part_offset_s": 0.0, "duration_s": 10.0}],
    }
    data.update({k: v for k, v in over.items() if k not in ("word", "p")})
    (tmp_path / "call.wav").write_bytes(b"RIFF")
    path = tmp_path / "run.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def _run(env: dict[str, Any], path: Path, inputs: _Inputs, **kw: Any) -> int:
    kw.setdefault("extract_fn", env["extract_fn"])
    kw.setdefault("play_fn", env["play_fn"])
    return tcli.run_terms(
        path,
        input_fn=inputs,
        tty_isatty=lambda: True,
        **kw,
    )


class TestListMode:
    def test_lists_least_sure_first_with_counts(self, env: dict[str, Any]) -> None:
        path = _sidecar(env["cwd"])
        rc = _run(env, path, _Inputs(""))
        assert rc == 0
        echoes = env["echoes"]
        assert "Terms in this call, least sure first:" in echoes
        row = echoes[1]
        assert row.startswith("  1 APS")
        assert "kokous" not in " ".join(echoes)  # ordinary prose is not listed

    def test_count_shown_only_above_one(self, env: dict[str, Any]) -> None:
        path = _sidecar(env["cwd"])
        data = json.loads(path.read_text(encoding="utf-8"))
        data["words"].append({"word": "APS", "start": 3.4, "end": 3.7, "p": 0.3})
        data["paragraphs"][0]["text"] += " APS"
        path.write_text(json.dumps(data), encoding="utf-8")
        _run(env, path, _Inputs(""))
        assert "  1 APS (2\u00d7)" in env["echoes"][1]

    def test_no_candidates_message(self, env: dict[str, Any]) -> None:
        path = _sidecar(env["cwd"], words=[])
        assert _run(env, path, _Inputs("")) == 0
        assert "no terms to check in this transcript" in env["echoes"]

    def test_malformed_sidecar_exits_1(self, env: dict[str, Any]) -> None:
        path = env["cwd"] / "broken.json"
        path.write_text("{nope", encoding="utf-8")
        assert _run(env, path, _Inputs("")) == 1

    def test_non_tty_exits_2(self, env: dict[str, Any]) -> None:
        path = _sidecar(env["cwd"])
        assert tcli.run_terms(path, tty_isatty=lambda: False) == 2
        assert any("requires an interactive TTY" in e for e in env["echoes"])


class TestHearAndFix:
    def test_pick_prints_context_line_and_plays(self, env: dict[str, Any]) -> None:
        path = _sidecar(env["cwd"], speaker_names={"SPEAKER_1": "Matti"})
        rc = _run(env, path, _Inputs("1", "-", ""))
        assert rc == 0
        line = next(e for e in env["echoes"] if "[00:00:02]" in e)
        assert "Matti" in line and "[APS]" in line
        assert len(env["plays"]) == 1

    def test_correction_saves_pair_and_term_and_rerenders(
        self, env: dict[str, Any]
    ) -> None:
        path = _sidecar(env["cwd"])
        rc = _run(env, path, _Inputs("1", "AVS", "", "", "y"))
        assert rc == 0
        glossary = env["cwd"] / ".vemoizer" / "glossary.txt"
        assert glossary.read_text(encoding="utf-8") == "APS => AVS\nAVS\n"
        md = (env["cwd"] / "run.md").read_text(encoding="utf-8")
        assert "AVS" in md and "APS" not in md
        assert "wrote run.md" in env["echoes"]

    def test_enter_confirms_bare_term_only(self, env: dict[str, Any]) -> None:
        path = _sidecar(env["cwd"])
        rc = _run(env, path, _Inputs("1", "", "", "y"))
        assert rc == 0
        glossary = env["cwd"] / ".vemoizer" / "glossary.txt"
        assert glossary.read_text(encoding="utf-8") == "APS\n"
        assert "=>" not in glossary.read_text(encoding="utf-8")

    def test_surface_case_suffix_saves_the_stem_and_rerenders(
        self, env: dict[str, Any]
    ) -> None:
        """``AVS:n`` corrects to AWS as ``AVS => AWS``; the render keeps the
        suffix: the .md shows ``AWS:n``."""
        path = _sidecar(env["cwd"], word="AVS:n")
        rc = _run(env, path, _Inputs("1", "AWS", "", "", "y"))
        assert rc == 0
        glossary = env["cwd"] / ".vemoizer" / "glossary.txt"
        assert glossary.read_text(encoding="utf-8") == "AVS => AWS\nAWS\n"
        md = (env["cwd"] / "run.md").read_text(encoding="utf-8")
        assert "AWS:n" in md and "AVS" not in md

    def test_replay_calls_play_twice(self, env: dict[str, Any]) -> None:
        path = _sidecar(env["cwd"])
        _run(env, path, _Inputs("1", "r", "", "", "n"))
        assert len(env["plays"]) == 2

    def test_dash_skips_without_an_answer(self, env: dict[str, Any]) -> None:
        path = _sidecar(env["cwd"])
        rc = _run(env, path, _Inputs("1", "-", ""))
        assert rc == 0
        assert not (env["cwd"] / ".vemoizer").exists()
        assert not any("Save to" in e for e in env["echoes"])

    def test_invalid_number_reprompts(self, env: dict[str, Any]) -> None:
        path = _sidecar(env["cwd"])
        inputs = _Inputs("9", "abc", "")
        assert _run(env, path, inputs) == 0
        assert (
            env["echoes"].count("type a number from the list, or Enter when done") == 2
        )
        assert inputs.prompts.count(tcli._PICK_PROMPT) == 3

    def test_revisit_last_answer_wins(self, env: dict[str, Any]) -> None:
        path = _sidecar(env["cwd"])
        _run(env, path, _Inputs("1", "X", "y", "1", "", "", "y"))
        glossary = env["cwd"] / ".vemoizer" / "glossary.txt"
        assert glossary.read_text(encoding="utf-8") == "APS\n"

    def test_no_play_skips_extraction(self, env: dict[str, Any]) -> None:
        path = _sidecar(env["cwd"])
        _run(env, path, _Inputs("1", "", "", "n"), no_play=True)
        assert env["plays"] == []
        assert any("[APS]" in e for e in env["echoes"])

    def test_missing_source_is_the_line_without_play(self, env: dict[str, Any]) -> None:
        path = _sidecar(env["cwd"], source=[])
        rc = _run(env, path, _Inputs("1", "", "", "n"))
        assert rc == 0
        assert env["plays"] == []
        assert not any("error" in e.lower() for e in env["echoes"])

    def test_r_without_a_clip_reprompts_and_is_never_a_spelling(
        self, env: dict[str, Any]
    ) -> None:
        """--no-play: ``r`` re-prompts (message once) and cannot save "r"."""
        path = _sidecar(env["cwd"])
        rc = _run(env, path, _Inputs("1", "r", "r", "AVS", "", "", "y"), no_play=True)
        assert rc == 0
        assert env["echoes"].count("  no clip to replay") == 1
        glossary = env["cwd"] / ".vemoizer" / "glossary.txt"
        assert glossary.read_text(encoding="utf-8") == "APS => AVS\nAVS\n"

    def test_r_without_a_clip_and_dash_saves_nothing(self, env: dict[str, Any]) -> None:
        path = _sidecar(env["cwd"], source=[])
        rc = _run(env, path, _Inputs("1", "r", "-", ""))
        assert rc == 0
        assert env["echoes"].count("  no clip to replay") == 1
        assert not any("=> r" in e for e in env["echoes"])
        assert not (env["cwd"] / ".vemoizer").exists()


class TestSaveAndAbort:
    def test_ctrl_d_keeps_answers_and_saves(self, env: dict[str, Any]) -> None:
        path = _sidecar(env["cwd"])
        # pick, fix, keep, then Ctrl-D at the pick prompt; "y" answers the save.
        rc = _run(env, path, _Inputs("1", "AVS", "", None, "y"))
        assert rc == 0
        glossary = env["cwd"] / ".vemoizer" / "glossary.txt"
        assert glossary.read_text(encoding="utf-8") == "APS => AVS\nAVS\n"

    def test_ctrl_c_at_prompt_keeps_answers(self, env: dict[str, Any]) -> None:
        path = _sidecar(env["cwd"])

        left: list[str] = ["1", ""]

        def inputs(prompt: str) -> str:
            # scripted: first round answers, then a Ctrl-C at the pick prompt.
            if not left:
                raise KeyboardInterrupt
            return left.pop(0)

        assert (
            tcli.run_terms(
                path,
                input_fn=inputs,
                tty_isatty=lambda: True,
                extract_fn=env["extract_fn"],
                play_fn=env["play_fn"],
            )
            == 0
        )
        assert any(e.startswith("Save to") for e in env["echoes"])

    def test_ctrl_c_during_playback_keeps_answers_and_saves(
        self, env: dict[str, Any]
    ) -> None:
        """Ctrl-C while playing (not at a prompt) acts like Ctrl-D at one."""
        path = _sidecar(env["cwd"])
        plays: list[Path] = []

        def play_fn(p: Path) -> None:
            plays.append(p)
            if len(plays) == 2:
                raise KeyboardInterrupt

        # pick+fix+keep, then a second pick whose playback is interrupted;
        # "y" saves.
        rc = _run(env, path, _Inputs("1", "AVS", "y", "1", "y"), play_fn=play_fn)
        assert rc == 0
        assert len(plays) == 2
        glossary = env["cwd"] / ".vemoizer" / "glossary.txt"
        assert glossary.read_text(encoding="utf-8") == "APS => AVS\nAVS\n"

    def test_ctrl_c_during_extraction_keeps_answers_and_saves(
        self, env: dict[str, Any]
    ) -> None:
        path = _sidecar(env["cwd"])
        calls: list[int] = []
        real_extract = env["extract_fn"]

        def extract_fn(source, windows, tmp):
            calls.append(1)
            if len(calls) == 2:
                raise KeyboardInterrupt
            return real_extract(source, windows, tmp)

        rc = _run(env, path, _Inputs("1", "AVS", "y", "1", "y"), extract_fn=extract_fn)
        assert rc == 0
        assert len(calls) == 2
        glossary = env["cwd"] / ".vemoizer" / "glossary.txt"
        assert glossary.read_text(encoding="utf-8") == "APS => AVS\nAVS\n"

    def test_home_folder_is_never_written(
        self, env: dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """HOME == cwd, no project glossary: refuse, no write, no re-render."""
        monkeypatch.setenv("HOME", str(env["cwd"]))
        path = _sidecar(env["cwd"])
        rc = _run(env, path, _Inputs("1", "AVS", "", "", "y"))
        assert rc == 0
        assert (
            "not saved: run this from the meeting's folder, not your home folder"
            in env["echoes"]
        )
        assert not (env["cwd"] / ".vemoizer").exists()
        assert not (env["cwd"] / "run.md").exists()

    def test_n_at_save_writes_nothing(self, env: dict[str, Any]) -> None:
        path = _sidecar(env["cwd"])
        assert _run(env, path, _Inputs("1", "AVS", "", "", "n")) == 0
        assert "not saved" in env["echoes"]
        assert not (env["cwd"] / ".vemoizer").exists()
        assert not (env["cwd"] / "run.md").exists()

    def test_save_prompt_lists_the_lines(self, env: dict[str, Any]) -> None:
        path = _sidecar(env["cwd"])
        _run(env, path, _Inputs("1", "AVS", "", "", "y"))
        i = env["echoes"].index("Save to .vemoizer/glossary.txt?")
        assert env["echoes"][i + 1 : i + 3] == ["  APS => AVS", "  AVS"]

    def test_counts_only_log(
        self, env: dict[str, Any], caplog: pytest.LogCaptureFixture
    ) -> None:
        path = _sidecar(env["cwd"])
        with caplog.at_level(logging.INFO, logger="vemoizer.term_check_cli"):
            _run(env, path, _Inputs("1", "AVS", "", "", "y"))
        line = next(r for r in caplog.records if "term check:" in r.getMessage())
        assert line.getMessage() == (
            "term check: listed 1, heard 1, corrected 1, confirmed 0, call-only 0"
        )
        assert "APS" not in line.getMessage()


class TestCommand:
    def test_terms_command_registered_and_non_tty_exit_2(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        isolate_home(monkeypatch, tmp_path, tmp_path)
        path = tmp_path / "run.json"
        path.write_text(json.dumps({"text": "joo"}), encoding="utf-8")
        result = runner.invoke(app, ["terms", str(path)])
        assert result.exit_code == 2


def _two_term_sidecar(tmp_path: Path) -> Path:
    """The fixture sidecar plus a second term-shaped word: pick 1 is APS
    (score 1.2), pick 2 is KXQ (score 1.05) — both acronyms, so deterministic."""
    path = _sidecar(tmp_path)
    data = json.loads(path.read_text(encoding="utf-8"))
    data["words"].append({"word": "KXQ", "start": 3.5, "end": 3.8, "p": 0.3})
    data["paragraphs"][0]["text"] += " KXQ"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


class TestCallOnlyFixes:
    """Issue #270: a typed fix can be kept for THIS call only (sidecar
    ``call_corrections``) instead of the folder glossary."""

    def test_n_at_keep_prompt_saves_call_only_not_glossary(
        self, env: dict[str, Any]
    ) -> None:
        path = _sidecar(env["cwd"])
        rc = _run(env, path, _Inputs("1", "AVS", "n", "", "y"))
        assert rc == 0
        assert not (env["cwd"] / ".vemoizer").exists()
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["call_corrections"] == {"APS": "AVS"}
        md = (env["cwd"] / "run.md").read_text(encoding="utf-8")
        assert "AVS" in md and "APS" not in md
        # No glossary lines: the header degrades to "Save?" and only the
        # call-only group prints.
        assert "Save?" in env["echoes"]
        i = env["echoes"].index("This call only:")
        assert env["echoes"][i + 1] == "  APS => AVS"

    def test_y_at_keep_prompt_keeps_glossary_behaviour(
        self, env: dict[str, Any]
    ) -> None:
        path = _sidecar(env["cwd"])
        rc = _run(env, path, _Inputs("1", "AVS", "y", "", "y"))
        assert rc == 0
        glossary = env["cwd"] / ".vemoizer" / "glossary.txt"
        assert glossary.read_text(encoding="utf-8") == "APS => AVS\nAVS\n"
        assert "call_corrections" not in json.loads(path.read_text(encoding="utf-8"))

    def test_mixed_glossary_and_call_only_in_one_session(
        self, env: dict[str, Any]
    ) -> None:
        path = _two_term_sidecar(env["cwd"])
        rc = _run(env, path, _Inputs("1", "AVS", "y", "2", "KXW", "n", "", "y"))
        assert rc == 0
        glossary = env["cwd"] / ".vemoizer" / "glossary.txt"
        assert glossary.read_text(encoding="utf-8") == "APS => AVS\nAVS\n"
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["call_corrections"] == {"KXQ": "KXW"}
        md = (env["cwd"] / "run.md").read_text(encoding="utf-8")
        assert "AVS" in md and "APS" not in md
        assert "KXW" in md and "KXQ" not in md
        i = env["echoes"].index("Save to .vemoizer/glossary.txt?")
        assert env["echoes"][i + 3 : i + 5] == ["This call only:", "  KXQ => KXW"]

    def test_repick_call_only_then_glossary_last_answer_wins(
        self, env: dict[str, Any]
    ) -> None:
        path = _sidecar(env["cwd"])
        _run(env, path, _Inputs("1", "AVS", "n", "1", "AWS", "y", "", "y"))
        glossary = env["cwd"] / ".vemoizer" / "glossary.txt"
        assert glossary.read_text(encoding="utf-8") == "APS => AWS\nAWS\n"
        data = json.loads(path.read_text(encoding="utf-8"))
        assert "call_corrections" not in data
        md = (env["cwd"] / "run.md").read_text(encoding="utf-8")
        assert "AWS" in md and "AVS" not in md and "APS" not in md

    def test_existing_call_corrections_merge_and_overwrite(
        self, env: dict[str, Any]
    ) -> None:
        path = _sidecar(env["cwd"])
        data = json.loads(path.read_text(encoding="utf-8"))
        data["call_corrections"] = {"APS": "VANHA", "XOV": "Y"}
        path.write_text(json.dumps(data), encoding="utf-8")
        _run(env, path, _Inputs("1", "AVS", "no", "", "y"))
        saved = json.loads(path.read_text(encoding="utf-8"))["call_corrections"]
        assert saved == {"APS": "AVS", "XOV": "Y"}

    def test_abort_at_keep_prompt_drops_that_word_keeps_earlier(
        self, env: dict[str, Any]
    ) -> None:
        path = _two_term_sidecar(env["cwd"])
        # APS is kept for later calls; the KXQ follow-up gets Ctrl-D.
        rc = _run(env, path, _Inputs("1", "AVS", "y", "2", "KXW", None, "y"))
        assert rc == 0
        glossary = env["cwd"] / ".vemoizer" / "glossary.txt"
        assert glossary.read_text(encoding="utf-8") == "APS => AVS\nAVS\n"
        assert "call_corrections" not in json.loads(path.read_text(encoding="utf-8"))
        assert any(e.startswith("Save to") for e in env["echoes"])

    def test_n_at_save_confirmation_leaves_sidecar_unchanged(
        self, env: dict[str, Any]
    ) -> None:
        path = _sidecar(env["cwd"])
        before = path.read_text(encoding="utf-8")
        assert _run(env, path, _Inputs("1", "AVS", "n", "", "n")) == 0
        assert path.read_text(encoding="utf-8") == before
        assert not (env["cwd"] / ".vemoizer").exists()
        assert not (env["cwd"] / "run.md").exists()

    def test_home_refusal_saves_no_call_corrections_either(
        self, env: dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("HOME", str(env["cwd"]))
        path = _two_term_sidecar(env["cwd"])
        before = path.read_text(encoding="utf-8")
        rc = _run(env, path, _Inputs("1", "AVS", "y", "2", "KXW", "n", "", "y"))
        assert rc == 0
        assert (
            "not saved: run this from the meeting's folder, not your home folder"
            in env["echoes"]
        )
        assert not (env["cwd"] / ".vemoizer").exists()
        assert path.read_text(encoding="utf-8") == before
        assert not (env["cwd"] / "run.md").exists()

    def test_log_line_counts_call_only_without_the_words(
        self, env: dict[str, Any], caplog: pytest.LogCaptureFixture
    ) -> None:
        path = _sidecar(env["cwd"])
        with caplog.at_level(logging.INFO, logger="vemoizer.term_check_cli"):
            _run(env, path, _Inputs("1", "AVS", "n", "", "y"))
        line = next(r for r in caplog.records if "term check:" in r.getMessage())
        assert line.getMessage().endswith("call-only 1")
        assert "APS" not in line.getMessage() and "AVS" not in line.getMessage()

    def test_ei_at_keep_prompt_stores_a_call_only_fix(
        self, env: dict[str, Any]
    ) -> None:
        """Finnish ``ei`` is "this call only", same as ``n``/``no``."""
        path = _sidecar(env["cwd"])
        rc = _run(env, path, _Inputs("1", "AVS", "ei", "", "y"))
        assert rc == 0
        assert not (env["cwd"] / ".vemoizer").exists()
        assert json.loads(path.read_text(encoding="utf-8"))["call_corrections"] == {
            "APS": "AVS"
        }
        assert "AVS" in (env["cwd"] / "run.md").read_text(encoding="utf-8")

    def test_invalid_keep_reply_reasks_the_same_prompt(
        self, env: dict[str, Any]
    ) -> None:
        """Only the known replies are accepted; ``KYLLÄ`` is the glossary one."""
        path = _sidecar(env["cwd"])
        inputs = _Inputs("1", "AVS", "maybe", "KYLLÄ", "", "y")
        rc = _run(env, path, inputs)
        assert rc == 0
        assert inputs.prompts.count(tcli._KEEP_PROMPT) == 2
        glossary = env["cwd"] / ".vemoizer" / "glossary.txt"
        assert glossary.read_text(encoding="utf-8") == "APS => AVS\nAVS\n"
        assert "call_corrections" not in json.loads(path.read_text(encoding="utf-8"))

    def test_confirm_in_a_later_session_removes_the_stored_call_fix(
        self, env: dict[str, Any]
    ) -> None:
        """An Enter on a word the sidecar still carries a fix for retires the
        fix: the map key goes and the .md shows the original word again."""
        path = _sidecar(env["cwd"])
        data = json.loads(path.read_text(encoding="utf-8"))
        data["call_corrections"] = {"APS": "AVS"}
        path.write_text(json.dumps(data), encoding="utf-8")
        rc = _run(env, path, _Inputs("1", "", "", "y"))
        assert rc == 0
        assert "call_corrections" not in json.loads(path.read_text(encoding="utf-8"))
        md = (env["cwd"] / "run.md").read_text(encoding="utf-8")
        assert "APS" in md and "AVS" not in md

    def test_glossary_answer_in_a_later_session_replaces_the_stored_call_fix(
        self, env: dict[str, Any]
    ) -> None:
        path = _sidecar(env["cwd"])
        data = json.loads(path.read_text(encoding="utf-8"))
        data["call_corrections"] = {"APS": "VANHA"}
        path.write_text(json.dumps(data), encoding="utf-8")
        rc = _run(env, path, _Inputs("1", "AVS", "y", "", "y"))
        assert rc == 0
        assert "call_corrections" not in json.loads(path.read_text(encoding="utf-8"))
        glossary = env["cwd"] / ".vemoizer" / "glossary.txt"
        assert glossary.read_text(encoding="utf-8") == "APS => AVS\nAVS\n"
        md = (env["cwd"] / "run.md").read_text(encoding="utf-8")
        assert "AVS" in md and "VANHA" not in md and "APS" not in md

    def test_call_only_save_from_the_home_folder_writes_the_sidecar(
        self, env: dict[str, Any], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """HOME == cwd refuses the HOME GLOSSARY only: a fix kept for this
        call goes into the sidecar anyway (issue #270 review)."""
        monkeypatch.setenv("HOME", str(env["cwd"]))
        path = _sidecar(env["cwd"])
        rc = _run(env, path, _Inputs("1", "AVS", "n", "", "y"))
        assert rc == 0
        assert not any("home folder" in e for e in env["echoes"])
        assert not (env["cwd"] / ".vemoizer").exists()
        assert json.loads(path.read_text(encoding="utf-8"))["call_corrections"] == {
            "APS": "AVS"
        }
        md = (env["cwd"] / "run.md").read_text(encoding="utf-8")
        assert "AVS" in md and "APS" not in md
