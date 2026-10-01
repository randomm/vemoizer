"""Tests for the ``vemoizer names X.json`` command (issue #93, M5c-1).

Covers the non-TTY guard, prompt loop, talk-share ordering, empty-answer
skip, --no-play quotes-only, no-labelled-paragraphs, old sidecar without
source, same-name collapse, people-config add prompt, and CLI registration.

All tests use ``isolate_home`` (chdir + HOME isolation), synthetic sidecar
JSON, and monkeypatched subprocess for ffmpeg/afplay. No real audio,
models, or network.
"""

from __future__ import annotations

import contextlib
import json
import struct
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
from _cli_helpers import isolate_home
from typer.testing import CliRunner

from vemoizer.cli import app
from vemoizer.names_cli import run_names

runner = CliRunner()


# --- Helpers ---


def _sidecar(**extra: Any) -> dict[str, Any]:
    """A minimal sidecar with two labelled speakers."""
    return {
        "text": "Puhuttiin asioista.",
        "paragraphs": [
            {"start": 0.0, "end": 10.0, "text": "Moikka.", "speaker": "SPEAKER_1"},
            {"start": 12.0, "end": 20.0, "text": "Kylla.", "speaker": "SPEAKER_2"},
        ],
        "notes": {"title": "Kokous", "summary": "Puhuttiin asioista."},
        "options": {
            "command": "meeting",
            "glossary_files": [],
            "glossary_sha256": None,
        },
        "speaker_names": {},
        **extra,
    }


def _make_wav(path: Path, seconds: float = 0.5) -> Path:
    """A tiny mono 16 kHz 16-bit PCM WAV."""
    rate = 16000
    n = int(seconds * rate)
    samples = b"".join(struct.pack("<h", 3000) for _ in range(n))
    hdr = (
        b"RIFF"
        + struct.pack("<I", 36 + len(samples) - 8)
        + b"WAVE"
        + b"fmt "
        + struct.pack("<I", 16)
        + struct.pack("<H", 1)
        + struct.pack("<H", 1)
        + struct.pack("<I", rate)
        + struct.pack("<I", rate * 2)
        + struct.pack("<H", 2)
        + struct.pack("<H", 16)
        + b"data"
        + struct.pack("<I", len(samples))
    )
    path.write_bytes(hdr + samples)
    return path


def _write_sidecar(
    tmp_path: Path, data: dict[str, Any], name: str = "sidecar.json"
) -> Path:
    path = tmp_path / name
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def _fake_ffmpeg_ok():
    def fake_run(argv, *a, **kw):
        _make_wav(Path(argv[-1]))
        return subprocess.CompletedProcess(argv, 0)

    return fake_run


def _fake_ffmpeg_fail():
    return lambda argv, *a, **kw: subprocess.CompletedProcess(argv, 1)


def _capture_stderr(fn, *args, **kwargs):
    """Run *fn* and capture its stderr; return (result, stderr_text)."""
    import io

    old_stderr = sys.stderr
    sys.stderr = io.StringIO()
    try:
        result = fn(*args, **kwargs)
    finally:
        captured = sys.stderr
        sys.stderr = old_stderr
    return result, captured.getvalue()


def _run(
    sc: Path, inputs: list[str] | None = None, no_play: bool = True, input_fn=None, **kw
) -> int:
    """Helper to run run_names with an input iterator."""
    if input_fn is None and inputs is not None:
        it = iter(inputs)

        def input_fn(p: str) -> str:
            return next(it)

    return run_names(
        sc, no_play=no_play, input_fn=input_fn, tty_isatty=lambda: True, **kw
    )


# --- Non-TTY guard ---


class TestNonTTYGuard:
    def test_non_tty_exits_2(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        """Non-TTY stdin → exit 2 with one stderr line, before any other logic."""
        isolate_home(monkeypatch, tmp_path, tmp_path)
        sc = _write_sidecar(tmp_path, _sidecar())
        rc, stderr = _capture_stderr(run_names, sc, tty_isatty=lambda: False)
        assert rc == 2 and "TTY" in stderr

    def test_non_tty_before_no_play(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """Non-TTY guard fires before --no-play processing (exit 2)."""
        isolate_home(monkeypatch, tmp_path, tmp_path)
        sc = _write_sidecar(tmp_path, _sidecar())
        assert run_names(sc, no_play=True, tty_isatty=lambda: False) == 2

    def test_non_tty_via_cli(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        """Non-TTY via CliRunner → exit 2 (CliRunner's stdin is non-TTY)."""
        isolate_home(monkeypatch, tmp_path, tmp_path)
        sc = _write_sidecar(tmp_path, _sidecar())
        result = runner.invoke(app, ["names", str(sc)])
        assert result.exit_code == 2 and "TTY" in result.stderr

    def test_non_tty_missing_sidecar(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """Non-TTY + missing sidecar → exit 2 (TTY guard fires first)."""
        isolate_home(monkeypatch, tmp_path, tmp_path)
        assert run_names(tmp_path / "nonexistent.json", tty_isatty=lambda: False) == 2

    def test_non_tty_malformed_sidecar(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """Non-TTY + malformed sidecar → exit 2 (TTY guard fires first)."""
        isolate_home(monkeypatch, tmp_path, tmp_path)
        sc = tmp_path / "bad.json"
        sc.write_text("not json", encoding="utf-8")
        assert run_names(sc, tty_isatty=lambda: False) == 2


# --- No labelled speakers ---


class TestNoLabelledSpeakers:
    def test_no_labelled_paragraphs(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """Sidecar with no labelled paragraphs → exit 0 with clear message."""
        isolate_home(monkeypatch, tmp_path, tmp_path)
        data = _sidecar()
        for p in data["paragraphs"]:
            p["speaker"] = None
        sc = _write_sidecar(tmp_path, data)
        rc, stderr = _capture_stderr(run_names, sc, tty_isatty=lambda: True)
        assert rc == 0 and "no labelled speakers" in stderr

    def test_no_paragraphs(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        """Sidecar with empty paragraphs → exit 0."""
        isolate_home(monkeypatch, tmp_path, tmp_path)
        data = _sidecar()
        data["paragraphs"] = []
        sc = _write_sidecar(tmp_path, data)
        assert run_names(sc, tty_isatty=lambda: True) == 0


# --- Malformed sidecar ---


class TestMalformedSidecar:
    def test_missing_sidecar(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        """Missing sidecar → exit 1."""
        isolate_home(monkeypatch, tmp_path, tmp_path)
        assert run_names(tmp_path / "nonexistent.json", tty_isatty=lambda: True) == 1

    def test_malformed_json(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        """Malformed JSON → exit 1."""
        isolate_home(monkeypatch, tmp_path, tmp_path)
        sc = tmp_path / "bad.json"
        sc.write_text("not json", encoding="utf-8")
        assert run_names(sc, tty_isatty=lambda: True) == 1


# --- Prompt loop ---


class TestPromptLoop:
    def test_single_label_named(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        """One label named → sidecar updated, .md written."""
        isolate_home(monkeypatch, tmp_path, tmp_path)
        data = _sidecar()
        data.pop("source", None)
        data["paragraphs"] = data["paragraphs"][:1]
        sc = _write_sidecar(tmp_path, data)
        assert _run(sc, ["Mikko", "n"]) == 0
        updated = json.loads(sc.read_text(encoding="utf-8"))
        assert updated["speaker_names"].get("SPEAKER_1") == "Mikko"

    def test_two_labels_both_named(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """Two labels both named → both persisted in one write."""
        isolate_home(monkeypatch, tmp_path, tmp_path)
        data = _sidecar()
        data.pop("source", None)
        sc = _write_sidecar(tmp_path, data)
        assert _run(sc, ["Mikko", "n", "Aino", "n"]) == 0
        updated = json.loads(sc.read_text(encoding="utf-8"))
        assert updated["speaker_names"] == {"SPEAKER_1": "Mikko", "SPEAKER_2": "Aino"}
        assert len(list(tmp_path.glob("*.md"))) == 1

    def test_empty_answer_skips_label(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """Empty answer for a label → that label not in persisted names."""
        isolate_home(monkeypatch, tmp_path, tmp_path)
        data = _sidecar()
        data.pop("source", None)
        sc = _write_sidecar(tmp_path, data)
        assert _run(sc, ["Mikko", "n", ""]) == 0
        updated = json.loads(sc.read_text(encoding="utf-8"))
        assert "SPEAKER_1" in updated["speaker_names"]
        assert "SPEAKER_2" not in updated["speaker_names"]

    def test_talk_share_ordering(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        """Labels listed in descending talk_share order."""
        isolate_home(monkeypatch, tmp_path, tmp_path)
        data = _sidecar()
        data["paragraphs"] = [
            {"start": 0.0, "end": 5.0, "text": "Hi.", "speaker": "SPEAKER_1"},
            {"start": 10.0, "end": 30.0, "text": "Longer.", "speaker": "SPEAKER_2"},
        ]
        data.pop("source", None)
        sc = _write_sidecar(tmp_path, data)
        prompts: list[str] = []

        def recording_input(prompt: str) -> str:
            if "Name for" in prompt:
                prompts.append(prompt)
            return (
                "n"
                if "Add" in prompt
                else ("MoreTalk" if "SPEAKER_2" in prompt else "LessTalk")
            )

        run_names(sc, no_play=True, input_fn=recording_input, tty_isatty=lambda: True)
        assert "SPEAKER_2" in prompts[0] and "SPEAKER_1" in prompts[1]


# --- --no-play quotes-only ---


class TestNoPlay:
    def test_no_play_quotes_only(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        """--no-play skips clip extraction and playback entirely."""
        isolate_home(monkeypatch, tmp_path, tmp_path)
        wav = _make_wav(tmp_path / "rec.wav")
        data = _sidecar()
        data["source"] = [{"path": str(wav), "part_offset_s": 0.0, "duration_s": 30.0}]
        sc = _write_sidecar(tmp_path, data)

        extract_calls = 0
        import vemoizer.speaker_clips as sc_mod

        original_extract = sc_mod.extract_clips

        def counting_extract(*a, **kw):
            nonlocal extract_calls
            extract_calls += 1
            return original_extract(*a, **kw)

        monkeypatch.setattr(sc_mod, "extract_clips", counting_extract)
        assert _run(sc, ["Name1", "n", "Name2", "n"]) == 0
        assert extract_calls == 0


# --- Old sidecar without source ---


class TestOldSidecar:
    def test_old_sidecar_no_source(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """Old sidecar without source → quotes-only with one notice line."""
        isolate_home(monkeypatch, tmp_path, tmp_path)
        sc = _write_sidecar(tmp_path, _sidecar())
        rc, stderr = _capture_stderr(
            _run, sc, ["Mikko", "n", "Aino", "n"], no_play=False
        )
        assert rc == 0 and ("unavailable" in stderr or "quotes only" in stderr)

    def test_old_sidecar_exactly_one_notice(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """At most ONE aggregate notice line per run."""
        isolate_home(monkeypatch, tmp_path, tmp_path)
        sc = _write_sidecar(tmp_path, _sidecar())
        _, stderr = _capture_stderr(
            _run, sc, ["Mikko", "n", "Aino", "n"], no_play=False
        )
        notice_lines = [
            line
            for line in stderr.strip().splitlines()
            if "unavailable" in line or "quotes only" in line
        ]
        assert len(notice_lines) <= 1


# --- Same-name collapse ---


class TestSameNameCollapse:
    def test_two_labels_same_name(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """Two labels given the same name → collapse to earliest via render merge."""
        isolate_home(monkeypatch, tmp_path, tmp_path)
        data = _sidecar()
        data.pop("source", None)
        sc = _write_sidecar(tmp_path, data)
        assert _run(sc, ["Mikko", "n", "Mikko", "n"]) == 0
        updated = json.loads(sc.read_text(encoding="utf-8"))
        assert updated["speaker_names"].get("SPEAKER_1") == "Mikko"
        assert updated["speaker_names"].get("SPEAKER_2") == "Mikko"
        md_content = list(tmp_path.glob("*.md"))[0].read_text(encoding="utf-8")
        assert "SPEAKER_2" not in md_content


# --- People config: add prompt ---


class TestPeopleConfig:
    def _setup_config(self, tmp_path: Path, content: str = "people = []\n") -> Path:
        d = tmp_path / ".vemoizer"
        d.mkdir(exist_ok=True)
        cfg = d / "config.toml"
        cfg.write_text(content, encoding="utf-8")
        return cfg

    def test_new_name_no_add(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        """New name not in people → asks to add; no/Enter leaves config unchanged."""
        isolate_home(monkeypatch, tmp_path, tmp_path)
        config = self._setup_config(tmp_path)
        data = _sidecar()
        data.pop("source", None)
        sc = _write_sidecar(tmp_path, data)
        assert _run(sc, ["Mikko", "n", "Aino", "n"]) == 0
        content = config.read_text(encoding="utf-8")
        assert "Mikko" not in content and "Aino" not in content

    def test_new_name_add_yes(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        """New name + yes → added to people list in config."""
        isolate_home(monkeypatch, tmp_path, tmp_path)
        config = self._setup_config(tmp_path)
        data = _sidecar()
        data.pop("source", None)
        sc = _write_sidecar(tmp_path, data)
        assert _run(sc, ["Mikko", "y", "Aino", "n"]) == 0
        assert "Mikko" in config.read_text(encoding="utf-8")

    def test_existing_name_no_add_prompt(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """Name already in people (case-insensitive) → no add prompt."""
        isolate_home(monkeypatch, tmp_path, tmp_path)
        self._setup_config(tmp_path, 'people = ["mikko"]\n')
        data = _sidecar()
        data.pop("source", None)
        sc = _write_sidecar(tmp_path, data)
        prompts: list[str] = []

        def recording_input(prompt: str) -> str:
            prompts.append(prompt)
            return (
                "n"
                if "Add" in prompt
                else ("Mikko" if "SPEAKER_1" in prompt else "Aino")
            )

        run_names(sc, no_play=True, input_fn=recording_input, tty_isatty=lambda: True)
        assert len([p for p in prompts if "Add" in p and "Mikko" in p]) == 0

    def test_people_list_read_fail_open(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """Missing/invalid people → empty list, no error."""
        isolate_home(monkeypatch, tmp_path, tmp_path)
        data = _sidecar()
        data.pop("source", None)
        sc = _write_sidecar(tmp_path, data)
        assert _run(sc, ["Mikko", "n", "Aino", "n"]) == 0


# --- Ctrl-C mid-prompt ---


class TestCtrlC:
    def test_ctrl_c_sidecar_byte_identical(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """Ctrl-C mid-prompt → sidecar byte-identical, no clip files remain."""
        isolate_home(monkeypatch, tmp_path, tmp_path)
        wav = _make_wav(tmp_path / "rec.wav")
        data = _sidecar()
        data["source"] = [{"path": str(wav), "part_offset_s": 0.0, "duration_s": 30.0}]
        sc = _write_sidecar(tmp_path, data)
        original_bytes = sc.read_bytes()

        call_count = 0

        def input_fn(prompt: str) -> str:
            nonlocal call_count
            call_count += 1
            if "Add" in prompt:
                return "n"
            if call_count <= 2:
                return "Mikko"
            raise KeyboardInterrupt

        monkeypatch.setattr("vemoizer.speaker_clips.subprocess.run", _fake_ffmpeg_ok())
        with contextlib.suppress(KeyboardInterrupt):
            run_names(sc, no_play=False, input_fn=input_fn, tty_isatty=lambda: True)

        assert sc.read_bytes() == original_bytes
        assert list(tmp_path.glob("clip_*.wav")) == []


# --- Persist failure ---


class TestPersistFailure:
    def test_persist_readonly_dir_returns_1(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """Read-only sidecar dir → clean error (exit 1), no traceback."""
        isolate_home(monkeypatch, tmp_path, tmp_path)
        ro = tmp_path / "ro"
        ro.mkdir()
        data = _sidecar()
        data.pop("source", None)
        data["paragraphs"] = data["paragraphs"][:1]
        sc = ro / "sidecar.json"
        sc.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        ro.chmod(0o555)
        try:
            rc, stderr = _capture_stderr(_run, sc, ["Mikko", "n"])
        finally:
            ro.chmod(0o755)
        assert rc == 1 and "could not persist speaker names" in stderr

    def test_md_write_readonly_dir_returns_1(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """Persist succeeds but .md write fails → clean error (exit 1), no traceback."""
        isolate_home(monkeypatch, tmp_path, tmp_path)
        data = _sidecar()
        data.pop("source", None)
        data["paragraphs"] = data["paragraphs"][:1]
        sc = _write_sidecar(tmp_path, data)

        import vemoizer.names_cli as names_cli_mod
        import vemoizer.render_cli as render_cli_mod

        original_persist = render_cli_mod._persist_speaker_names

        def persist_then_lock(*a, **kw):
            original_persist(*a, **kw)
            tmp_path.chmod(0o555)

        monkeypatch.setattr(render_cli_mod, "_persist_speaker_names", persist_then_lock)
        monkeypatch.setattr(names_cli_mod, "_persist_speaker_names", persist_then_lock)
        try:
            rc, stderr = _capture_stderr(_run, sc, ["Mikko", "n"])
        finally:
            tmp_path.chmod(0o755)
        assert rc == 1 and "could not write" in stderr


# --- Completer restore ---


class TestCompleterRestore:
    def test_completer_restored_after_run(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """Command-level: the prompt loop restores readline's completer to
        its pre-call value (decision 7)."""
        try:
            import readline
        except ImportError:
            pytest.skip("readline not available")

        isolate_home(monkeypatch, tmp_path, tmp_path)

        def sentinel(text: str, state: int) -> str | None:
            return None

        monkeypatch.setattr(readline, "get_completer", lambda: sentinel)
        monkeypatch.setattr(sys.stdin, "isatty", lambda: True)
        monkeypatch.setattr(sys.stdout, "isatty", lambda: True)

        data = _sidecar()
        data.pop("source", None)
        sc = _write_sidecar(tmp_path, data)
        assert _run(sc, ["Mikko", "n", "Aino", "n"]) == 0
        assert readline.get_completer() is sentinel


# --- Relative source path resolution ---


class TestRelativeSourcePath:
    def test_relative_path_sidecar_dir(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """Relative source path resolves against sidecar's directory first."""
        isolate_home(monkeypatch, tmp_path, tmp_path)
        _make_wav(tmp_path / "rec.wav")
        data = _sidecar()
        data["source"] = [{"path": "rec.wav", "part_offset_s": 0.0, "duration_s": 30.0}]
        sc = _write_sidecar(tmp_path, data)
        monkeypatch.setattr("vemoizer.speaker_clips.subprocess.run", _fake_ffmpeg_ok())
        assert _run(sc, ["Mikko", "n", "Aino", "n"], no_play=False) == 0

    def test_relative_path_cwd_fallback(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """Relative source path not in sidecar dir → try CWD."""
        isolate_home(monkeypatch, tmp_path, tmp_path)
        sub = tmp_path / "subdir"
        sub.mkdir()
        _make_wav(tmp_path / "rec.wav")
        data = _sidecar()
        data["source"] = [{"path": "rec.wav", "part_offset_s": 0.0, "duration_s": 30.0}]
        sc = _write_sidecar(sub, data)
        monkeypatch.setattr("vemoizer.speaker_clips.subprocess.run", _fake_ffmpeg_ok())
        assert _run(sc, ["Mikko", "n", "Aino", "n"], no_play=False) == 0

    def test_unresolvable_path_degrades(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """Unresolvable relative path → degrades to quotes-only, no abort."""
        isolate_home(monkeypatch, tmp_path, tmp_path)
        data = _sidecar()
        data["source"] = [
            {"path": "nonexistent.wav", "part_offset_s": 0.0, "duration_s": 30.0}
        ]
        sc = _write_sidecar(tmp_path, data)
        rc, stderr = _capture_stderr(
            _run, sc, ["Mikko", "n", "Aino", "n"], no_play=False
        )
        assert rc == 0 and ("unavailable" in stderr or "quotes only" in stderr)


# --- ffmpeg failure degradation ---


class TestFfmpegFailure:
    def test_ffmpeg_failure_degrades(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """ffmpeg failure → degrades to quotes-only, one notice, exit 0."""
        isolate_home(monkeypatch, tmp_path, tmp_path)
        wav = _make_wav(tmp_path / "rec.wav")
        data = _sidecar()
        data["source"] = [{"path": str(wav), "part_offset_s": 0.0, "duration_s": 30.0}]
        sc = _write_sidecar(tmp_path, data)
        monkeypatch.setattr(
            "vemoizer.speaker_clips.subprocess.run", _fake_ffmpeg_fail()
        )
        rc, stderr = _capture_stderr(
            _run, sc, ["Mikko", "n", "Aino", "n"], no_play=False
        )
        assert rc == 0 and ("unavailable" in stderr or "quotes only" in stderr)


# --- EOFError mid-prompt (Ctrl-D) ---


class TestEOFError:
    def test_eof_mid_name_prompt(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        """EOFError at the second name prompt → clean exit 0, name persisted."""
        isolate_home(monkeypatch, tmp_path, tmp_path)
        data = _sidecar()
        data.pop("source", None)
        sc = _write_sidecar(tmp_path, data)

        def eof_input(prompt: str) -> str:
            if "SPEAKER_1" in prompt:
                return "Mikko"
            raise EOFError

        rc, stderr = _capture_stderr(_run, sc, input_fn=eof_input)
        assert rc == 0
        assert "Ctrl-D" in stderr or "input ended" in stderr
        updated = json.loads(sc.read_text(encoding="utf-8"))
        assert updated["speaker_names"].get("SPEAKER_1") == "Mikko"
        assert "SPEAKER_2" not in updated["speaker_names"]

    def test_eof_at_add_to_people_prompt(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """EOFError at the Add-to-people prompt → clean exit 0, config unchanged."""
        isolate_home(monkeypatch, tmp_path, tmp_path)
        vemoizer_dir = tmp_path / ".vemoizer"
        vemoizer_dir.mkdir()
        config = vemoizer_dir / "config.toml"
        config.write_text("people = []\n", encoding="utf-8")
        data = _sidecar()
        data.pop("source", None)
        sc = _write_sidecar(tmp_path, data)
        config_before = config.read_bytes()

        def eof_input(prompt: str) -> str:
            if "Add" in prompt:
                raise EOFError
            return "Mikko"

        rc, stderr = _capture_stderr(_run, sc, input_fn=eof_input)
        assert rc == 0
        assert "Ctrl-D" in stderr or "input ended" in stderr
        updated = json.loads(sc.read_text(encoding="utf-8"))
        assert updated["speaker_names"].get("SPEAKER_1") == "Mikko"
        assert config.read_bytes() == config_before


# --- Non-dict speaker_names in sidecar ---


class TestNonDictSpeakerNames:
    def test_list_speaker_names_treated_as_empty(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """A sidecar with speaker_names as a list → treated as empty, no
        AttributeError."""
        isolate_home(monkeypatch, tmp_path, tmp_path)
        data = _sidecar()
        data.pop("source", None)
        data["speaker_names"] = ["not", "a", "dict"]
        sc = _write_sidecar(tmp_path, data)
        assert _run(sc, ["Mikko", "n", "Aino", "n"]) == 0
        updated = json.loads(sc.read_text(encoding="utf-8"))
        assert updated["speaker_names"].get("SPEAKER_1") == "Mikko"
        assert updated["speaker_names"].get("SPEAKER_2") == "Aino"

    def test_string_speaker_names_treated_as_empty(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """A sidecar with speaker_names as a string → treated as empty, no
        AttributeError."""
        isolate_home(monkeypatch, tmp_path, tmp_path)
        data = _sidecar()
        data.pop("source", None)
        data["speaker_names"] = "garbage"
        data["paragraphs"] = data["paragraphs"][:1]
        sc = _write_sidecar(tmp_path, data)
        assert _run(sc, ["Mikko", "n"]) == 0
        updated = json.loads(sc.read_text(encoding="utf-8"))
        assert updated["speaker_names"].get("SPEAKER_1") == "Mikko"


# --- CLI registration ---


class TestCLIRegistration:
    def test_names_help(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        """vemoizer names --help shows the command."""
        isolate_home(monkeypatch, tmp_path, tmp_path)
        result = runner.invoke(app, ["names", "--help"])
        assert result.exit_code == 0 and "--no-play" in result.output

    def test_names_in_main_help(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        """vemoizer --help lists the names command."""
        isolate_home(monkeypatch, tmp_path, tmp_path)
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0 and "names" in result.output
