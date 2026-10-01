"""Tests for the ``vemoizer names X.json`` command (issue #93, M5c-1).

Covers the non-TTY guard, prompt loop, talk-share ordering, empty-answer
skip, --no-play quotes-only, no-labelled-paragraphs, old sidecar without
source, same-name collapse, people-config add prompt, and CLI registration.

All tests use ``isolate_home`` (chdir + HOME isolation), synthetic sidecar
JSON, and monkeypatched subprocess for ffmpeg/afplay. No real audio,
models, or network.
"""

from __future__ import annotations

import json
import struct
import subprocess
from pathlib import Path
from typing import Any

import pytest
from _cli_helpers import isolate_home
from typer.testing import CliRunner

from vemoizer.cli import app
from vemoizer.names_cli import run_names

runner = CliRunner()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sidecar(**extra: Any) -> dict[str, Any]:
    """A minimal sidecar with two labelled speakers."""
    base: dict[str, Any] = {
        "text": "Puhuttiin asioista.",
        "paragraphs": [
            {
                "start": 0.0,
                "end": 10.0,
                "text": "Moikka, aloitellaan tässä.",
                "speaker": "SPEAKER_1",
            },
            {
                "start": 12.0,
                "end": 20.0,
                "text": "Kyllä, mietin samaa.",
                "speaker": "SPEAKER_2",
            },
        ],
        "notes": {
            "title": "Kokous",
            "summary": "Puhuttiin asioista.",
        },
        "options": {
            "command": "meeting",
            "glossary_files": [],
            "glossary_sha256": None,
        },
        "speaker_names": {},
        **extra,
    }
    return base


def _make_wav(path: Path, seconds: float) -> Path:
    """A tiny mono 16 kHz 16-bit PCM WAV."""
    rate = 16000
    n = int(seconds * rate)
    samples = b"".join(struct.pack("<h", 3000) for _ in range(n))
    filesize = 36 + len(samples)
    hdr = (
        b"RIFF"
        + struct.pack("<I", filesize - 8)
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
    """A fake subprocess.run that creates a valid-enough WAV for extract_clips."""

    def fake_run(argv, *a, **kw):
        out_path = argv[-1]
        _make_wav(Path(out_path), 0.5)
        return subprocess.CompletedProcess(argv, 0)

    return fake_run


def _fake_ffmpeg_fail():
    """A fake subprocess.run that always fails."""

    def fake_run(argv, *a, **kw):
        return subprocess.CompletedProcess(argv, 1)

    return fake_run


def _capture_stderr(fn, *args, **kwargs):
    """Run *fn* and capture its stderr; return (result, stderr_text)."""
    import io
    import sys

    old_stderr = sys.stderr
    sys.stderr = io.StringIO()
    try:
        result = fn(*args, **kwargs)
    finally:
        captured = sys.stderr  # type: ignore[misc]
        sys.stderr = old_stderr
    return result, captured.getvalue()


# ---------------------------------------------------------------------------
# Non-TTY guard
# ---------------------------------------------------------------------------


class TestNonTTYGuard:
    def test_non_tty_exits_2(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Non-TTY stdin → exit 2 with one stderr line, before any other logic."""
        isolate_home(monkeypatch, tmp_path, tmp_path)
        sc = _write_sidecar(tmp_path, _sidecar())
        rc, stderr = _capture_stderr(run_names, sc, tty_isatty=lambda: False)
        assert rc == 2
        assert "TTY" in stderr

    def test_non_tty_before_no_play(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Non-TTY guard fires before --no-play processing (exit 2)."""
        isolate_home(monkeypatch, tmp_path, tmp_path)
        sc = _write_sidecar(tmp_path, _sidecar())
        rc = run_names(sc, no_play=True, tty_isatty=lambda: False)
        assert rc == 2

    def test_non_tty_via_cli(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Non-TTY via CliRunner → exit 2 (CliRunner's stdin is non-TTY)."""
        isolate_home(monkeypatch, tmp_path, tmp_path)
        sc = _write_sidecar(tmp_path, _sidecar())
        result = runner.invoke(app, ["names", str(sc)])
        assert result.exit_code == 2
        assert "TTY" in result.stderr

    def test_non_tty_with_missing_sidecar(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Non-TTY + missing sidecar → exit 2 (TTY guard fires first)."""
        isolate_home(monkeypatch, tmp_path, tmp_path)
        sc = tmp_path / "nonexistent.json"
        rc = run_names(sc, tty_isatty=lambda: False)
        assert rc == 2

    def test_non_tty_with_malformed_sidecar(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Non-TTY + malformed sidecar → exit 2 (TTY guard fires first)."""
        isolate_home(monkeypatch, tmp_path, tmp_path)
        sc = tmp_path / "bad.json"
        sc.write_text("not json", encoding="utf-8")
        rc = run_names(sc, tty_isatty=lambda: False)
        assert rc == 2


# ---------------------------------------------------------------------------
# No labelled speakers
# ---------------------------------------------------------------------------


class TestNoLabelledSpeakers:
    def test_no_labelled_paragraphs(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Sidecar with no labelled paragraphs → exit 0 with clear message."""
        isolate_home(monkeypatch, tmp_path, tmp_path)
        data = _sidecar()
        for p in data["paragraphs"]:
            p["speaker"] = None
        sc = _write_sidecar(tmp_path, data)
        rc, stderr = _capture_stderr(run_names, sc, tty_isatty=lambda: True)
        assert rc == 0
        assert "no labelled speakers" in stderr

    def test_no_paragraphs(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Sidecar with empty paragraphs → exit 0."""
        isolate_home(monkeypatch, tmp_path, tmp_path)
        data = _sidecar()
        data["paragraphs"] = []
        sc = _write_sidecar(tmp_path, data)
        rc = run_names(sc, tty_isatty=lambda: True)
        assert rc == 0


# ---------------------------------------------------------------------------
# Malformed sidecar
# ---------------------------------------------------------------------------


class TestMalformedSidecar:
    def test_missing_sidecar(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Missing sidecar → exit 1."""
        isolate_home(monkeypatch, tmp_path, tmp_path)
        sc = tmp_path / "nonexistent.json"
        rc = run_names(sc, tty_isatty=lambda: True)
        assert rc == 1

    def test_malformed_json(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Malformed JSON → exit 1."""
        isolate_home(monkeypatch, tmp_path, tmp_path)
        sc = tmp_path / "bad.json"
        sc.write_text("not json", encoding="utf-8")
        rc = run_names(sc, tty_isatty=lambda: True)
        assert rc == 1


# ---------------------------------------------------------------------------
# Prompt loop happy path
# ---------------------------------------------------------------------------


class TestPromptLoop:
    def test_single_label_named(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """One label named → sidecar updated, .md written."""
        isolate_home(monkeypatch, tmp_path, tmp_path)
        data = _sidecar()
        data.pop("source", None)
        data["paragraphs"] = data["paragraphs"][:1]  # only SPEAKER_1
        sc = _write_sidecar(tmp_path, data)

        # Input: name for SPEAKER_1, then "n" for add-to-people
        inputs = iter(["Mikko", "n"])
        rc = run_names(
            sc,
            no_play=True,
            input_fn=lambda prompt: next(inputs),
            tty_isatty=lambda: True,
        )
        assert rc == 0

        updated = json.loads(sc.read_text(encoding="utf-8"))
        assert updated["speaker_names"].get("SPEAKER_1") == "Mikko"

    def test_two_labels_both_named(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Two labels both named → both persisted in one write."""
        isolate_home(monkeypatch, tmp_path, tmp_path)
        data = _sidecar()
        data.pop("source", None)
        sc = _write_sidecar(tmp_path, data)

        # SPEAKER_1 (10s) before SPEAKER_2 (8s). Each: name + add(y/n)
        inputs = iter(["Mikko", "n", "Aino", "n"])
        rc = run_names(
            sc,
            no_play=True,
            input_fn=lambda prompt: next(inputs),
            tty_isatty=lambda: True,
        )
        assert rc == 0

        updated = json.loads(sc.read_text(encoding="utf-8"))
        assert updated["speaker_names"] == {"SPEAKER_1": "Mikko", "SPEAKER_2": "Aino"}

        md_files = list(tmp_path.glob("*.md"))
        assert len(md_files) == 1

    def test_empty_answer_skips_label(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Empty answer for a label → that label not in persisted names."""
        isolate_home(monkeypatch, tmp_path, tmp_path)
        data = _sidecar()
        data.pop("source", None)
        sc = _write_sidecar(tmp_path, data)

        # SPEAKER_1: name "Mikko" + skip add. SPEAKER_2: empty (skip).
        inputs = iter(["Mikko", "n", ""])
        rc = run_names(
            sc,
            no_play=True,
            input_fn=lambda prompt: next(inputs),
            tty_isatty=lambda: True,
        )
        assert rc == 0

        updated = json.loads(sc.read_text(encoding="utf-8"))
        assert "SPEAKER_1" in updated["speaker_names"]
        assert "SPEAKER_2" not in updated["speaker_names"]

    def test_talk_share_ordering(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Labels listed in descending talk_share order."""
        isolate_home(monkeypatch, tmp_path, tmp_path)
        data = _sidecar()
        # SPEAKER_2 has more talk time (20s vs 5s)
        data["paragraphs"] = [
            {"start": 0.0, "end": 5.0, "text": "Hi.", "speaker": "SPEAKER_1"},
            {
                "start": 10.0,
                "end": 30.0,
                "text": "Longer talk here.",
                "speaker": "SPEAKER_2",
            },
        ]
        data.pop("source", None)
        sc = _write_sidecar(tmp_path, data)

        name_prompts: list[str] = []

        def recording_input(prompt: str) -> str:
            if "Name for" in prompt:
                name_prompts.append(prompt)
            if "Add" in prompt:
                return "n"
            if "SPEAKER_2" in prompt:
                return "MoreTalk"
            return "LessTalk"

        run_names(
            sc,
            no_play=True,
            input_fn=recording_input,
            tty_isatty=lambda: True,
        )
        # First name prompt should be for SPEAKER_2 (more talk)
        assert "SPEAKER_2" in name_prompts[0]
        assert "SPEAKER_1" in name_prompts[1]


# ---------------------------------------------------------------------------
# --no-play quotes-only
# ---------------------------------------------------------------------------


class TestNoPlay:
    def test_no_play_quotes_only(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """--no-play skips clip extraction and playback entirely."""
        isolate_home(monkeypatch, tmp_path, tmp_path)
        wav = _make_wav(tmp_path / "rec.wav", 1.0)
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

        inputs = iter(["Name1", "n", "Name2", "n"])
        rc = run_names(
            sc,
            no_play=True,
            input_fn=lambda prompt: next(inputs),
            tty_isatty=lambda: True,
        )
        assert rc == 0
        assert extract_calls == 0


# ---------------------------------------------------------------------------
# Old sidecar without source
# ---------------------------------------------------------------------------


class TestOldSidecar:
    def test_old_sidecar_no_source(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Old sidecar without source → quotes-only with one notice line."""
        isolate_home(monkeypatch, tmp_path, tmp_path)
        data = _sidecar()
        # No source key
        sc = _write_sidecar(tmp_path, data)

        inputs = iter(["Mikko", "n", "Aino", "n"])
        rc, stderr = _capture_stderr(
            run_names,
            sc,
            no_play=False,
            input_fn=lambda prompt: next(inputs),
            tty_isatty=lambda: True,
        )
        assert rc == 0
        assert "unavailable" in stderr or "quotes only" in stderr

    def test_old_sidecar_exactly_one_notice(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """At most ONE aggregate notice line per run."""
        isolate_home(monkeypatch, tmp_path, tmp_path)
        data = _sidecar()
        sc = _write_sidecar(tmp_path, data)

        inputs = iter(["Mikko", "n", "Aino", "n"])
        _, stderr = _capture_stderr(
            run_names,
            sc,
            no_play=False,
            input_fn=lambda prompt: next(inputs),
            tty_isatty=lambda: True,
        )
        notice_lines = [
            line
            for line in stderr.strip().splitlines()
            if "unavailable" in line or "quotes only" in line
        ]
        assert len(notice_lines) <= 1


# ---------------------------------------------------------------------------
# Same-name collapse
# ---------------------------------------------------------------------------


class TestSameNameCollapse:
    def test_two_labels_same_name(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Two labels given the same name → collapse to earliest via render merge."""
        isolate_home(monkeypatch, tmp_path, tmp_path)
        data = _sidecar()
        data.pop("source", None)
        sc = _write_sidecar(tmp_path, data)

        inputs = iter(["Mikko", "n", "Mikko", "n"])
        rc = run_names(
            sc,
            no_play=True,
            input_fn=lambda prompt: next(inputs),
            tty_isatty=lambda: True,
        )
        assert rc == 0

        updated = json.loads(sc.read_text(encoding="utf-8"))
        assert updated["speaker_names"].get("SPEAKER_1") == "Mikko"
        assert updated["speaker_names"].get("SPEAKER_2") == "Mikko"

        md_files = list(tmp_path.glob("*.md"))
        assert len(md_files) == 1
        md_content = md_files[0].read_text(encoding="utf-8")
        # SPEAKER_2 should not appear as a separate speaker in the markdown
        assert "SPEAKER_2" not in md_content


# ---------------------------------------------------------------------------
# People config: add prompt
# ---------------------------------------------------------------------------


class TestPeopleConfig:
    def test_new_name_no_add(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """New name not in people → asks to add; no/Enter leaves config unchanged."""
        isolate_home(monkeypatch, tmp_path, tmp_path)

        vemoizer_dir = tmp_path / ".vemoizer"
        vemoizer_dir.mkdir()
        config = vemoizer_dir / "config.toml"
        config.write_text("people = []\n", encoding="utf-8")

        data = _sidecar()
        data.pop("source", None)
        sc = _write_sidecar(tmp_path, data)

        inputs = iter(["Mikko", "n", "Aino", "n"])
        rc = run_names(
            sc,
            no_play=True,
            input_fn=lambda prompt: next(inputs),
            tty_isatty=lambda: True,
        )
        assert rc == 0

        config_content = config.read_text(encoding="utf-8")
        assert "Mikko" not in config_content
        assert "Aino" not in config_content

    def test_new_name_add_yes(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """New name + yes → added to people list in config."""
        isolate_home(monkeypatch, tmp_path, tmp_path)

        vemoizer_dir = tmp_path / ".vemoizer"
        vemoizer_dir.mkdir()
        config = vemoizer_dir / "config.toml"
        config.write_text("people = []\n", encoding="utf-8")

        data = _sidecar()
        data.pop("source", None)
        sc = _write_sidecar(tmp_path, data)

        inputs = iter(["Mikko", "y", "Aino", "n"])
        rc = run_names(
            sc,
            no_play=True,
            input_fn=lambda prompt: next(inputs),
            tty_isatty=lambda: True,
        )
        assert rc == 0

        config_content = config.read_text(encoding="utf-8")
        assert "Mikko" in config_content

    def test_existing_name_no_add_prompt(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Name already in people (case-insensitive) → no add prompt."""
        isolate_home(monkeypatch, tmp_path, tmp_path)

        vemoizer_dir = tmp_path / ".vemoizer"
        vemoizer_dir.mkdir()
        config = vemoizer_dir / "config.toml"
        config.write_text('people = ["mikko"]\n', encoding="utf-8")

        data = _sidecar()
        data.pop("source", None)
        sc = _write_sidecar(tmp_path, data)

        prompts: list[str] = []

        def recording_input(prompt: str) -> str:
            prompts.append(prompt)
            if "Add" in prompt:
                return "n"
            if "SPEAKER_1" in prompt:
                return "Mikko"
            return "Aino"

        run_names(
            sc,
            no_play=True,
            input_fn=recording_input,
            tty_isatty=lambda: True,
        )

        # No "Add ... to people" prompt for "Mikko" (already in people).
        # But "Aino" IS new, so it gets one add prompt.
        mikko_adds = [p for p in prompts if "Add" in p and "Mikko" in p]
        assert len(mikko_adds) == 0

    def test_people_list_read_fail_open(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Missing/invalid people → empty list, no error."""
        isolate_home(monkeypatch, tmp_path, tmp_path)

        data = _sidecar()
        data.pop("source", None)
        sc = _write_sidecar(tmp_path, data)

        inputs = iter(["Mikko", "n", "Aino", "n"])
        rc = run_names(
            sc,
            no_play=True,
            input_fn=lambda prompt: next(inputs),
            tty_isatty=lambda: True,
        )
        assert rc == 0


# ---------------------------------------------------------------------------
# Ctrl-C mid-prompt
# ---------------------------------------------------------------------------


class TestCtrlC:
    def test_ctrl_c_sidecar_byte_identical(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Ctrl-C mid-prompt → sidecar byte-identical, no clip files remain."""
        isolate_home(monkeypatch, tmp_path, tmp_path)

        wav = _make_wav(tmp_path / "rec.wav", 1.0)
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
            if call_count <= 2:  # first name prompt
                return "Mikko"
            raise KeyboardInterrupt

        import contextlib

        monkeypatch.setattr("vemoizer.speaker_clips.subprocess.run", _fake_ffmpeg_ok())

        with contextlib.suppress(KeyboardInterrupt):
            run_names(
                sc,
                no_play=False,
                input_fn=input_fn,
                tty_isatty=lambda: True,
            )

        assert sc.read_bytes() == original_bytes

        # No clip files remain (clip_session cleans up)
        wav_files = list(tmp_path.glob("clip_*.wav"))
        assert wav_files == []


# ---------------------------------------------------------------------------
# Relative source path resolution
# ---------------------------------------------------------------------------


class TestRelativeSourcePath:
    def test_relative_path_sidecar_dir(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Relative source path resolves against sidecar's directory first."""
        isolate_home(monkeypatch, tmp_path, tmp_path)

        _make_wav(tmp_path / "rec.wav", 1.0)
        data = _sidecar()
        data["source"] = [{"path": "rec.wav", "part_offset_s": 0.0, "duration_s": 30.0}]
        sc = _write_sidecar(tmp_path, data)

        monkeypatch.setattr("vemoizer.speaker_clips.subprocess.run", _fake_ffmpeg_ok())

        inputs = iter(["Mikko", "n", "Aino", "n"])
        rc = run_names(
            sc,
            no_play=False,
            input_fn=lambda prompt: next(inputs),
            tty_isatty=lambda: True,
        )
        assert rc == 0

    def test_relative_path_cwd_fallback(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Relative source path not in sidecar dir → try CWD."""
        isolate_home(monkeypatch, tmp_path, tmp_path)

        sub = tmp_path / "subdir"
        sub.mkdir()
        _make_wav(tmp_path / "rec.wav", 1.0)  # wav in CWD (tmp_path)

        data = _sidecar()
        data["source"] = [{"path": "rec.wav", "part_offset_s": 0.0, "duration_s": 30.0}]
        sc = _write_sidecar(sub, data)  # sidecar in subdir, wav in CWD

        monkeypatch.setattr("vemoizer.speaker_clips.subprocess.run", _fake_ffmpeg_ok())

        inputs = iter(["Mikko", "n", "Aino", "n"])
        rc = run_names(
            sc,
            no_play=False,
            input_fn=lambda prompt: next(inputs),
            tty_isatty=lambda: True,
        )
        assert rc == 0

    def test_unresolvable_path_degrades(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Unresolvable relative path → degrades to quotes-only, no abort."""
        isolate_home(monkeypatch, tmp_path, tmp_path)

        data = _sidecar()
        data["source"] = [
            {"path": "nonexistent.wav", "part_offset_s": 0.0, "duration_s": 30.0}
        ]
        sc = _write_sidecar(tmp_path, data)

        inputs = iter(["Mikko", "n", "Aino", "n"])
        rc, stderr = _capture_stderr(
            run_names,
            sc,
            no_play=False,
            input_fn=lambda prompt: next(inputs),
            tty_isatty=lambda: True,
        )
        assert rc == 0
        assert "unavailable" in stderr or "quotes only" in stderr


# ---------------------------------------------------------------------------
# ffmpeg failure degradation
# ---------------------------------------------------------------------------


class TestFfmpegFailure:
    def test_ffmpeg_failure_degrades(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """ffmpeg failure → degrades to quotes-only, one notice, exit 0."""
        isolate_home(monkeypatch, tmp_path, tmp_path)

        wav = _make_wav(tmp_path / "rec.wav", 1.0)
        data = _sidecar()
        data["source"] = [{"path": str(wav), "part_offset_s": 0.0, "duration_s": 30.0}]
        sc = _write_sidecar(tmp_path, data)

        monkeypatch.setattr(
            "vemoizer.speaker_clips.subprocess.run", _fake_ffmpeg_fail()
        )

        inputs = iter(["Mikko", "n", "Aino", "n"])
        rc, stderr = _capture_stderr(
            run_names,
            sc,
            no_play=False,
            input_fn=lambda prompt: next(inputs),
            tty_isatty=lambda: True,
        )
        assert rc == 0
        assert "unavailable" in stderr or "quotes only" in stderr


# ---------------------------------------------------------------------------
# CLI registration
# ---------------------------------------------------------------------------


class TestCLIRegistration:
    def test_names_help(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """vemoizer names --help shows the command."""
        isolate_home(monkeypatch, tmp_path, tmp_path)
        result = runner.invoke(app, ["names", "--help"])
        assert result.exit_code == 0
        assert "--no-play" in result.output

    def test_names_in_main_help(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """vemoizer --help lists the names command."""
        isolate_home(monkeypatch, tmp_path, tmp_path)
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "names" in result.output
