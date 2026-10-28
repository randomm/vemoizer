"""Listen naming UX fixes (issue #202).

Feedback while clips are prepared, type-ahead discarded before each prompt,
the two-track owner never prompted (and always clipped from the mic stem),
and the sidecar's own ``.md`` updated in place. TTY, prompts, clip extraction
and playback are all faked: no audio plays, no ffmpeg runs.
"""

from __future__ import annotations

import io
import json
import struct
import sys
from pathlib import Path
from typing import Any

import pytest
from _cli_helpers import isolate_home

import vemoizer.names_cli as names_cli
from vemoizer.names_cli import _discard_pending_input, run_names
from vemoizer.sidecar import build_sidecar

OWNER = "Janni"
OWNER_TEXT = "Testin puhua myös suomea tässä, että saadaan testattua molempia kieliä."
OTHERS_TEXT = "Test your call quality, record a short message after the beep."


def _make_wav(path: Path, seconds: float = 0.5) -> Path:
    rate = 16000
    samples = b"".join(struct.pack("<h", 3000) for _ in range(int(seconds * rate)))
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
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(hdr + samples)
    return path


def _two_track_sidecar(tmp_path: Path, *, owner: str | None = OWNER) -> Path:
    """A two-track call: the mix plus the owner's mic stem, both on disk."""
    _make_wav(tmp_path / "calls" / "t" / "call.flac")
    _make_wav(tmp_path / "calls" / "t" / "me.flac")
    options: dict[str, Any] = {
        "command": "meeting",
        "glossary_files": [],
        "glossary_sha256": None,
    }
    if owner is not None:
        options["owner"] = owner
        options["owner_track"] = "calls/t/me.flac"
    data = {
        "text": f"{OWNER_TEXT} {OTHERS_TEXT}",
        "paragraphs": [
            {"start": 0.0, "end": 13.2, "text": OWNER_TEXT, "speaker": owner or "A"},
            {"start": 14.0, "end": 19.0, "text": OTHERS_TEXT, "speaker": "Others"},
        ],
        "source": [{"path": "calls/t/call.flac", "part_offset_s": 0.0}],
        "options": options,
        "speaker_names": {owner: owner} if owner else {},
    }
    path = tmp_path / "2026-10-28 Testi.json"
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return path


class _Recorder:
    """Scripted prompt answers plus an ordered event log (prompts and discards)."""

    def __init__(self, answers: list[str]) -> None:
        self.events: list[tuple[str, str]] = []
        self._answers = iter(answers)

    def discard(self) -> None:
        self.events.append(("discard", ""))

    def ask(self, prompt: str) -> str:
        self.events.append(("prompt", prompt))
        return next(self._answers)

    @property
    def prompts(self) -> list[str]:
        return [text for kind, text in self.events if kind == "prompt"]


@pytest.fixture
def clips(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> dict[str, list]:
    """Fake clip extraction (records each call's source file names) and playback."""
    calls: dict[str, list] = {"sources": [], "played": []}

    def fake_extract(source, windows, tmp_dir):
        calls["sources"].append([Path(e["path"]).name for e in source])
        out = {}
        for i, window in enumerate(windows):
            wav = Path(tmp_dir) / f"clip_{len(calls['sources'])}_{i}.wav"
            wav.write_bytes(b"RIFF")
            out[window] = wav
        return out

    monkeypatch.setattr(names_cli, "extract_clips", fake_extract)
    monkeypatch.setattr(names_cli, "play", lambda p: calls["played"].append(p) or True)
    return calls


def _run(sc: Path, rec: _Recorder, *, no_play: bool = False) -> int:
    return run_names(
        sc,
        no_play=no_play,
        input_fn=rec.ask,
        tty_isatty=lambda: True,
        discard_input=rec.discard,
        config_path=sc.parent / "people.toml",
    )


# --- 1. feedback and type-ahead ---


class TestFeedback:
    def test_preparing_line_printed_before_clips_are_cut(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
        clips: dict[str, list],
    ) -> None:
        isolate_home(monkeypatch, tmp_path, tmp_path)
        sc = _two_track_sidecar(tmp_path)
        seen_before_cut: list[str] = []
        real_extract = names_cli.extract_clips

        def watching_extract(source, windows, tmp_dir):
            seen_before_cut.append(capsys.readouterr().out)
            return real_extract(source, windows, tmp_dir)

        monkeypatch.setattr(names_cli, "extract_clips", watching_extract)
        assert _run(sc, _Recorder(["Bot", "n"])) == 0
        assert "Preparing audio clips" in seen_before_cut[0]

    def test_no_preparing_line_under_no_play(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        isolate_home(monkeypatch, tmp_path, tmp_path)
        sc = _two_track_sidecar(tmp_path)
        assert _run(sc, _Recorder(["Bot", "n"]), no_play=True) == 0
        assert "Preparing audio clips" not in capsys.readouterr().out

    def test_pending_input_discarded_before_every_prompt(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, clips: dict
    ) -> None:
        isolate_home(monkeypatch, tmp_path, tmp_path)
        sc = _two_track_sidecar(tmp_path, owner=None)
        rec = _Recorder(["Aino", "n", "Mikko", "n"])
        assert _run(sc, rec, no_play=True) == 0
        # Every prompt (name and people-add) follows a discard, so type-ahead
        # typed during clip preparation cannot answer it.
        assert [kind for kind, _ in rec.events] == ["discard", "prompt"] * 4

    def test_discard_flushes_terminal_input_queue(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        flushed: list[tuple[int, int]] = []

        class _Stdin:
            def fileno(self) -> int:
                return 7

        monkeypatch.setattr(sys, "stdin", _Stdin())
        monkeypatch.setattr(
            names_cli.termios, "tcflush", lambda fd, q: flushed.append((fd, q))
        )
        _discard_pending_input()
        assert flushed == [(7, names_cli.termios.TCIFLUSH)]

    def test_discard_is_harmless_without_a_terminal(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class _NoFd(io.StringIO):
            def fileno(self) -> int:
                raise io.UnsupportedOperation("no fileno")

        monkeypatch.setattr(sys, "stdin", _NoFd())
        _discard_pending_input()  # must not raise


# --- 2. owner is known: never prompted, never offered for people ---


class TestOwner:
    def test_owner_never_prompted_nor_offered_for_people(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, clips: dict
    ) -> None:
        isolate_home(monkeypatch, tmp_path, tmp_path)
        sc = _two_track_sidecar(tmp_path)
        rec = _Recorder(["Bot", "n"])
        assert _run(sc, rec, no_play=True) == 0
        assert rec.prompts == [
            "Name for Others (empty to skip): ",
            "Add Bot to people in " + str(tmp_path / "people.toml") + "? [y/N] ",
        ]

    def test_owner_gets_quote_and_clip_from_mic_stem(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
        clips: dict[str, list],
    ) -> None:
        isolate_home(monkeypatch, tmp_path, tmp_path)
        sc = _two_track_sidecar(tmp_path)
        assert _run(sc, _Recorder(["Bot", "n"])) == 0
        out = capsys.readouterr().out
        assert f'"{OWNER_TEXT[:40]}' in out
        # The owner's clip comes from the mic stem; the other side from the mix.
        assert clips["sources"][0] == ["me.flac"]
        assert clips["sources"][1] == ["call.flac"]
        assert len(clips["played"]) == 2

    def test_owner_name_kept_and_other_side_named(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, clips: dict
    ) -> None:
        isolate_home(monkeypatch, tmp_path, tmp_path)
        sc = _two_track_sidecar(tmp_path)
        assert _run(sc, _Recorder(["Bot", "n"]), no_play=True) == 0
        stored = json.loads(sc.read_text(encoding="utf-8"))["speaker_names"]
        assert stored == {OWNER: OWNER, "Others": "Bot"}

    def test_build_sidecar_seeds_owner_label(self, tmp_path: Path) -> None:
        result: dict[str, Any] = {"text": "moi", "paragraphs": []}
        build_sidecar(
            result,
            command="meeting",
            glossary_files=None,
            owner=OWNER,
            owner_track=Path("calls/t/me.flac"),
        )
        assert result["options"]["owner"] == OWNER
        assert result["options"]["owner_track"] == "calls/t/me.flac"
        assert result["speaker_names"] == {OWNER: OWNER}

    def test_build_sidecar_without_owner_adds_no_owner_keys(self) -> None:
        result: dict[str, Any] = {"text": "moi", "paragraphs": []}
        build_sidecar(result, command="meeting", glossary_files=None)
        assert "owner" not in result["options"]
        assert "owner_track" not in result["options"]
        assert result["speaker_names"] == {}


# --- 3. exactly one Markdown per sidecar ---


class TestOneMarkdown:
    def test_naming_updates_the_sidecar_markdown_in_place(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, clips: dict
    ) -> None:
        isolate_home(monkeypatch, tmp_path, tmp_path)
        sc = _two_track_sidecar(tmp_path)
        original = tmp_path / "2026-10-28 Testi.md"
        original.write_text("STALE RENDER", encoding="utf-8")
        assert _run(sc, _Recorder(["Bot", "n"]), no_play=True) == 0
        assert sorted(p.name for p in tmp_path.glob("*.md")) == [original.name]
        text = original.read_text(encoding="utf-8")
        assert "STALE RENDER" not in text
        assert "Bot" in text
