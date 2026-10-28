"""The two-track run path never prompts for the owner (issue #202).

Drives the real ``run_preset`` plain loop with ``tracks=`` and ``me=`` and
checks what it writes: the sidecar carries ``options.owner``,
``options.owner_track`` and the seeded ``speaker_names``, and the end-of-run
naming hook never asks for the owner. ``transcribe_file`` is a fake (no
models), the naming TTY gate is patched on, and the clip extraction and
playback seams are faked: no audio plays and no ffmpeg runs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from _cli_helpers import isolate_home, touch_files

import vemoizer.names_cli as names_cli
import vemoizer.pipeline as pipeline_module
from vemoizer.batch_preset import run_preset
from vemoizer.owner_config import RunOwner
from vemoizer.tracks import TrackPaths

OWNER = "Janni"
OWNER_TEXT = "Testin puhua myös suomea tässä, että saadaan testattua molempia kieliä."
OTHERS_TEXT = "Test your call quality, record a short message after the beep."


class _Prompt:
    """Scripted ``input_fn`` that records every prompt it was asked."""

    def __init__(self, *answers: str) -> None:
        self._answers = list(answers)
        self.prompts: list[str] = []

    def __call__(self, text: str) -> str:
        self.prompts.append(text)
        return self._answers.pop(0) if self._answers else ""


def _fake_two_track_transcribe(seen: list[dict[str, Any]], label: str = OWNER):
    """A transcript shaped like the two-track pipeline: the owner's words are
    labelled with the owner label (from the mic stem) and the far side is
    ``Others``."""

    def fake(path, **kwargs):
        seen.append(kwargs)
        return {
            "text": f"{OWNER_TEXT} {OTHERS_TEXT}",
            "paragraphs": [
                {"start": 0.0, "end": 13.2, "text": OWNER_TEXT, "speaker": label},
                {"start": 14.0, "end": 19.0, "text": OTHERS_TEXT, "speaker": "Others"},
            ],
            "segments": [],
            "notes": {"title": "Testi"},
        }

    return fake


@pytest.fixture
def run_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    """An isolated HOME and cwd; the run writes its pair into this directory."""
    isolate_home(monkeypatch, tmp_path, tmp_path)
    return tmp_path


@pytest.fixture
def fake_clips(monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    """Fake clip extraction and playback; records each extraction's sources."""
    sources: list[list[str]] = []

    def fake_extract(source, windows, tmp_dir):
        sources.append([Path(e["path"]).name for e in source])
        out = {}
        for i, window in enumerate(windows):
            wav = Path(tmp_dir) / f"clip_{len(sources)}_{i}.wav"
            wav.write_bytes(b"RIFF")
            out[window] = wav
        return out

    monkeypatch.setattr(names_cli, "extract_clips", fake_extract)
    monkeypatch.setattr(names_cli, "play", lambda p: True)
    monkeypatch.setattr(names_cli, "_discard_pending_input", lambda: None)
    return sources


def _run_two_track(
    files: list[Path],
    tracks: TrackPaths,
    *,
    input_fn: _Prompt,
    seen: list[dict[str, Any]],
    monkeypatch: pytest.MonkeyPatch,
    owner: RunOwner | None = None,
) -> int:
    label = owner.label if owner is not None else OWNER
    monkeypatch.setattr(
        pipeline_module, "transcribe_file", _fake_two_track_transcribe(seen, label)
    )
    # The naming hook only asks when stdout is a terminal; make it one so the
    # real end-of-run prompt flow is what the test observes.
    monkeypatch.setattr("vemoizer.naming_hook._stdout_isatty", lambda: True)
    return run_preset(
        files,
        command="meeting",
        config_path=None,
        glossary_path=None,
        tracks=tracks,
        me=OWNER,
        owner=owner,
        quiet=True,
        input_fn=input_fn,
        tty_isatty=lambda: True,
    )


def _only_json(directory: Path) -> dict[str, Any]:
    sidecars = sorted(directory.glob("*.json"))
    assert len(sidecars) == 1, sidecars
    return json.loads(sidecars[0].read_text(encoding="utf-8"))


def test_two_track_run_seeds_the_owner_into_the_sidecar(
    run_dir: Path, monkeypatch: pytest.MonkeyPatch, fake_clips: list[list[str]]
) -> None:
    files = touch_files(["call.m4a"], run_dir)
    touch_files(["me.flac"], run_dir)
    me = run_dir / "me.flac"
    them = run_dir / "them.flac"
    seen: list[dict[str, Any]] = []
    asked = _Prompt("n")  # decline the naming hook; only the owner matters here

    code = _run_two_track(
        files,
        TrackPaths(me=me, them=them),
        input_fn=asked,
        seen=seen,
        monkeypatch=monkeypatch,
    )

    assert code == 0
    assert seen[0]["owner"] == OWNER
    sidecar = _only_json(run_dir)
    assert sidecar["options"]["owner"] == OWNER
    assert sidecar["options"]["owner_track"] == str(me)
    assert sidecar["speaker_names"] == {OWNER: OWNER}


def test_two_track_naming_never_prompts_for_the_owner(
    run_dir: Path, monkeypatch: pytest.MonkeyPatch, fake_clips: list[list[str]]
) -> None:
    files = touch_files(["call.m4a"], run_dir)
    touch_files(["me.flac"], run_dir)
    tracks = TrackPaths(me=run_dir / "me.flac", them=run_dir / "them.flac")
    asked = _Prompt("y", "")  # accept the naming hook, skip the far side

    code = _run_two_track(
        files, tracks, input_fn=asked, seen=[], monkeypatch=monkeypatch
    )

    assert code == 0
    assert asked.prompts, "the naming hook should have offered to name the speakers"
    assert not [p for p in asked.prompts if OWNER in p], asked.prompts
    assert [p for p in asked.prompts if "Others" in p], asked.prompts
    # The owner still gets a clip, from the mic stem rather than the mix.
    assert fake_clips, "naming should have cut the owner's clip"
    assert fake_clips[0] == ["me.flac"]


def test_fallback_owner_sidecar_still_prompts_for_the_owner(
    run_dir: Path, monkeypatch: pytest.MonkeyPatch, fake_clips: list[list[str]]
) -> None:
    """Issue #202 review: the language fallback label ("Minä") is not a name.
    With no --me and no configured owner the sidecar records no owner, so
    naming still asks for the "Minä" label and the user can name themselves."""
    files = touch_files(["call.m4a"], run_dir)
    touch_files(["me.flac"], run_dir)
    tracks = TrackPaths(me=run_dir / "me.flac", them=run_dir / "them.flac")
    fallback = RunOwner("Minä", named=False)
    asked = _Prompt("y", "Janni", "")  # accept the naming hook, name Minä, skip Others

    code = _run_two_track(
        files,
        tracks,
        input_fn=asked,
        seen=[],
        monkeypatch=monkeypatch,
        owner=fallback,
    )

    assert code == 0
    sidecar = _only_json(run_dir)
    assert "owner" not in sidecar["options"]
    assert "owner_track" not in sidecar["options"]
    assert [p for p in asked.prompts if "Minä" in p], asked.prompts
    assert sidecar["speaker_names"] == {"Minä": "Janni"}
