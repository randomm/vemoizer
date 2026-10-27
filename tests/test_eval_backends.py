"""Eval decode backends live in ``vemoizer.eval_backends`` (issues #175, #176).

The CLI module only wires the Typer command; the registry and the decode
helpers are reached through ``eval_backends`` with no re-export, so tests
patch the module that owns the attribute.
"""

from __future__ import annotations

import threading
from pathlib import Path

import numpy as np
import pytest

import vemoizer.eval_backends as eval_backends
import vemoizer.eval_cli as eval_cli
import vemoizer.pipeline as pipeline_module


def test_backend_registry_lives_in_eval_backends() -> None:
    assert set(eval_backends.BACKENDS) == {
        "parakeet",
        "canary",
        "consensus",
        "whisper-turbo",
    }


def test_decode_helper_lives_in_eval_backends() -> None:
    assert callable(eval_backends.transcribe_decode_only)


def test_eval_cli_does_not_define_or_reexport_backends() -> None:
    assert not hasattr(eval_cli, "BACKENDS")
    assert not hasattr(eval_cli, "transcribe_decode_only")


def _patch_meeting_seams(monkeypatch, decode) -> None:
    """Stub ingest, VAD slicing and glossary so only ``decode_meeting`` is live."""
    audio = np.zeros(16_000, dtype=np.float32)
    monkeypatch.setattr(pipeline_module, "ingest_audio", lambda path: audio)
    monkeypatch.setattr(pipeline_module, "_speech_slices", lambda a: ([(0, a)], True))
    monkeypatch.setattr(pipeline_module, "load_glossary", lambda path: [])
    monkeypatch.setattr(pipeline_module, "glossary_prompt", lambda terms: None)
    monkeypatch.setattr(pipeline_module, "decode_meeting", decode)


def test_whisper_turbo_routes_through_decode_meeting_with_language_none(
    monkeypatch,
) -> None:
    seen: dict = {}

    def fake_decode_meeting(audio, slices, **kwargs):
        seen.update(kwargs)
        return {"text": " moro maailma "}

    _patch_meeting_seams(monkeypatch, fake_decode_meeting)
    text = eval_backends.BACKENDS["whisper-turbo"](Path("memo.m4a"))
    assert text == "moro maailma"
    assert seen["language"] is None
    assert seen["vad_slices"] == [(0, 16_000)]


def test_whisper_turbo_fails_open_to_empty_hypothesis(monkeypatch) -> None:
    _patch_meeting_seams(monkeypatch, lambda audio, slices, **kwargs: None)
    assert eval_backends.transcribe_meeting_only(Path("memo.m4a")) == {
        "text": "",
        "segments": [],
    }


def test_whisper_turbo_never_falls_back_to_dictation_decoders(monkeypatch) -> None:
    def _must_not_run(*args, **kwargs):
        raise AssertionError("dictation decoder ran under whisper-turbo")

    _patch_meeting_seams(monkeypatch, lambda audio, slices, **kwargs: None)
    monkeypatch.setattr(pipeline_module, "ParakeetTranscriber", _must_not_run)
    monkeypatch.setattr(pipeline_module, "CanaryTranscriber", _must_not_run)
    eval_backends.transcribe_meeting_only(Path("memo.m4a"))


def test_whisper_turbo_ingest_failure_is_empty_with_error(monkeypatch) -> None:
    from vemoizer.ingest import IngestError

    def _broken(path):
        raise IngestError("no audio stream")

    monkeypatch.setattr(pipeline_module, "ingest_audio", _broken)
    result = eval_backends.transcribe_meeting_only(Path("memo.m4a"))
    assert result == {"text": "", "segments": [], "error": "no audio stream"}


def test_whisper_turbo_empty_audio_skips_the_decode(monkeypatch) -> None:
    monkeypatch.setattr(
        pipeline_module, "ingest_audio", lambda path: np.zeros(0, dtype=np.float32)
    )

    def _must_not_run(*args, **kwargs):
        raise AssertionError("decode ran on empty audio")

    monkeypatch.setattr(pipeline_module, "decode_meeting", _must_not_run)
    assert eval_backends.transcribe_meeting_only(Path("memo.m4a")) == {
        "text": "",
        "segments": [],
    }


def test_on_large_stack_runs_the_call_on_a_worker_thread() -> None:
    names: list[str] = []

    def _decode(wav: Path) -> str:
        names.append(threading.current_thread().name)
        return wav.stem

    assert eval_backends.on_large_stack(_decode)(Path("a.wav")) == "a"
    assert names and names[0] != threading.main_thread().name


def test_on_large_stack_reraises_the_call_error() -> None:
    def _decode(wav: Path) -> str:
        raise ValueError("decode broke")

    with pytest.raises(ValueError, match="decode broke"):
        eval_backends.on_large_stack(_decode)(Path("a.wav"))


def test_on_large_stack_requests_a_large_stack_then_restores_the_default(
    monkeypatch,
) -> None:
    requested: list[int] = []

    def _spy(size: int = 0) -> int:
        requested.append(size)
        return 0

    monkeypatch.setattr(threading, "stack_size", _spy)
    assert eval_backends.on_large_stack(lambda wav: "ok")(Path("a.wav")) == "ok"
    assert requested[0] >= 64 * 1024 * 1024
    assert requested[-1] == 0
