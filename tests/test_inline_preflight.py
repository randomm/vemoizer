"""Inline preflight tests (issue #79, M7).

The inline preflight in ``transcribe_file`` (issue #79, M7) runs before
any decode: ffmpeg, config parse, all 5 pinned models cached, and the HF
token when the meeting profile runs or diarization is requested.  A red
check aborts before a single decode is spent.  These tests exercise the
preflight's interaction with the pipeline entry point.

All stages are mocked — no models, no network, no ffmpeg.
"""

from __future__ import annotations

from test_pipeline import (
    _patch_decoders,
    _patch_ingest,
    _patch_preflight_ffmpeg_fail,
    _patch_preflight_pass,
    _patch_preflight_token_fail,
    _patch_vad,
)

import vemoizer.pipeline as pipeline
import vemoizer.preflight as preflight
from vemoizer.pipeline import transcribe_file


def test_preflight_red_aborts_before_any_decode(monkeypatch) -> None:
    """A red preflight must short-circuit with an error key and no ingest."""
    _patch_preflight_ffmpeg_fail(monkeypatch)

    def _boom(path):
        raise AssertionError("ingest must not run when preflight is red")

    monkeypatch.setattr(pipeline, "ingest_audio", _boom)
    result = transcribe_file("/nonexistent.m4a")
    assert "error" in result
    assert "preflight" in result["error"]
    assert result["text"] == ""
    assert result["segments"] == []


def test_preflight_token_check_only_when_diarizing(monkeypatch) -> None:
    """The HF-token check fires only for diarizing runs (memo: diarize=False)."""
    _patch_preflight_token_fail(monkeypatch)
    _patch_ingest(monkeypatch)
    _patch_vad(monkeypatch)
    _patch_decoders(
        monkeypatch,
        {"text": "hei", "words": [{"word": "hei", "start": 0.0, "end": 0.3}]},
        None,
    )
    # diarize=False: the token check must NOT run → green → decode proceeds
    result = transcribe_file("/nonexistent.m4a", diarize=False)
    assert "error" not in result
    assert result["text"] == "hei"

    # diarize=True: the token check runs → red → abort before decode
    def _boom_decode():
        raise AssertionError("decode must not run when preflight is red")

    monkeypatch.setattr(pipeline, "ingest_audio", _boom_decode)
    result = transcribe_file("/nonexistent.m4a", diarize=True)
    assert "error" in result
    assert "token" in result["error"]


def test_preflight_token_check_fires_for_meeting_profile(monkeypatch) -> None:
    """Spec d4: the HF-token check fires when profile=="meeting" OR diarize,
    so a meeting run with --no-diarize still goes through the gated path.
    """
    _patch_preflight_token_fail(monkeypatch)

    def _boom_decode(*a, **kw):
        raise AssertionError("decode must not run when preflight is red")

    monkeypatch.setattr(pipeline, "ingest_audio", _boom_decode)
    result = transcribe_file("/nonexistent.m4a", profile="meeting", diarize=False)
    assert "error" in result
    assert "token" in result["error"]


def test_preflight_green_runs_pipeline_normally(monkeypatch, tmp_path) -> None:
    _patch_preflight_pass(monkeypatch)
    _patch_ingest(monkeypatch)
    _patch_vad(monkeypatch)
    _patch_decoders(
        monkeypatch,
        {"text": "hei maailma", "words": [{"word": "hei", "start": 0.0, "end": 0.4}]},
        None,
    )
    result = transcribe_file("/nonexistent.m4a")
    assert result["text"] == "hei maailma"
    assert "error" not in result


def test_preflight_models_missing_is_red(monkeypatch) -> None:
    """A missing pinned model (cache size 0) fails preflight before decode."""
    monkeypatch.setattr(preflight, "ffmpeg_ok", lambda: True)
    monkeypatch.setattr(preflight, "config_parse_ok", lambda: True)
    monkeypatch.setattr(preflight, "models_cached", lambda: ["parakeet"])
    monkeypatch.setattr(preflight, "hf_token_present", lambda: True)

    def _boom(path):
        raise AssertionError("ingest must not run when preflight is red")

    monkeypatch.setattr(pipeline, "ingest_audio", _boom)
    result = transcribe_file("/nonexistent.m4a")
    assert "error" in result
    assert "parakeet" in result["error"]


def test_preflight_config_parse_failure_is_red(monkeypatch) -> None:
    monkeypatch.setattr(preflight, "ffmpeg_ok", lambda: True)
    monkeypatch.setattr(preflight, "config_parse_ok", lambda: False)
    monkeypatch.setattr(preflight, "models_cached", lambda: [])
    monkeypatch.setattr(preflight, "hf_token_present", lambda: True)

    def _boom(path):
        raise AssertionError("ingest must not run when preflight is red")

    monkeypatch.setattr(pipeline, "ingest_audio", _boom)
    result = transcribe_file("/nonexistent.m4a")
    assert "error" in result
    assert "config" in result["error"]
