"""Tests for src/vemoizer/diarization.py (issue #13).

All tests mock pyannote: the library is NOT installed in the dev
environment, the weights are gated, and AGENTS.md forbids model or
network access in unit tests.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest import mock

import numpy as np

from vemoizer.diarization import (
    ATTRIBUTION,
    DIARIZATION_REPO_ID,
    DiarizationResult,
    _load_pipeline,
    diarize,
)

_AUDIO = np.zeros(16000, dtype=np.float32)
_TURNS = [
    (0.0, 2.0, "SPEAKER_00"),
    (2.5, 4.0, "SPEAKER_01"),
]


def _fake_diarization() -> mock.Mock:
    diarization = mock.Mock()
    diarization.itertracks.return_value = [
        SimpleNamespace(start=s, end=e, speaker=sp) for s, e, sp in _TURNS
    ]
    return diarization


def _fake_pipeline_for(device: str) -> mock.Mock:
    """A pipeline whose .to() records the device and whose call succeeds.

    If *device* is ``"mps"`` and ``_fake_pipeline_for.mps_fails`` is set,
    the call raises (simulating the M4 kernel crash).
    """
    pipeline = mock.Mock()
    pipeline.to.side_effect = None
    pipeline.return_value = _fake_diarization()
    if device == "mps" and getattr(_fake_pipeline_for, "mps_fails", False):
        pipeline.__call__ = mock.Mock(side_effect=RuntimeError("MPS kernel crash"))
    return pipeline


def test_diarize_returns_result(monkeypatch):
    pipeline = _fake_pipeline_for("cpu")
    monkeypatch.setattr("vemoizer.diarization._load_pipeline", lambda device: pipeline)
    result = diarize(_AUDIO, device="cpu")
    assert isinstance(result, DiarizationResult)
    assert result.segments == _TURNS


def test_mps_failure_falls_back_to_cpu(monkeypatch):
    """device='auto' tries MPS, then retries the whole pipeline on CPU."""
    calls = []

    def fake_load(device: str) -> mock.Mock:
        calls.append(device)
        if device == "mps":
            raise RuntimeError("MPS unavailable")
        pipeline = _fake_pipeline_for("cpu")
        return pipeline

    monkeypatch.setattr("vemoizer.diarization._load_pipeline", fake_load)
    result = diarize(_AUDIO)  # device="auto"
    assert calls == ["mps", "cpu"]
    assert isinstance(result, DiarizationResult)
    assert result.segments == _TURNS


def test_load_pipeline_lazy_imports_pyannote(monkeypatch):
    """pyannote is imported only inside _load_pipeline, with the gated repo."""
    import sys
    import types

    fake_pipeline_obj = mock.Mock()
    fake_pipeline_cls = mock.Mock()
    fake_pipeline_cls.from_pretrained.return_value = fake_pipeline_obj

    module = types.ModuleType("pyannote.audio")
    module.Pipeline = fake_pipeline_cls  # ty: ignore[unresolved-attribute]
    parent = types.ModuleType("pyannote")
    parent.audio = module  # ty: ignore[unresolved-attribute]
    monkeypatch.setitem(sys.modules, "pyannote", parent)
    monkeypatch.setitem(sys.modules, "pyannote.audio", module)
    monkeypatch.setitem(sys.modules, "torch", mock.Mock())
    monkeypatch.setenv("HF_TOKEN", "hf_test_token")

    pipeline = _load_pipeline("cpu")
    assert pipeline is fake_pipeline_obj
    fake_pipeline_cls.from_pretrained.assert_called_once_with(
        DIARIZATION_REPO_ID, use_auth_token="hf_test_token"
    )


def test_attribution_string_is_cc_by():
    assert ATTRIBUTION.startswith(
        "Speaker diarization: pyannote/speaker-diarization-community-1"
    )
    assert "CC-BY-4.0" in ATTRIBUTION
    assert DIARIZATION_REPO_ID == "pyannote/speaker-diarization-community-1"
