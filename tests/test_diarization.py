"""Tests for src/vemoizer/diarization.py (issue #13).

All tests mock pyannote: the library is NOT installed in the dev
environment, the weights are gated, and AGENTS.md forbids model or
network access in unit tests.

The lazy-import tests below stub ``sys.modules`` with typed fake module
objects (issue #141): the stubs mirror the real module surface the
production code reaches (``pyannote.audio.Pipeline``,
``huggingface_hub.snapshot_download``) so no suppression comments are
needed to satisfy the checker.
"""

from __future__ import annotations

from collections.abc import Callable
from types import ModuleType, SimpleNamespace
from typing import Any
from unittest import mock

import numpy as np
import pytest

from vemoizer.diarization import (
    ATTRIBUTION,
    DIARIZATION_REPO_ID,
    DIARIZATION_REVISION,
    DiarizationResult,
    _load_pipeline,
    diarize,
    run_diarization_stage,
)

_AUDIO = np.zeros(16000, dtype=np.float32)
_TURNS = [
    (0.0, 2.0, "SPEAKER_00"),
    (2.5, 4.0, "SPEAKER_01"),
]


def _fake_diarization() -> mock.Mock:
    """Annotation fake matching the REAL pyannote contract:
    itertracks(yield_label=True) yields (segment, track, label) tuples."""
    diarization = mock.Mock()
    diarization.itertracks.return_value = [
        (SimpleNamespace(start=s, end=e), "track", sp) for s, e, sp in _TURNS
    ]
    return diarization


def _fake_pipeline_for(device: str) -> mock.Mock:
    """A pipeline whose .to() records the device and whose call succeeds.

    If *device* is ``"mps"`` and ``_fake_pipeline_for.mps_fails`` is set,
    the call raises (simulating the M4 kernel crash).
    """
    pipeline = mock.Mock()
    pipeline.to.side_effect = None
    wrapper = mock.Mock(spec=["speaker_diarization", "speaker_embeddings"])
    wrapper.speaker_diarization = _fake_diarization()
    pipeline.return_value = wrapper
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


class _PyannoteAudioModule(ModuleType):
    """Typed stand-in for the ``pyannote.audio`` module (issue #141).

    The lazy import in ``_load_pipeline`` resolves ``from pyannote.audio
    import Pipeline`` through ``sys.modules``; a plain ``types.ModuleType``
    has no ``Pipeline`` attribute in the checker's view, which is what
    forced the old suppressions. A subclass with a declared class attribute
    keeps the same runtime semantics with a clean static surface.
    """

    Pipeline: Any

    def __init__(self, pipeline_cls: Any) -> None:
        super().__init__("pyannote.audio")
        self.Pipeline = pipeline_cls


class _HuggingfaceHubModule(ModuleType):
    """Typed stand-in for the ``huggingface_hub`` module (issue #141)."""

    snapshot_download: Callable[..., Any]

    def __init__(self, snapshot_download_fn: Callable[..., Any]) -> None:
        super().__init__("huggingface_hub")
        self.snapshot_download = snapshot_download_fn


class _PyannoteParentModule(ModuleType):
    """Typed stand-in for the ``pyannote`` parent package (issue #141)."""

    audio: Any

    def __init__(self, audio_module: ModuleType) -> None:
        super().__init__("pyannote")
        self.audio = audio_module


def _install_fake_modules(
    monkeypatch: pytest.MonkeyPatch, pipeline_cls: mock.Mock
) -> mock.Mock:
    """Register typed fakes for pyannote and huggingface_hub in sys.modules."""
    import sys

    module = _PyannoteAudioModule(pipeline_cls)
    parent = _PyannoteParentModule(module)
    monkeypatch.setitem(sys.modules, "pyannote", parent)
    monkeypatch.setitem(sys.modules, "pyannote.audio", module)
    monkeypatch.setitem(sys.modules, "torch", mock.Mock())

    fake_snapshot = mock.Mock(return_value="/fake/hf-cache/snapshot")
    hf_mod = _HuggingfaceHubModule(fake_snapshot)
    monkeypatch.setitem(sys.modules, "huggingface_hub", hf_mod)
    return fake_snapshot


def test_load_pipeline_lazy_imports_pyannote(monkeypatch):
    """pyannote is imported only inside _load_pipeline, weights are pinned.

    ``snapshot_download`` is called with the full-SHA revision and the
    pipeline is loaded from the returned local snapshot path — never the
    bare repo ID (invariant #4).
    """
    import re

    assert re.fullmatch(r"[0-9a-f]{40}", DIARIZATION_REVISION)

    fake_pipeline_obj = mock.Mock()
    fake_pipeline_cls = mock.Mock()
    fake_pipeline_cls.from_pretrained.return_value = fake_pipeline_obj

    fake_snapshot = _install_fake_modules(monkeypatch, fake_pipeline_cls)
    monkeypatch.setenv("HF_TOKEN", "hf_test_token")

    pipeline = _load_pipeline("cpu")
    assert pipeline is fake_pipeline_obj
    # Warm-cache silencing probe (local-only) + the real pinned call; the
    # token is forwarded to the real call only, never to the local-only probe.
    for call in fake_snapshot.call_args_list:
        _, kwargs = call
        assert kwargs.get("revision") == DIARIZATION_REVISION
    real = [
        c for c in fake_snapshot.call_args_list if not c.kwargs.get("local_files_only")
    ]
    assert len(real) == 1
    assert real[0].args[0] == DIARIZATION_REPO_ID
    assert real[0].kwargs.get("token") == "hf_test_token"


def test_load_pipeline_warm_cache_silences_progress_bars(monkeypatch):
    """The diarization loader resolves its weights through the shared models
    seam (issue #147): a locally-complete snapshot is fetched silently, and
    the gated-repo ``token`` is forwarded to the real download — the
    local-only probe never carries it."""
    from unittest.mock import patch

    from huggingface_hub import utils as hf_utils

    fake_pipeline_obj = mock.Mock()
    fake_pipeline_cls = mock.Mock()
    fake_pipeline_cls.from_pretrained.return_value = fake_pipeline_obj
    _install_fake_modules(monkeypatch, fake_pipeline_cls)
    monkeypatch.setenv("HF_TOKEN", "hf_test_token")

    seen: list[bool] = []
    token_calls: list[str | None] = []

    def fake_snapshot(repo_id, **kwargs):
        if not kwargs.get("local_files_only"):
            token_calls.append(kwargs.get("token"))
        seen.append(hf_utils.are_progress_bars_disabled())
        return "/fake/hf-cache/snapshot"

    with (
        patch("huggingface_hub.snapshot_download", side_effect=fake_snapshot) as snap,
        patch("vemoizer.model_cache.snapshot_locally_complete", return_value=True),
    ):
        pipeline = _load_pipeline("cpu")

    assert pipeline is fake_pipeline_obj
    assert not hf_utils.are_progress_bars_disabled()  # no leaked disable
    # One call, inside the silencing switch (warm cache).
    assert seen == [True]
    # The token reached exactly the one real download call (the local-only
    # probe does not carry it).
    assert snap.call_count == 1
    assert token_calls == ["hf_test_token"]
    # Pipeline loads from the local snapshot path, never the bare repo ID.
    fake_pipeline_cls.from_pretrained.assert_called_once_with("/fake/hf-cache/snapshot")


def test_load_pipeline_raises_on_none(monkeypatch):
    """A snapshot that yields no loadable pipeline fails loudly with a clear
    error (instead of ``None.to``) rather than a confusing downstream crash."""
    fake_pipeline_cls = mock.Mock()
    fake_pipeline_cls.from_pretrained.return_value = None

    _install_fake_modules(monkeypatch, fake_pipeline_cls)
    monkeypatch.delenv("HF_TOKEN", raising=False)

    with pytest.raises(RuntimeError, match="returned None"):
        _load_pipeline("cpu")


def test_attribution_string_is_cc_by():
    assert ATTRIBUTION.startswith(
        "Speaker diarization: pyannote/speaker-diarization-community-1"
    )
    assert "CC-BY-4.0" in ATTRIBUTION
    assert DIARIZATION_REPO_ID == "pyannote/speaker-diarization-community-1"


def test_load_pipeline_disables_telemetry(monkeypatch):
    """After _load_pipeline with pyannote mocked, PYANNOTE_METRICS_ENABLED is false."""
    monkeypatch.delenv("PYANNOTE_METRICS_ENABLED", raising=False)
    monkeypatch.delenv("OTEL_SDK_DISABLED", raising=False)

    fake_pipeline_obj = mock.Mock()
    fake_pipeline_cls = mock.Mock()
    fake_pipeline_cls.from_pretrained.return_value = fake_pipeline_obj

    _install_fake_modules(monkeypatch, fake_pipeline_cls)
    monkeypatch.delenv("HF_TOKEN", raising=False)

    _load_pipeline("cpu")

    import os

    assert os.environ["PYANNOTE_METRICS_ENABLED"] == "false"
    assert os.environ["OTEL_SDK_DISABLED"] == "true"


def test_load_pipeline_preserves_user_opt_in(monkeypatch):
    """A pre-existing PYANNOTE_METRICS_ENABLED=true is not overwritten."""
    monkeypatch.setenv("PYANNOTE_METRICS_ENABLED", "true")
    monkeypatch.delenv("OTEL_SDK_DISABLED", raising=False)

    fake_pipeline_obj = mock.Mock()
    fake_pipeline_cls = mock.Mock()
    fake_pipeline_cls.from_pretrained.return_value = fake_pipeline_obj

    _install_fake_modules(monkeypatch, fake_pipeline_cls)
    monkeypatch.delenv("HF_TOKEN", raising=False)

    _load_pipeline("cpu")

    import os

    # User's deliberate opt-in is preserved (setdefault, not set).
    assert os.environ["PYANNOTE_METRICS_ENABLED"] == "true"
    # OTEL_SDK_DISABLED is set via setdefault (belt-and-braces).
    assert os.environ["OTEL_SDK_DISABLED"] == "true"


def test_pipeline_receives_waveform_tensor_not_ndarray(monkeypatch):
    """pyannote 4.x expects {"waveform": Tensor(channel, time), "sample_rate"}.

    The old {"audio": ndarray} key means "a file path" to pyannote and is
    rejected at runtime — the stage could never actually run.
    """
    received: dict[str, object] = {}

    def fake_pipeline(waveforms: dict[str, Any]) -> mock.Mock:
        received.update(waveforms)
        wrapper = mock.Mock(spec=["speaker_diarization"])
        wrapper.speaker_diarization = _fake_diarization()
        return wrapper

    pipeline = mock.Mock(side_effect=fake_pipeline)
    monkeypatch.setattr("vemoizer.diarization._load_pipeline", lambda device: pipeline)
    diarize(_AUDIO, device="cpu")

    assert "waveform" in received
    assert "audio" not in received
    assert received["sample_rate"] == 16_000
    waveform: Any = received["waveform"]
    # (channel, time) with a leading singleton channel dim
    assert tuple(waveform.shape) == (1, len(_AUDIO))


def _received_kwargs(monkeypatch, speakers) -> dict:
    received: dict[str, object] = {}

    def fake_pipeline(waveforms: dict[str, object], **kwargs: object) -> mock.Mock:
        received.update(kwargs)
        wrapper = mock.Mock(spec=["speaker_diarization"])
        wrapper.speaker_diarization = _fake_diarization()
        return wrapper

    pipeline = mock.Mock(side_effect=fake_pipeline)
    monkeypatch.setattr("vemoizer.diarization._load_pipeline", lambda device: pipeline)
    diarize(_AUDIO, device="cpu", num_speakers=speakers)
    return received


def test_exact_speaker_count_pins_num_speakers(monkeypatch):
    assert _received_kwargs(monkeypatch, 4) == {"num_speakers": 4}


def test_speaker_range_bounds_clustering(monkeypatch):
    assert _received_kwargs(monkeypatch, (3, 5)) == {
        "min_speakers": 3,
        "max_speakers": 5,
    }


def test_no_speaker_count_leaves_clustering_free(monkeypatch):
    assert _received_kwargs(monkeypatch, None) == {}


def test_keyboard_interrupt_in_diarize_propagates_out_of_stage(
    monkeypatch,
) -> None:
    """A Ctrl-C raised inside the real ``diarize`` seam bound in
    ``run_diarization_stage`` must propagate OUT of the stage (issue #148):
    the stage boundary only fails open for ``Exception``; ``KeyboardInterrupt``
    is a ``BaseException`` and the Ctrl-C interrupt line must be able to
    name the stage the run was in."""

    def fake_diarize(audio, *, device="auto", num_speakers=None):
        raise KeyboardInterrupt()

    monkeypatch.setattr("vemoizer.diarization.diarize", fake_diarize)
    with pytest.raises(KeyboardInterrupt):
        run_diarization_stage(_AUDIO)
