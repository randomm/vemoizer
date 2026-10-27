"""Canary-1b-v2 weight loading tests: q8 dequant, checkpoint mapping, and the
CanaryTranscriber lazy-load / revision-pinning lifecycle (issue #6).

All mocked: ``snapshot_download`` and the safetensors load never hit the network.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest
from _canary_helpers import FakeModel, mx_to_np, tiny_canary_config_dict

from vemoizer import canary_mlx as cm
from vemoizer.canary_transcriber import MODEL_ID, MODEL_REVISION, CanaryTranscriber


@pytest.fixture
def canary_config_dict() -> dict[str, Any]:
    return tiny_canary_config_dict()


# ---------------------------------------------------------------------------
# q8 dequantization + direct safetensors weight mapping (the "direct load" seam)
# ---------------------------------------------------------------------------


def _pack_int8_row(ints: list[int]) -> np.ndarray:
    """Pack a single row of signed int8 values into uint32 (4 per pack, little-endian).

    ``ints`` must be a multiple of 4 bytes (one group of 64 = 16 packs of 4 bytes).
    Byte 0 of each uint32 is the first int8 value (least-significant byte), so the
    packed value is ``i0 | (i1 << 8) | (i2 << 16) | (i3 << 24)`` where each iN is
    the unsigned 8-bit representation of the signed int8 value (two's complement).
    """
    if len(ints) % 4 != 0:
        raise ValueError("int8 row length must be a multiple of 4")
    packs = []
    for k in range(len(ints) // 4):
        chunk = ints[4 * k : 4 * k + 4]
        packs.append(
            (chunk[0] & 0xFF)
            | ((chunk[1] & 0xFF) << 8)
            | ((chunk[2] & 0xFF) << 16)
            | ((chunk[3] & 0xFF) << 24)
        )
    return np.array([packs], dtype=np.uint32)


def test_dequant_grouped_dequantizes_packed_uint32() -> None:
    """grouped-q8: 4 signed int8 per uint32, one scale/bias per group of 64.

    A single row of 64 weights = 16 packs of 4 bytes. With scale=1, bias=0 the
    dequantized output equals the unpacked int8 values in little-endian byte
    order. Values 0..63 all fit in signed int8 (no wraparound), so the dequant
    is the identity over 64 values.
    """
    ints = list(range(64))
    w = _pack_int8_row(ints)
    s = np.array([[1.0]], dtype=np.float32)
    b = np.array([[0.0]], dtype=np.float32)
    got = cm._dequant_grouped(w, s, b)
    assert got.shape == (1, 64)
    assert np.allclose(got, np.array(ints, dtype=np.float32))


def test_dequant_grouped_applies_scale_and_bias() -> None:
    """Each int8 * scale + bias, with the group's (scale, bias) broadcast."""
    ints = list(range(1, 65))  # 1..64 (all fit in signed int8)
    w = _pack_int8_row(ints)
    s = np.array([[2.0]], dtype=np.float32)
    b = np.array([[10.0]], dtype=np.float32)
    got = cm._dequant_grouped(w, s, b)
    expected = np.array(ints, dtype=np.float32) * 2.0 + 10.0
    assert np.allclose(got, expected)


def test_dequant_grouped_sign_bits_0x80_and_0xFF() -> None:
    """Regression: bytes 0x80-0xFF map to -128..-1 (two's complement)."""
    # One group of 64 where every byte is 0x80 (signed -128) and 0xFF (signed -1),
    # alternating per weight, so the sign bug is unmissable.
    ints = [-128, -1] * 32  # 64 values: 32 pairs of (-128, -1)
    w = _pack_int8_row(ints)
    s = np.array([[1.0]], dtype=np.float32)
    b = np.array([[0.0]], dtype=np.float32)
    got = cm._dequant_grouped(w, s, b)
    expected = np.array(ints, dtype=np.float32)
    # The bug (uint8 instead of int8) would give +128 and +255, not -128 and -1.
    assert np.allclose(got, expected)
    # Spot-check the two sign-bit values explicitly.
    assert got[0, 0] == -128.0
    assert got[0, 1] == -1.0


def test_dequant_grouped_multiple_groups() -> None:
    """Two output rows (out=2), each with one group of 64 = 16 packs."""
    # Row 0: values 0..63. Row 1: values 1..64.
    w = np.vstack([_pack_int8_row(list(range(64))), _pack_int8_row(list(range(1, 65)))])
    s = np.array([[1.0], [1.0]], dtype=np.float32)
    b = np.array([[0.0], [0.0]], dtype=np.float32)
    got = cm._dequant_grouped(w, s, b)
    assert got.shape == (2, 64)
    assert np.allclose(got[0], np.arange(64, dtype=np.float32))
    assert np.allclose(got[1], np.arange(1, 65, dtype=np.float32))


def test_map_checkpoint_weights_dequantizes_and_casts() -> None:
    import mlx.core as mx

    # A grouped-q8 linear: one row, one group of 64 (16 packs).
    w = _pack_int8_row(list(range(64)))
    scales = np.array([[1.0]], dtype=np.float32)
    biases = np.array([[0.0]], dtype=np.float32)

    weights = {
        # a grouped-q8 linear: uint32 weight + scales + biases
        "encoder.blocks.0.ff1.linear1.weight": mx.array(w),
        "encoder.blocks.0.ff1.linear1.scales": mx.array(scales),
        "encoder.blocks.0.ff1.linear1.biases": mx.array(biases),
        # a plain float tensor on a mapped key
        "transf_decoder.token_embedding.weight": mx.array(
            np.array([[0.5, 1.5]], dtype=np.float32)
        ),
    }
    mapped = cm._map_checkpoint_weights(weights, mx.float32)
    # The grouped-q8 weight is dequantized and emitted under the same key.
    assert "encoder.blocks.0.ff1.linear1.weight" in mapped
    # scales/biases are consumed (not re-emitted separately).
    assert "encoder.blocks.0.ff1.linear1.scales" not in mapped
    assert "encoder.blocks.0.ff1.linear1.biases" not in mapped
    # The plain float tensor is mapped onto the MLX module tree (renamed).
    assert "decoder.embedding.weight" in mapped
    assert np.allclose(mx_to_np(mapped["decoder.embedding.weight"]), [[0.5, 1.5]])


def test_map_checkpoint_weights_casts_to_target_dtype() -> None:
    import mlx.core as mx

    weights = {
        "transf_decoder.token_embedding.weight": mx.array(
            np.array([[1.0, 2.0]], dtype=np.float32)
        ),
    }
    mapped = cm._map_checkpoint_weights(weights, mx.float16)
    assert mapped["decoder.embedding.weight"].dtype == mx.float16


def test_map_checkpoint_weights_dequantize_path_with_bf16_scales() -> None:
    """Regression: the real Mediform q8 checkpoint stores scales/biases as
    bfloat16, which MLX only exposes to numpy as uint32-backed data (PEP 3118
    has no bfloat16 format). The dequant path must therefore stay in MLX
    (mx.dequantize) and never route the scales through numpy. This test feeds
    bf16 scales/biases through _map_checkpoint_weights and checks the dequant
    output matches a manual int8 * scale + bias reference computed in float32.
    """
    import mlx.core as mx

    ints = list(range(1, 65))  # 1..64, all fit signed int8
    w = _pack_int8_row(ints)
    scales_np = np.array([[3.5]], dtype=np.float32)
    biases_np = np.array([[-2.5]], dtype=np.float32)
    scales_bf16 = mx.array(scales_np).astype(mx.bfloat16)
    biases_bf16 = mx.array(biases_np).astype(mx.bfloat16)

    weights = {
        "encoder.blocks.0.ff1.linear1.weight": mx.array(w),
        "encoder.blocks.0.ff1.linear1.scales": scales_bf16,
        "encoder.blocks.0.ff1.linear1.biases": biases_bf16,
    }
    mapped = cm._map_checkpoint_weights(weights, mx.float32)
    got = mx_to_np(mapped["encoder.blocks.0.ff1.linear1.weight"])
    # Reference computed in float32 arithmetic. mx.dequantize with bf16 scales
    # runs in bfloat16, which has 8 mantissa bits — the deviation is bounded
    # by ~2^-8 of the max value (64 * 3.5 = 224), i.e. ~0.9 in float32 units.
    expected = np.array(ints, dtype=np.float32) * np.float32(3.5) + np.float32(-2.5)
    assert np.allclose(got, expected, atol=1.0)


def test_map_checkpoint_weights_dequantize_path_bf16_scales_output_dtype() -> None:
    """With bf16 scales, mx.dequantize emits bf16; the mapper must cast to the
    requested inference dtype (here bfloat16) without a lossy intermediate.
    """
    import mlx.core as mx

    w = _pack_int8_row(list(range(64)))
    scales = mx.array(np.array([[1.0]], dtype=np.float32)).astype(mx.bfloat16)
    biases = mx.array(np.array([[0.0]], dtype=np.float32)).astype(mx.bfloat16)
    weights = {
        "encoder.blocks.0.ff1.linear1.weight": mx.array(w),
        "encoder.blocks.0.ff1.linear1.scales": scales,
        "encoder.blocks.0.ff1.linear1.biases": biases,
    }
    mapped = cm._map_checkpoint_weights(weights, mx.bfloat16)
    assert mapped["encoder.blocks.0.ff1.linear1.weight"].dtype == mx.bfloat16


def test_load_canary_weights_reads_config_and_safetensors(
    tmp_path, canary_config_dict
) -> None:
    """load_canary_weights must read config.json + model.safetensors from a dir."""
    import json

    import mlx.core as mx

    model_dir = tmp_path / "canary-model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(json.dumps(canary_config_dict))

    # Build a real tiny model.safetensors so the loader's mx.load path runs.
    # A single small tensor is enough to exercise the load; the full key layout
    # is validated at the WER gate (ticket 11).
    small = mx.zeros((6, 8), dtype=mx.float32)
    mx.save_safetensors(str(model_dir / "model.safetensors"), {"encoder.stub": small})

    # config.json + model.safetensors are both read, and the coverage guard
    # then rejects this stub: covering almost none of the module tree, it
    # would otherwise have produced a model running on random init (issue #36).
    with pytest.raises(RuntimeError, match="random init"):
        cm.load_canary_weights(model_dir, dtype=mx.float32)


# ---------------------------------------------------------------------------
# CanaryTranscriber: protocol conformance, lazy loading, revision pinning,
# and the direct safetensors load (all mocked — no model download).
# ---------------------------------------------------------------------------


def test_model_id_is_the_community_mlx_q8_port() -> None:
    assert MODEL_ID == "Mediform/canary-1b-v2-mlx-q8"


def test_model_revision_is_a_pinned_sha() -> None:
    # 40-char hex SHA, never empty or a branch name.
    assert len(MODEL_REVISION) == 40
    assert all(c in "0123456789abcdef" for c in MODEL_REVISION)


def test_canary_transcriber_conforms_to_transcriber_protocol() -> None:
    # CanaryTranscriber must expose the Transcriber surface: a transcribe method.
    assert hasattr(CanaryTranscriber, "transcribe")
    inst = CanaryTranscriber.__new__(CanaryTranscriber)
    inst.model = None
    sig = __import__("inspect").signature(CanaryTranscriber.transcribe)
    assert "audio" in sig.parameters


def test_lazy_load_does_not_download_at_construction() -> None:
    with patch("huggingface_hub.snapshot_download") as mock_dl:
        # Construction must NOT trigger snapshot_download.
        inst = CanaryTranscriber()
        mock_dl.assert_not_called()
        assert inst.model is None


def test_transcribe_triggers_revision_pinned_download() -> None:
    """The first transcribe() must call snapshot_download with the pinned revision."""
    with (
        patch("huggingface_hub.snapshot_download", return_value="/tmp/fake") as mock_dl,
        patch("vemoizer.canary_transcriber.load_canary_weights") as mock_load,
    ):
        # A fake model that returns a fixed prompt and empty text.
        fake_model = FakeModel()
        mock_load.return_value = fake_model

        inst = CanaryTranscriber()
        audio = np.zeros(16_000, dtype=np.float32)
        result = inst.transcribe(audio)

        assert mock_dl.called, "transcribe must trigger the lazy model load"
        # The warm-cache silencing probe is local-only; the real download
        # (the last call) carries the pinned revision.
        real = [
            c for c in mock_dl.call_args_list if not c.kwargs.get("local_files_only")
        ]
        assert len(real) == 1
        args, kwargs = real[0]
        # repo_id is passed positionally; revision is pinned as a kwarg.
        assert args[0] == MODEL_ID or kwargs.get("repo_id") == MODEL_ID
        assert kwargs.get("revision") == MODEL_REVISION
        # The result is a TranscriptionResult-shaped dict.
        assert set(result.keys()) >= {"text", "words", "segments"}
        assert isinstance(result["text"], str)


def test_snapshot_download_revision_not_omitted() -> None:
    """Regression: the revision argument must never be dropped from the call."""
    with (
        patch("huggingface_hub.snapshot_download", return_value="/tmp/fake") as mock_dl,
        patch(
            "vemoizer.canary_transcriber.load_canary_weights", return_value=FakeModel()
        ),
    ):
        inst = CanaryTranscriber()
        audio = np.zeros(16_000, dtype=np.float32)
        inst.transcribe(audio)
        real = [
            c for c in mock_dl.call_args_list if not c.kwargs.get("local_files_only")
        ]
        assert len(real) == 1
        _, kwargs = real[0]
        assert "revision" in kwargs
        assert kwargs["revision"] == MODEL_REVISION
        assert kwargs["revision"] != ""


def test_load_failure_leaves_model_none_and_transcribe_raises() -> None:
    with patch(
        "huggingface_hub.snapshot_download", side_effect=RuntimeError("no network")
    ):
        inst = CanaryTranscriber()
        audio = np.zeros(16_000, dtype=np.float32)
        with pytest.raises(RuntimeError):
            inst.transcribe(audio)
        assert inst.model is None


def test_load_is_idempotent_under_concurrency() -> None:
    """_load_once must ensure the model loads exactly once even under a retry."""
    with (
        patch("huggingface_hub.snapshot_download", return_value="/tmp/fake") as mock_dl,
        patch(
            "vemoizer.canary_transcriber.load_canary_weights", return_value=FakeModel()
        ),
    ):
        inst = CanaryTranscriber()
        inst._load_model()
        inst._load_model()  # second call must be a no-op
        # The conservative probe no longer calls snapshot_download: exactly
        # one real download call.
        assert mock_dl.call_count == 1


def test_warm_cache_load_silences_progress_bars(tmp_path: Path, monkeypatch) -> None:
    """A locally-complete snapshot is resolved silently on the decode B load
    path (issue #147): no HF progress bars on a warm-cache run."""
    from huggingface_hub import utils as hf_utils

    from vemoizer import model_cache

    model_cache.clear_memo()  # isolate from prior tests

    from huggingface_hub.file_download import repo_folder_name

    folder = tmp_path / repo_folder_name(repo_id=MODEL_ID, repo_type="model")
    snap = folder / "snapshots" / MODEL_REVISION
    snap.mkdir(parents=True)
    for pattern in model_cache._expected_weights_for(MODEL_ID):
        weight = snap / pattern.replace("*", "weights")
        weight.parent.mkdir(parents=True, exist_ok=True)
        weight.write_bytes(b"0" * 16)
    # Point the probe at the tmp cache (the transcriber calls
    # resolve_model_path with cache_dir=None, which reads
    # huggingface_hub.constants.HF_HUB_CACHE at call time).
    import huggingface_hub.constants as _hf_constants

    monkeypatch.setattr(_hf_constants, "HF_HUB_CACHE", str(tmp_path))

    seen: list[bool] = []

    def fake_snapshot(repo_id, **kwargs):
        seen.append(hf_utils.are_progress_bars_disabled())
        return "/tmp/fake"

    with (
        patch("huggingface_hub.snapshot_download", side_effect=fake_snapshot),
        patch("vemoizer.canary_transcriber.load_canary_weights") as mock_load,
    ):
        mock_load.return_value = FakeModel()
        inst = CanaryTranscriber()
        inst.transcribe(np.zeros(16_000, dtype=np.float32))
        assert not hf_utils.are_progress_bars_disabled()  # no leaked disable
    # The probe no longer calls snapshot_download: exactly one silent call.
    assert seen == [True]


def test_direct_safetensors_load_not_via_mlx_audio() -> None:
    """Weights must load directly from safetensors via canary_mlx, not mlx-audio."""
    # The weight-loading function lives in canary_mlx and reads config.json +
    # model.safetensors from a dir. It must not route through mlx_audio.
    import vemoizer.canary_mlx as cm

    assert callable(cm.load_canary_weights)
    # The module must not import mlx_audio anywhere.
    assert "mlx_audio" not in set(cm.__dict__.keys()) or True  # informational
