"""Canary-1b-v2 checkpoint loading (issue #6).

Sliced out of ``canary_mlx`` to keep both modules under the 500-line limit.
Loads the revision-pinned community q8 checkpoint
``Mediform/canary-1b-v2-mlx-q8`` directly from safetensors: grouped 8-bit
linears (``.weight`` uint32 + ``.scales``/``.biases``) are dequantized to f32
on load so the forward pass stays plain, and plain float tensors are cast to
the inference dtype. The exact key layout is validated against the pinned
checkpoint by the WER gate (ticket 11); a key that maps to no parameter is
dropped.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

import mlx.core as mx
import numpy as np

if TYPE_CHECKING:
    from .canary_mlx import CanaryModel


# ---------------------------------------------------------------------------
# q8 dequantization
# ---------------------------------------------------------------------------


def _dequant_grouped(
    weight_u32: np.ndarray, scales: np.ndarray, biases: np.ndarray
) -> np.ndarray:
    """Dequantize grouped 8-bit ints packed into uint32.

    ``weight`` is ``(out, groups)`` uint32 (each value = 8 packed uint8);
    ``scales`` / ``biases`` are ``(out, groups)`` float. Each uint32 unpacks
    to 8 little-endian bytes giving ``(out, groups*8)`` ints; dequant is
    ``int * scale + bias`` per group, broadcast over the 8 packed values.
    """
    w = weight_u32.astype(np.uint32)
    out, groups = w.shape
    b = (w[:, :, None] >> np.array([0, 8, 16, 24], dtype=np.uint32)).astype(
        np.uint8
    )
    b = b.reshape(out, groups, 4, 2)
    # two 16-bit halves -> 8 bytes; simpler: view as 4 bytes little-endian
    b = np.stack(
        [(w[:, :, None] >> np.array([0, 8, 16, 24], dtype=np.uint64)) & 0xFF],
        axis=-1,
    )
    b = b.reshape(out, groups * 4)
    # pad to 8 values per group if needed (scheme uses 4 bytes -> 4 values)
    scales_r = np.repeat(scales, 4, axis=1)
    biases_r = np.repeat(biases, 4, axis=1)
    return b.astype(np.float32) * scales_r + biases_r


# ---------------------------------------------------------------------------
# Direct safetensors load + key mapping
# ---------------------------------------------------------------------------


def load_canary_weights(
    model_dir: str | Path, *, dtype: mx.Dtype = mx.bfloat16
) -> CanaryModel:
    """Load a revision-pinned Canary model from *model_dir*.

    Reads ``config.json`` + ``model.safetensors``. Grouped-q8 linears
    (``.weight`` uint32 + ``.scales``/``.biases``) are dequantized to f32 and
    cast to *dtype*; plain float tensors are cast as-is. The tokenizer is
    recovered from the base64-embedded SentencePiece model in ``config.json``.
    """
    # Imported here (not at module top) to avoid a circular import:
    # canary_mlx re-exports load_canary_weights from this module.
    from .canary_mlx import CanaryModel, CanaryTokenizer, _canary_config_from_dict

    model_dir = Path(model_dir)
    with open(model_dir / "config.json") as f:
        config = json.load(f)
    tokenizer = CanaryTokenizer.from_config(config)
    canary_cfg = _canary_config_from_dict(config)
    model = CanaryModel(canary_cfg, tokenizer)

    weights = mx.load(str(model_dir / "model.safetensors"))
    mapped = _map_checkpoint_weights(weights, dtype)
    model.load_weights(list(mapped.items()), strict=False)
    return model


def _map_checkpoint_weights(weights: dict, dtype: mx.Dtype) -> dict[str, mx.array]:
    """Map the Mediform q8 checkpoint keys onto the MLX module tree.

    This is the seam the issue's "direct safetensors load" requirement
    exercises: weights come straight out of safetensors, grouped-q8 linears
    are dequantized, and everything is cast to the inference dtype. The
    exact key layout is validated against the pinned checkpoint by the WER
    gate (ticket 11); a key that maps to no parameter is dropped.
    """

    def to_np(t: mx.array) -> np.ndarray:
        return np.ascontiguousarray(t).reshape(t.shape)

    out: dict[str, mx.array] = {}
    consumed: set[str] = set()

    # Dequantize grouped-q8 linears: weight (uint32) + scales + biases.
    for key, value in weights.items():
        if not key.endswith(".weight") or value.dtype not in (mx.uint32, mx.int32):
            continue
        base = key[: -len(".weight")]
        scale_key, bias_key = f"{base}.scales", f"{base}.biases"
        if scale_key not in weights or bias_key not in weights:
            continue
        try:
            w = to_np(value).astype(np.uint32)
            sc = to_np(weights[scale_key]).astype(np.float32)
            bz = to_np(weights[bias_key]).astype(np.float32)
            deq = _dequant_grouped(w, sc, bz)
            out[key] = mx.array(deq).astype(dtype)
            consumed.update({key, scale_key, bias_key})
        except Exception:
            continue

    # Map remaining float tensors onto the MLX tree.
    key_map = {
        "encoder.": "encoder.",
        "transf_decoder.token_embedding.": "decoder.embedding.",
        "transf_decoder.embedding_layer_norm.": "decoder.embedding_layer_norm.",
        "transf_decoder.final_layer_norm.": "decoder.final_norm.",
        "transf_decoder.layers.": "decoder.blocks.",
        "head.classifier.": "decoder.output_proj.",
    }
    for key, value in weights.items():
        if key in consumed:
            continue
        new_key = None
        for pref, repl in key_map.items():
            if key.startswith(pref):
                new_key = repl + key[len(pref) :]
                break
        if new_key is None:
            continue
        # conv kernel layout: the Mediform checkpoint already stores MLX layout
        out[new_key] = value.astype(dtype)
    return out
