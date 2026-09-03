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
import logging
from pathlib import Path
from typing import TYPE_CHECKING

import mlx.core as mx
import numpy as np

if TYPE_CHECKING:
    from .canary_mlx import CanaryModel

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# q8 dequantization
# ---------------------------------------------------------------------------


def _dequant_grouped(
    weight_u32: np.ndarray, scales: np.ndarray, biases: np.ndarray
) -> np.ndarray:
    """Dequantize grouped 8-bit ints packed into uint32.

    ``weight`` is ``(out, n_packs)`` uint32; each uint32 packs **4 signed int8**
    in little-endian byte order (u32 = 4 bytes = 4 x int8), so the unpacked
    weight is ``(out, n_packs * 4)``. ``scales`` / ``biases`` are
    ``(out, n_groups)`` float — one pair per group of 64 weights
    (``mx.dequantize(w, scales, biases, 64, 8)`` semantics).

    The relationship: ``n_packs = 16 * n_groups`` (since 64 weights per group
    = 16 packs of 4 bytes). Each group's (scale, bias) is broadcast over its
    64 unpacked int8 weights.

    Returns ``(out, n_packs * 4)`` float32.
    """
    w = weight_u32.astype(np.uint32)
    out, n_packs = w.shape
    n_groups = scales.shape[1] if scales.ndim == 2 else scales.shape[0]

    if scales.ndim != 2 or scales.shape != (out, n_groups):
        raise ValueError(
            f"scales shape mismatch: scales={scales.shape}, weight={w.shape}"
        )
    if biases.ndim != 2 or biases.shape != (out, n_groups):
        raise ValueError(
            f"biases shape mismatch: biases={biases.shape}, weight={w.shape}"
        )
    if n_packs != 16 * n_groups:
        raise ValueError(
            f"weight has {n_packs} packs per row; expected 16 * {n_groups} = "
            f"{16 * n_groups} (4 bytes per pack, 64 weights per group)"
        )

    # Unpack: 4 signed int8 per uint32, little-endian byte order (b0 LSB first).
    ints = np.ascontiguousarray(w).view(np.int8).reshape(out, n_packs * 4)
    # Reshape to (out, n_groups, 64) so each group's 64 weights are contiguous.
    ints_g = ints.reshape(out, n_groups, 64)
    # Broadcast each group's (scale, bias) over its 64 weights.
    scale_r = scales[:, :, None]  # (out, n_groups, 1)
    bias_r = biases[:, :, None]  # (out, n_groups, 1)
    deq_g = ints_g.astype(np.float32) * scale_r + bias_r
    return deq_g.reshape(out, n_packs * 4)


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
    if not isinstance(weights, dict):
        raise TypeError(f"Expected dict from mx.load, got {type(weights)}")
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
            logger.warning(
                "Grouped-q8 weight %s is missing its .scales/.biases; dropping "
                "the linear (it will run on random init).",
                key,
            )
            continue
        # Fail loud on a corrupt weight: a silent drop here is exactly the
        # bug that once ran the whole model on random init (issue #36).
        try:
            w = to_np(value).astype(np.uint32)
            sc = to_np(weights[scale_key]).astype(np.float32)
            bz = to_np(weights[bias_key]).astype(np.float32)
            deq = _dequant_grouped(w, sc, bz)
        except Exception as e:
            raise RuntimeError(
                f"Failed to dequantize grouped-q8 linear {key}: {e!r}"
            ) from e
        out[key] = mx.array(deq).astype(dtype)
        consumed.update({key, scale_key, bias_key})

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
