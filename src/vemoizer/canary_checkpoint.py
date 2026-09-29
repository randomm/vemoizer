"""Canary-1b-v2 checkpoint loading (issue #6).

Sliced out of ``canary_mlx`` to keep both modules under the 500-line limit.
Loads the revision-pinned community q8 checkpoint
``Mediform/canary-1b-v2-mlx-q8`` directly from safetensors: grouped 8-bit
linears (``.weight`` uint32 + ``.scales``/``.biases``) are dequantized via
``mx.dequantize`` on load so the forward pass stays plain, and plain float
tensors are cast to the inference dtype.

The key layout is enforced at load time: every parameter in the module tree
must be filled by the checkpoint, or the load raises. A silently dropped key
leaves that parameter at its random initialization and yields a model that
loads cleanly and emits noise (issue #36).
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

import mlx.core as mx
import numpy as np
from mlx.utils import tree_flatten

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

    Note: kept only for testability / diagnostics — the load path uses
    ``mx.dequantize`` (below), which never touches the numpy buffer interface
    for the scales/biases and so cannot fail on the checkpoint's uint32-packed
    bfloat16 scales (PEP 3118 exposes only float32 from MLX).
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
    _assert_full_coverage(model, mapped)
    model.load_weights(list(mapped.items()), strict=False)
    # MLX modules construct with ``training=True``, which makes the encoder's
    # BatchNorm use per-utterance batch statistics and ignore the checkpoint's
    # running_mean/running_var. Inference must use the running stats.
    model.eval()
    return model


class _HasParameters(Protocol):
    """The only thing the coverage guard needs from a model."""

    def parameters(self) -> dict: ...


def _assert_full_coverage(model: _HasParameters, mapped: dict[str, mx.array]) -> None:
    """Fail loud when any model parameter would keep its random init.

    ``load_weights(strict=False)`` leaves an unmatched parameter at its
    initialization values and reports nothing, so a key-mapping gap produces a
    model that loads cleanly, runs at full speed and emits noise. That is
    issue #36, and it regressed once already on the decoder path: the
    ``transf_decoder.layers.*`` sub-layer names never matched, so 8 blocks x 26
    tensors stayed random while the encoder loaded fine.

    Checking coverage here — against the module tree, not against the
    checkpoint — is what makes that class of bug impossible to reintroduce
    silently: any parameter the mapping fails to fill aborts the load.
    """
    expected = {name for name, _ in tree_flatten(model.parameters())}
    missing = sorted(expected - set(mapped))
    if missing:
        shown = ", ".join(missing[:8])
        more = f" (+{len(missing) - 8} more)" if len(missing) > 8 else ""
        raise RuntimeError(
            f"Canary checkpoint covers {len(expected) - len(missing)}/"
            f"{len(expected)} model parameters; {len(missing)} would run on "
            f"random init: {shown}{more}"
        )
    unused = sorted(set(mapped) - expected)
    if unused:
        logger.warning(
            "Checkpoint supplied %d weights with no matching parameter: %s",
            len(unused),
            ", ".join(unused[:8]),
        )


#: NeMo's ``TransformerDecoderBlock`` names its sub-layers and their pre-norms
#: *positionally* (first = self-attention, second = cross-attention, third =
#: feed-forward), while the MLX port names them semantically. The decoder half
#: of the checkpoint therefore needs a real translation, not the prefix swap
#: the encoder gets: ``transf_decoder.layers.N.first_sub_layer.linear_q`` and
#: ``decoder.blocks.N.self_attn.q_proj`` share no substring at all.
_DECODER_SUBLAYER_MAP = {
    "first_sub_layer.linear_q": "self_attn.q_proj",
    "first_sub_layer.linear_k": "self_attn.k_proj",
    "first_sub_layer.linear_v": "self_attn.v_proj",
    "first_sub_layer.linear_out": "self_attn.out_proj",
    "second_sub_layer.linear_q": "cross_attn.q_proj",
    "second_sub_layer.linear_k": "cross_attn.k_proj",
    "second_sub_layer.linear_v": "cross_attn.v_proj",
    "second_sub_layer.linear_out": "cross_attn.out_proj",
    "third_sub_layer.linear1": "ff1",
    "third_sub_layer.linear2": "ff2",
    "layer_norm_1": "self_attn_norm",
    "layer_norm_2": "cross_attn_norm",
    "layer_norm_3": "ff_norm",
}

#: Whole-subtree renames. The encoder's entry is an identity map: the MLX
#: encoder mirrors the checkpoint's Conformer names 1:1.
_PREFIX_MAP = {
    "encoder.": "encoder.",
    "transf_decoder.token_embedding.": "decoder.embedding.",
    "transf_decoder.embedding_layer_norm.": "decoder.embedding_layer_norm.",
    "transf_decoder.final_layer_norm.": "decoder.final_norm.",
    "head.classifier.": "decoder.output_proj.",
}

_DECODER_LAYER_RE = re.compile(r"^transf_decoder\.layers\.(\d+)\.(.+)$")


def _map_key(key: str) -> str | None:
    """Translate one checkpoint key to its MLX parameter path (``None`` = drop)."""
    match = _DECODER_LAYER_RE.match(key)
    if match is not None:
        index, rest = match.group(1), match.group(2)
        for nemo_name, mlx_name in _DECODER_SUBLAYER_MAP.items():
            if rest.startswith(f"{nemo_name}."):
                leaf = rest[len(nemo_name) + 1 :]
                return f"decoder.blocks.{index}.{mlx_name}.{leaf}"
        return None
    for pref, repl in _PREFIX_MAP.items():
        if key.startswith(pref):
            return repl + key[len(pref) :]
    return None


def _map_checkpoint_weights(weights: dict, dtype: mx.Dtype) -> dict[str, mx.array]:
    """Map the Mediform q8 checkpoint keys onto the MLX module tree.

    This is the seam the issue's "direct safetensors load" requirement
    exercises: weights come straight out of safetensors, grouped-q8 linears
    are dequantized, and everything is cast to the inference dtype.

    Both halves — the dequantized q8 linears and the plain float tensors —
    go through :func:`_map_key`. Dequantized weights used to be emitted under
    their raw checkpoint name, which was invisible for the encoder (its
    rename is the identity) and silently left every quantized decoder linear
    on random init. :func:`load_canary_weights` asserts full coverage, so a
    key that maps to nothing is now a loud failure rather than a wrong model.
    """

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
        # Dequantize via mx.dequantize: it keeps the weight AND the
        # uint32-packed bfloat16 scales/biases in MLX land, so it avoids the
        # numpy PEP 3118 conversion that failed on the packed bf16 scales
        # ("bfloat16 is not a valid PEP 3118 buffer format string"). It
        # expects uint32 weights with 64 int8 per (scale, bias) group.
        # mx.dequantize propagates the scales dtype to the output, so cast
        # explicitly to the inference dtype afterwards.
        # Fail loud on a corrupt weight: a silent drop here is exactly the
        # bug that once ran the whole model on random init (issue #36).
        try:
            w = value.astype(mx.uint32)
            deq = mx.dequantize(w, weights[scale_key], weights[bias_key], 64, 8)
        except Exception as e:
            raise RuntimeError(
                f"Failed to dequantize grouped-q8 linear {key}: {e!r}"
            ) from e
        consumed.update({key, scale_key, bias_key})
        mapped_key = _map_key(key)
        if mapped_key is None:
            continue
        out[mapped_key] = deq.astype(dtype)

    # Map the remaining plain float tensors onto the MLX tree.
    for key, value in weights.items():
        if key in consumed:
            continue
        new_key = _map_key(key)
        if new_key is None:
            continue
        # conv kernel layout: the Mediform checkpoint already stores MLX layout
        out[new_key] = value.astype(dtype)
    return out
