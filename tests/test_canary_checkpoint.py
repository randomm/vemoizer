"""Checkpoint key mapping and load-time coverage guard (issue #36 regression).

The Mediform q8 checkpoint names the decoder after NeMo's positional
sub-layers while the MLX port names them semantically. When that translation
is missing, ``load_weights(strict=False)`` silently leaves those parameters at
their random initialization and the model loads clean, runs fast, and emits
noise. These tests pin both the translation and the guard that makes such a
gap loud.

Pure logic — no model download, no network.
"""

from __future__ import annotations

import mlx.core as mx
import pytest

from vemoizer.canary_checkpoint import (
    _assert_full_coverage,
    _map_checkpoint_weights,
    _map_key,
)


class _FakeModel:
    """Just enough of a CanaryModel to exercise the coverage guard."""

    def __init__(self, names: list[str]) -> None:
        self._names = names

    def parameters(self) -> dict:
        return {n: mx.zeros((1,)) for n in self._names}


@pytest.mark.parametrize(
    ("checkpoint_key", "expected"),
    [
        # self-attention (NeMo "first" sub-layer)
        (
            "transf_decoder.layers.0.first_sub_layer.linear_q.weight",
            "decoder.blocks.0.self_attn.q_proj.weight",
        ),
        (
            "transf_decoder.layers.7.first_sub_layer.linear_out.bias",
            "decoder.blocks.7.self_attn.out_proj.bias",
        ),
        # cross-attention ("second")
        (
            "transf_decoder.layers.3.second_sub_layer.linear_k.weight",
            "decoder.blocks.3.cross_attn.k_proj.weight",
        ),
        # feed-forward ("third")
        (
            "transf_decoder.layers.1.third_sub_layer.linear1.weight",
            "decoder.blocks.1.ff1.weight",
        ),
        (
            "transf_decoder.layers.1.third_sub_layer.linear2.bias",
            "decoder.blocks.1.ff2.bias",
        ),
        # the three pre-norms are positional in NeMo, semantic in the port
        (
            "transf_decoder.layers.2.layer_norm_1.weight",
            "decoder.blocks.2.self_attn_norm.weight",
        ),
        (
            "transf_decoder.layers.2.layer_norm_2.weight",
            "decoder.blocks.2.cross_attn_norm.weight",
        ),
        (
            "transf_decoder.layers.2.layer_norm_3.bias",
            "decoder.blocks.2.ff_norm.bias",
        ),
        # whole-subtree renames
        ("transf_decoder.token_embedding.weight", "decoder.embedding.weight"),
        ("transf_decoder.final_layer_norm.bias", "decoder.final_norm.bias"),
        ("head.classifier.weight", "decoder.output_proj.weight"),
        # the encoder's rename is the identity
        ("encoder.layers.0.self_attn.linear_q.weight", None),
    ],
)
def test_map_key_translates_decoder_sublayers(
    checkpoint_key: str, expected: str | None
) -> None:
    mapped = _map_key(checkpoint_key)
    if expected is None:  # encoder keys pass through unchanged
        assert mapped == checkpoint_key
    else:
        assert mapped == expected


def test_map_key_drops_unknown_keys() -> None:
    assert _map_key("optimizer.state.step") is None
    assert _map_key("transf_decoder.layers.0.fourth_sub_layer.linear_q.weight") is None


def test_dequantized_weights_are_mapped_not_emitted_raw() -> None:
    """A q8 decoder linear must land on its MLX name, not its checkpoint name.

    Dequantized weights used to be emitted under the raw checkpoint key. That
    was invisible for the encoder (identity rename) and left every quantized
    decoder linear on random init.
    """
    base = "transf_decoder.layers.0.first_sub_layer.linear_q"
    weights = {
        f"{base}.weight": mx.zeros((8, 16), dtype=mx.uint32),
        f"{base}.scales": mx.ones((8, 1), dtype=mx.float32),
        f"{base}.biases": mx.zeros((8, 1), dtype=mx.float32),
    }
    mapped = _map_checkpoint_weights(weights, mx.float32)

    assert "decoder.blocks.0.self_attn.q_proj.weight" in mapped
    assert f"{base}.weight" not in mapped
    # 16 uint32 packs -> 64 int8 weights per row
    assert mapped["decoder.blocks.0.self_attn.q_proj.weight"].shape == (8, 64)


def test_coverage_guard_accepts_a_complete_mapping() -> None:
    model = _FakeModel(["encoder.a", "decoder.b"])
    _assert_full_coverage(
        model, {"encoder.a": mx.zeros((1,)), "decoder.b": mx.zeros((1,))}
    )


def test_coverage_guard_rejects_a_partial_mapping() -> None:
    model = _FakeModel(["encoder.a", "decoder.b", "decoder.c"])
    with pytest.raises(RuntimeError, match="random init") as excinfo:
        _assert_full_coverage(model, {"encoder.a": mx.zeros((1,))})
    message = str(excinfo.value)
    assert "1/3" in message  # covered / expected
    assert "decoder.b" in message


def test_coverage_guard_logs_but_allows_extra_checkpoint_weights(caplog) -> None:
    """Unused checkpoint tensors are a warning, not a failure.

    Only a *missing* parameter means random init; a spare tensor is harmless.
    """
    caplog.set_level("WARNING", logger="vemoizer.canary_checkpoint")
    model = _FakeModel(["encoder.a"])
    _assert_full_coverage(
        model, {"encoder.a": mx.zeros((1,)), "encoder.unused": mx.zeros((1,))}
    )
    assert "no matching parameter" in caplog.text
