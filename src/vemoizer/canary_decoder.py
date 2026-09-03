"""Canary-1b-v2 decoder stack (transformer decoder + classifier head).

Sliced out of ``canary_mlx`` to keep each module under the 500-line limit
(issue #6). The encoder/conformer side stays in ``canary_mlx``; checkpoint
loading lives in ``canary_checkpoint``.
"""

from __future__ import annotations

import functools
import math
from typing import TYPE_CHECKING

import mlx.core as mx
import mlx.nn as nn

if TYPE_CHECKING:
    from .canary_mlx import CanaryConfig


@functools.lru_cache(maxsize=None)  # noqa: UP033
def _fixed_pos_enc(d: int, n: int) -> mx.array:
    pos = mx.arange(n)[:, None]
    div = mx.exp(mx.arange(0, d, 2) * (-math.log(10000.0) / d))
    pe = mx.zeros((n, d))
    pe[:, 0::2] = mx.sin(pos * div)
    pe[:, 1::2] = mx.cos(pos * div)
    return pe / math.sqrt(d)


class _BlockCache:
    """Per-block decode state: self-attention K/V history + cross-attention K/V.

    Autoregressive decoding feeds one token per step. Without this the step
    only ever saw itself — a length-1 self-attention with no history — so the
    decoder had no memory of what it had already emitted and collapsed to a
    near-constant output. The cross-attention K/V are a pure function of the
    encoder output, so they are computed once per slice instead of once per
    token.
    """

    __slots__ = ("keys", "values", "cross_k", "cross_v")

    def __init__(self) -> None:
        self.keys: mx.array | None = None
        self.values: mx.array | None = None
        self.cross_k: mx.array | None = None
        self.cross_v: mx.array | None = None

    def append(self, k: mx.array, v: mx.array) -> tuple[mx.array, mx.array]:
        """Append this step's K/V and return the full history."""
        keys, values = self.keys, self.values
        if keys is None or values is None:
            keys, values = k, v
        else:
            keys = mx.concatenate([keys, k], axis=2)
            values = mx.concatenate([values, v], axis=2)
        self.keys, self.values = keys, values
        return keys, values


class _MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim**-0.5
        self.q_proj = nn.Linear(d_model, d_model, bias=True)
        self.k_proj = nn.Linear(d_model, d_model, bias=True)
        self.v_proj = nn.Linear(d_model, d_model, bias=True)
        self.out_proj = nn.Linear(d_model, d_model, bias=True)

    def __call__(
        self,
        x: mx.array,
        mask: mx.array | None = None,
        cache: _BlockCache | None = None,
    ) -> mx.array:
        b, t, _ = x.shape
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        q = q.reshape(b, t, self.n_heads, self.head_dim).swapaxes(1, 2)
        k = k.reshape(b, t, self.n_heads, self.head_dim).swapaxes(1, 2)
        v = v.reshape(b, t, self.n_heads, self.head_dim).swapaxes(1, 2)
        if cache is not None:
            k, v = cache.append(k, v)
        o = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=mask)
        o = o.swapaxes(1, 2).reshape(b, t, -1)
        return self.out_proj(o)


class _MultiHeadCrossAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim**-0.5
        self.q_proj = nn.Linear(d_model, d_model, bias=True)
        self.k_proj = nn.Linear(d_model, d_model, bias=True)
        self.v_proj = nn.Linear(d_model, d_model, bias=True)
        self.out_proj = nn.Linear(d_model, d_model, bias=True)

    def __call__(
        self, x: mx.array, enc: mx.array, cache: _BlockCache | None = None
    ) -> mx.array:
        b, t, _ = x.shape
        eb, es, _ = enc.shape
        q = self.q_proj(x).reshape(b, t, self.n_heads, self.head_dim).swapaxes(1, 2)
        cross_k = cache.cross_k if cache is not None else None
        cross_v = cache.cross_v if cache is not None else None
        if cross_k is not None and cross_v is not None:
            k, v = cross_k, cross_v
        else:
            k = (
                self.k_proj(enc)
                .reshape(eb, es, self.n_heads, self.head_dim)
                .swapaxes(1, 2)
            )
            v = (
                self.v_proj(enc)
                .reshape(eb, es, self.n_heads, self.head_dim)
                .swapaxes(1, 2)
            )
            if cache is not None:
                cache.cross_k, cache.cross_v = k, v
        o = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale)
        o = o.swapaxes(1, 2).reshape(b, t, -1)
        return self.out_proj(o)


class _DecoderBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, inner: int) -> None:
        super().__init__()
        self.self_attn_norm = nn.LayerNorm(d_model)
        self.self_attn = _MultiHeadSelfAttention(d_model, n_heads)
        self.cross_attn_norm = nn.LayerNorm(d_model)
        self.cross_attn = _MultiHeadCrossAttention(d_model, n_heads)
        self.ff_norm = nn.LayerNorm(d_model)
        self.ff1 = nn.Linear(d_model, inner, bias=True)
        self.ff2 = nn.Linear(inner, d_model, bias=True)

    def __call__(
        self,
        x: mx.array,
        enc: mx.array,
        mask: mx.array | None,
        cache: _BlockCache | None = None,
    ) -> mx.array:
        x = x + self.self_attn(self.self_attn_norm(x), mask, cache)
        x = x + self.cross_attn(self.cross_attn_norm(x), enc, cache)
        x = x + self.ff2(nn.relu(self.ff1(self.ff_norm(x))))
        return x


class CanaryDecoder(nn.Module):
    def __init__(self, args: CanaryConfig) -> None:
        super().__init__()
        d = args.dec_hidden
        self.embedding = nn.Embedding(args.vocab_size, d)
        self.embedding_layer_norm = nn.LayerNorm(d)
        self.blocks = [
            _DecoderBlock(d, args.dec_num_heads, args.dec_inner)
            for _ in range(args.dec_num_layers)
        ]
        self.final_norm = nn.LayerNorm(d)
        self.output_proj = nn.Linear(d, args.head_num_classes, bias=True)

    def make_cache(self) -> list[_BlockCache]:
        """One :class:`_BlockCache` per block, for a single decode run."""
        return [_BlockCache() for _ in self.blocks]

    def __call__(
        self,
        input_ids: mx.array,
        enc: mx.array,
        start_pos: int = 0,
        mask: mx.array | None = None,
        cache: list[_BlockCache] | None = None,
    ) -> mx.array:
        b, t = input_ids.shape
        x = self.embedding(input_ids)
        d = self.blocks[0].self_attn_norm.weight.shape[0]
        # ``start_pos`` is the absolute position of this call's first token.
        # It was previously accepted and ignored, so every incremental step
        # re-used position 0 and the model could not tell token 5 from token 50.
        pos = _fixed_pos_enc(d, start_pos + t)[start_pos : start_pos + t]
        x = x + pos[None, :, :].astype(x.dtype)
        x = self.embedding_layer_norm(x)
        if mask is None and t > 1:
            mask = _causal_mask(t, x.dtype)
        for i, block in enumerate(self.blocks):
            x = block(x, enc, mask, cache[i] if cache is not None else None)
        return self.output_proj(self.final_norm(x))


@functools.lru_cache(maxsize=None)  # noqa: UP033
def _causal_mask_f32(t: int) -> mx.array:
    idx = mx.arange(t)
    return mx.where(idx[None, :] > idx[:, None], -1e9, 0.0)


def _causal_mask(t: int, dtype: mx.Dtype) -> mx.array:
    """Additive causal mask for a *t*-token prompt pass.

    Multi-token passes previously ran unmasked, letting each prompt token
    attend to the ones after it. The dtype must follow Q/K/V: a float32 mask
    against bfloat16 inputs is rejected by
    ``mx.fast.scaled_dot_product_attention``.
    """
    return _causal_mask_f32(t).astype(dtype)
