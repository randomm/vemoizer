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
        self, x: mx.array, mask: mx.array | None = None
    ) -> tuple[mx.array, tuple[mx.array, mx.array]]:
        b, t, _ = x.shape
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        q = q.reshape(b, t, self.n_heads, self.head_dim).swapaxes(1, 2)
        k = k.reshape(b, t, self.n_heads, self.head_dim).swapaxes(1, 2)
        v = v.reshape(b, t, self.n_heads, self.head_dim).swapaxes(1, 2)
        o = mx.fast.scaled_dot_product_attention(q, k, v, scale=self.scale, mask=mask)
        o = o.swapaxes(1, 2).reshape(b, t, -1)
        return self.out_proj(o), (k, v)


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

    def __call__(self, x: mx.array, enc: mx.array) -> mx.array:
        b, t, _ = x.shape
        eb, es, _ = enc.shape
        q = self.q_proj(x).reshape(b, t, self.n_heads, self.head_dim).swapaxes(1, 2)
        k = self.k_proj(enc).reshape(eb, es, self.n_heads, self.head_dim).swapaxes(1, 2)
        v = self.v_proj(enc).reshape(eb, es, self.n_heads, self.head_dim).swapaxes(1, 2)
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

    def __call__(self, x: mx.array, enc: mx.array, mask: mx.array | None) -> mx.array:
        x = x + self.self_attn(self.self_attn_norm(x), mask)[0]
        x = x + self.cross_attn(self.cross_attn_norm(x), enc)
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

    def __call__(
        self,
        input_ids: mx.array,
        enc: mx.array,
        start_pos: int = 0,
        mask: mx.array | None = None,
    ) -> mx.array:
        b, t = input_ids.shape
        x = self.embedding(input_ids)
        d = self.blocks[0].self_attn_norm.weight.shape[0]
        x = x + _fixed_pos_enc(d, t)[None, :, :]
        x = self.embedding_layer_norm(x)
        for block in self.blocks:
            x = block(x, enc, mask)
        return self.output_proj(self.final_norm(x))
