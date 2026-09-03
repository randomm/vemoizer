"""Canary-1b-v2 MLX port: model architecture, features, and inference.

Self-contained port of ``nvidia/canary-1b-v2`` to MLX. The tokenizer
lives in :mod:`vemoizer.canary_tokenizer`, the decoder in
:mod:`vemoizer.canary_decoder`, and checkpoint loading in
:mod:`vemoizer.canary_checkpoint`.
"""

from __future__ import annotations

import functools
import math
from dataclasses import dataclass
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from .canary_checkpoint import (  # noqa: E402
    _dequant_grouped,  # noqa: F401 - re-export for test access
    _map_checkpoint_weights,  # noqa: F401 - re-export for test access
    load_canary_weights,  # noqa: F401 - re-export
)
from .canary_decoder import CanaryDecoder  # noqa: E402,F401
from .canary_tokenizer import (  # noqa: E402
    CanaryTokenizer,
    parse_sentencepiece_model,  # noqa: F401 - re-export for test access
)

#: Internal audio contract (project invariant #6).
SAMPLE_RATE = 16_000

# Canary special-token strings (stable across the Canary model family); their
# ids are resolved from the tokenizer vocabulary at load time.
CANARY_BOS = "<|startoftranscript|>"
CANARY_EOS = chr(0x03)  # the SentencePiece end-of-transcript control token
CANARY_PAD = "<pad>"
CANARY_NOSPEECH = "<|nospeech|>"
CANARY_PNC = "<|pnc|>"
CANARY_NOPNC = "<|nopnc|>"
CANARY2_BOCTX = "<|startofcontext|>"


# ---------------------------------------------------------------------------
# Audio feature extraction (numpy STFT -> mel -> log -> per-feature norm)
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=None)  # noqa: UP033
def _hann_window(n: int) -> np.ndarray:
    return np.hanning(n + 1)[:-1].astype(np.float32)


@functools.lru_cache(maxsize=None)  # noqa: UP033
def _mel_filterbank(n_mels: int, n_fft: int, sr: int) -> np.ndarray:
    """Hand-rolled mel filterbank (no scipy/librosa)."""

    def hz_to_mel(hz):
        return 2595.0 * np.log10(1.0 + hz / 700.0)

    def mel_to_hz(m):
        return 700.0 * (10.0 ** (m / 2595.0) - 1.0)

    mel_pts = np.linspace(0.0, hz_to_mel(sr / 2.0), n_mels + 2)
    hz_pts = mel_to_hz(mel_pts)
    bins = np.floor((n_fft + 1) * hz_pts / sr).astype(np.int32)
    f = np.zeros((n_mels, n_fft // 2 + 1), dtype=np.float32)
    for i in range(1, n_mels + 1):
        left, center, right = bins[i - 1], bins[i], bins[i + 1]
        for j in range(left, center):
            if center != left:
                f[i - 1, j] = (j - left) / (center - left)
        for j in range(center, right):
            if right != center:
                f[i - 1, j] = (right - j) / (right - center)
    norm = np.linalg.norm(f, axis=1, keepdims=True)
    return f / (norm + 1e-10)


def compute_features(audio: np.ndarray, *, dtype: mx.Dtype = mx.float32) -> mx.array:
    """Compute the 128-dim log-mel features the Canary encoder expects.

    Contract: *audio* is 16 kHz mono float32. This mirrors the NeMo
    ``DynamicSignalNormalizer`` / ``SignalFbank`` pipeline: preemphasis,
    hann-windowed STFT, power, mel projection, log, then per-feature
    (per-time-frame) normalization.
    """
    x = np.asarray(audio, dtype=np.float32).ravel()
    n_fft, hop, win, n_mels = 512, 160, 400, 128
    preemph = 0.97

    if x.size < win:
        return mx.zeros((1, 1, n_mels), dtype=dtype)

    x = np.concatenate([x[:1], x[1:] - preemph * x[:-1]]).astype(np.float32)
    window = _hann_window(win)
    pad = n_fft // 2
    x = np.pad(x, pad, mode="reflect")
    t = (x.size - win) // hop + 1
    step = hop * x.strides[0]
    frames = np.lib.stride_tricks.as_strided(
        x, shape=(t, win), strides=(step, x.strides[0])
    )
    spec = np.ascontiguousarray(frames) * window
    spec_c = np.fft.rfft(spec, n=n_fft)
    power = spec_c.real**2 + spec_c.imag**2
    mel = power @ _mel_filterbank(n_mels, n_fft, SAMPLE_RATE).T
    mel = np.log(mel + 1e-5)
    # per-feature (per mel-bin) normalization over time
    mean = mel.mean(axis=0, keepdims=True)
    std = mel.std(axis=0, keepdims=True)
    norm = (mel - mean) / (std + 1e-5)
    return mx.array(norm.T.astype(np.float32))[None, ...].astype(dtype)


# ---------------------------------------------------------------------------
# Model architecture (mirrors the Blaizzy/mlx-audio Canary reference)
# ---------------------------------------------------------------------------


@dataclass
class CanaryConfig:
    feat_in: int
    n_layers: int
    d_model: int
    n_heads: int
    ff_expansion_factor: int
    subsampling_factor: int
    conv_kernel_size: int
    subsampling_conv_channels: int
    use_bias: bool
    # decoder
    vocab_size: int
    dec_hidden: int
    dec_inner: int
    dec_num_layers: int
    dec_num_heads: int
    max_sequence_length: int
    # classifier head
    head_num_layers: int
    head_num_classes: int


def _canary_config_from_dict(cfg: dict[str, Any]) -> CanaryConfig:
    enc = cfg["encoder"]
    dec = cfg["transf_decoder"]
    head = cfg["head"]
    return CanaryConfig(
        feat_in=enc["feat_in"],
        n_layers=enc["n_layers"],
        d_model=enc["d_model"],
        n_heads=enc["n_heads"],
        ff_expansion_factor=enc["ff_expansion_factor"],
        subsampling_factor=enc["subsampling_factor"],
        conv_kernel_size=enc["conv_kernel_size"],
        subsampling_conv_channels=enc["subsampling_conv_channels"],
        use_bias=enc.get("use_bias", True),
        vocab_size=dec["vocab_size"],
        dec_hidden=dec["hidden_size"],
        dec_inner=dec["inner_size"],
        dec_num_layers=dec["num_layers"],
        dec_num_heads=dec["num_attention_heads"],
        max_sequence_length=dec.get("max_sequence_length", 1024),
        head_num_layers=head["num_layers"],
        head_num_classes=head["num_classes"],
    )


class RelPositionalEncoding(nn.Module):
    """Sinusoidal relative positional encoding buffer (T5-style, ±max_len)."""

    def __init__(self, d_model: int, max_len: int = 5000) -> None:
        super().__init__()
        assert d_model % 2 == 0
        positions = (
            mx.arange(max_len - 1, -max_len, -1, dtype=mx.int32)[:, None]
        ).astype(mx.float32)
        div_term = mx.exp(
            mx.arange(0, d_model, 2, dtype=mx.float32) * -(math.log(10000.0) / d_model)
        )
        pe = mx.zeros((2 * max_len - 1, d_model), dtype=mx.float32)
        pe[:, 0::2] = mx.sin(positions * div_term)
        pe[:, 1::2] = mx.cos(positions * div_term)
        self._pe = pe

    def __call__(self, length: int) -> mx.array:
        buf = self._pe.shape[0]
        mid = buf // 2
        return self._pe[mid - (length - 1) : mid + length, :].astype(mx.float32)


class RelPosMultiHeadAttention(nn.Module):
    """Multi-head attention with T5-style relative positional bias."""

    def __init__(self, d_model: int, n_heads: int, use_bias: bool = True) -> None:
        super().__init__()
        assert d_model % n_heads == 0
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim**-0.5
        self.linear_q = nn.Linear(d_model, d_model, bias=use_bias)
        self.linear_k = nn.Linear(d_model, d_model, bias=use_bias)
        self.linear_v = nn.Linear(d_model, d_model, bias=use_bias)
        self.linear_out = nn.Linear(d_model, d_model, bias=use_bias)
        self.linear_pos = nn.Linear(d_model, d_model, bias=False)
        self.pos_bias_u = mx.zeros((n_heads, self.head_dim))
        self.pos_bias_v = mx.zeros((n_heads, self.head_dim))

    @staticmethod
    def _rel_shift(x: mx.array) -> mx.array:
        b, h, tq, pos_len = x.shape
        x = mx.pad(x, [(0, 0), (0, 0), (0, 0), (1, 0)])
        x = x.reshape(b, h, pos_len + 1, tq)
        return x[:, :, 1:, :].reshape(b, h, tq, pos_len)

    def __call__(self, x: mx.array, pos_emb: mx.array) -> mx.array:
        q = self.linear_q(x)
        k = self.linear_k(x)
        v = self.linear_v(x)
        p = self.linear_pos(pos_emb)  # (1, P, d)

        batch = q.shape[0]
        q_seq = q.shape[1]
        pos_len = p.shape[1]

        q = q.reshape(batch, q_seq, -1, self.head_dim)
        k = k.reshape(batch, q_seq, -1, self.head_dim).swapaxes(1, 2)
        v = v.reshape(batch, q_seq, -1, self.head_dim).swapaxes(1, 2)
        p = p.reshape(p.shape[0], pos_len, -1, self.head_dim).swapaxes(1, 2)
        p = mx.broadcast_to(p, (batch, p.shape[1], p.shape[2], self.head_dim))

        q_u = (q + self.pos_bias_u[None, None]).swapaxes(1, 2)
        q_v = (q + self.pos_bias_v[None, None]).swapaxes(1, 2)

        # relative bias: q_v · p^T, shifted so that position (i, j) sees (i - j)
        matrix_bd = self._rel_shift(q_v @ p.swapaxes(-2, -1))[:, :, :, :q_seq]
        matrix_bd = matrix_bd * self.scale
        # mx.fast.scaled_dot_product_attention expects a mask of at most rank 4,
        # so fold (batch, head) into a single axis: (b,1,h,q,p)->(b*h,q,p).
        matrix_bd = mx.expand_dims(matrix_bd, 1).reshape(
            -1, matrix_bd.shape[2], matrix_bd.shape[3]
        )

        o = mx.fast.scaled_dot_product_attention(
            q_u, k, v, scale=self.scale, mask=matrix_bd
        )
        o = o.swapaxes(1, 2).reshape(batch, q_seq, -1)
        return self.linear_out(o)


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, use_bias: bool = True) -> None:
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff, bias=use_bias)
        self.linear2 = nn.Linear(d_ff, d_model, bias=use_bias)

    def __call__(self, x: mx.array) -> mx.array:
        return self.linear2(nn.silu(self.linear1(x)))


class ConformerConvolution(nn.Module):
    def __init__(self, args: CanaryConfig) -> None:
        super().__init__()
        self.padding = (args.conv_kernel_size - 1) // 2
        d = args.d_model
        self.pointwise_conv1 = nn.Conv1d(
            d, 2 * d, 1, stride=1, padding=0, bias=args.use_bias
        )
        self.depthwise_conv = nn.Conv1d(
            d,
            d,
            args.conv_kernel_size,
            stride=1,
            padding=0,
            groups=d,
            bias=args.use_bias,
        )
        self.batch_norm = nn.BatchNorm(d)
        self.pointwise_conv2 = nn.Conv1d(
            d, d, 1, stride=1, padding=0, bias=args.use_bias
        )

    def __call__(self, x: mx.array) -> mx.array:
        x = self.pointwise_conv1(x)
        x = nn.glu(x, axis=2)
        x = mx.pad(x, ((0, 0), (self.padding, self.padding), (0, 0)))
        x = self.depthwise_conv(x)
        x = self.batch_norm(x)
        x = nn.silu(x)
        return self.pointwise_conv2(x)


class ConformerBlock(nn.Module):
    def __init__(self, args: CanaryConfig) -> None:
        super().__init__()
        d = args.d_model
        self.norm_feed_forward1 = nn.LayerNorm(d)
        self.feed_forward1 = FeedForward(d, d * args.ff_expansion_factor, args.use_bias)
        self.norm_self_att = nn.LayerNorm(d)
        self.self_attn = RelPosMultiHeadAttention(d, args.n_heads, args.use_bias)
        self.norm_mha_out = nn.LayerNorm(d)
        self.norm_conv = nn.LayerNorm(d)
        self.conv = ConformerConvolution(args)
        self.norm_feed_forward2 = nn.LayerNorm(d)
        self.feed_forward2 = FeedForward(d, d * args.ff_expansion_factor, args.use_bias)
        self.norm_out = nn.LayerNorm(d)

    def __call__(self, x: mx.array, pos_emb: mx.array) -> mx.array:
        x = x + 0.5 * self.feed_forward1(self.norm_feed_forward1(x))
        x = x + self.self_attn(self.norm_self_att(x), pos_emb)
        x = x + self.conv(self.norm_conv(x))
        x = x + 0.5 * self.feed_forward2(self.norm_feed_forward2(x))
        return self.norm_out(x)


class DwStridingSubsampling(nn.Module):
    """Depthwise-striding conv subsampling (NeMo ``dw_striding``).

    Built from a single ordered list of conv / ReLU layers matching the NeMo
    ``encoder.pre_encode.conv`` layout so the weights map 1:1.
    """

    def __init__(self, args: CanaryConfig) -> None:
        super().__init__()
        factor = args.subsampling_factor
        assert factor > 0 and (factor & (factor - 1)) == 0
        self._num = int(math.log2(factor))
        stride, kernel, padding = 2, 3, 1
        final_freq = args.feat_in
        for _ in range(self._num):
            final_freq = (final_freq + 2 * padding - kernel) // stride + 1
        self.conv: list[nn.Module] = [
            nn.Conv2d(1, args.subsampling_conv_channels, kernel, stride, padding),
            nn.ReLU(),
        ]
        in_ch = args.subsampling_conv_channels
        for _ in range(self._num - 1):
            self.conv.append(
                nn.Conv2d(in_ch, in_ch, kernel, stride, padding, groups=in_ch)
            )
            self.conv.append(
                nn.Conv2d(in_ch, args.subsampling_conv_channels, 1, 1, 0, groups=1)
            )
            self.conv.append(nn.ReLU())
        self.out = nn.Linear(args.subsampling_conv_channels * final_freq, args.d_model)

    def __call__(self, x: mx.array) -> tuple[mx.array, mx.array]:
        # x: (B, T, feat); conv_forward transposes to NCHW, runs convs, back
        x = mx.expand_dims(x, 1)
        x = x.transpose(0, 2, 3, 1)  # (B, feat, T, 1) -> MLX NCHW (B, 1, T, feat)
        for layer in self.conv:
            x = layer(x)
        b, c, t, f = x.shape
        x = self.out(x.reshape(b, t, -1))
        lengths = mx.full((x.shape[0],), x.shape[1], dtype=mx.int32)
        return x, lengths


class ConformerEncoder(nn.Module):
    def __init__(self, args: CanaryConfig) -> None:
        super().__init__()
        self.pre_encode = DwStridingSubsampling(args)
        self.pos_enc = RelPositionalEncoding(args.d_model)
        self.layers = [ConformerBlock(args) for _ in range(args.n_layers)]

    def __call__(self, mel: mx.array) -> tuple[mx.array, mx.array]:
        lengths = mx.full((mel.shape[0],), mel.shape[1], dtype=mx.int64)
        enc, lengths = self.pre_encode(mel)
        pos_emb = self.pos_enc(enc.shape[1])  # (P, d)
        pos_emb = pos_emb[None, :, :]  # (1, P, d)
        for layer in self.layers:
            enc = layer(enc, pos_emb)
        return enc, lengths


class CanaryModel(nn.Module):
    def __init__(self, config: CanaryConfig, tokenizer: CanaryTokenizer) -> None:
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.encoder = ConformerEncoder(config)
        self.decoder = CanaryDecoder(config)

    def generate(
        self,
        mel: mx.array,
        prompt_ids: list[int],
        source_lang: str = "en",
        target_lang: str = "en",
        *,
        max_tokens: int = 256,
    ) -> str:
        enc, _ = self.encoder(mel)
        prompt_ids = prompt_ids or self.tokenizer.build_prompt(source_lang, target_lang)
        generated = list(prompt_ids)
        eos_id = self.tokenizer.eos_id
        for step in range(max_tokens):
            input_ids = mx.array(
                [generated[-1:]] if generated else prompt_ids, dtype=mx.int32
            )
            if step == 0:
                input_ids = mx.array([prompt_ids], dtype=mx.int32)
            logits = self.decoder(input_ids, enc, start_pos=step)
            next_token = int(mx.argmax(logits[:, -1], axis=-1).item())
            if next_token == eos_id:
                break
            generated.append(next_token)
        special_vals = self.tokenizer.special_tokens.values()
        decoded = [t for t in generated[len(prompt_ids) :] if t not in special_vals]
        return self.tokenizer.decode(decoded)
