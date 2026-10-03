"""Canary-1b-v2 mel feature extraction (issue #117).

Sliced out of ``canary_mlx`` to keep both modules under the 500-line limit.
Pure numpy/MLX math: no model architecture, no checkpoint loading, no
tokenizer — just the 16 kHz mono float32 -> 128-dim log-mel conversion the
Canary encoder expects.
"""

from __future__ import annotations

import functools

import mlx.core as mx
import numpy as np

#: Internal audio contract (project invariant #6).
SAMPLE_RATE = 16_000


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
    # (B, T, n_mels) — the layout DwStridingSubsampling documents and the one
    # the short-audio early return above already produces. Transposing to
    # (B, n_mels, T) here fed frequency into the conv's time axis.
    return mx.array(norm.astype(np.float32))[None, ...].astype(dtype)
