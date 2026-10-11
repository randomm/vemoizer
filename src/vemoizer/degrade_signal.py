"""Far-field degradation signal math (issue #154, informational).

Pure numpy: no corpus I/O, no ffmpeg, no eval. :mod:`vemoizer.degrade`
owns the corpus, the loudnorm arm and the table.

Profiles (all seeded, so the same seed gives byte-identical WAV output):

- ``far``: attenuation of 20 dB plus a synthetic exponential-decay room
  reverb (impulse response, T60 = 0.4 s).
- ``noisy``: white Gaussian noise mixed in at a fixed SNR of 10 dB.
- ``far-noisy``: ``far`` first, then ``noisy`` at 10 dB relative to the
  attenuated, reverberant speech.

Input and output are the 16 kHz mono float32 arrays of the ingest contract
(invariant #6).
"""

from __future__ import annotations

import zlib

import numpy as np

from vemoizer.audio_contract import SAMPLE_RATE

#: Base seed recorded in every run's output; per-clip seeds derive from it.
BASE_SEED = 154

ATTENUATION_DB = 20.0
REVERB_T60_S = 0.4
REVERB_LEN_S = 0.5
NOISE_SNR_DB = 10.0

#: Profile name -> ordered processing steps. Unknown names are rejected.
STEPS: dict[str, tuple[str, ...]] = {
    "far": ("far",),
    "noisy": ("noisy",),
    "far-noisy": ("far", "noisy"),
}
PROFILES: tuple[str, ...] = tuple(STEPS)


def clip_seed(stem: str, profile: str) -> int:
    """Stable per-(clip, profile) seed derived from :data:`BASE_SEED`."""
    return BASE_SEED * 1_000_003 + zlib.crc32(f"{profile}/{stem}".encode())


def attenuate(x: np.ndarray, db: float = ATTENUATION_DB) -> np.ndarray:
    """Scale *x* down by *db* decibels."""
    return (x * (10.0 ** (-db / 20.0))).astype(np.float32)


def reverb(x: np.ndarray, seed: int) -> np.ndarray:
    """Convolve *x* with a seeded exponential-decay impulse response.

    The response is white noise shaped by ``exp(-6.9 t / T60)`` (the -60 dB
    point at T60), with the direct sound at index 0 so the first arrival is
    unchanged in timing. Energy is normalised to the input's energy. The
    convolution runs in the frequency domain (zero-padded, then truncated to
    ``len(x)``), which equals the direct linear convolution up to float
    rounding and costs O(N log N) instead of O(N * M).

    Memory is several full-length float64 copies of the clip: this is sized
    for the seconds-long corpus clips. Reusing it on long memos needs
    block-wise overlap-add first.
    """
    rng = np.random.default_rng(seed)
    n = int(REVERB_LEN_S * SAMPLE_RATE)
    t = np.arange(n) / SAMPLE_RATE
    ir = rng.standard_normal(n) * np.exp(-6.9 * t / REVERB_T60_S)
    ir[0] = 1.0
    ir = ir / np.sqrt(np.sum(ir**2))
    xd = x.astype(np.float64)
    size = len(xd) + n - 1
    full = np.fft.irfft(np.fft.rfft(xd, size) * np.fft.rfft(ir, size), size)
    y = full[: len(xd)]
    in_rms = np.sqrt(np.mean(xd**2)) or 1.0
    out_rms = np.sqrt(np.mean(y**2)) or 1.0
    return (y * (in_rms / out_rms)).astype(np.float32)


def add_noise(x: np.ndarray, seed: int, snr_db: float = NOISE_SNR_DB) -> np.ndarray:
    """Mix seeded white noise into *x* at *snr_db* (signal power / noise power)."""
    # Separate stream from reverb(): both get the same clip seed, and a shared
    # generator would make the noise a known copy of the room impulse response.
    rng = np.random.default_rng([seed, 1])
    xd = x.astype(np.float64)
    sig_power = float(np.mean(xd**2))
    noise = rng.standard_normal(len(xd))
    noise_power = float(np.mean(noise**2)) or 1.0
    target = sig_power / (10.0 ** (snr_db / 10.0))
    noise *= np.sqrt(target / noise_power)
    return np.clip(xd + noise, -1.0, 1.0).astype(np.float32)


def degrade(x: np.ndarray, profile: str, seed: int) -> np.ndarray:
    """Apply *profile* to the float32 clip *x*. Unknown profiles raise."""
    if profile not in STEPS:
        known = ", ".join(PROFILES)
        raise ValueError(f"unknown degrade profile {profile!r} (known: {known})")
    if len(x) == 0:
        return np.asarray(x, dtype=np.float32)
    y = x
    for step in STEPS[profile]:
        y = reverb(attenuate(y), seed) if step == "far" else add_noise(y, seed)
    return y
