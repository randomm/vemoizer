"""Unit tests for the far-field degradation signal math (issue #154).

Pure numpy: no models, no network, no ffmpeg.
"""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
import pytest

from vemoizer.audio_contract import SAMPLE_RATE, write_wav
from vemoizer.degrade import Row, format_table
from vemoizer.degrade_signal import (
    ATTENUATION_DB,
    NOISE_SNR_DB,
    PROFILES,
    add_noise,
    attenuate,
    clip_seed,
    degrade,
)


def _speech_like(seconds: float = 2.0) -> np.ndarray:
    t = np.arange(int(seconds * SAMPLE_RATE)) / SAMPLE_RATE
    rng = np.random.default_rng(1)
    # Amplitude-modulated tone plus a little broadband content: non-trivial
    # power, deterministic.
    env = 0.5 + 0.5 * np.sin(2 * np.pi * 3 * t)
    sig = 0.4 * env * np.sin(2 * np.pi * 220 * t) + 0.02 * rng.standard_normal(len(t))
    return sig.astype(np.float32)


def _db(ratio: float) -> float:
    return 10.0 * float(np.log10(ratio))


def test_same_seed_gives_identical_degraded_bytes(tmp_path: Path) -> None:
    x = _speech_like()
    for profile in PROFILES:
        seed = clip_seed("sample", profile)
        a = write_wav(tmp_path / f"a-{profile}.wav", degrade(x, profile, seed))
        b = write_wav(tmp_path / f"b-{profile}.wav", degrade(x, profile, seed))
        assert a.read_bytes() == b.read_bytes(), profile


def test_different_seed_changes_noise(tmp_path: Path) -> None:
    x = _speech_like()
    a = degrade(x, "noisy", clip_seed("s", "noisy"))
    b = degrade(x, "noisy", clip_seed("t", "noisy"))
    assert not np.array_equal(a, b)


def test_noisy_profile_hits_target_snr() -> None:
    x = _speech_like()
    noisy = add_noise(x, seed=7)
    residual = noisy.astype(np.float64) - x.astype(np.float64)
    # Clipping at +-1 is negligible for this signal; measured SNR must sit
    # within 0.5 dB of the configured target.
    snr = _db(float(np.mean(x.astype(np.float64) ** 2)) / float(np.mean(residual**2)))
    assert snr == pytest.approx(NOISE_SNR_DB, abs=0.5)


def test_far_profile_attenuates_by_about_20_db() -> None:
    x = _speech_like()
    att = attenuate(x)
    p_att = float(np.mean(att.astype(np.float64) ** 2))
    p_in = float(np.mean(x.astype(np.float64) ** 2))
    ratio = _db(p_att / p_in)
    assert ratio == pytest.approx(-ATTENUATION_DB, abs=0.01)


def test_far_profile_keeps_energy_of_reverberant_output_near_attenuated_input() -> None:
    x = _speech_like()
    far = degrade(x, "far", clip_seed("s", "far"))
    att_rms = float(np.sqrt(np.mean(attenuate(x).astype(np.float64) ** 2)))
    far_rms = float(np.sqrt(np.mean(far.astype(np.float64) ** 2)))
    assert far_rms == pytest.approx(att_rms, rel=0.01)


def test_unknown_profile_raises() -> None:
    with pytest.raises(ValueError, match="unknown degrade profile"):
        degrade(_speech_like(0.1), "nope", 1)


def test_write_wav_is_16k_mono_16bit(tmp_path: Path) -> None:
    p = write_wav(tmp_path / "x.wav", _speech_like(0.5))
    with wave.open(str(p), "rb") as w:
        assert w.getnchannels() == 1
        assert w.getsampwidth() == 2
        assert w.getframerate() == SAMPLE_RATE


def test_format_table_reports_delta_and_clean_reference() -> None:
    lines = format_table(
        [Row("clean", "parakeet", 0.05, 0.06), Row("far", "parakeet", 0.30, 0.20)]
    )
    assert lines[0] == "profile\tbackend\tplain\tloudnorm\tdelta"
    assert lines[1] == "clean\tparakeet\t0.0500\t0.0600\t+0.0100"
    assert lines[2] == "far\tparakeet\t0.3000\t0.2000\t-0.1000"


def test_noise_is_uncorrelated_with_reverb_excitation() -> None:
    # reverb() draws its impulse response from default_rng(seed); the noise
    # must not be a copy of that stream, or far-noisy is not independent noise.
    seed = 7
    x = _speech_like(1.0)
    residual = add_noise(x, seed).astype(np.float64) - x.astype(np.float64)
    excitation = np.random.default_rng(seed).standard_normal(len(x))
    corr = float(np.corrcoef(residual, excitation)[0, 1])
    assert abs(corr) < 0.05


def test_empty_clip_degrades_to_empty() -> None:
    empty = np.zeros(0, dtype=np.float32)
    for profile in PROFILES:
        out = degrade(empty, profile, 1)
        assert out.shape == (0,)
