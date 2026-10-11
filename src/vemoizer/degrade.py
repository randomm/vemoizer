"""Far-field degradation benchmark (issue #154, informational).

Builds degraded variants of the committed eval clips so the question
"does ``--preprocess loudnorm`` help recognition at a distance?" is answered
by a WER measurement instead of a listening test. Nothing here is gated:
the table is evidence, and the default stays a separate decision.

Profiles (all seeded, so the same seed gives byte-identical WAV output):

- ``far``: attenuation of 20 dB plus a synthetic exponential-decay room
  reverb (impulse response, T60 = 0.4 s).
- ``noisy``: white Gaussian noise mixed in at a fixed SNR of 10 dB.
- ``far-noisy``: ``far`` first, then ``noisy`` at 10 dB relative to the
  attenuated, reverberant speech.

Variants are written to a caller-supplied scratch directory and are never
committed (AGENTS.md: audio fixtures stay small; no degraded audio in
``tests/fixtures``). Degradation arithmetic runs on the 16 kHz mono float32
array that :func:`vemoizer.ingest.ingest_audio` produces, so the audio
contract (invariant #6) holds; ffmpeg does the decoding, no new dependency.
"""

from __future__ import annotations

import shutil
import wave
import zlib
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from vemoizer.audio_contract import SAMPLE_RATE

#: Base seed recorded in every run's output; per-clip seeds derive from it.
BASE_SEED = 154

ATTENUATION_DB = 20.0
REVERB_T60_S = 0.4
REVERB_LEN_S = 0.5
NOISE_SNR_DB = 10.0

#: Profile name -> (attenuate+reverb?, add noise?).
PROFILES: dict[str, tuple[bool, bool]] = {
    "far": (True, False),
    "noisy": (False, True),
    "far-noisy": (True, True),
}


@dataclass(frozen=True)
class Variant:
    """One degraded clip: which profile, which seed, where it was written."""

    profile: str
    stem: str
    seed: int
    path: Path


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
    unchanged in timing. Energy is normalised to the input's energy.
    """
    rng = np.random.default_rng(seed)
    n = int(REVERB_LEN_S * SAMPLE_RATE)
    t = np.arange(n) / SAMPLE_RATE
    ir = rng.standard_normal(n) * np.exp(-6.9 * t / REVERB_T60_S)
    ir[0] = 1.0
    ir = ir / np.sqrt(np.sum(ir**2))
    y = np.convolve(x.astype(np.float64), ir)[: len(x)]
    in_rms = np.sqrt(np.mean(x.astype(np.float64) ** 2)) or 1.0
    out_rms = np.sqrt(np.mean(y**2)) or 1.0
    return (y * (in_rms / out_rms)).astype(np.float32)


def add_noise(x: np.ndarray, seed: int, snr_db: float = NOISE_SNR_DB) -> np.ndarray:
    """Mix seeded white noise into *x* at *snr_db* (signal power / noise power)."""
    # Separate stream from reverb(): both get the same clip seed, and a shared
    # generator would make the noise a known copy of the room impulse response.
    rng = np.random.default_rng([seed, 1])
    xd = x.astype(np.float64)
    sig_power = float(np.mean(xd**2)) if len(xd) else 0.0
    noise = rng.standard_normal(len(xd))
    noise_power = float(np.mean(noise**2)) or 1.0
    target = sig_power / (10.0 ** (snr_db / 10.0))
    noise *= np.sqrt(target / noise_power)
    return np.clip(xd + noise, -1.0, 1.0).astype(np.float32)


def degrade(x: np.ndarray, profile: str, seed: int) -> np.ndarray:
    """Apply *profile* to the float32 clip *x*. Unknown profiles raise."""
    if profile not in PROFILES:
        known = ", ".join(PROFILES)
        raise ValueError(f"unknown degrade profile {profile!r} (known: {known})")
    far, noisy = PROFILES[profile]
    y = x
    if len(x) == 0:
        return np.asarray(x, dtype=np.float32)
    if far:
        y = reverb(attenuate(y), seed)
    if noisy:
        y = add_noise(y, seed)
    return y


def write_wav(path: Path, x: np.ndarray) -> Path:
    """Write *x* as 16 kHz mono 16-bit PCM WAV (deterministic bytes)."""
    pcm = (np.clip(x, -1.0, 1.0) * 32767.0).round().astype("<i2")
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(pcm.tobytes())
    return path


def build_variants(
    corpus: Path,
    profile: str,
    out_dir: Path,
    decode: Callable[[Path], np.ndarray],
) -> list[Variant]:
    """Write a degraded copy of every ``<stem>.wav``/``<stem>.txt`` pair.

    *decode* maps a WAV path to the 16 kHz mono float32 array (the ingest
    contract; the caller passes :func:`vemoizer.ingest.ingest_audio`).
    The reference ``.txt`` is copied next to each variant so the output
    directory is itself a valid corpus for :func:`vemoizer.eval_harness.run_eval`.
    """
    if profile not in PROFILES:
        raise ValueError(f"unknown degrade profile {profile!r}")
    target = out_dir / profile
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
    variants: list[Variant] = []
    for wav in sorted(corpus.glob("*.wav")):
        txt = wav.with_suffix(".txt")
        if not txt.is_file():
            continue
        seed = clip_seed(wav.stem, profile)
        audio = np.asarray(decode(wav), dtype=np.float32)
        out = write_wav(target / wav.name, degrade(audio, profile, seed))
        shutil.copyfile(txt, target / txt.name)
        variants.append(Variant(profile, wav.stem, seed, out))
    return variants


class DegradeError(RuntimeError):
    """The degradation benchmark cannot produce a trustworthy number.

    Raised instead of reporting a score that would be a measurement artefact
    (empty corpus, a crashing backend scored as WER 1.0).
    """


class LoudnormMeasurementError(DegradeError):
    """Pass 1 produced no usable measurement for a clip.

    Raised instead of falling back to the plain decode: a fallback would
    make the loudnorm arm report a fake "no effect" delta.
    """


def loudnorm_copy(wav: Path, out_dir: Path) -> Path:
    """Return a WAV holding the loudnorm-preprocessed audio of *wav*.

    This is the only place the loudnorm arm is applied: the processed
    audio is written next to the plain variant so the backend sees the
    same kind of path in both arms. ``preprocess_audio`` fails open to the
    plain decode, so the measurement is checked first and a failed
    measurement raises :class:`LoudnormMeasurementError` (never a delta
    computed from the fallback).
    """
    from vemoizer.loudnorm import measurement_for, preprocess_audio

    if measurement_for(wav) is None:
        raise LoudnormMeasurementError(
            f"loudnorm measurement failed for {wav.name}; refusing to report "
            "a plain-decode fallback as a loudnorm result"
        )
    audio = preprocess_audio(wav, "loudnorm")
    return write_wav(out_dir / wav.name, np.asarray(audio, dtype=np.float32))


def loudnorm_corpus(src: Path, out_dir: Path) -> Path:
    """Mirror corpus *src* into *out_dir* with every WAV loudnorm-processed.

    The reference ``.txt`` files are copied unchanged, so the result is a
    valid corpus for :func:`vemoizer.eval_harness.run_eval`.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    for wav in sorted(src.glob("*.wav")):
        txt = wav.with_suffix(".txt")
        if not txt.is_file():
            continue
        loudnorm_copy(wav, out_dir)
        shutil.copyfile(txt, out_dir / txt.name)
    return out_dir


def run_degrade_eval(
    corpus: Path,
    profile: str,
    backends: dict[str, Callable[[Path], str]],
    work_dir: Path,
    decode: Callable[[Path], np.ndarray],
) -> tuple[list[tuple[str, str, float, float]], list[tuple[str, float, float]]]:
    """Score each backend plain and with loudnorm on *profile* variants.

    Returns ``(rows, clean_rows)`` for :func:`format_table`. The clean
    reference uses the committed clips unchanged; the degraded arm uses
    the seeded variants written under *work_dir*. Each arm is walked by
    :func:`vemoizer.eval_harness.run_eval` on a directory, so the plain
    and loudnorm numbers come from the same harness as the gate.
    """
    from vemoizer.eval_harness import AGGREGATE_KEY, run_eval

    def score(directory: Path, transcribe: Callable[[Path], str]) -> float:
        # run_eval scores a crashing backend as an empty hypothesis (WER 1.0);
        # here that would show up as a fake regression, so count and refuse.
        failed: list[str] = []

        def guarded(wav: Path) -> str:
            try:
                return transcribe(wav)
            except Exception:
                failed.append(wav.name)
                raise

        mean = run_eval(directory, guarded)[AGGREGATE_KEY]
        if failed:
            raise DegradeError(
                f"backend failed on {len(failed)} clip(s) in {directory.name} "
                f"({', '.join(failed[:5])}); refusing to score it as WER"
            )
        return mean

    degraded = build_variants(corpus, profile, work_dir, decode)
    if not degraded:
        raise DegradeError(f"no <stem>.wav + <stem>.txt pair in {corpus}")
    deg_dir = work_dir / profile
    clean_ln = loudnorm_corpus(corpus, work_dir / "clean-loudnorm")
    deg_ln = loudnorm_corpus(deg_dir, work_dir / f"{profile}-loudnorm")
    rows: list[tuple[str, str, float, float]] = []
    clean_rows: list[tuple[str, float, float]] = []
    for name, transcribe in backends.items():
        clean_plain = score(corpus, transcribe)
        clean_norm = score(clean_ln, transcribe)
        clean_rows.append((name, clean_plain, clean_norm))
        plain = score(deg_dir, transcribe)
        norm = score(deg_ln, transcribe)
        rows.append((profile, name, plain, norm))
    return rows, clean_rows


def format_table(
    rows: list[tuple[str, str, float, float]],
    clean_rows: list[tuple[str, float, float]],
) -> list[str]:
    """Format the informational table.

    *rows*: ``(profile, backend, plain_wer, loudnorm_wer)``.
    *clean_rows*: ``(backend, plain_wer, loudnorm_wer)`` for reference.
    The delta is ``loudnorm - plain``: negative means loudnorm helped.
    """
    header = "profile\tbackend\tplain\tloudnorm\tdelta"
    lines = [header]
    for backend, plain, norm in clean_rows:
        lines.append(f"clean\t{backend}\t{plain:.4f}\t{norm:.4f}\t{norm - plain:+.4f}")
    for profile, backend, plain, norm in rows:
        lines.append(
            f"{profile}\t{backend}\t{plain:.4f}\t{norm:.4f}\t{norm - plain:+.4f}"
        )
    return lines
