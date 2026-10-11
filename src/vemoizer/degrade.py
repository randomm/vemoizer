"""Far-field degradation benchmark: corpus and eval arms (issue #154, informational).

Builds degraded variants of the committed eval clips and scores each backend
plain and with ``loudnorm``, so "does ``--preprocess loudnorm`` help at a
distance?" is answered by a WER measurement. Nothing here is gated: the
table is evidence, and the default stays a separate decision.

The signal arithmetic lives in :mod:`vemoizer.degrade_signal`; this module
owns the corpus I/O, the loudnorm arm and the table.

Variants are written to a caller-supplied scratch directory and are never
committed (AGENTS.md: audio fixtures stay small). Degraded and loudnorm
variants are 16-bit PCM WAVs re-decoded by the backends; see the eval-only
quantisation note in docs/pipeline-spec.md.
"""

from __future__ import annotations

import logging
import shutil
import wave
from collections.abc import Callable
from pathlib import Path
from typing import NamedTuple

import numpy as np

from vemoizer import loudnorm
from vemoizer.audio_contract import SAMPLE_RATE
from vemoizer.degrade_signal import clip_seed, degrade
from vemoizer.eval_harness import AGGREGATE_KEY, run_eval

logger = logging.getLogger(__name__)


class DegradeError(RuntimeError):
    """The degradation benchmark cannot produce a trustworthy number.

    Raised instead of reporting a score that would be a measurement artefact
    (empty corpus, a crashing backend scored as WER 1.0, a loudnorm
    fallback reported as a loudnorm result).
    """


class DegradeRow(NamedTuple):
    """One degraded-arm result: a backend's WER plain and with loudnorm."""

    profile: str
    backend: str
    plain_wer: float
    loudnorm_wer: float


class CleanRow(NamedTuple):
    """One clean-arm reference result: a backend's WER plain and with loudnorm."""

    backend: str
    plain_wer: float
    loudnorm_wer: float


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


def _corpus_pairs(corpus: Path) -> list[tuple[Path, Path]]:
    """Stem-paired ``(wav, txt)`` clips in *corpus*, sorted by name.

    A WAV without a sibling ``.txt`` is skipped with a warning, so the clip
    drops out of every arm and the operator can see it happened.
    """
    pairs: list[tuple[Path, Path]] = []
    for wav in sorted(corpus.glob("*.wav")):
        txt = wav.with_suffix(".txt")
        if txt.is_file():
            pairs.append((wav, txt))
        else:
            logger.warning("degrade: skipping %s (no reference .txt)", wav.name)
    return pairs


def build_variants(
    corpus: Path,
    profile: str,
    out_dir: Path,
    decode: Callable[[Path], np.ndarray],
) -> list[Path]:
    """Write a degraded copy of every ``<stem>.wav``/``<stem>.txt`` pair.

    *decode* maps a WAV path to the 16 kHz mono float32 array (the ingest
    contract; the caller passes :func:`vemoizer.ingest.ingest_audio`).
    The reference ``.txt`` is copied next to each variant so the output
    directory is itself a valid corpus for :func:`vemoizer.eval_harness.run_eval`.
    Returns the written WAV paths.
    """
    target = out_dir / profile
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
    written: list[Path] = []
    for wav, txt in _corpus_pairs(corpus):
        audio = np.asarray(decode(wav), dtype=np.float32)
        seed = clip_seed(wav.stem, profile)
        written.append(write_wav(target / wav.name, degrade(audio, profile, seed)))
        shutil.copyfile(txt, target / txt.name)
    return written


def loudnorm_copy(wav: Path, out_dir: Path) -> Path:
    """Write the loudnorm-processed audio of *wav* into *out_dir*; return it.

    Measured once and never fail-open: a pass-1 failure raises
    :class:`DegradeError`, and a pass-2 ffmpeg failure raises
    :class:`vemoizer.ingest.IngestError`. Either would otherwise be reported
    as a loudnorm result that is really the plain decode.
    """
    measurement = loudnorm.measurement_for(wav)
    if measurement is None:
        raise DegradeError(
            f"loudnorm measurement failed for {wav.name}; refusing to report "
            "a plain-decode fallback as a loudnorm result"
        )
    audio = loudnorm.decode_measured(wav, measurement)
    return write_wav(out_dir / wav.name, audio)


def loudnorm_corpus(src: Path, out_dir: Path) -> Path:
    """Mirror corpus *src* into *out_dir* with every WAV loudnorm-processed.

    The reference ``.txt`` files are copied unchanged, so the result is a
    valid corpus for :func:`vemoizer.eval_harness.run_eval`.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    for wav, txt in _corpus_pairs(src):
        loudnorm_copy(wav, out_dir)
        shutil.copyfile(txt, out_dir / txt.name)
    return out_dir


def _score(directory: Path, transcribe: Callable[[Path], str], *, clean: bool) -> float:
    """Mean WER of *transcribe* over *directory*, refusing artefacts.

    A crash raises :class:`DegradeError` (``on_error="raise"``, so it is
    never scored as WER 1.0). On the clean arm an empty transcript is also
    refused: committed speech always decodes to text, so an empty result is
    a backend failing open. Degraded-arm emptiness is the measured effect
    and is scored as-is.
    """
    hyps: dict[str, str] = {}
    try:
        mean = run_eval(directory, transcribe, hyps, on_error="raise")[AGGREGATE_KEY]
    except Exception as exc:
        raise DegradeError(f"backend failed on {directory.name}: {exc}") from exc
    if clean:
        empty = sorted(stem for stem, text in hyps.items() if not text.strip())
        if empty:
            raise DegradeError(
                f"empty transcript for {len(empty)} clean clip(s) in "
                f"{directory.name} ({', '.join(empty[:5])}); refusing to score "
                "a fail-open decode as WER"
            )
    return mean


def run_degrade_eval(
    corpus: Path,
    profile: str,
    backends: dict[str, Callable[[Path], str]],
    work_dir: Path,
    decode: Callable[[Path], np.ndarray],
) -> tuple[list[DegradeRow], list[CleanRow]]:
    """Score each backend plain and with loudnorm on *profile* variants.

    Returns ``(rows, clean_rows)`` for :func:`format_table`. The clean
    reference uses the committed clips unchanged; the degraded arm uses
    the seeded variants written under *work_dir*. Each arm is walked by
    :func:`vemoizer.eval_harness.run_eval` on a directory, so the plain
    and loudnorm numbers come from the same harness as the gate.
    """
    degraded = build_variants(corpus, profile, work_dir, decode)
    if not degraded:
        raise DegradeError(f"no <stem>.wav + <stem>.txt pair in {corpus}")
    deg_dir = work_dir / profile
    clean_ln = loudnorm_corpus(corpus, work_dir / "clean-loudnorm")
    deg_ln = loudnorm_corpus(deg_dir, work_dir / f"{profile}-loudnorm")
    rows: list[DegradeRow] = []
    clean_rows: list[CleanRow] = []
    for name, transcribe in backends.items():
        clean_plain = _score(corpus, transcribe, clean=True)
        clean_norm = _score(clean_ln, transcribe, clean=True)
        clean_rows.append(CleanRow(name, clean_plain, clean_norm))
        plain = _score(deg_dir, transcribe, clean=False)
        norm = _score(deg_ln, transcribe, clean=False)
        rows.append(DegradeRow(profile, name, plain, norm))
    return rows, clean_rows


def format_table(rows: list[DegradeRow], clean_rows: list[CleanRow]) -> list[str]:
    """Format the informational table.

    The delta is ``loudnorm - plain``: negative means loudnorm helped.
    """
    header = "profile\tbackend\tplain\tloudnorm\tdelta"
    lines = [header]
    for c in clean_rows:
        delta = c.loudnorm_wer - c.plain_wer
        lines.append(
            f"clean\t{c.backend}\t{c.plain_wer:.4f}\t{c.loudnorm_wer:.4f}\t{delta:+.4f}"
        )
    for r in rows:
        delta = r.loudnorm_wer - r.plain_wer
        lines.append(
            f"{r.profile}\t{r.backend}\t{r.plain_wer:.4f}\t{r.loudnorm_wer:.4f}\t{delta:+.4f}"
        )
    return lines
