"""Far-field degradation benchmark: corpus and eval arms (issue #154, informational).

Builds degraded variants of the committed eval clips and scores each backend
plain and with ``loudnorm``, so "does ``--preprocess loudnorm`` help at a
distance?" is answered by a WER measurement. Nothing here is gated: the
table is evidence, and the default stays a separate decision.

The signal arithmetic lives in :mod:`vemoizer.degrade_signal`; this module
owns the corpus I/O, the loudnorm arm and the table.

Failure policy: the benchmark never reports a number that is a measurement
artefact. Any of these raises :class:`DegradeError` (the CLI maps it to exit
2): an unpaired WAV, an unknown profile, no clip pairs, a crashing backend
(never scored as WER 1.0), an empty clean transcript (a fail-open decode),
or a scoring failure. Loudnorm runs in strict mode, so a measurement or
pass-2 failure raises instead of falling back to the plain decode.

Variants are written to a caller-supplied scratch directory and are never
committed (AGENTS.md: audio fixtures stay small). Degraded and loudnorm
variants are 16-bit PCM WAVs re-decoded by the backends; see the eval-only
quantisation note in docs/pipeline-spec.md.
"""

from __future__ import annotations

import logging
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import NamedTuple

import numpy as np

from vemoizer import loudnorm
from vemoizer.audio_contract import write_wav
from vemoizer.degrade_signal import PROFILES, clip_seed, degrade
from vemoizer.eval_harness import AGGREGATE_KEY, run_eval

logger = logging.getLogger(__name__)

#: Profile label of the committed clips, unchanged (the reference arm).
CLEAN = "clean"


class DegradeError(RuntimeError):
    """The degradation benchmark cannot produce a trustworthy number."""


class Row(NamedTuple):
    """One result: a backend's WER plain and with loudnorm on *profile*."""

    profile: str
    backend: str
    plain_wer: float
    loudnorm_wer: float


def _corpus_pairs(corpus: Path) -> list[tuple[Path, Path]]:
    """Stem-paired ``(wav, txt)`` clips in *corpus*, sorted by name.

    An unpaired WAV raises: every arm must score the same samples.
    """
    pairs: list[tuple[Path, Path]] = []
    for wav in sorted(corpus.glob("*.wav")):
        txt = wav.with_suffix(".txt")
        if not txt.is_file():
            raise DegradeError(
                f"{wav.name} has no reference .txt in {corpus}; refusing to "
                "benchmark unpaired clips"
            )
        pairs.append((wav, txt))
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

    Strict: a measurement failure or a pass-2 ffmpeg failure raises
    :class:`vemoizer.ingest.IngestError` rather than reporting the plain
    decode as a loudnorm result.
    """
    audio = loudnorm.preprocess_audio(wav, "loudnorm", strict=True)
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

    ``run_eval`` scores a crashing sample as an empty hypothesis, so the
    wrapper records the first backend exception and it is re-checked after
    the run: a crash never becomes a WER. On the clean arm an empty
    transcript is also refused: committed speech always decodes to text, so
    an empty result is a backend failing open. Degraded-arm emptiness is the
    measured effect and is scored as-is.
    """
    failures: list[Exception] = []

    def recording(wav: Path) -> str:
        try:
            return transcribe(wav)
        except Exception as exc:
            failures.append(exc)
            raise

    hyps: dict[str, str] = {}
    try:
        mean = run_eval(directory, recording, hyps)[AGGREGATE_KEY]
    except OSError as exc:
        # Only I/O while reading the clips is an expected failure here; any
        # other exception from the scoring path is a bug and must raise.
        raise DegradeError(
            f"scoring failed on {directory.name} ({type(exc).__name__}): {exc}"
        ) from exc
    if failures:
        raise DegradeError(
            f"backend failed on {directory.name}: {failures[0]}"
        ) from failures[0]
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
) -> list[Row]:
    """Score each backend plain and with loudnorm on *profile* variants.

    Returns the clean-reference rows followed by the degraded rows, for
    :func:`format_table`. The clean reference uses the committed clips
    unchanged; the degraded arm uses the seeded variants written under
    *work_dir*. Each arm is walked by :func:`vemoizer.eval_harness.run_eval`
    on a directory, so the plain and loudnorm numbers come from the same
    harness as the gate.
    """
    if profile not in PROFILES:
        known = ", ".join(PROFILES)
        raise DegradeError(f"unknown degrade profile {profile!r} (known: {known})")
    degraded = build_variants(corpus, profile, work_dir, decode)
    if not degraded:
        raise DegradeError(f"no <stem>.wav + <stem>.txt pair in {corpus}")
    deg_dir = work_dir / profile
    clean_ln = loudnorm_corpus(corpus, work_dir / "clean-loudnorm")
    deg_ln = loudnorm_corpus(deg_dir, work_dir / f"{profile}-loudnorm")
    clean_rows: list[Row] = []
    degraded_rows: list[Row] = []
    for name, transcribe in backends.items():
        clean_plain = _score(corpus, transcribe, clean=True)
        clean_norm = _score(clean_ln, transcribe, clean=True)
        clean_rows.append(Row(CLEAN, name, clean_plain, clean_norm))
        plain = _score(deg_dir, transcribe, clean=False)
        norm = _score(deg_ln, transcribe, clean=False)
        degraded_rows.append(Row(profile, name, plain, norm))
    return [*clean_rows, *degraded_rows]


def format_table(rows: list[Row]) -> list[str]:
    """Format the informational table.

    The delta is ``loudnorm - plain``: negative means loudnorm helped.
    """
    lines = ["profile\tbackend\tplain\tloudnorm\tdelta"]
    for r in rows:
        delta = r.loudnorm_wer - r.plain_wer
        lines.append(
            f"{r.profile}\t{r.backend}\t{r.plain_wer:.4f}\t{r.loudnorm_wer:.4f}\t{delta:+.4f}"
        )
    return lines
