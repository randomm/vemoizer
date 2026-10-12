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
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import NamedTuple

import numpy as np

from vemoizer import loudnorm
from vemoizer.audio_contract import write_wav
from vemoizer.degrade_signal import (
    ATTENUATION_DB,
    BASE_SEED,
    NOISE_SNR_DB,
    PROFILES,
    REVERB_LEN_S,
    REVERB_T60_S,
    STEPS,
    clip_seed,
    degrade,
)
from vemoizer.eval_harness import AGGREGATE_KEY, run_eval
from vemoizer.models import MODELS

logger = logging.getLogger(__name__)

#: Profile label of the committed clips, unchanged (the reference arm).
CLEAN = "clean"


class DegradeError(RuntimeError):
    """The degradation benchmark cannot produce a trustworthy number."""


class Row(NamedTuple):
    """One result: a backend's WER plain, level-only and with loudnorm.

    ``gain_wer`` is ``None`` on the clean reference, where no control applies.
    """

    profile: str
    backend: str
    plain_wer: float
    gain_wer: float | None
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


def _mirror_pairs(
    corpus: Path,
    out_dir: Path,
    produce: Callable[[Path], np.ndarray],
) -> list[Path]:
    """Write ``produce(wav)`` as ``out_dir/<wav>`` for every clip in *corpus*.

    The reference ``.txt`` is copied next to each output, so *out_dir* is a
    valid corpus for :func:`vemoizer.eval_harness.run_eval`. The caller owns
    *out_dir* (creation and any cleanup); this only fills it.
    """
    written: list[Path] = []
    for wav, txt in _corpus_pairs(corpus):
        written.append(write_wav(out_dir / wav.name, produce(wav)))
        shutil.copyfile(txt, out_dir / txt.name)
    return written


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

    def produce(wav: Path) -> np.ndarray:
        audio = np.asarray(decode(wav), dtype=np.float32)
        return degrade(audio, profile, clip_seed(wav.stem, profile))

    return _mirror_pairs(corpus, target, produce)


def loudnorm_corpus(src: Path, out_dir: Path) -> Path:
    """Mirror corpus *src* into *out_dir* with every WAV loudnorm-processed.

    The reference ``.txt`` files are copied unchanged, so the result is a
    valid corpus for :func:`vemoizer.eval_harness.run_eval`. Strict: a
    measurement or pass-2 failure raises :class:`vemoizer.ingest.IngestError`
    rather than reporting the plain decode as a loudnorm result.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    _mirror_pairs(
        src,
        out_dir,
        lambda wav: loudnorm.preprocess_audio(wav, "loudnorm", strict=True),
    )
    return out_dir


def requantise_corpus(
    corpus: Path, out_dir: Path, decode: Callable[[Path], np.ndarray]
) -> Path:
    """Mirror *corpus* with every clip passed through the 16-bit write path.

    The clean reference goes through the same :func:`write_wav` as the
    degraded arms, so no arm gets a quantisation the others do not. The
    committed fixtures are 16-bit already, but ``decode`` -> ``write_wav`` is
    not byte-identical to them (the rounding differs), so the copy is needed.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    _mirror_pairs(
        corpus, out_dir, lambda wav: np.asarray(decode(wav), dtype=np.float32)
    )
    return out_dir


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(x**2))) or 1.0


def gain_corpus(
    corpus: Path,
    deg_dir: Path,
    out_dir: Path,
    decode: Callable[[Path], np.ndarray],
) -> Path:
    """Mirror *deg_dir* with each clip's RMS scaled to its clean counterpart.

    The oracle level-only control: a plain gain change with no loudnorm and no
    dereverberation, so the table can separate level restoration from what
    loudnorm adds. The target RMS comes from the clean reference, which an
    inference-time system does not have, so this is an oracle control, not an
    inference-time method. An RMS match is not WER-optimal, so it is not a
    proven upper bound on what level alone can recover. Clipped to [-1, 1]
    and written as 16-bit like the variants. *corpus* must be the requantised
    clean reference.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    def produce(wav: Path) -> np.ndarray:
        clean = np.asarray(decode(wav), dtype=np.float64)
        deg = np.asarray(decode(deg_dir / wav.name), dtype=np.float64)
        scaled = np.clip(deg * (_rms(clean) / _rms(deg)), -1.0, 1.0)
        return scaled.astype(np.float32)

    _mirror_pairs(corpus, out_dir, produce)
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
    """Score each backend plain, level-only and with loudnorm on *profile*.

    Returns the clean-reference rows followed by the degraded rows, for
    :func:`format_table`. The clean reference is the committed clips passed
    through the same 16-bit write as every degraded arm (see
    :func:`requantise_corpus`); the degraded arm uses the seeded variants
    written under *work_dir*. Each arm is walked by
    :func:`vemoizer.eval_harness.run_eval` on a directory, so the plain and
    loudnorm numbers come from the same harness as the gate.
    """
    if profile not in PROFILES:
        known = ", ".join(PROFILES)
        raise DegradeError(f"unknown degrade profile {profile!r} (known: {known})")
    degraded = build_variants(corpus, profile, work_dir, decode)
    if not degraded:
        raise DegradeError(f"no <stem>.wav + <stem>.txt pair in {corpus}")
    deg_dir = work_dir / profile
    clean_dir = requantise_corpus(corpus, work_dir / "clean", decode)
    clean_ln = loudnorm_corpus(clean_dir, work_dir / "clean-loudnorm")
    deg_ln = loudnorm_corpus(deg_dir, work_dir / f"{profile}-loudnorm")
    gain_dir = gain_corpus(clean_dir, deg_dir, work_dir / f"{profile}-gain", decode)
    clean_rows: list[Row] = []
    degraded_rows: list[Row] = []
    for name, transcribe in backends.items():
        clean_plain = _score(clean_dir, transcribe, clean=True)
        clean_norm = _score(clean_ln, transcribe, clean=True)
        clean_rows.append(Row(CLEAN, name, clean_plain, None, clean_norm))
        plain = _score(deg_dir, transcribe, clean=False)
        gain = _score(gain_dir, transcribe, clean=False)
        norm = _score(deg_ln, transcribe, clean=False)
        degraded_rows.append(Row(profile, name, plain, gain, norm))
    return [*clean_rows, *degraded_rows]


def format_table(rows: list[Row]) -> list[str]:
    """Format the informational table.

    ``gain`` is the level-only control (``-`` on clean rows). ``delta`` is
    ``loudnorm - plain``: negative means loudnorm helped. ``loudnorm - gain``
    is the part of the loudnorm effect that level restoration does not explain.
    """
    lines = ["profile\tbackend\tplain\tgain\tloudnorm\tdelta"]
    for r in rows:
        gain = "-" if r.gain_wer is None else f"{r.gain_wer:.4f}"
        delta = r.loudnorm_wer - r.plain_wer
        lines.append(
            f"{r.profile}\t{r.backend}\t{r.plain_wer:.4f}\t{gain}\t"
            f"{r.loudnorm_wer:.4f}\t{delta:+.4f}"
        )
    return lines


def _ffmpeg_version() -> str:
    """First line of ``ffmpeg -version``, or ``unknown`` if it cannot be read."""
    try:
        proc = subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    first = proc.stdout.splitlines()[:1]
    return first[0] if first else "unknown"


def run_header(profile: str) -> list[str]:
    """Reproducibility header printed above the table (issue #154, decision 2).

    Seeds, profile parameters, the ffmpeg version and every pinned model
    revision, so a printed table can be traced to its inputs.
    """
    steps = "->".join(STEPS[profile])
    models = " ".join(f"{m.name}={m.repo_id}@{m.revision}" for m in MODELS)
    return [
        f"# degrade={profile} base_seed={BASE_SEED} steps={steps} "
        "seed=clip_seed(stem, profile) (src/vemoizer/degrade_signal.py)",
        f"# attenuation_db={ATTENUATION_DB} reverb_t60_s={REVERB_T60_S} "
        f"reverb_len_s={REVERB_LEN_S} snr_db={NOISE_SNR_DB} noise=white",
        f"# ffmpeg={_ffmpeg_version()}",
        f"# models {models}",
        "# gain=oracle level control: each clip scaled to its clean RMS (needs the "
        "clean reference, so it is an oracle, not an inference-time method; "
        "not a proven upper bound).",
        "# compare the gain column with loudnorm (same clips; measured results "
        "in docs/pipeline-spec.md). whisper not covered "
        "(parakeet/canary/consensus only).",
        "# informational, not gated: no default changes. WER is uncapped: values "
        "above 1.0 come from insertions (e.g. repetition loops); check per-clip "
        "output before citing.",
    ]
