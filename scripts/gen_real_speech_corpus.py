"""Generate the real-speech (FLEURS ``fi_fi``) fixture clips under
``tests/fixtures/corpus/``.

The Piper TTS corpus (``scripts/gen_fixtures.py``) is a regression gate, not
a challenge set: on clean synthetic speech both decoders are often *similarly
wrong*, and two-way comparison cannot detect agreement on the wrong answer
(issue #62). This script adds a harder additive corpus of genuinely human
speech: ``N`` seeded, deterministic clips from the FLEURS Finnish train set
(``google/fleurs``, config ``fi_fi``, CC-BY-4.0 — the attribution file
``tests/fixtures/corpus/CORPUS_ATTRIBUTION.md`` is required by the licence).

FLEURS Finnish is ungated; the selection reads the dataset's public
``parquet-data/fi_fi/train-00000-of-00001.parquet`` directly with
``pyarrow`` (a dev-time dependency only — never imported at runtime). The
selection is fully deterministic: the seed, the candidate window (duration
and word-count bounds), and the sample size are all fixed here, so a
re-run produces byte-identical clips.

Dev-time usage (network required for the one-time parquet download)::

    uv pip install pyarrow
    uv run python scripts/gen_real_speech_corpus.py

After regenerating, re-measure the WER baseline
(``vemoizer eval --backend all --update-baseline``) in a DEDICATED commit
(per AGENTS.md; never mixed into a feature commit).

Clips are written as ``fleurs_fi_<id:04d>.wav`` (16 kHz mono, per the
source parquet) side by side with a same-stem ``.txt`` reference transcript
— the stem-pair contract ``tests/test_fixture_corpus.py`` enforces.

Note: the FLEURS parquet keys rows by ``id`` (utterance), not by clip —
several rows share an ``id`` (different takes of the same reference). The
stem is the 4-digit ``id``; on a collision (two selected takes of the same
utterance) the second+ take in ``(id, row_index)`` order gets a ``b``,
``c``, ... suffix, so no two clips collide on a stem (the ``.wav``/``.txt``
pair contract is per-stem, not per-utterance).
"""

from __future__ import annotations

import argparse
import random
import shutil
import subprocess
import sys
import wave
from dataclasses import dataclass
from pathlib import Path

from vemoizer.audio_contract import SAMPLE_RATE  # single home for the 16 kHz contract

#: FLEURS dataset (CC-BY-4.0) — ungated, public on HuggingFace.
FLEURS_REPO = "google/fleurs"
FLEURS_SPLIT = "fi_fi"
FLEURS_PARQUET = f"parquet-data/{FLEURS_SPLIT}/train-00000-of-00001.parquet"
FLEURS_URL = (
    f"https://huggingface.co/datasets/{FLEURS_REPO}/resolve/main/{FLEURS_PARQUET}"
)

#: Selection parameters — the seed and bounds ARE the reproducibility
#: contract. Changing any of them changes which clips are selected; a
#: re-run with the same values must produce byte-identical WAVs.
SEED = 20261003
N_CLIPS = 28
MIN_SECONDS = 3.5
MAX_SECONDS = 5.5
MIN_WORDS = 8
MAX_WORDS = 18

#: The source parquet stores 16 kHz mono 16-bit PCM — the audio contract
#: already, so the resampling step is a sanity check (and a guard against
#: a future dataset revision changing the format) rather than a
#: transformation.
SOURCE_WIDTH = 2  # 16-bit PCM


@dataclass(frozen=True)
class Clip:
    """One selected FLEURS clip.

    ``row_index`` is the parquet row position (stable across re-reads of
    the same file); ``clip_id`` is the dataset's ``id`` column. Two rows
    may share ``clip_id`` (different takes of the same reference) — the
    stem is disambiguated on collision (see the module docstring).
    """

    row_index: int
    clip_id: int
    transcript: str
    audio_bytes: bytes


def resolve_corpus_dir(out: Path) -> Path:
    """Resolve *out* and verify it stays under the project root.

    Mirrors ``scripts/gen_fixtures.py::resolve_corpus_dir`` — a symlinked
    corpus dir would let ffmpeg follow the link and overwrite an arbitrary
    file when resampling in place.
    """
    resolved = out.resolve()
    project_root = Path(__file__).resolve().parent.parent
    if not (resolved == project_root or project_root in resolved.parents):
        raise RuntimeError(
            f"refusing to write corpus outside the project root: {resolved} "
            f"(project root: {project_root})"
        )
    return resolved


def load_fleurs_rows(parquet_path: Path) -> list[dict[str, object]]:
    """Read the FLEURS train parquet into plain dicts (one row per clip).

    The ``to_pylist()`` call returns dicts whose field types depend on the
    parquet schema (``int32`` for ``id``, ``string`` for ``transcription``,
    ``struct`` for ``audio``). The caller accesses only the fields it cares
    about and coerces with ``str()`` / ``int()`` as needed; the ``object``
    type keeps the function signature stable across schema revisions.
    """
    try:
        import pyarrow.parquet as pq
    except ImportError as e:
        raise RuntimeError(
            "pyarrow is not installed. Install it with `uv pip install pyarrow` "
            "and re-run — it is a dev-time dependency only (never a runtime "
            "dependency of vemoizer)."
        ) from e
    table = pq.read_table(parquet_path)
    rows: list[dict[str, object]] = table.to_pylist()  # type: ignore[assignment]
    return rows


def select_clips(rows: list[dict[str, object]], *, seed: int, n: int) -> list[Clip]:
    """Deterministic seeded selection of *n* clips from *rows*.

    Candidates are filtered to the duration/word-count window, sorted by
    clip id (the natural order — independent of parquet row order), then
    drawn with ``random.Random(seed).sample``. The same input + same
    parameters always yields the same set, so a re-run is byte-identical.
    """
    candidates: list[Clip] = []
    for row_index, r in enumerate(rows):
        audio = r["audio"]
        assert isinstance(audio, dict)  # FLEURS stores {bytes, path}
        raw = audio["bytes"]
        if not isinstance(raw, (bytes, bytearray)):
            continue
        duration = len(raw) / 4 / SAMPLE_RATE  # 16-bit mono @ 16 kHz
        words = str(r["transcription"]).split()
        if (
            MIN_SECONDS <= duration <= MAX_SECONDS
            and MIN_WORDS <= len(words) <= MAX_WORDS
        ):
            candidates.append(
                Clip(
                    row_index=row_index,
                    clip_id=_as_int(r["id"]),
                    transcript=str(r["transcription"]),
                    audio_bytes=_as_bytes(raw),
                )
            )
    if not candidates:
        raise RuntimeError(
            f"no FLEURS rows matched the selection window "
            f"[{MIN_SECONDS}, {MAX_SECONDS}]s / {MIN_WORDS}-{MAX_WORDS} words; "
            f"the dataset shape may have changed — update the window and re-run"
        )
    # Sort by (clip_id, row_index): the natural dataset order, stable
    # across re-reads of the same parquet file, independent of row order
    # in the file (which is not guaranteed to be sorted by id).
    candidates.sort(key=lambda c: (c.clip_id, c.row_index))
    if n > len(candidates):
        raise RuntimeError(
            f"selected {n} clips but only {len(candidates)} candidates exist "
            f"in the window; lower --n or widen the window"
        )
    rng = random.Random(seed)
    return rng.sample(candidates, n)


def resample_to_contract(src: Path, dst: Path) -> None:
    """Resample *src* to the 16 kHz mono 16-bit contract via ffmpeg.

    When *src* and *dst* are the same path (in-place resampling), ffmpeg
    refuses to open the output because it would truncate the input; the
    implementation writes to ``<dst>.tmp`` and renames, so a mid-write
    failure cannot leave a corrupt file behind.
    """
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg not found on PATH; required for resampling.")
    same_file = src.resolve() == dst.resolve()
    tmp_path = dst.parent / f"{dst.stem}.wav.tmp" if same_file else dst
    # Ensure the temp file ends with .wav (so ffmpeg infers the wav muxer)
    if same_file:
        tmp_path = dst.parent / f"{dst.stem}.tmp.wav"
    try:
        subprocess.run(
            [
                ffmpeg,
                "-nostdin",
                "-v",
                "error",
                "-y",
                "-i",
                str(src),
                "-ac",
                "1",
                "-ar",
                str(SAMPLE_RATE),
                "-c:a",
                "pcm_s16le",
                str(tmp_path),
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=300,
        )
    except subprocess.TimeoutExpired as e:
        _clean_tmp(tmp_path, same_file)
        raise RuntimeError(f"ffmpeg timed out resampling {src} -> {dst}") from e
    except subprocess.CalledProcessError as e:
        _clean_tmp(tmp_path, same_file)
        stderr_tail = (e.stderr or "").strip()[-2000:]
        raise RuntimeError(
            f"ffmpeg failed (returncode {e.returncode}) resampling {src} -> {dst}:\n"
            f"{stderr_tail}"
        ) from e
    if same_file:
        tmp_path.replace(dst)


def _clean_tmp(tmp: Path, same_file: bool) -> None:
    """Remove the temp file on failure (only when it is a true temp)."""
    from contextlib import suppress

    if same_file and tmp.exists():
        with suppress(OSError):
            tmp.unlink()


def _as_int(v: object) -> int:
    """Coerce an *object* to *int*, raising a clear error on failure.

    FLEURS rows are typed as ``dict[str, object]`` (see :func:`load_fleurs_rows`)
    to keep the signature stable across schema revisions; this helper is
    the single coercion point so the ``int(...)`` call site does not need
    a ``# type: ignore`` and the failure mode is a clean ``ValueError``
    (not a silent ``TypeError`` from the ``int`` constructor).
    """
    if isinstance(v, int):
        return v
    if isinstance(v, (str, bytes, bytearray)):
        return int(v)
    raise ValueError(f"cannot coerce {type(v).__name__} to int: {v!r}")


def _as_bytes(v: object) -> bytes:
    """Coerce an *object* to *bytes*, raising a clear error on failure.

    See :func:`_as_int` for the rationale.
    """
    if isinstance(v, bytes):
        return v
    if isinstance(v, bytearray):
        return bytes(v)
    raise ValueError(f"cannot coerce {type(v).__name__} to bytes: {v!r}")


def _clean_tmp(tmp: Path, same_file: bool) -> None:
    """Remove the temp file on failure (only when it is a true temp)."""
    from contextlib import suppress

    if same_file and tmp.exists():
        with suppress(OSError):
            tmp.unlink()


def write_clip(clip: Clip, corpus_dir: Path, stem: str) -> tuple[Path, Path]:
    """Write one clip as ``<stem>.wav`` + same-stem ``.txt``."""
    wav_path = corpus_dir / f"{stem}.wav"
    txt_path = corpus_dir / f"{stem}.txt"
    wav_path.write_bytes(clip.audio_bytes)
    resample_to_contract(wav_path, wav_path)
    # Sanity-check the resampled WAV against the contract before writing
    # the transcript — a contract violation here is a script bug (or a
    # changed dataset), not a fixture defect.
    with wave.open(str(wav_path), "rb") as w:
        if (w.getnchannels(), w.getsampwidth(), w.getframerate()) != (
            1,
            SOURCE_WIDTH,
            SAMPLE_RATE,
        ):
            raise RuntimeError(
                f"{wav_path.name}: resampled WAV violates the contract "
                f"({w.getnchannels()} ch, {w.getsampwidth() * 8}-bit, "
                f"{w.getframerate()} Hz)"
            )
    txt_path.write_text(clip.transcript + "\n", encoding="utf-8")
    return wav_path, txt_path


def generate_real_speech(
    corpus_dir: Path, parquet_path: Path | None, n: int
) -> list[Path]:
    """(Re)generate the FLEURS clips; returns the written WAV paths.

    Only the ``fleurs_fi_*`` stems are touched — the Piper stems
    (``fi_*``, ``en_*``, ``meeting_sample``) are left alone, so this
    script is additive to the existing corpus.
    """
    if parquet_path is None:
        raise RuntimeError(
            "no parquet path — pass --parquet or set FLEURS_PARQUET; the "
            "download is a one-time dev-time step, never a runtime fetch"
        )
    rows = load_fleurs_rows(parquet_path)
    clips = select_clips(rows, seed=SEED, n=n)
    clips.sort(key=lambda c: (c.clip_id, c.row_index))  # stable write order
    # Assign stems: two selected clips may share clip_id (different takes of
    # the same reference). The stem is the 4-digit clip id; on a collision,
    # the second+ take in (clip_id, row_index) order gets a "b", "c", ...
    # suffix so no two clips collide on a stem (the .wav/.txt pair contract
    # is per-stem, not per-utterance).
    counter: dict[int, int] = {}
    written: list[Path] = []
    for clip in clips:
        k = counter.get(clip.clip_id, 0)
        suffix = "" if k == 0 else chr(ord("a") + k)
        stem = f"fleurs_fi_{clip.clip_id:04d}{suffix}"
        counter[clip.clip_id] = k + 1
        wav_path, _ = write_clip(clip, corpus_dir, stem)
        written.append(wav_path)
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--parquet",
        type=Path,
        default=None,
        help="local path to the FLEURS fi_fi test parquet (downloaded once; "
        "re-runs reuse it). The URL is in the module docstring.",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=N_CLIPS,
        help=f"number of clips to select (default: {N_CLIPS})",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path(__file__).resolve().parent.parent
        / "tests"
        / "fixtures"
        / "corpus",
        help="corpus directory (default: tests/fixtures/corpus)",
    )
    args = parser.parse_args(argv)

    if args.parquet is None:
        print(
            f"error: --parquet is required (one-time download from {FLEURS_URL}); "
            f"the script does not fetch on its own — see the module docstring",
            file=sys.stderr,
        )
        return 2
    if not args.parquet.is_file():
        print(f"error: parquet not found: {args.parquet}", file=sys.stderr)
        return 2

    corpus_dir = resolve_corpus_dir(args.out)
    corpus_dir.mkdir(parents=True, exist_ok=True)
    written = generate_real_speech(corpus_dir, args.parquet, args.n)
    for path in written:
        print(f"wrote {path}")
    print(f"corpus: {len(written)} FLEURS clips (seed {SEED}) under {corpus_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
