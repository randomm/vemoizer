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
``parquet-data/fi_fi/train-00000-of-00001.parquet`` directly with ``pyarrow``
(a dev-time dependency only — never imported at runtime). By default the
script regenerates exactly the committed 28 clips via their recorded FLEURS
ids (``--ids``; the authoritative set is in ``CORPUS_ATTRIBUTION.md``);
``--ids seeded`` reproduces the original seeded window draw (seed + duration/
word bounds fixed in :mod:`fleurs_source`), which is how the ids were
originally selected (issue #62).

The download URL is pinned to the dataset commit recorded in
``CORPUS_ATTRIBUTION.md`` (``FLEURS_REVISION``, not ``main``) and the local
parquet's SHA-256 is verified against ``FLEURS_PARQUET_SHA256`` *before*
regenerating; override deliberately with ``--expected-sha256`` or skip with
``--skip-parquet-check``.

On **both** paths the duration/word window is part of the selection: FLEURS
rows repeat an ``id`` across speakers, and the window picks the wanted take(s)
among them (on the ``--ids`` path the in-window take count per id is checked
against the recorded take count and must match, so a drift can never
re-letter a stem).

Dev-time usage (network required for the one-time parquet download)::

    uv pip install pyarrow
    uv run python scripts/gen_real_speech_corpus.py --parquet <local.parquet>

After regenerating, re-measure the WER baseline
(``vemoizer eval --backend all --update-baseline``) in a DEDICATED commit
(per AGENTS.md; never mixed into a feature commit).

Clips are written as ``fleurs_fi_<id:04d>.wav`` (16 kHz mono 16-bit, the
contract — the source parquet is 16 kHz mono **32-bit float**) side by side
with a same-stem ``.txt`` reference transcript (the stem-pair contract
``tests/test_fixture_corpus.py`` enforces). Duration is computed from the
real sample count read out of the WAV header
(:func:`wav_duration_seconds`), never from a hard-coded byte width — the
source is float32, and the committed corpus was drawn with this exact
selection (the 28 clip ids are the authoritative set, recorded in
``CORPUS_ATTRIBUTION.md`` and usable via ``--ids``).

The FLEURS parquet keys rows by ``id`` (utterance), not by clip — several rows
share an ``id`` (different takes). The stem is the 4-digit ``id``; on a
collision (two selected takes of the same utterance) the second+ take in
``(id, row_index)`` order gets a ``b``, ``c``, ... suffix, so no two clips
collide on a stem (the ``.wav``/``.txt`` pair contract is per-stem).

The dataset-source side (source constants, digest guard, row loading, clip
selection) lives in the sibling module :mod:`fleurs_source` (reached via the
shared :func:`scripts._sibling_loader.load_sibling` helper, issue #138);
this script is the CLI plus the writing side (corpus-dir guard, resampling,
clip writing).
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import wave
from pathlib import Path

from vemoizer.audio_contract import SAMPLE_RATE  # single home for the 16 kHz contract

try:
    from scripts._sibling_loader import load_sibling
except ImportError:
    # The bare fallback (script directory on ``sys.path``) is not resolvable
    # to a first-party module from the project root, so ty cannot see it.
    # ty: ignore[unresolved-import] - runtime-optional path
    from _sibling_loader import load_sibling

_fleurs_source = load_sibling("fleurs_source")
Clip = _fleurs_source.Clip
N_CLIPS = _fleurs_source.N_CLIPS
SEED = _fleurs_source.SEED
COMMITTED_CLIP_IDS = _fleurs_source.COMMITTED_CLIP_IDS
FLEURS_PARQUET_SHA256 = _fleurs_source.FLEURS_PARQUET_SHA256
FLEURS_URL = _fleurs_source.FLEURS_URL
load_fleurs_rows = _fleurs_source.load_fleurs_rows
select_clips = _fleurs_source.select_clips
verify_parquet_digest = _fleurs_source.verify_parquet_digest

#: Contract sample width (bytes) of the clips written: 16-bit PCM, 16 kHz
#: mono (``resample_to_contract`` converts the float32 source to this).
SOURCE_WIDTH = 2  # 16-bit PCM


def resolve_corpus_dir(out: Path) -> Path:
    """Resolve *out* and verify it stays strictly under the project root.

    A symlinked corpus dir would let ffmpeg follow the link and overwrite an
    arbitrary file when resampling in place. The project root itself is not
    allowed: ``Path.resolve()`` on a symlink to the root yields the root, which
    an ``==`` or naive ``in parents`` check would accept, so we require a
    *strict* descendant — the root must appear in ``resolved.parents``,
    excluding the equality case.
    """
    resolved = out.resolve()
    project_root = Path(__file__).resolve().parent.parent
    if project_root not in resolved.parents:
        raise RuntimeError(
            f"refusing to write corpus outside the project root: {resolved} "
            f"(project root: {project_root})"
        )
    return resolved


def resample_to_contract(src: Path, dst: Path) -> None:
    """Resample *src* to the 16 kHz mono 16-bit contract via ffmpeg.

    In-place resampling (*src* == *dst*) writes to ``<dst>.tmp.wav`` and
    renames, because ffmpeg refuses to truncate its own input — a mid-write
    failure cannot leave a corrupt file behind. The temp name ends in ``.wav``
    so ffmpeg infers the wav muxer.
    """
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg not found on PATH; required for resampling.")
    same_file = src.resolve() == dst.resolve()
    # In-place resampling writes to a temp file (renamed after) because
    # ffmpeg refuses to truncate its own input.
    tmp_path = dst.parent / f"{dst.stem}.tmp.wav" if same_file else dst
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
            f"ffmpeg failed (returncode {e.returncode}) resampling {src} -> {dst}: "
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


def write_clip(clip: Clip, corpus_dir: Path, stem: str) -> tuple[Path, Path]:
    """Write one clip as ``<stem>.wav`` + same-stem ``.txt``."""
    wav_path = corpus_dir / f"{stem}.wav"
    txt_path = corpus_dir / f"{stem}.txt"
    wav_path.write_bytes(clip.audio_bytes)
    resample_to_contract(wav_path, wav_path)
    # Sanity-check the resampled WAV against the contract before writing the
    # transcript — a violation here is a script bug (or a changed dataset).
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
    corpus_dir: Path,
    parquet_path: Path | None,
    n: int,
    ids: list[int] | None = None,
    *,
    expected_sha256: str | None = None,
    skip_parquet_check: bool = False,
) -> list[Path]:
    """(Re)generate the FLEURS clips; returns the written WAV paths.

    Only the ``fleurs_fi_*`` stems are touched (the Piper stems are left
    alone, so this script is additive to the corpus). With *ids* the clip
    set is exactly those ids; otherwise the seeded window draws *n* clips.
    Before any clip is written, the parquet's SHA-256 is verified against
    *expected_sha256* (default :data:`FLEURS_PARQUET_SHA256`)
    unless *skip_parquet_check* is set (see :func:`verify_parquet_digest`).
    """
    if parquet_path is None:
        raise RuntimeError(
            "no parquet path — pass --parquet or set FLEURS_PARQUET; the "
            "download is a one-time dev-time step, never a runtime fetch"
        )
    if not skip_parquet_check:
        verify_parquet_digest(parquet_path, expected_sha256)
    rows = load_fleurs_rows(parquet_path)
    clips = select_clips(rows, seed=SEED, n=n, ids=ids)
    clips.sort(key=lambda c: (c.clip_id, c.row_index))  # stable write order
    # Assign stems: two selected clips may share clip_id (different takes);
    # the stem is the 4-digit id, with a b/c/... suffix on a collision so no
    # two clips share a stem (the .wav/.txt pair contract is per-stem).
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
        help="local FLEURS fi_fi train parquet path (downloaded once; URL is "
        "FLEURS_URL in the module docstring)",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=N_CLIPS,
        help=f"number of clips to select (default: {N_CLIPS})",
    )
    parser.add_argument(
        "--ids",
        default=None,
        help="comma-separated FLEURS clip ids to regenerate (default: the "
        "committed 28 ids; 'seeded' uses the seeded window draw)",
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
    parser.add_argument(
        "--expected-sha256",
        default=None,
        help="SHA-256 the parquet must match (default: FLEURS_PARQUET_SHA256); "
        "pass the file's digest to regenerate from a newer revision",
    )
    parser.add_argument(
        "--skip-parquet-check",
        action="store_true",
        help="skip the parquet SHA-256 check (use deliberately)",
    )
    args = parser.parse_args(argv)

    if args.parquet is None:
        print(
            f"error: --parquet is required (one-time download from {FLEURS_URL})",
            file=sys.stderr,
        )
        return 2
    if not args.parquet.is_file():
        print(f"error: parquet not found: {args.parquet}", file=sys.stderr)
        return 2

    ids = _parse_ids(args.ids)
    corpus_dir = resolve_corpus_dir(args.out)
    corpus_dir.mkdir(parents=True, exist_ok=True)
    written = generate_real_speech(
        corpus_dir,
        args.parquet,
        args.n,
        ids=ids,
        expected_sha256=args.expected_sha256,
        skip_parquet_check=args.skip_parquet_check,
    )
    for path in written:
        print(f"wrote {path}")
    mode = f"ids {len(ids)}" if ids else f"seed {SEED}"
    print(f"corpus: {len(written)} FLEURS clips ({mode}) under {corpus_dir}")
    return 0


def _parse_ids(raw: str | None) -> list[int] | None:
    """Parse ``--ids``: the committed corpus ids, or ``None`` for the seeded draw."""
    if raw is None:
        return list(COMMITTED_CLIP_IDS)  # default: regenerate exactly what is committed
    if raw.strip() == "seeded":
        return None
    try:
        return [int(p) for p in raw.split(",") if p.strip()]
    except ValueError as e:
        raise SystemExit(
            f"error: --ids must be comma-separated ints or 'seeded': {e}"
        ) from e


if __name__ == "__main__":
    raise SystemExit(main())
