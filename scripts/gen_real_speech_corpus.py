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
word bounds fixed here), which is how the ids were originally selected
(issue #62).

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
"""

from __future__ import annotations

import argparse
import hashlib
import random
import shutil
import subprocess
import sys
import wave
from dataclasses import dataclass
from pathlib import Path

from vemoizer.audio_contract import SAMPLE_RATE  # single home for the 16 kHz contract

try:
    # Direct script run (``python scripts/gen_real_speech_corpus.py``) puts the
    # script's directory on sys.path, so the sibling module imports directly.
    from wav_header import wav_duration_seconds
except ImportError:  # pragma: no cover - covered by the tests' importlib path
    # Imported without the sibling on sys.path (e.g. via importlib): resolve
    # the sibling next to this file.
    import importlib.util as _importlib_util

    _spec = _importlib_util.spec_from_file_location(
        "_fleurs_wav_header", Path(__file__).resolve().parent / "wav_header.py"
    )
    _sib = _importlib_util.module_from_spec(_spec)
    assert _spec.loader is not None
    _spec.loader.exec_module(_sib)
    wav_duration_seconds = _sib.wav_duration_seconds

#: FLEURS dataset (CC-BY-4.0) — ungated, public on HuggingFace.
FLEURS_REPO = "google/fleurs"
FLEURS_SPLIT = "fi_fi"
FLEURS_PARQUET = f"parquet-data/{FLEURS_SPLIT}/train-00000-of-00001.parquet"
#: Dataset commit the committed corpus was drawn from (CORPUS_ATTRIBUTION.md)
#: — the URL is pinned to it rather than ``main`` so a re-run always resolves
#: the same revision (AGENTS.md invariant 4: revision pinning).
FLEURS_REVISION = "70bb2e84b976b7e960aa89f1c648e09c59f894dd"
FLEURS_URL = f"https://huggingface.co/datasets/{FLEURS_REPO}/resolve/{FLEURS_REVISION}/{FLEURS_PARQUET}"
#: SHA-256 of ``FLEURS_PARQUET`` at ``FLEURS_REVISION`` (single source of truth;
#: ``CORPUS_ATTRIBUTION.md`` records the same value and points here). Compared
#: by ``verify_parquet_digest`` before regenerating.
FLEURS_PARQUET_SHA256 = (
    "1fe57ed16edcf35014fd8b3fb6ed85b1a45b9478251fca44c2ae9acee07185c9"
)

#: Selection parameters — the seed and bounds ARE the reproducibility
#: contract; a re-run with the same values must produce byte-identical WAVs.
SEED = 20261003
N_CLIPS = 28
MIN_SECONDS = 3.5
MAX_SECONDS = 5.5
MIN_WORDS = 8
MAX_WORDS = 18

#: Contract sample width (bytes) of the clips written: 16-bit PCM, 16 kHz
#: mono (``resample_to_contract`` converts the float32 source to this).
SOURCE_WIDTH = 2  # 16-bit PCM

#: FLEURS clip ids of the committed corpus — the authoritative selection
#: (issue #62); a re-run reproduces exactly these clips, independent of the
#: selection parameters remaining valid for the dataset's current shape.
COMMITTED_CLIP_IDS = [
    24,
    25,
    34,
    36,
    36,
    204,
    235,
    238,
    246,
    252,
    533,
    596,
    604,
    630,
    656,
    692,
    711,
    732,
    748,
    748,
    800,
    991,
    1039,
    1046,
    1096,
    1290,
    1324,
    1343,
]


@dataclass(frozen=True)
class Clip:
    """One selected FLEURS clip.

    ``row_index`` is the parquet row position (stable across re-reads);
    ``clip_id`` is the dataset's ``id`` column. Two rows may share
    ``clip_id`` (different takes) — the stem is disambiguated on collision
    (see the module docstring).
    """

    row_index: int
    clip_id: int
    transcript: str
    audio_bytes: bytes


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


def verify_parquet_digest(
    parquet_path: Path, expected_sha256: str | None = None
) -> str:
    """Verify the local parquet's SHA-256 before regenerating from it.

    *expected_sha256* defaults to :data:`FLEURS_PARQUET_SHA256`. Compared
    *before* selection/write so a drifted revision is caught before any clip
    is touched; pass ``--expected-sha256`` to regenerate from a newer
    revision. Raises ``ValueError`` naming both digests on mismatch; pure
    (no ``pyarrow`` import), so it runs before the parquet is loaded.
    """
    expected = expected_sha256 or FLEURS_PARQUET_SHA256
    h = hashlib.sha256()
    with parquet_path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    actual = h.hexdigest()
    if actual != expected:
        raise ValueError(
            f"parquet SHA-256 mismatch: expected {expected} but {parquet_path} has "
            f"{actual}; the dataset revision likely drifted — pass --expected-sha256 "
            f"{actual} to regenerate deliberately, or --skip-parquet-check to bypass"
        )
    return expected


def load_fleurs_rows(parquet_path: Path) -> list[dict[str, object]]:
    """Read the FLEURS train parquet into plain dicts (one row per clip).

    Field types depend on the schema; the caller coerces as needed.
    ``dict[str, object]`` keeps the signature stable across revisions.
    """
    try:
        import pyarrow.parquet as pq
    except ImportError as e:
        raise RuntimeError(
            "pyarrow is not installed; install it with `uv pip install pyarrow` "
            "(a dev-time-only dependency) and re-run"
        ) from e
    table = pq.read_table(parquet_path)
    rows = table.to_pylist()
    return rows


def select_clips(
    rows: list[dict[str, object]], *, seed: int, n: int, ids: list[int] | None = None
) -> list[Clip]:
    """Deterministic selection of *n* clips from *rows*.

    Without *ids* (``--ids seeded``): filter to the duration/word window
    (via :func:`wav_duration_seconds`), sort by ``(clip_id, row_index)``, draw
    with ``random.Random(seed).sample`` (issue #62); same input + parameters
    always yields the same set, so a re-run is byte-identical.

    With *ids* (the default recorded-id path): the recorded id list is
    **authoritative** but the window is still applied — FLEURS rows repeat an
    ``id`` across speakers, and the window picks the wanted take. Each id's
    in-window row count must equal its recorded take count (ids 36 and 748
    have two takes; all others one); a drift is a ``ValueError`` so a silent
    corpus swap can never re-letter a stem relative to
    ``CORPUS_ATTRIBUTION.md``. Sorted by ``(clip_id, row_index)`` for stable
    same-id order (``0036`` then ``0036b``).
    """
    if ids is not None:
        if not ids:
            raise ValueError("--ids was given but is empty")
        expected_counts: dict[int, int] = {}
        for cid in ids:
            expected_counts[cid] = expected_counts.get(cid, 0) + 1
        window_rows: dict[int, list[tuple[int, dict[str, object]]]] = {}
        for row_index, r in enumerate(rows):
            if r.get("id") is None:
                continue
            cid = _as_int(r["id"])
            if cid not in expected_counts:
                continue
            audio = r["audio"]
            if not isinstance(audio, dict):  # FLEURS stores {bytes, path}
                raise ValueError(
                    f"row {row_index}: expected a dict for 'audio', "
                    f"got {type(audio).__name__!r}"
                )
            raw = audio["bytes"]
            if not isinstance(raw, (bytes, bytearray)):
                continue
            duration = wav_duration_seconds(_as_bytes(raw))
            words = str(r["transcription"]).split()
            if not (
                MIN_SECONDS <= duration <= MAX_SECONDS
                and MIN_WORDS <= len(words) <= MAX_WORDS
            ):
                continue  # this take is outside the window; a same-id take
                # inside the window still stands in its place
            window_rows.setdefault(cid, []).append((row_index, r))
        # The in-window take count must match the recorded take count, or a
        # drift would re-letter stems relative to CORPUS_ATTRIBUTION.md.
        for cid, expected in sorted(expected_counts.items()):
            actual = len(window_rows.get(cid, []))
            if actual != expected:
                raise ValueError(
                    f"clip id {cid}: recorded {expected} take(s) but found "
                    f"{actual} in-window row(s) in the parquet — the window "
                    f"selection changed; do not regenerate silently"
                )
        picked: list[tuple[int, dict[str, object]]] = []
        for cid in sorted(window_rows):
            picked.extend(sorted(window_rows[cid], key=lambda t: t[0]))
        return [
            Clip(
                row_index=ri,
                clip_id=_as_int(r["id"]),
                transcript=str(r["transcription"]),
                audio_bytes=_clip_audio_bytes(r, row_index),
            )
            for ri, r in picked
        ]
    candidates: list[Clip] = []
    for row_index, r in enumerate(rows):
        audio = r["audio"]
        if not isinstance(audio, dict):  # FLEURS stores {bytes, path}
            raise ValueError(
                f"row {row_index}: expected a dict for 'audio', "
                f"got {type(audio).__name__!r}"
            )
        raw = audio["bytes"]
        if not isinstance(raw, (bytes, bytearray)):
            continue
        duration = wav_duration_seconds(_as_bytes(raw))
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
            f"no FLEURS rows matched the window [{MIN_SECONDS}, {MAX_SECONDS}]s / "
            f"{MIN_WORDS}-{MAX_WORDS} words; the dataset shape may have changed"
        )
    # Sort by (clip_id, row_index): stable order, independent of file row order.
    candidates.sort(key=lambda c: (c.clip_id, c.row_index))
    if n > len(candidates):
        raise RuntimeError(
            f"selected {n} clips but only {len(candidates)} candidates exist in "
            f"the window; lower --n or widen the window"
        )
    rng = random.Random(seed)
    return rng.sample(candidates, n)


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


def _clip_audio_bytes(r: dict[str, object], row_index: int) -> bytes:
    """Coerce a row's audio bytes to ``bytes`` (single coercion point for the
    ``dict[str, object]`` row type; a clean ``ValueError`` on a bad shape).
    """
    audio = r.get("audio")
    if not isinstance(audio, dict):
        raise ValueError(
            f"row {row_index}: expected a dict for 'audio', "
            f"got {type(audio).__name__!r}"
        )
    return _as_bytes(audio["bytes"])


def _as_int(v: object) -> int:
    """Coerce an *object* to *int*, raising a clear error on failure.

    Accepts ``int`` (as-is) or ``str``/``bytes``/``bytearray`` (via
    ``int()``). A ``bool`` raises even though it is an ``int`` subclass —
    accepting it as 0/1 would hide a schema drift (FLEURS ``id`` is ``int32``).
    """
    if isinstance(v, bool):
        raise ValueError(
            f"cannot coerce bool to int (bool is an int subclass, rejected): {v!r}"
        )
    if isinstance(v, int):
        return v
    if isinstance(v, (str, bytes, bytearray)):
        return int(v)
    raise ValueError(f"cannot coerce {type(v).__name__} to int: {v!r}")


def _as_bytes(v: object) -> bytes:
    """Coerce an *object* to *bytes* (see :func:`_as_int`)."""
    if isinstance(v, bytes):
        return v
    if isinstance(v, bytearray):
        return bytes(v)
    raise ValueError(f"cannot coerce {type(v).__name__} to bytes: {v!r}")


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
    *expected_sha256* (default :data:`FLEURS_PARQUET_SHA256`) unless
    *skip_parquet_check* is set (see :func:`verify_parquet_digest`).
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
