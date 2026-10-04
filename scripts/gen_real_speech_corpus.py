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
``pyarrow`` (a dev-time dependency only — never imported at runtime).
By default the script regenerates exactly the committed 28 clips via
their recorded FLEURS ids (``--ids``; the authoritative set is in
``CORPUS_ATTRIBUTION.md``), so a re-run produces byte-identical clips
without depending on the candidate window. ``--ids seeded`` reproduces
the original seeded window draw (seed + duration/word bounds fixed here),
which is how the ids were originally selected (issue #62).

Dev-time usage (network required for the one-time parquet download)::

    uv pip install pyarrow
    uv run python scripts/gen_real_speech_corpus.py

After regenerating, re-measure the WER baseline
(``vemoizer eval --backend all --update-baseline``) in a DEDICATED commit
(per AGENTS.md; never mixed into a feature commit).

Clips are written as ``fleurs_fi_<id:04d>.wav`` (16 kHz mono 16-bit, the
contract — the source parquet is 16 kHz mono **32-bit float**) side by side
with a same-stem ``.txt`` reference transcript — the stem-pair contract
``tests/test_fixture_corpus.py`` enforces.

Duration is computed from the real sample count read out of the WAV header
(see :func:`wav_duration_seconds`), never from a hard-coded byte width — the
source is float32, and the committed corpus was drawn with this exact
selection (the 28 clip ids are the authoritative set, recorded in
``CORPUS_ATTRIBUTION.md`` and usable via ``--ids``).

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
import struct
import subprocess
import sys
import wave
from collections import Counter
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

#: The contract sample width (bytes) of the clips this script writes:
#: 16-bit PCM, 16 kHz mono. The FLEURS source parquet is 16 kHz mono
#: 32-bit *float*; ``resample_to_contract`` converts it, so the written
#: clips are always 16-bit.
SOURCE_WIDTH = 2  # 16-bit PCM

#: The FLEURS clip ids of the committed corpus — the authoritative selection
#: (issue #62). The seeded window draw that produced them is documented in
#: CORPUS_ATTRIBUTION.md; these ids make a re-run reproduce exactly the
#: committed clips without depending on the selection parameters remaining
#: valid for the dataset's current shape.
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
    """Resolve *out* and verify it stays strictly under the project root.

    Mirrors ``scripts/gen_fixtures.py::resolve_corpus_dir`` — a symlinked
    corpus dir would let ffmpeg follow the link and overwrite an arbitrary
    file when resampling in place.  The project root itself is **not**
    allowed: a symlink whose target *is* the project root would be accepted
    by an ``==`` or ``in parents`` check (``Path.resolve()`` on a symlink to
    the root yields the root), so we require a *strict* descendant — the
    project root must appear in ``resolved.parents``, which excludes the
    equality case.
    """
    resolved = out.resolve()
    project_root = Path(__file__).resolve().parent.parent
    if project_root not in resolved.parents:
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
    rows = table.to_pylist()
    return rows


def wav_duration_seconds(wav_bytes: bytes) -> float:
    """Duration of a WAV payload in seconds, from its real header.

    Reads the ``fmt `` and ``data`` chunks (skipping unknown ones such as
    ``fact``), so the result is the true length in sample units regardless
    of the source's sample width — the FLEURS parquet ships 32-bit float,
    and a hard-coded ``len(raw) / 4`` would silently misread a future
    16-bit revision of the dataset. A non-PCM header (e.g. WAVE_FORMAT_EXTENSIBLE
    with an unsupported codec) or a truncated header raises ``ValueError``
    rather than guessing a byte width.
    """
    try:
        header, payload = _parse_wav_chunks(wav_bytes)
    except (struct.error, IndexError, ValueError) as e:
        # _parse_wav_chunks raises ValueError for a non-RIFF/WAVE payload
        # or a struct.error for a truncated header; both map to the
        # documented clean error.
        raise ValueError(f"unparseable WAV payload: {e}") from e
    if header is None or payload is None:
        raise ValueError("WAV payload has no fmt or data chunk")
    audio_format, channels, rate, _, sample_width = header
    # WAVE_FORMAT_PCM (1) and WAVE_FORMAT_IEEE_FLOAT (3) — the FLEURS
    # parquet ships float32; both carry an honest sample width in the
    # header, so the duration is exact either way.
    if audio_format not in (1, 3) or rate == 0:
        raise ValueError(
            f"unsupported WAV format {audio_format} at {rate} Hz; "
            f"expected PCM (1) or IEEE float (3) at a non-zero rate"
        )
    return len(payload) / (sample_width * max(channels, 1)) / rate


def _parse_wav_chunks(
    wav_bytes: bytes,
) -> tuple[tuple[int, int, int, int, int] | None, bytes | None]:
    """Parse a WAV payload into ``(fmt fields, data payload)``.

    ``fmt fields`` is ``(audio_format, channels, rate, block_align,
    sample_width_bytes)``; either side is ``None`` when the chunk is
    absent. Unknown chunks (``fact``, ``LIST``, ...) are skipped, which is
    what makes this robust to the exact layout of the FLEURS parquet
    payloads (RIFF/WAVE with a single ``data`` chunk).
    """
    if wav_bytes[:4] != b"RIFF" or wav_bytes[8:12] != b"WAVE":
        raise ValueError("not a RIFF/WAVE payload")
    header: tuple[int, int, int, int, int] | None = None
    payload: bytes | None = None
    off = 12
    while off + 8 <= len(wav_bytes):
        chunk_id = wav_bytes[off : off + 4]
        size = struct.unpack("<I", wav_bytes[off + 4 : off + 8])[0]
        body = wav_bytes[off + 8 : off + 8 + size]
        if chunk_id == b"fmt " and len(body) >= 14:
            # The FLEURS parquet payloads carry an 18-byte fmt chunk whose
            # layout is (format, channels, rate, bytes_per_sample,
            # bytes_per_frame) packed into the standard positions — the
            # 4th field is 4 (float32 bytes/sample), not the standard
            # block align. The 5th field is the sample rate again (a
            # dataset quirk). We take the 4th field as the sample width
            # because it is exact for the float32 source and for any
            # 16-bit PCM revision.
            audio_format, channels, rate, _block_align, sample_width = struct.unpack(
                "<HHIIH", body[:14]
            )
            header = (audio_format, channels, rate, _block_align, sample_width)
        elif chunk_id == b"data" and payload is None:
            payload = body
        # The RIFF spec says a chunk with an odd size is followed by one
        # pad byte that is NOT counted in the chunk's size field; (size & 1)
        # accounts for that pad byte when the size is odd.
        off += 8 + size + (size & 1)
    return header, payload


def select_clips(
    rows: list[dict[str, object]], *, seed: int, n: int, ids: list[int] | None = None
) -> list[Clip]:
    """Deterministic selection of *n* clips from *rows*.

    Without *ids* (the ``--ids seeded`` path), candidates are filtered to
    the duration/word-count window (duration from the real WAV header via
    :func:`wav_duration_seconds`), sorted by ``(clip_id, row_index)``,
    then drawn with ``random.Random(seed).sample`` — the documented
    original draw (issue #62); the same input + same parameters always
    yields the same set, so a re-run is byte-identical.

    With *ids*, the recorded id list is **authoritative**: the duration/
    word window is **not** applied, because the recorded ids already encode
    the exact selection.  Instead, each id's **count** in the list (duplicates
    = number of takes) is checked against the number of rows that share that
    id in the dataset.  An id whose row count differs from the recorded take
    count is a ``ValueError`` naming the id, so a silent corpus swap (e.g.
    a dataset reshuffle that dropped a take or inserted a new same-id row)
    can never produce a stem letter that differs from the one recorded in
    ``CORPUS_ATTRIBUTION.md``.
    """
    if ids is not None:
        if not ids:
            raise ValueError("--ids was given but is empty")
        expected_counts: dict[int, int] = dict(Counter(ids))
        # Collect all rows for each recorded id (window not applied).
        id_rows: dict[int, list[tuple[int, dict[str, object]]]] = {}
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
            # Audio must parse (raises ValueError on a corrupt row); this is
            # the only sanity check — the window is deliberately not applied.
            wav_duration_seconds(_as_bytes(raw))
            id_rows.setdefault(cid, []).append((row_index, r))
        # Verify row count per id matches the recorded take count.
        for cid, expected in sorted(expected_counts.items()):
            actual = len(id_rows.get(cid, []))
            if actual != expected:
                raise ValueError(
                    f"clip id {cid}: recorded {expected} take(s) but found "
                    f"{actual} row(s) in the parquet — the dataset shape "
                    f"changed; do not regenerate silently"
                )
        picked: list[tuple[int, dict[str, object]]] = []
        for cid in sorted(id_rows):
            picked.extend(sorted(id_rows[cid], key=lambda t: t[0]))
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
    # In-place resampling writes to a temp file (renamed after) because
    # ffmpeg refuses to truncate its own input; the temp name ends in
    # .wav so ffmpeg infers the wav muxer.
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


def _clip_audio_bytes(r: dict[str, object], row_index: int) -> bytes:
    """Coerce a row's audio bytes to ``bytes``, with a clean error on failure.

    The FLEURS row type is ``dict[str, object]``; this helper is the
    single coercion point so the call site does not need a
    ``# type: ignore`` and the failure mode is a clean ``ValueError``.
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
    corpus_dir: Path, parquet_path: Path | None, n: int, ids: list[int] | None = None
) -> list[Path]:
    """(Re)generate the FLEURS clips; returns the written WAV paths.

    Only the ``fleurs_fi_*`` stems are touched — the Piper stems
    (``fi_*``, ``en_*``, ``meeting_sample``) are left alone, so this
    script is additive to the existing corpus. With *ids* the clip set
    is exactly those ids (the committed corpus); otherwise the seeded
    window selection draws *n* clips.
    """
    if parquet_path is None:
        raise RuntimeError(
            "no parquet path — pass --parquet or set FLEURS_PARQUET; the "
            "download is a one-time dev-time step, never a runtime fetch"
        )
    rows = load_fleurs_rows(parquet_path)
    clips = select_clips(rows, seed=SEED, n=n, ids=ids)
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
        help="local path to the FLEURS fi_fi train parquet (downloaded once; "
        "re-runs reuse it). The URL is in the module docstring.",
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
        help="comma-separated FLEURS clip ids to regenerate exactly "
        "(the committed corpus's ids; see CORPUS_ATTRIBUTION.md). "
        "Defaults to the committed 28 ids; pass 'seeded' to use the "
        "seeded window selection instead.",
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

    ids = _parse_ids(args.ids)
    corpus_dir = resolve_corpus_dir(args.out)
    corpus_dir.mkdir(parents=True, exist_ok=True)
    written = generate_real_speech(corpus_dir, args.parquet, args.n, ids=ids)
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
