"""FLEURS source handling for the real-speech corpus generator.

The dataset-source side of ``scripts/gen_real_speech_corpus.py``: the
pinned source constants (URL, revision, the recorded parquet digest, the
committed clip ids, the selection window), the parquet SHA-256 guard, the
pyarrow row loader, the row/field coercion helpers, and the deterministic
clip selection — extracted into its own single-responsibility module so the
main script stays under the source-line cap.

Importable both as a ``scripts`` module (``import fleurs_source``) and via
``importlib.util.spec_from_file_location`` (the tests), mirroring
``scripts/wav_header.py``. Sibling imports go through the shared
:func:`scripts._sibling_loader.load_sibling` helper (issue #138).
"""

from __future__ import annotations

import hashlib
import random
import sys
from dataclasses import dataclass
from pathlib import Path


def _load_helper():
    """Load the sibling loader helper, handling all three import paths."""
    try:
        from scripts._sibling_loader import load_sibling

        return load_sibling
    except ImportError:
        pass
    try:
        from _sibling_loader import load_sibling  # ty: ignore[unresolved-import]

        return load_sibling
    except ImportError:
        pass
    # Last resort: the helper itself is not importable (the tests' importlib
    # path, where no sibling is on sys.path at all). Load it from disk.
    import importlib.util as _ilu

    _helper_path = Path(__file__).resolve().parent / "_sibling_loader.py"
    _spec = _ilu.spec_from_file_location("scripts._sibling_loader", _helper_path)
    if _spec is None or _spec.loader is None:
        raise ImportError(f"cannot build an import spec for {_helper_path}")
    _mod = _ilu.module_from_spec(_spec)
    sys.modules["scripts._sibling_loader"] = _mod
    _spec.loader.exec_module(_mod)
    return _mod.load_sibling


_load_sibling = _load_helper()
wav_duration_seconds = _load_sibling("wav_header").wav_duration_seconds

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
