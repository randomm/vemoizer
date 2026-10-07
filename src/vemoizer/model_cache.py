"""Local-cache probe for revision-pinned HF snapshots (issue #147).

The single place that decides whether a pinned snapshot is already
resolvable from the local cache, so that :func:`vemoizer.models.resolve_model_path`
can silence the warm-cache download (no HF progress bars) while leaving
the cold-cache download with its progress bar.

The probe is a **conservative, filesystem-only completeness check** (no
``snapshot_download``, no network): it reports a snapshot as complete only
when the pinned revision's snapshot directory exists and holds at least one
real weights file for that model type (per
:data:`_EXPECTED_WEIGHT_FILES`), every entry in the snapshot directory
resolves to a real file, and the repo's blob directory has no
``*.incomplete`` file. Anything missing, partial, or unreadable counts as
*not complete* — the caller then runs the real download with progress bars
enabled (the safe direction: a silent multi-GB fetch would look like a
hang).

The result is memoized per ``(repo_id, revision)``; only ``False`` results
are cached (a failed probe is re-checked on the next call, and a failed
real download never poisons the memo with ``True``).
"""

from __future__ import annotations

import os
from collections.abc import Generator
from pathlib import Path

__all__ = ["snapshot_locally_complete", "clear_memo"]

#: Per-process memo of "pinned snapshot resolvable from local cache" probes,
#: keyed by ``(repo_id, revision)``. Only negative results are cached: a
#: failed probe is re-checked on the next call, and a ``True`` is never
#: cached across a failed real download.
_COMPLETE_SNAPSHOTS: dict[tuple[str, str], bool] = {}

#: Minimal weight-file expectations per registry model, derived from the
#: actual pinned snapshots in the local HF cache (the single source of
#: truth for the registry: ``vemoizer.models.MODELS``):
#:
#: - parakeet: ``model.safetensors`` (MLX conversion of the Parakeet TDT)
#: - canary: ``model.safetensors`` (mlx-q8 MLX conversion)
#: - whisper-finnish / whisper-turbo: ``weights.safetensors`` (MLX
#:   whisper conversions)
#: - pyannote: subfolder-local PyTorch/NumPy blobs (``embedding/``,
#:   ``plda/``, ``segmentation/``)
#:
#: ``*`` in a path is a one-level wildcard (``Path.glob``-style, single
#: segment) so a partial snapshot carrying only ``embedding/`` does not
#: count as complete for the pyannote model.
_EXPECTED_WEIGHT_FILES: dict[str, tuple[str, ...]] = {
    "mlx-community/parakeet-tdt-0.6b-v3": ("model.safetensors",),
    "Mediform/canary-1b-v2-mlx-q8": ("model.safetensors",),
    "FredrikKarlssonSpeech/whisper-large-finnish-v3-mlx": ("weights.safetensors",),
    "mlx-community/whisper-large-v3-turbo": ("weights.safetensors",),
    "pyannote/speaker-diarization-community-1": (
        "embedding/*.bin",
        "plda/*.npz",
        "segmentation/*.bin",
    ),
}


def clear_memo() -> None:
    """Clear the per-process snapshot-completeness memo (for tests)."""
    _COMPLETE_SNAPSHOTS.clear()


def snapshot_locally_complete(
    repo_id: str, revision: str, cache_dir: str | Path | None = None
) -> bool:
    """True when the pinned snapshot is provably complete in the local cache.

    Conservative by design: any missing directory, missing weights file,
    dangling symlink, or ``*.incomplete`` blob file means *not complete*
    (the real download then runs with progress bars — the safe direction).
    The probe touches only the local filesystem; it never talks to the hub.

    Only ``False`` results are memoized, per ``(repo_id, revision)``; a
    ``True`` is re-verified on every call (cheap filesystem reads) so a
    failed real download can never leave a poisoned ``True`` behind.
    """
    key = (repo_id, revision)
    if _COMPLETE_SNAPSHOTS.get(key) is False:
        return False
    try:
        complete = _snapshot_is_complete(repo_id, revision, cache_dir)
    except Exception:  # noqa: BLE001 - any probe failure keeps the bars on
        complete = False
    if not complete:
        _COMPLETE_SNAPSHOTS[key] = False
    return complete


def _storage_root(cache_dir: str | Path | None) -> Path:
    """The cache directory under which ``models--<org>--<repo>`` lives."""
    if cache_dir is not None:
        return Path(cache_dir)
    from huggingface_hub import constants

    return Path(constants.HF_HUB_CACHE)


def _storage_folder(repo_id: str, cache_dir: str | Path | None) -> Path:
    try:
        from huggingface_hub.file_download import repo_folder_name
    except ImportError:
        # huggingface_hub is a hard dependency; this branch is only reached
        # in tests where a fake module object is installed in sys.modules
        # without a real file_download submodule.
        parts = ["models", *repo_id.split("/")]
        return _storage_root(cache_dir) / ("--".join(parts))

    return _storage_root(cache_dir) / repo_folder_name(
        repo_id=repo_id, repo_type="model"
    )


def _snapshot_is_complete(
    repo_id: str, revision: str, cache_dir: str | Path | None
) -> bool:
    """Conservative filesystem completeness check for a pinned snapshot.

    Complete only when ALL of the following hold:

    1. the snapshot directory for the pinned revision SHA exists;
    2. it contains at least one real weights file for the model type
       (per :data:`_EXPECTED_WEIGHT_FILES`);
    3. every entry in the snapshot directory (recursively) resolves —
       no dangling symlink into ``blobs/``;
    4. the repo's ``blobs`` directory has no ``*.incomplete`` file.

    All four checks are required: a missing snapshot dir (cold cache), a
    partial snapshot (refs + config.json only), a dangling symlink, or an
    interrupted download (``*.incomplete`` blob) all report *not complete*.
    """
    folder = _storage_folder(repo_id, cache_dir)
    snapshot_dir = folder / "snapshots" / revision
    # Check 1: pinned snapshot dir must exist.
    if not snapshot_dir.is_dir():
        return False
    # Check 4: no interrupted-download marker blobs.
    blobs = folder / "blobs"
    if blobs.is_dir() and any(blobs.glob("*.incomplete")):
        return False
    # Check 3: every entry in the snapshot tree must resolve.
    if not _all_entries_resolve(snapshot_dir):
        return False
    # Check 2: at least one real weights file for this model type.
    return _has_expected_weight_file(repo_id, snapshot_dir)


def _all_entries_resolve(snapshot_dir: Path) -> bool:
    """True when every entry under *snapshot_dir* resolves to a real file.

    A symlink whose target does not exist (e.g. a removed or still-being-
    written blob) fails the check; the walk never follows symlinks (so a
    symlink cycle cannot hang the probe).
    """
    for root, dirs, files in os.walk(snapshot_dir):
        for name in (*dirs, *files):
            entry = Path(root) / name
            try:
                if not entry.exists():
                    return False
            except OSError:  # pragma: no cover - race with a concurrent cleanup
                return False
    return True


def _expected_weights_for(repo_id: str) -> tuple[str, ...]:
    """Expected weight-file patterns for *repo_id*.

    For the five registry models the patterns come from
    :data:`_EXPECTED_WEIGHT_FILES`; for any other repo a generic set of
    weight-file suffixes is used (so the probe works for future registry
    additions without a per-model entry, while still requiring at least
    one real weight file).
    """
    patterns = _EXPECTED_WEIGHT_FILES.get(repo_id)
    if patterns is not None:
        return patterns
    return ("*.safetensors", "*.bin", "*.npz", "*.pt")


def _has_expected_weight_file(repo_id: str, snapshot_dir: Path) -> bool:
    """True when at least one expected weights file resolves under the snapshot."""
    for pattern in _expected_weights_for(repo_id):
        if _any_match(snapshot_dir, pattern):
            return True
    return False


def _any_match(snapshot_dir: Path, pattern: str) -> bool:
    """``Path.glob``-style match for a one-level wildcard pattern.

    ``embedding/*.bin`` matches any single segment under ``embedding/``;
    a plain name matches that file only. Each matched path must resolve
    (a dangling symlink to a weight file does not count).
    """
    parts = pattern.split("/")
    if len(parts) == 1:
        matches: Generator[Path, None, None] = snapshot_dir.glob(parts[0])
        return any(p.is_file() for p in matches)
    head, wildcard = "/".join(parts[:-1]), parts[-1]
    parent = snapshot_dir / head
    if not parent.is_dir():
        return False
    matches = parent.glob(wildcard)
    return any(p.is_file() for p in matches)
