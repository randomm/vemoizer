"""Local-cache probe for revision-pinned HF snapshots (issue #147).

The single place that decides whether a pinned snapshot is already
resolvable from the local cache, so that :func:`vemoizer.models.resolve_model_path`
can silence the warm-cache download (no HF progress bars) while leaving
the cold-cache download with its progress bar.

The probe is a local-only ``snapshot_download`` call; the result is
memoized per (repo, revision, cache_dir) so it runs once per model per
process. A probe failure means "unknown" — the caller then downloads
with progress bars enabled (the safe direction).
"""

from __future__ import annotations

__all__ = ["snapshot_locally_complete"]

#: Per-process memo of "pinned snapshot resolvable from local cache" probes,
#: keyed by ``(repo_id, revision, cache_dir)``.
_COMPLETE_SNAPSHOTS: dict[tuple[str, str, str | None], bool] = {}


def clear_memo() -> None:
    """Clear the per-process snapshot-completeness memo (for tests)."""
    _COMPLETE_SNAPSHOTS.clear()


def snapshot_locally_complete(
    repo_id: str, revision: str, cache_dir: str | None
) -> bool:
    """True when the pinned snapshot can be served from the local cache alone.

    Probed by a local-only ``snapshot_download`` call; the result is memoized
    per (repo, revision, cache_dir) so the probe runs once per model per
    process. A probe failure means "unknown" — the caller then downloads
    with progress bars enabled (the safe direction).
    """
    key = (repo_id, revision, cache_dir)
    if key in _COMPLETE_SNAPSHOTS:
        return _COMPLETE_SNAPSHOTS[key]
    try:
        _snapshot_download_probe(repo_id, revision, cache_dir)
        complete = True
    except Exception:  # noqa: BLE001 - probe failure means "not provable"
        complete = False
    _COMPLETE_SNAPSHOTS[key] = complete
    return complete


def _snapshot_download_probe(
    repo_id: str, revision: str, cache_dir: str | None
) -> None:
    """Probe whether the pinned snapshot is resolvable from the local cache.

    Uses a local-only ``snapshot_download`` call; the revision-pinned SHA's
    file list is cached on the first download, so a complete snapshot
    resolves without a network call. An incomplete cache raises.
    """
    from huggingface_hub import snapshot_download

    snapshot_download(
        repo_id,
        revision=revision,
        cache_dir=cache_dir,
        local_files_only=True,
    )
