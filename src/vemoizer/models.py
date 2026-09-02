"""Revision-pinned model registry + ``models pull`` support (issue #3).

The consensus pipeline downloads its weights through
``huggingface_hub.snapshot_download`` pinned to a full-SHA revision
(project invariant #4): loading from a bare repo ID would cache a moving
ref, so every entry in :data:`MODELS` carries a 40-char SHA, and
:func:`assert_all_revisions_pinned` is a guard test that fails if an
entry ever regresses to a branch name or short SHA.

This module is the single source of truth for which repo each transcriber
loads from. The transcriber modules keep their own ``MODEL_ID`` /
``MODEL_REVISION`` constants (they were written first); a drift test pins the
two against each other so they cannot drift. The re-decode model
(whisper-large-finnish-v3, issue #8) lives only here until its
transcriber lands.

``MODELS`` is the canonical tuple of :class:`ModelSpec` (friendly name,
HF repo, pinned SHA) in pipeline order. ``MODEL_REGISTRY`` and
:func:`get_model` expose the same models as :class:`ModelEntry` for
name-keyed lookup. ``pull_model`` / ``pull_all`` download a single model or
all three and return local snapshot paths; ``pull_models`` is the
CLI-facing variant that captures per-model failures instead of aborting.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from pathlib import Path

__all__ = [
    "ModelSpec",
    "ModelEntry",
    "PulledModel",
    "MODELS",
    "MODEL_REGISTRY",
    "get_model",
    "pull_models",
    "pull_model",
    "pull_all",
    "assert_all_revisions_pinned",
    "cache_dir",
    "cache_size",
    "cache_size_bytes",
    "format_size",
    "format_pull_error",
    "render_pull_report",
]

#: Full-SHA commit, 40 hex chars — a branch name or short SHA is a violation
#: of invariant #4.
_FULL_SHA_RE = re.compile(r"^[0-9a-f]{40}$")


@dataclass(frozen=True)
class ModelSpec:
    """One revision-pinned model in the consensus set."""

    name: str  # friendly name: "parakeet", "canary", "whisper-finnish"
    repo_id: str
    revision: str  # full-SHA commit, never a branch name


@dataclass(frozen=True)
class PulledModel:
    """Outcome of pulling one model."""

    spec: ModelSpec
    local_path: str | None
    error: str | None
    seconds: float


MODELS: tuple[ModelSpec, ...] = (
    ModelSpec(
        name="parakeet",
        repo_id="mlx-community/parakeet-tdt-0.6b-v3",
        revision="ed2b7e8c15f9aaa0b5772e2efb986255eaef7e15",
    ),
    ModelSpec(
        name="canary",
        repo_id="Mediform/canary-1b-v2-mlx-q8",
        revision="0b6b32ee10f30c89e3ead7249bb636445e3019ee",
    ),
    ModelSpec(
        name="whisper-finnish",
        repo_id="Finnish-NLP/whisper-large-finnish-v3",
        revision="b23deb0b3855c829ffe04cb1c6709757ff16d49c",
    ),
)


@dataclass(frozen=True)
class ModelEntry:
    """One consensus-pipeline model: registry name, HF repo, pinned SHA."""

    name: str
    repo_id: str
    revision: str

    @property
    def display(self) -> str:
        return f"{self.name} ({self.repo_id}@{self.revision[:8]}…)"


_MODELS_ENTRIES: tuple[ModelEntry, ...] = tuple(
    ModelEntry(name=s.name, repo_id=s.repo_id, revision=s.revision) for s in MODELS
)

#: Lookup by name: ``{name: entry}``, stable for dict consumers.
MODEL_REGISTRY: dict[str, ModelEntry] = {e.name: e for e in _MODELS_ENTRIES}


def get_model(name: str) -> ModelEntry:
    """Look up a registry entry by name.

    Raises ``KeyError`` with the set of known names when *name* is unknown.
    """
    if name in MODEL_REGISTRY:
        return MODEL_REGISTRY[name]
    raise KeyError(f"unknown model {name!r}; known: {sorted(MODEL_REGISTRY)}")


def assert_all_revisions_pinned(models: tuple[ModelSpec, ...] = MODELS) -> None:
    """Raise ``ValueError`` if any entry is not pinned to a full-SHA commit.

    Regression guard for invariant #4: ``snapshot_download`` without a
    revision (or with a branch name / short SHA) caches a moving ref.
    """
    for spec in models:
        if not _FULL_SHA_RE.match(spec.revision):
            raise ValueError(
                f"model {spec.name!r} ({spec.repo_id}) is not pinned to a "
                f"full-SHA commit: {spec.revision!r}"
            )


def cache_dir() -> Path:
    """The HuggingFace hub cache directory (honours ``HF_HOME``).

    ``HF_HOME`` is read at call time (not cached at import) so tests can
    monkeypatch the environment variable.
    """
    import os

    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        return Path(hf_home) / "hub"
    try:
        from huggingface_hub.constants import HF_HUB_CACHE
    except ImportError:  # pragma: no cover - huggingface_hub is a hard dep
        return Path.home() / ".cache" / "huggingface" / "hub"
    return Path(HF_HUB_CACHE)


def _model_cache_name(repo_id: str) -> str:
    """Cache directory name for a repo (e.g. ``models--org--name``).

    Delegates to ``huggingface_hub``'s own ``repo_folder_name`` so the
    naming can never drift from what ``snapshot_download`` writes on disk.
    """
    from huggingface_hub.file_download import repo_folder_name

    return repo_folder_name(repo_id=repo_id, repo_type="model")


def cache_size(models: tuple[ModelSpec, ...] = MODELS) -> dict[str, int]:
    """On-disk size (bytes) of each model's cache dir in the HF cache.

    Missing cache dirs report ``0``. Reads only the local cache; no network.
    """
    root = cache_dir()
    sizes: dict[str, int] = {}
    for spec in models:
        model_dir = root / _model_cache_name(spec.repo_id)
        sizes[spec.name] = _dir_size(model_dir) if model_dir.is_dir() else 0
    return sizes


def cache_size_bytes(name: str, cache_dir: str | None = None) -> int | None:
    """Byte size of the pinned snapshot for *name*, or ``None`` if absent.

    Uses ``huggingface_hub.scan_cache_dir`` so the walk respects the HF
    cache layout (``models--<org>--<name>``); no raw filesystem globbing.
    """
    from huggingface_hub import scan_cache_dir

    entry = get_model(name)
    info = scan_cache_dir(cache_dir)
    for repo in info.repos:
        if repo.repo_id == entry.repo_id:
            return repo.size_on_disk
    return None


def _dir_size(path: Path) -> int:
    total = 0
    for file in path.rglob("*"):
        if file.is_file():
            total += file.stat().st_size
    return total


def format_size(nbytes: int) -> str:
    """Human-readable byte count with one decimal (e.g. ``1.2 GiB``)."""
    value = float(nbytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024.0 or unit == "TiB":
            if unit == "B":
                return f"{int(value)} B"
            return f"{value:.1f} {unit}"
        value /= 1024.0
    raise AssertionError("unreachable")  # pragma: no cover


def pull_model(name: str, cache_dir: str | None = None) -> str:
    """Download (revision-pinned) the *name* model and return the local path.

    ``snapshot_download`` is idempotent: a warm cache makes no network call
    and returns the cached snapshot. If ``HF_HUB_OFFLINE=1`` is set and the
    snapshot is not cached, ``huggingface_hub.errors.OfflineModeIsEnabled``
    is raised; use :func:`format_pull_error` to turn it into a user-facing
    message.
    """
    from huggingface_hub import snapshot_download

    entry = get_model(name)
    if cache_dir is None:
        return str(snapshot_download(entry.repo_id, revision=entry.revision))
    return str(
        snapshot_download(entry.repo_id, revision=entry.revision, cache_dir=cache_dir)
    )


def pull_all(cache_dir: str | None = None) -> dict[str, str]:
    """Pre-warm every registry model; returns ``{name: local_path}``.

    Iterates in pipeline order (parakeet, canary, whisper-finnish).
    """
    return {
        entry.name: pull_model(entry.name, cache_dir=cache_dir)
        for entry in _MODELS_ENTRIES
    }


def pull_models(
    models: tuple[ModelSpec, ...] = MODELS,
) -> list[PulledModel]:
    """Pre-download every model, revision-pinned, idempotent.

    Returns one :class:`PulledModel` per model, in order. A per-model
    failure (404, gated, offline, …) is captured on that entry rather than
    aborting the run: the remaining models still get pulled, and the caller
    reports the failure. ``snapshot_download`` is a no-op when the pinned
    snapshot is already cached, so a warm cache completes quickly with no
    network traffic.
    """
    assert_all_revisions_pinned(models)
    from huggingface_hub import snapshot_download

    results: list[PulledModel] = []
    for spec in models:
        start = time.monotonic()
        try:
            local_path = snapshot_download(spec.repo_id, revision=spec.revision)
            results.append(
                PulledModel(spec, str(local_path), None, time.monotonic() - start)
            )
        except Exception as exc:  # noqa: BLE001 - per-model failure captured
            results.append(
                PulledModel(
                    spec, None, _describe_hf_error(exc), time.monotonic() - start
                )
            )
    return results


def _describe_hf_error(exc: Exception) -> str:
    """Turn a ``huggingface_hub`` failure into an actionable one-liner.

    ``OfflineModeIsEnabled`` and ``RepositoryNotFoundError`` are named
    explicitly (ported from kuiskaus's error-formatting tests); anything
    else gets a generic message that includes the type name but never the
    raw exception text.
    """
    try:
        from huggingface_hub.errors import (
            GatedRepoError,
            HfHubHTTPError,
            IncompleteSnapshotError,
            LocalEntryNotFoundError,
            OfflineModeIsEnabled,
            RepositoryNotFoundError,
            RevisionNotFoundError,
        )
    except ImportError:  # pragma: no cover - huggingface_hub is a hard dep
        return f"failed to download model: {type(exc).__name__}"

    if isinstance(exc, GatedRepoError):
        return (
            "model repository is gated; set HF_TOKEN and accept the access "
            "terms, then re-run 'vemoizer models pull'"
        )
    if isinstance(exc, OfflineModeIsEnabled):
        return (
            "HF_HUB_OFFLINE=1 is set and the model is not cached; unset "
            "HF_HUB_OFFLINE and re-run 'vemoizer models pull'"
        )
    if isinstance(exc, RepositoryNotFoundError):
        return (
            f"model repository not found on the hub: "
            f"{getattr(exc, 'message', str(exc))}"
        )
    if isinstance(exc, RevisionNotFoundError):
        return f"pinned revision not found: {getattr(exc, 'message', str(exc))}"
    if isinstance(exc, (IncompleteSnapshotError, LocalEntryNotFoundError)):
        return (
            "cached snapshot is incomplete; delete the model's cache "
            "directory and re-run 'vemoizer models pull'"
        )
    if isinstance(exc, HfHubHTTPError):
        status = getattr(exc, "status_code", None)
        detail = getattr(exc, "message", str(exc))
        return f"model download failed (HTTP {status}): {detail}".strip()
    return f"failed to download model: {type(exc).__name__}"


def format_pull_error(exc: Exception) -> str:
    """Format a :func:`pull_model` failure for user-facing display.

    ``OfflineModeIsEnabled`` gets a distinct message that names
    ``HF_HUB_OFFLINE`` — the generic bucket would not explain that the fix
    is either to clear the flag or to run ``models pull`` online first.
    """
    from huggingface_hub.errors import (
        GatedRepoError,
        HfHubHTTPError,
        OfflineModeIsEnabled,
    )

    if isinstance(exc, OfflineModeIsEnabled):
        return (
            "HuggingFace is in offline mode (HF_HUB_OFFLINE=1) and the "
            "pinned snapshot is not in the local cache. Run 'vemoizer "
            "models pull' once with network access, or unset "
            "HF_HUB_OFFLINE to let the download proceed."
        )
    if isinstance(exc, GatedRepoError):
        return (
            f"{exc} — accept the license at "
            f"https://huggingface.co and provide an access token."
        )
    response = getattr(exc, "response", None)
    status = getattr(response, "status_code", None) if response is not None else None
    if isinstance(exc, HfHubHTTPError) and status in {401, 403}:
        return (
            f"Authentication failed for this model ({status}). "
            "Check your HF access token."
        )
    if isinstance(exc, HfHubHTTPError) and status == 404:
        return (
            "Repository not found (404). The pinned revision may have been "
            "deleted upstream."
        )
    return (
        f"Model download failed ({type(exc).__name__}). "
        "Check network access and the HuggingFace repository."
    )


def render_pull_report(results: list[PulledModel], sizes: dict[str, int] | None) -> str:
    """Render the ``models pull`` summary (stdout-safe, no color)."""
    lines: list[str] = []
    for result in results:
        spec = result.spec
        if result.error is None:
            lines.append(
                f"{spec.name}: pulled {spec.repo_id}@{spec.revision[:12]} "
                f"({spec.revision} pinned, {result.seconds:.1f}s)"
            )
        else:
            lines.append(f"{spec.name}: FAILED {spec.repo_id} — {result.error}")
    if sizes is not None:
        total = sum(sizes.values())
        lines.append(
            "cache: "
            + "  ".join(f"{name} {format_size(n)}" for name, n in sizes.items())
            + f"  (total {format_size(total)})"
        )
    return "\n".join(lines)
