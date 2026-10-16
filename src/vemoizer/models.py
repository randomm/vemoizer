"""Revision-pinned model registry + ``models pull`` support (issue #3).

The consensus pipeline downloads its weights through
``huggingface_hub.snapshot_download`` pinned to a full-SHA revision
(project invariant #4): loading from a bare repo ID would cache a moving
ref, so every entry in :data:`MODELS` carries a 40-char SHA, and
:func:`assert_all_revisions_pinned` is a guard test that fails if an
entry ever regresses to a branch name or short SHA.

This module is the single source of truth for which repo each transcriber
loads from (issue #79): the whisper-turbo, whisper-finnish (re-decode) and
pyannote (diarization) modules read ``repo_id`` + ``revision`` from the
registry via :func:`get_model`, so no repo/SHA pair lives in two places.

``MODELS`` is the canonical tuple of :class:`ModelSpec` (friendly name,
HF repo, pinned SHA) in pipeline order. ``pull_model`` / ``pull_all``
download a single model or all five and return local snapshot paths;
``pull_models`` is the CLI-facing variant that captures per-model failures
instead of aborting. Warm-cache downloads are silenced (no HF progress
bars, issue #147); a real download keeps its progress bar.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# The ``models pull`` report helpers (issue #67) and the HuggingFace error copy
# live in :mod:`vemoizer.model_pull_report` (issue #151): one coherent
# responsibility — pure presentation, no registry or download state. Imported
# here (and listed in ``__all__``) so ``from vemoizer.models import ...`` keeps
# working for the CLI and the test suite; ``_describe_hf_error`` is also called
# directly by ``pull_models``. The report module imports nothing from here, so
# this direction introduces no import cycle. One statement per name: ruff's
# isort splits a parenthesised ``from`` import whose members are all ``X as X``
# re-export bindings, so the multi-name block is not a lint-stable form.
from .model_pull_report import _EXPECTED_SIZE_GIB as _EXPECTED_SIZE_GIB
from .model_pull_report import STAGES as STAGES
from .model_pull_report import _describe_hf_error as _describe_hf_error
from .model_pull_report import expected_size_str as expected_size_str
from .model_pull_report import format_pull_error as format_pull_error
from .model_pull_report import format_size as format_size
from .model_pull_report import render_pull_report as render_pull_report

__all__ = [
    "ModelSpec",
    "ModelEntry",
    "PulledModel",
    "MODELS",
    "MODEL_REGISTRY",
    "STAGES",
    "get_model",
    "resolve_model_path",
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

    name: str  # friendly name: "parakeet", "canary", "whisper-finnish", ...
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
        repo_id="FredrikKarlssonSpeech/whisper-large-finnish-v3-mlx",
        revision="f51f0310c1b2a3e5acb16905c1a7245bb9476846",
    ),
    ModelSpec(
        name="whisper-turbo",
        repo_id="mlx-community/whisper-large-v3-turbo",
        revision="a4aaeec0636e6fef84abdcbe3544cb2bf7e9f6fb",
    ),
    ModelSpec(
        name="pyannote",
        repo_id="pyannote/speaker-diarization-community-1",
        revision="3533c8cf8e369892e6b79ff1bf80f7b0286a54ee",
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

    Regression guard for invariant #4: a moving ref is a bug.
    """
    for spec in models:
        if not _FULL_SHA_RE.match(spec.revision):
            raise ValueError(
                f"model {spec.name!r} ({spec.repo_id}) is not pinned to a "
                f"full-SHA commit: {spec.revision!r}"
            )


def cache_dir() -> Path:
    """The HuggingFace hub cache directory (honours ``HF_HOME``).

    ``HF_HOME`` is read at call time so tests can monkeypatch it.
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

    Delegates to ``huggingface_hub``'s own ``repo_folder_name``.
    """
    from huggingface_hub.file_download import repo_folder_name

    return repo_folder_name(repo_id=repo_id, repo_type="model")


def cache_size(models: tuple[ModelSpec, ...] = MODELS) -> dict[str, int]:
    """On-disk size (bytes) of each model's cache dir in the HF cache.

    Missing cache dirs report ``0``. No network.
    """
    root = cache_dir()
    sizes: dict[str, int] = {}
    for spec in models:
        model_dir = root / _model_cache_name(spec.repo_id)
        sizes[spec.name] = _dir_size(model_dir) if model_dir.is_dir() else 0
    return sizes


def cache_size_bytes(name: str, cache_dir: str | None = None) -> int | None:
    """Byte size of the pinned snapshot for *name*, or ``None`` if absent.

    Uses ``huggingface_hub.scan_cache_dir`` to respect the HF cache layout.
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


def resolve_model_path(
    repo_id: str, revision: str, cache_dir: str | None = None, **extra: Any
) -> str:
    """Resolve a revision-pinned snapshot to its local path (issue #147).

    The single seam through which every model in this project reaches
    ``huggingface_hub.snapshot_download``: the ``models pull`` command
    (via :func:`pull_model` / :func:`pull_models`) and the pipeline loaders
    (decode A/B, re-decode, diarization) all call this, so the warm-cache
    silencing lives in exactly one place.

    When the pinned snapshot is already fully cached the download is
    silenced through the library's own ``disable_progress_bars`` switch
    (no "Fetching" / "Download complete" / "Reconstruction" bars); a real
    download keeps its progress bar so a multi-GB silent fetch does not
    look like a hang. ``*extra`` carries loader-specific kwargs (e.g.
    ``token`` for the gated diarization repo) and is forwarded to both the
    probe and the real call (``cache_dir`` is a named param and is also
    forwarded to both).

    Returns the local snapshot path as a ``str``. Errors propagate
    unchanged (offline mode, gated repos, …).
    """
    from huggingface_hub import snapshot_download

    from .model_cache import snapshot_locally_complete

    all_kwargs: dict[str, Any] = {"revision": revision, **extra}
    if cache_dir is not None:
        all_kwargs["cache_dir"] = cache_dir

    if snapshot_locally_complete(repo_id, revision, cache_dir):
        # Warm cache: no network call, and no "Fetching" / "Download
        # complete" / "Reconstruction" progress bars (issue #147).
        try:
            from huggingface_hub.utils import disable_progress_bars
        except ImportError:
            # In tests that install a fake ``huggingface_hub`` module object
            # in ``sys.modules`` without a real ``utils`` submodule, fall
            # back to a no-op context manager so the warm-cache path still
            # works.
            import contextlib

            def disable_progress_bars() -> Any:
                return contextlib.nullcontext()

        with disable_progress_bars():
            return str(snapshot_download(repo_id, **all_kwargs))
    return str(snapshot_download(repo_id, **all_kwargs))


def pull_model(name: str, cache_dir: str | None = None) -> str:
    """Download (revision-pinned) the *name* model and return the local path.

    Warm-cache downloads are silenced (no HF progress bars, issue #147);
    a real download keeps its progress bar.
    """
    entry = get_model(name)
    return resolve_model_path(entry.repo_id, entry.revision, cache_dir=cache_dir)


def pull_all(cache_dir: str | None = None) -> dict[str, str]:
    """Pre-warm every registry model; returns ``{name: local_path}``.

    Iterates in pipeline order (parakeet, canary, whisper-finnish,
    whisper-turbo, pyannote).
    """
    return {
        entry.name: pull_model(entry.name, cache_dir=cache_dir)
        for entry in _MODELS_ENTRIES
    }


def pull_models(
    models: tuple[ModelSpec, ...] = MODELS,
    cache_dir: str | None = None,
) -> list[PulledModel]:
    """Pre-download every model, revision-pinned, idempotent.

    Returns one :class:`PulledModel` per model, in order. A per-model
    failure (404, gated, offline, …) is captured on that entry rather than
    aborting the run: the remaining models still get pulled, and the caller
    reports the failure. ``snapshot_download`` is a no-op when the pinned
    snapshot is already cached, so a warm cache completes quickly with no
    network traffic. Warm-cache downloads are quiet (no "Fetching" /
    "Download" / "Reconstruction" progress bars, issue #147); a real
    download keeps its progress bar.
    """
    assert_all_revisions_pinned(models)

    results: list[PulledModel] = []
    for spec in models:
        start = time.monotonic()
        try:
            local_path = resolve_model_path(
                spec.repo_id, spec.revision, cache_dir=cache_dir
            )
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
