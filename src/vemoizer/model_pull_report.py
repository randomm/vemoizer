"""Rendering and error copy for the ``models pull`` command (issue #151).

One coherent responsibility: everything the ``vemoizer models pull`` report
says to the operator — the per-model line (pipeline stage + expected
on-disk size, issue #67), the cache-total line, and the two error
formatters that turn a ``huggingface_hub`` failure into an actionable
one-liner. Pure presentation: no downloads, no filesystem walks, no
registry lookups; :func:`~vemoizer.models.pull_models` feeds it
:class:`~vemoizer.models.PulledModel` results and cache sizes.

Extracted verbatim from :mod:`vemoizer.models` to restore headroom under
the 500-line source cap (AGENTS.md); behaviour is unchanged.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

# Imported, never restated: the error formatters below echo
# ``huggingface_hub`` exception text into the operator's ``models pull``
# report, a trust boundary the run log does not cover — ``RedactingFormatter``
# scrubs the *log file* only, never stdout. Sharing one implementation means
# one shape cannot be redacted in one path and leak in the other (issue #151
# review, SECURITY).
from vemoizer.run_log_io import scrub_credentials

if TYPE_CHECKING:
    from vemoizer.models import PulledModel

__all__ = [
    "STAGES",
    "_EXPECTED_SIZE_GIB",
    "_describe_hf_error",
    "expected_size_str",
    "format_pull_error",
    "format_size",
    "render_pull_report",
]

#: Pipeline stage each model serves (issue #67): shown in the
#: ``models pull`` report so users know what each download is for.
STAGES: dict[str, str] = {
    "parakeet": "decode A",
    "canary": "decode B",
    "whisper-finnish": "re-decode",
    "whisper-turbo": "meeting decode",
    "pyannote": "diarization",
}

#: Expected on-disk size of each model's pinned snapshot (issue #67):
#: measured against the local HF cache, rounded. Shown in the
#: ``models pull`` report so users on 16 GB machines know what to
#: expect before downloading.
_EXPECTED_SIZE_GIB: dict[str, float] = {
    "parakeet": 2.3,
    "canary": 1.1,
    "whisper-finnish": 2.9,
    "whisper-turbo": 1.5,
    "pyannote": 0.032,
}


def expected_size_str(name: str) -> str:
    """Human-readable expected size for a model (e.g. ``~2.3 GiB``)."""
    gib = _EXPECTED_SIZE_GIB.get(name)
    if gib is None:
        return "?"
    if gib < 0.1:  # the GiB form would show 0.1; MiB is more readable below this
        return f"~{int(gib * 1024)} MiB"
    return f"~{gib:.1f} GiB"


def format_size(nbytes: int) -> str:
    """Human-readable byte count with one decimal (e.g. ``1.2 GiB``).

    Moved here from :mod:`vemoizer.models` with its only caller, the pull
    report (issue #151): ``models.py`` imports this module, so keeping the
    helper on the registry side would force a deferred import to avoid a
    cycle. Re-exported from ``vemoizer.models`` for the CLI surface.
    """
    value = float(nbytes)
    for unit in ("B", "KiB", "MiB", "GiB", "TiB"):
        if value < 1024.0 or unit == "TiB":
            if unit == "B":
                return f"{int(value)} B"
            return f"{value:.1f} {unit}"
        value /= 1024.0
    raise AssertionError("unreachable")  # pragma: no cover


def render_pull_report(results: list[PulledModel], sizes: dict[str, int] | None) -> str:
    """Render the ``models pull`` summary (stdout-safe, no color).

    Each model line carries its pipeline stage and expected on-disk size
    (issue #67) so users know what each download is for and what to expect
    before pulling the revision-pinned models.
    """
    lines: list[str] = []
    for result in results:
        spec = result.spec
        stage = STAGES.get(spec.name, "?")
        if result.error is None:
            expected = expected_size_str(spec.name)
            lines.append(
                f"{spec.name} [{stage}]: pulled {spec.repo_id}@{spec.revision[:12]} "
                f"({spec.revision} pinned, {result.seconds:.1f}s, {expected})"
            )
        else:
            lines.append(
                f"{spec.name} [{stage}]: FAILED {spec.repo_id} — {result.error}"
            )
    if sizes is not None:
        total = sum(sizes.values())
        lines.append(
            "cache: "
            + "  ".join(f"{name} {format_size(n)}" for name, n in sizes.items())
            + f"  (total {format_size(total)})"
        )
    return "\n".join(lines)


def _describe_hf_error(exc: Exception) -> str:
    """Turn a ``huggingface_hub`` failure into an actionable one-liner.

    ``OfflineModeIsEnabled`` and ``RepositoryNotFoundError`` are named
    explicitly (ported from kuiskaus's error-formatting tests); anything
    else gets a generic message that includes the type name but never the
    raw exception text.

    The two branches that do echo library text (``RepositoryNotFoundError``,
    ``RevisionNotFoundError``) and the ``GatedRepoError`` branch of
    :func:`format_pull_error` pass it through
    :func:`~vemoizer.run_log_io.scrub_credentials` first — this report goes to
    the operator's terminal, where the run-log ``RedactingFormatter`` does not
    apply.
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
            f"{scrub_credentials(getattr(exc, 'message', str(exc)))}"
        )
    if isinstance(exc, RevisionNotFoundError):
        return (
            f"pinned revision not found: "
            f"{scrub_credentials(getattr(exc, 'message', str(exc)))}"
        )
    if isinstance(exc, (IncompleteSnapshotError, LocalEntryNotFoundError)):
        return (
            "cached snapshot is incomplete; delete the model's cache "
            "directory and re-run 'vemoizer models pull'"
        )
    if isinstance(exc, HfHubHTTPError):
        status = getattr(exc, "status_code", None)
        detail = scrub_credentials(getattr(exc, "message", str(exc)))
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
            f"{scrub_credentials(f'{exc}')} — accept the license at "
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
