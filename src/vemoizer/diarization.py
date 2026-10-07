"""Optional speaker diarization (issue #13, opt-in ``--diarize``).

Uses ``pyannote.audio==4.0.7`` with the
``pyannote/speaker-diarization-community-1`` weights. The weights are
**CC-BY-4.0-licensed and gated on HuggingFace** — the user must accept the
license form and provide an access token before first use. Attribution is
mandatory under CC-BY-4.0 and is exposed via :data:`ATTRIBUTION`.

pyannote is lazily imported inside :func:`_load_pipeline` so the default
(no-``--diarize``) path never touches it and never needs it installed.
Device selection: MPS is attempted first (fixed in pyannote PR 1546); on
any load/inference exception we fall back to CPU.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from .models import get_model, resolve_model_path

#: HuggingFace repo for the diarization weights (CC-BY-4.0, gated), read
#: from the central registry so no repo/SHA pair lives in two places
#: (issue #79); the drift test in tests/test_cli_models.py pins it.
DIARIZATION_REPO_ID = get_model("pyannote").repo_id

#: Pinned full-SHA commit of the diarization weights (invariant #4): loading
#: from a bare repo ID would cache a moving ref.
DIARIZATION_REVISION = get_model("pyannote").revision

#: Mandatory CC-BY-4.0 attribution (weights license, not code license).
ATTRIBUTION = (
    "Speaker diarization: pyannote/speaker-diarization-community-1 "
    "(weights licensed under CC-BY-4.0, gated on HuggingFace; "
    "user accepted the license form and supplied an access token)."
)

#: Sample rate of the internal audio contract (AGENTS.md invariant #6).
_CONTRACT_SAMPLE_RATE = 16000

#: An exact speaker count, or ``(min, max)`` bounds for meetings where
#: people join and leave (no single count is right for the whole file).
SpeakerCount = int | tuple[int, int]

#: Environment variable holding the HuggingFace access token for the gated repo.
_HF_TOKEN_ENV = "HF_TOKEN"


@dataclass(frozen=True)
class DiarizationResult:
    """Speaker-labelled time segments covering the recording."""

    segments: list[tuple[float, float, str]]  # (start_s, end_s, speaker_label)


def _disable_pyannote_telemetry() -> None:
    """Disable pyannote's OpenTelemetry metrics (invariant #1, issue #103).

    pyannote 4.x ships usage metrics enabled by default and phones home to
    ``https://otel.pyannote.ai`` on every pipeline apply. The variable is
    read at import time, so this must run *before* ``import pyannote``.
    ``setdefault`` lets a user deliberately opt in by exporting ``true``;
    ``OTEL_SDK_DISABLED`` is set the same way, as belt-and-braces.
    """
    os.environ.setdefault("PYANNOTE_METRICS_ENABLED", "false")
    os.environ.setdefault("OTEL_SDK_DISABLED", "true")


class _DiarizePipeline(Protocol):
    """The callable seam the module relies on for pyannote pipelines (issue #141).

    pyannote's ``Pipeline`` is lazy-imported (issue #103: the import must
    stay inside :func:`_load_pipeline`, after
    :func:`_disable_pyannote_telemetry`) and is untyped, so this documents
    only the call boundary ``diarize`` actually uses — nothing more. The
    positional argument is pyannote 4.x's in-memory audio contract
    (``{"waveform": tensor, "sample_rate": int}``, i.e. an ``AudioFile``
    mapping), and the keyword set is the fixed speaker-count arguments
    pyannote's ``apply`` accepts: ``num_speakers`` (int) or the pair
    ``min_speakers``/``max_speakers``. The result stays ``Any`` because
    ``diarize`` only reads it via ``getattr`` fallbacks
    (``exclusive_speaker_diarization`` / ``speaker_diarization``) rather
    than a typed interface.
    """

    def __call__(
        self,
        waveforms: Mapping[str, Any],
        /,
        *,
        num_speakers: int | None = None,
        min_speakers: int | None = None,
        max_speakers: int | None = None,
    ) -> Any: ...


def _load_pipeline(device: str) -> _DiarizePipeline:
    """Lazily import pyannote and build the community pipeline on *device*.

    Weights are downloaded with ``snapshot_download`` pinned to
    :data:`DIARIZATION_REVISION` (invariant #4) and loaded from the local
    path, following the same pattern as :mod:`vemoizer.models`.
    """
    _disable_pyannote_telemetry()

    import torch
    from pyannote.audio import Pipeline

    local_path = resolve_model_path(
        DIARIZATION_REPO_ID,
        DIARIZATION_REVISION,
        token=os.environ.get(_HF_TOKEN_ENV),
    )
    pipeline = Pipeline.from_pretrained(local_path)
    if pipeline is None:
        raise RuntimeError(
            f"pyannote Pipeline.from_pretrained returned None for "
            f"{DIARIZATION_REPO_ID}@{DIARIZATION_REVISION}; "
            "the snapshot at the pinned revision is missing, unreadable, "
            "or not a loadable pyannote pipeline."
        )
    pipeline.to(torch.device(device))
    return pipeline


def diarize(
    audio: np.ndarray,
    *,
    device: str = "auto",
    num_speakers: SpeakerCount | None = None,
) -> DiarizationResult:
    """Run speaker diarization over 16 kHz mono float32 *audio*.

    ``device="auto"`` tries MPS first (Apple Silicon) and falls back to CPU
    on any load/inference exception. ``device`` may also be an explicit
    torch device name (e.g. ``"cpu"`` or ``"mps"``).

    ``num_speakers`` pins the cluster count when the caller knows how many
    people were in the room — unconstrained clustering split one of four
    speakers into two on the reference meeting (issue #71 forensics). A
    ``(min, max)`` tuple bounds the clustering instead: a pinned count too
    high splits one voice, too low merges two, when attendance changes.
    """
    import torch

    # pyannote 4.x contract: {"waveform": Tensor(channel, time),
    # "sample_rate": int}. The "audio" key means a file path and is
    # rejected for in-memory arrays — with it, the stage could never run.
    waveforms = {
        "waveform": torch.from_numpy(audio.astype(np.float32)).unsqueeze(0),
        "sample_rate": _CONTRACT_SAMPLE_RATE,
    }

    kwargs: dict = {}
    if isinstance(num_speakers, tuple):
        kwargs["min_speakers"], kwargs["max_speakers"] = num_speakers
    elif num_speakers is not None:
        kwargs["num_speakers"] = num_speakers
    if device == "auto":
        try:
            pipeline = _load_pipeline("mps")
            diarization = pipeline(waveforms, **kwargs)
        except Exception:
            pipeline = _load_pipeline("cpu")
            diarization = pipeline(waveforms, **kwargs)
    else:
        pipeline = _load_pipeline(device)
        diarization = pipeline(waveforms, **kwargs)

    # pyannote 4.x returns a DiarizeOutput wrapper. Prefer the exclusive
    # partition (non-overlapping, purpose-built for ASR alignment — no
    # overlap tie-breaking downstream); fall back to the plain annotation,
    # then to the object itself for older versions.
    annotation = getattr(diarization, "exclusive_speaker_diarization", None)
    if annotation is None:
        annotation = getattr(diarization, "speaker_diarization", diarization)
    segments: list[tuple[float, float, str]] = [
        (turn.start, turn.end, speaker)
        for turn, _track, speaker in annotation.itertracks(yield_label=True)
    ]
    return DiarizationResult(segments=segments)


def speaker_for_span(
    seg_start: float,
    seg_end: float,
    speaker_segments: list[tuple[float, float, str]],
) -> str | None:
    """Pick the speaker whose segment overlaps ``[seg_start, seg_end)`` the most.

    ``None`` when no speaker segment overlaps the disputed span (fail-open,
    so callers can omit the ``speaker`` key rather than guessing).
    """
    best: str | None = None
    best_overlap = 0.0
    for s_start, s_end, speaker in speaker_segments:
        overlap = min(seg_end, s_end) - max(seg_start, s_start)
        if overlap > best_overlap:
            best_overlap = overlap
            best = speaker
    return best


def run_diarization_stage(
    audio: np.ndarray,
    speakers: SpeakerCount | None = None,
) -> list[tuple[float, float, str]] | None:
    """Run the diarization stage; ``None`` (fail-open) on any failure.

    ``None`` and ``[]`` are distinct to callers: the orchestrator passes
    ``None`` when diarization was skipped or failed, and ``[]`` when the
    stage ran but found no speakers — either way the downstream overlap step
    leaves the ``speaker`` key off every segment.
    """
    try:
        result = diarize(audio, num_speakers=speakers)
    except Exception as e:  # noqa: BLE001 - fail-open stage boundary
        import logging

        logging.getLogger(__name__).warning(
            "diarization failed, continuing without speaker labels: %s", e
        )
        return None
    return list(result.segments)
