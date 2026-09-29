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

from dataclasses import dataclass

import numpy as np

#: HuggingFace repo for the diarization weights (CC-BY-4.0, gated).
DIARIZATION_REPO_ID = "pyannote/speaker-diarization-community-1"

#: Pinned full-SHA commit of the diarization weights (invariant #4): loading
#: from a bare repo ID would cache a moving ref.
DIARIZATION_REVISION = "3533c8cf8e369892e6b79ff1bf80f7b0286a54ee"

#: Mandatory CC-BY-4.0 attribution (weights license, not code license).
ATTRIBUTION = (
    "Speaker diarization: pyannote/speaker-diarization-community-1 "
    "(weights licensed under CC-BY-4.0, gated on HuggingFace; "
    "user accepted the license form and supplied an access token)."
)

#: Sample rate of the internal audio contract (AGENTS.md invariant #6).
_CONTRACT_SAMPLE_RATE = 16000

#: Environment variable holding the HuggingFace access token for the gated repo.
_HF_TOKEN_ENV = "HF_TOKEN"


@dataclass(frozen=True)
class DiarizationResult:
    """Speaker-labelled time segments covering the recording."""

    segments: list[tuple[float, float, str]]  # (start_s, end_s, speaker_label)


def _load_pipeline(device: str) -> object:
    """Lazily import pyannote and build the community pipeline on *device*.

    Weights are downloaded with ``snapshot_download`` pinned to
    :data:`DIARIZATION_REVISION` (invariant #4) and loaded from the local
    path, following the same pattern as :mod:`vemoizer.models`.
    """
    import os

    import torch
    from huggingface_hub import snapshot_download
    from pyannote.audio import Pipeline

    local_path = snapshot_download(
        DIARIZATION_REPO_ID,
        revision=DIARIZATION_REVISION,
        token=os.environ.get(_HF_TOKEN_ENV),
    )
    pipeline = Pipeline.from_pretrained(local_path)
    pipeline.to(torch.device(device))
    return pipeline


def diarize(audio: np.ndarray, *, device: str = "auto") -> DiarizationResult:
    """Run speaker diarization over 16 kHz mono float32 *audio*.

    ``device="auto"`` tries MPS first (Apple Silicon) and falls back to CPU
    on any load/inference exception. ``device`` may also be an explicit
    torch device name (e.g. ``"cpu"`` or ``"mps"``).
    """
    waveforms = {
        "audio": audio.astype(np.float32),
        "sample_rate": _CONTRACT_SAMPLE_RATE,
    }

    if device == "auto":
        try:
            pipeline = _load_pipeline("mps")
            diarization = pipeline(waveforms)  # ty: ignore[call-non-callable]
        except Exception:
            pipeline = _load_pipeline("cpu")
            diarization = pipeline(waveforms)  # ty: ignore[call-non-callable]
    else:
        pipeline = _load_pipeline(device)
        diarization = pipeline(waveforms)  # ty: ignore[call-non-callable]

    segments: list[tuple[float, float, str]] = [
        (turn.start, turn.end, turn.speaker)
        for turn in diarization.itertracks(yield_label=True)
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
