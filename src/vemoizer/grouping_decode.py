"""Boundary-decode half of the M3 split-recording grouping (issue #77).

The 20 s edge-window decode that feeds :func:`vemoizer.grouping.propose_groups`
with pre-decoded tail/head texts. Split from ``grouping.py`` so that the
pure heuristic (grouping.py) and the impure ffmpeg/Whisper decode (this
module) each have a single, small responsibility.

``decode_boundaries`` returns ``(tail_texts, head_texts)`` — one entry per
boundary, i.e. ``len(files) - 1`` of each (the last file's tail has no
successor, so it is never decoded). A shared ``transcribe_fn`` is loaded
once and cleaned up after the last boundary. The default is a lazily
created ``WhisperTranscriber(language=None)`` — the concrete class the
meeting profile decodes with: boundary text can be Finnish or English (no
language pinning), and the closing-cue heuristic only needs a coarse read,
not full accuracy. Any per-slice decode failure degrades that edge to
``""`` (no evidence) — never a raise.

Only the 20 s edge windows are decoded (ffmpeg ``-ss`` / ``-t``); the
file's full duration is probed with ffprobe for bounds only (the decoded
PCM still defines every offset — ffprobe never measures transcribed
audio). When ffprobe yields no duration (``0.0``) for an existing file,
the tail edge is skipped (``""``) rather than requesting an unbounded
decode window.
"""

from __future__ import annotations

import logging
import subprocess
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from vemoizer.ingest import IngestError
from vemoizer.transcriber import Transcriber

logger = logging.getLogger(__name__)


def _decode_edge_window(path: Path, start: float, end: float) -> np.ndarray:
    """Decode ``[start, end)`` seconds of *path* — just that window.

    ``-ss`` / ``-t`` are placed BEFORE ``-i``: ffmpeg seeks into the input
    (fast seek) and caps the decode length, so a 1-hour file costs a ~20 s
    decode instead of a full-file decode (~2.4 GB of transient float32
    PCM). Note: with input seeking, ``-ss`` lands on the nearest seekable
    point, so the decoded window is approximate at the start boundary —
    fine for a coarse closing-cue probe, never for transcription
    alignment. ``-t`` cannot read past the actual media end, so an
    over-estimated probed duration never over-reads; the caller clamps
    the tail start against the probed duration and skips the tail when
    the probe is empty (``0.0``).
    """
    argv = [
        "ffmpeg",
        "-nostdin",
        "-v",
        "error",
        "-ss",
        f"{start:.6f}",
        "-t",
        f"{end - start:.6f}",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_f32le",
        "-f",
        "f32le",
        "-",
        "-i",
        str(path),
    ]
    proc = subprocess.run(argv, capture_output=True, check=False, timeout=60.0)
    if proc.returncode != 0:
        # stderr is ffmpeg internals — truncate and strip newlines so the
        # user-facing line stays a single bounded paragraph that still
        # names the offending file (round 3 finding 8).
        stderr = proc.stderr.decode("utf-8", errors="replace").strip()
        stderr = " ".join(stderr.split())
        if len(stderr) > 200:
            stderr = stderr[:200] + "…"
        raise IngestError(f"ffmpeg edge decode failed for {path}: {stderr}")
    raw = proc.stdout
    n = len(raw) // 4
    if n == 0:
        return np.zeros(0, dtype=np.float32)
    return np.frombuffer(raw, dtype=np.float32, count=n).copy()


def decode_boundaries(
    files: Sequence[Path | str],
    transcribe_fn: Callable[[np.ndarray], dict[str, Any]] | None = None,
) -> tuple[list[str], list[str]]:
    """Decode each boundary's 20 s tail and head.

    Returns ``(tail_texts, head_texts)`` — one entry per boundary, i.e.
    ``len(files) - 1`` of each (the last file's tail has no successor,
    so it is never decoded). A shared *transcribe_fn* is loaded once and
    released after the last boundary. The default is a lazily created
    ``WhisperTranscriber(language=None)`` — the concrete class the meeting
    profile decodes with: boundary text can be Finnish or English (no
    language pinning), and the closing-cue heuristic only needs a coarse
    read, not full accuracy. Any per-slice decode failure degrades that
    edge to ``""`` (no evidence) — never a raise.

    Only the 20 s edge windows are decoded (ffmpeg ``-ss`` / ``-t``); the
    file's full duration is probed with ffprobe for bounds only (the
    decoded PCM still defines every offset — ffprobe never measures
    transcribed audio). When ffprobe yields no duration (``0.0``) for an
    existing file, the tail edge is skipped (``""``) rather than
    requesting an unbounded decode window.

    Memory: the boundary model's weights live in the shared
    ``mlx_whisper`` ``ModelHolder`` cache and — deliberately — remain
    resident for the main pipeline: the meeting profile's
    ``WhisperTranscriber`` reuses exactly those weights for the per-group
    decode, so keeping them resident costs no extra peak (bounded by the
    one shared copy); ``cleanup()`` is never called here because it
    would destroy that shared cache. In dictation mode the boundary model
    additionally stays resident alongside the consensus models (parakeet
    + canary) for the run's duration — a known follow-up, not an error.
    """
    # Deferred imports so that monkeypatch.setattr(grouping, "X", ...)
    # in tests patches the names that decode_boundaries actually looks up.
    from .grouping import BOUNDARY_SECONDS, natural_sort, probe_duration_seconds
    from .grouping import _decode_edge_window as _dew

    ordered = natural_sort(files)

    transcriber: Transcriber | None = None
    _window: np.ndarray | None = None  # loop-local; del'd in the finally
    _tr_snapshot: Transcriber | None = None
    if transcribe_fn is None:
        from vemoizer.whisper_transcriber import WhisperTranscriber

        transcriber = WhisperTranscriber(language=None)
        _tr_snapshot = transcriber
    try:

        def _edge_text(
            path: Path, start: float, end: float
        ) -> tuple[str, np.ndarray | None]:
            """Decode ``[start, end)`` seconds of *path*; ``("", audio)``.

            The decoded PCM array is returned alongside the text so the
            ``finally`` block can drop the reference deterministically;
            audio is the only large local this function owns besides the
            transcriber itself. ``None`` when nothing was decoded (empty
            window or failure) — there is then no array to drop.
            """
            tr = _tr_snapshot
            try:
                audio = _dew(path, start, end)
                if len(audio) == 0:
                    return "", None
                if transcribe_fn is not None:
                    text = transcribe_fn(audio).get("text", "")
                else:
                    if tr is None:
                        raise RuntimeError("boundary transcriber not loaded")
                    text = tr.transcribe(audio).get("text", "")
                return str(text).strip(), audio
            except (IngestError, RuntimeError, OSError) as e:
                # Log the full error for diagnostics; the user-facing path
                # is the degraded "" (no evidence) below.
                logger.warning("boundary decode failed for %s: %s", path, e)
                return "", None

        tail_texts: list[str] = []
        head_texts: list[str] = []
        for i in range(len(ordered) - 1):
            path = ordered[i]
            dur = probe_duration_seconds(path)
            # No probe evidence (``0.0``): skip the tail rather than
            # requesting an unbounded decode window (ffprobe failure) —
            # say so, since the boundary then degrades to a break with no
            # tail evidence.
            if dur > 0.0:
                tail, tail_audio = _edge_text(path, dur - BOUNDARY_SECONDS, dur)
            else:
                logger.warning(
                    "no duration evidence for %s (ffprobe failed or "
                    "unreadable); tail probe skipped, boundary will "
                    "degrade to a break",
                    path,
                )
                tail, tail_audio = "", None
            tail_texts.append(tail)
            # The head is the first 20 s of the NEXT file (the one the
            # boundary leads into), not of the current file.
            head, head_audio = _edge_text(ordered[i + 1], 0.0, BOUNDARY_SECONDS)
            head_texts.append(head)
            # Keep only the newest decoded window alive: once a boundary's
            # both edges are done, the previous window is no longer needed
            # and the single reference is reassigned (the loop variable
            # itself would otherwise pin the last window until this
            # finally).
            _window = head_audio if head_audio is not None else tail_audio
        if transcriber is not None:
            # Observable, intentional: the weights stay in the shared
            # mlx_whisper ModelHolder cache for the main pipeline's
            # per-group decode (cleanup() would destroy that cache).
            logger.info(
                "boundary decode done; whisper-large-v3-turbo weights are "
                "intentionally left resident in the shared ModelHolder cache "
                "for the main pipeline"
            )
        return tail_texts, head_texts
    finally:
        # cleanup() is deliberately NOT called here: it would set
        # mlx_whisper.transcribe.ModelHolder.model = None, destroying the
        # shared MLX model cache that the main pipeline's
        # WhisperTranscriber (meeting profile) reuses for the per-group
        # decode. The transcriber instance is released here instead by
        # dropping every local reference (the transcriber + the last
        # decoded PCM array) deterministically, rather than relying on GC
        # to find the scope exit first — the instance's __del__ then
        # releases it, while the shared ModelHolder cache (holding the
        # actual weights) survives for the main pipeline. Memory stays
        # bounded: the boundary model is the same whisper-large-v3-turbo
        # weights the main pipeline would load anyway, so no extra peak is
        # incurred (round 3 findings 9-16).
        del transcriber
        del _window
        if "tail_audio" in locals():
            del tail_audio
        if "head_audio" in locals():
            del head_audio
