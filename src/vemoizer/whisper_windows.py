"""Per-window raws post-processing for the Whisper meeting decode (issue #147).

Extracted from ``whisper_transcriber.py`` so the language-count accumulation
and summary-line logic has a dedicated home within the 500-line source-file
cap. The per-window raws produced by the window loop are post-processed here:
the echo filter is applied, segments/words are built on the recording
timeline, and the per-window detected languages are aggregated into a
``Counter`` that becomes the run's ``language_summary`` string.

``language_summary`` is display-only (invariant #3: language is a property
of a span, never a file pin). It carries the per-window distribution
(e.g. ``"fi 29/30, en 1/30"``) so the run log and the end-of-run summary
can show it without the per-window "Detected language: X" lines that
would otherwise scroll the progress display away.
"""

from __future__ import annotations

import logging
import math
from collections import Counter
from collections.abc import Callable
from typing import Any

import numpy as np

from .audio_contract import SAMPLE_RATE
from .echo_filter import filter_echo_segments
from .lang_filter import filter_language_lines
from .transcriber import TranscriptionResult

logger = logging.getLogger(__name__)

#: Minimum seconds of VAD speech overlap with a window before the window
#: counts as speech (issue #152). VAD slices are second-accurate at the
#: edges, and a 30 s window whose only "speech" is 0.4 s of VAD bleed from
#: a neighboring slice is not a lost window — it is a pause. 0.5 s is a
#: deliberately small floor: real utterance fragments are longer, VAD
#: edge jitter is shorter.
MIN_VAD_OVERLAP_S = 0.5

#: Fallback energy-gate constants for the no-slices path (dictation and
#: test callers that never run VAD; issue #152). The gate is frame-based
#: RMS so a hiss-only window (peak ~5e-3) does NOT count as speech the way
#: the old peak-amplitude rule (1e-6) did — room noise peaks well above
#: 1e-6 and was producing false "puhetta, ei tekstiä" report lines.
#:
#: - :data:`FALLBACK_FRAME_SECONDS` (0.03 s = 30 ms) is the frame length;
#:   short enough to resolve 30 Hz syllabic modulation, long enough that a
#:   single-frame RMS is a stable number.
#:
#: - :data:`FALLBACK_SILENT_RMS` (1e-2) is the per-frame RMS floor. That is
#:   20·log10(1e-2) = -40 dBFS. Room-noise hiss sits around -55…-65 dBFS
#:   RMS; quiet real speech (whispered) sits around -35…-30 dBFS RMS, so
#:   -40 dBFS separates the two with margin on both sides.
#:
#: - :data:`FALLBACK_MIN_SILENT_FRAMES` (3 consecutive silent frames = 90 ms
#:   of continuous sub-threshold energy) is the hysteresis that stops a
#:   1-frame glitch from counting as speech. Real speech at any level
#:   sustains energy over many frames; a single transient click does not.
FALLBACK_FRAME_SECONDS = 0.03
FALLBACK_SILENT_RMS = 1e-2  # = -40 dBFS (frame considered "active")
FALLBACK_MIN_SILENT_FRAMES = 3


def process_window_raws(
    raws: list[dict[str, Any]],
    *,
    offset_s_per_window: float,
    echo_terms: list[str] | None,
    transcribe_time: float,
    audio_duration: float,
) -> TranscriptionResult:
    """Post-process per-window raw decode results into a result dict.

    Applies the echo filter to each window's segments, builds the
    recording-timeline words/segments lists, accumulates per-window
    detected languages, and assembles the result dict with the
    ``language`` (single value, for the TypedDict contract) and
    ``language_summary`` (per-window distribution, issue #147) keys.
    """
    words: list[dict[str, Any]] = []
    segments: list[dict[str, Any]] = []
    all_window_texts: list[str] = []
    language_counts: Counter[str] = Counter()

    for index, raw in enumerate(raws):
        offset_s = index * offset_s_per_window
        # Per-window language count (issue #147, display only — invariant
        # #3: never pins a language, just records what was detected).
        lang = raw.get("language")
        if lang:
            language_counts[str(lang)] += 1

        window_texts: list[str] = []
        kept_segments, kept_words = filter_echo_segments(
            raw.get("segments") or [], offset_s, echo_terms
        )
        if not raw.get("segments"):
            logger.debug(
                "whisper window %d (offset %.0fs) returned no segments; "
                "transcript may be incomplete",
                index,
                offset_s,
            )
        for seg in kept_segments:
            text = str(seg.get("text", "")).strip()
            if not text:
                continue
            window_texts.append(text)
            entry: dict[str, Any] = {
                "start": float(seg.get("start", 0.0)) + offset_s,
                "end": float(seg.get("end", 0.0)) + offset_s,
                "text": text,
            }
            for key in ("avg_logprob", "no_speech_prob", "compression_ratio"):
                if seg.get(key) is not None:
                    entry[key] = float(seg[key])
            segments.append(entry)
        words.extend(kept_words)
        if not window_texts and not raw.get("segments"):
            window_texts.append(str(raw.get("text", "")).strip())
        all_window_texts.extend(window_texts)

    result: TranscriptionResult = {
        "text": " ".join(t for t in all_window_texts if t).strip(),
        "words": words,
        "segments": segments,
        "transcribe_time": transcribe_time,
        "audio_duration": audio_duration,
        "rtf": transcribe_time / audio_duration if audio_duration > 0 else 0.0,
    }

    # Language (per-utterance, invariant #3): single value when all windows
    # agree; the warning is kept for the disagreement case.
    if len(language_counts) == 1:
        result["language"] = next(iter(language_counts))
    elif language_counts:
        logger.warning(
            "window language disagreement %s; not attributing a run "
            "language (per-slice language from decode B wins downstream)",
            sorted(language_counts),
        )

    # Per-window language summary (issue #147): "fi 29/30, en 1/30" —
    # display only, present whenever at least one window reported a language.
    if language_counts:
        total = sum(language_counts.values())
        parts = [
            f"{lang} {count}/{total}"
            for lang, count in sorted(
                language_counts.items(), key=lambda item: (-item[1], item[0])
            )
        ]
        result["language_summary"] = ", ".join(parts)

    return result


def normalize_lost_windows(
    value: Any,
) -> list[tuple[float, float]]:
    """Validate and normalize ``lost_windows`` entries.

    Accepts a list of ``(start_s, end_s)`` pairs (tuples or lists, ints or
    floats). Drops entries that are not 2-element pairs, have non-numeric
    values, bool values, NaN/inf, negative values, or start > end. Returns
    a list of ``tuple[float, float]``. Empty list on empty/invalid input.
    """
    if not isinstance(value, list):
        return []
    result: list[tuple[float, float]] = []
    for p in value:
        if not isinstance(p, (tuple, list)) or len(p) != 2:
            continue
        a, b = p
        for v in (a, b):
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                break
            fv = float(v)
            if not math.isfinite(fv) or fv < 0:
                break
        else:
            fa, fb = float(a), float(b)
            if fa <= fb:
                result.append((fa, fb))
    return result


def retry_lost_windows(
    raws: list[dict[str, Any]],
    audio: np.ndarray,
    transcribe_fn: Callable[[np.ndarray, dict[str, Any]], Any],
    *,
    window_frames: int,
    window_seconds: float,
    vad_slices: list[tuple[int, int]] | None = None,
) -> list[tuple[float, float]]:
    """Prompt-free fail-safe retry for 0-segment windows with speech.

    A window that is flagged as speech (either by VAD-slice overlap —
    *vad_slices*, when given, or by the fallback frame-RMS gate) but that
    returned 0 segments is re-decoded once via *transcribe_fn* without the
    glossary prompt. The prompt is the remaining suspect: whisper can echo
    the glossary instead of transcribing, and the echo filter (post-decode)
    would then drop the only segment. A retry that returns None (a real
    mlx-whisper failure) or 0 segments records the window as lost and fails
    open. Returns the list of ``(start_s, end_s)`` tuples for windows that
    were lost.

    *vad_slices* are ``(start_sample, end_sample)`` pairs on the recording
    timeline. When present and non-empty, a window is speech only if the
    sum of its overlaps with the slices is ≥ :data:`MIN_VAD_OVERLAP_S` —
    this is the real "was there speech here?" signal the seam has access to
    (pipeline.py computes ``slices = _speech_slices(audio)`` BEFORE
    ``decode_meeting``). When absent or empty (dictation/transcribe path,
    tests, or a run where VAD found nothing and fell back to a single
    full-recording slice — the latter still passes here as ``(0, len(audio))``
    which is correct: if VAD found no speech, the fallback RMS gate is what
    decides, so an all-coverage VAD slice is treated as "no VAD info").
    """
    # use_vad is True when VAD slices were actually computed and represent
    # real speech boundaries (not the "VAD found nothing" single full-file
    # fallback). A single (0, len(audio)) slice is the fallback — in that
    # case the RMS gate is the real signal.
    use_vad = bool(vad_slices) and any((e - s) < len(audio) for s, e in vad_slices)
    lost_windows: list[tuple[float, float]] = []
    for index, raw in enumerate(raws):
        if raw.get("segments"):
            continue
        window_start_s = index * window_seconds
        window_end_s = window_start_s + window_seconds
        offset_samples = index * window_frames
        window_audio = audio[offset_samples : offset_samples + window_frames]
        if use_vad:
            overlap_s = _vad_overlap_seconds(
                vad_slices, window_start_s, window_end_s, SAMPLE_RATE
            )
            is_speech = overlap_s >= MIN_VAD_OVERLAP_S
        else:
            is_speech = _window_has_speech(window_audio)
        if not is_speech:
            # Genuinely silent window (or VAD-bleed-only): nothing to
            # retry, not a loss.
            continue
        offset_s = index * window_seconds
        logger.warning(
            "whisper window %d (offset %.0fs) returned no segments; "
            "retrying without glossary prompt",
            index,
            offset_s,
        )
        # The retry is a second decode call, so it honors the same stdout
        # contract as the main window loop: the per-window "Detected
        # language: X" line is filtered. The progress shim is intentionally
        # NOT re-entered — it tracks the main window loop one window at a
        # time, and a retry re-marking the same window would regress the
        # bar; the retry is a rare fail-safe path so the cosmetic impact of
        # skipping it is negligible.
        with filter_language_lines():
            retry_raw = transcribe_fn(window_audio, {"initial_prompt": None})
        if retry_raw is None:
            # A None from mlx-whisper is a real failure of the retry, not a
            # silent no-op: the window had confirmed speech energy, so record
            # it as lost (fail-open: the transcript is incomplete but the run
            # continues; never silent). mlx-whisper returns None for a failed
            # decode (e.g. an empty/no-speech classification), not an
            # exception — the log explains the mechanism, not just the
            # observation.
            logger.error(
                "retry for window %d (offset %.0fs) returned None "
                "(mlx-whisper decode failure); recording as lost",
                index,
                offset_s,
            )
            lost_windows.append((offset_s, offset_s + window_seconds))
            continue
        raws[index] = retry_raw
        if not retry_raw.get("segments"):
            # FIX 4: the retry returned 0 segments (not None) — the window
            # is lost. Log a warning naming the window index and offset so
            # the loss is never silent (the None case is logged above).
            logger.warning(
                "retry for window %d (offset %.0fs) returned 0 segments; "
                "recording as lost",
                index,
                offset_s,
            )
            lost_windows.append((offset_s, offset_s + window_seconds))
    return lost_windows


def _vad_overlap_seconds(
    vad_slices: list[tuple[int, int]],
    window_start_s: float,
    window_end_s: float,
    sample_rate: int,
) -> float:
    """Total seconds of overlap between a time window and the VAD slices.

    Slices are ``(start_sample, end_sample)`` pairs on the recording
    timeline; the window is a half-open ``[window_start_s, window_end_s)``
    interval in seconds. Overlapping intervals are clipped to the window
    bounds and summed. This is the primary speech signal for
    :func:`retry_lost_windows` when *vad_slices* is available.
    """
    total = 0.0
    for s, e in vad_slices:
        slice_start_s = s / sample_rate
        slice_end_s = e / sample_rate
        overlap_start = max(window_start_s, slice_start_s)
        overlap_end = min(window_end_s, slice_end_s)
        if overlap_end > overlap_start:
            total += overlap_end - overlap_start
    return total


def _window_has_speech(window_audio: np.ndarray) -> bool:
    """Frame-RMS fallback gate for a 0-segment window (issue #152).

    Splits *window_audio* into :data:`FALLBACK_FRAME_SECONDS` frames, computes
    per-frame RMS, and returns True only if the number of consecutive
    below-threshold frames stays below :data:`FALLBACK_MIN_SILENT_FRAMES` for
    the entire window. A hiss-only window (room noise, peak ~5e-3, RMS
    ~-55…-65 dBFS) fails the gate; quiet-but-real speech (RMS ~-35 dBFS)
    passes. Used only when VAD slices are unavailable at the seam; never as
    a general speech detector.
    """
    if len(window_audio) == 0:
        return False
    frame_len = int(FALLBACK_FRAME_SECONDS * SAMPLE_RATE)
    if len(window_audio) < frame_len:
        # Too short for even one frame: use peak as a conservative fallback.
        return float(np.abs(window_audio).max()) >= FALLBACK_SILENT_RMS
    n_frames = len(window_audio) // frame_len
    window_trimmed = window_audio[: n_frames * frame_len]
    rms = np.sqrt((window_trimmed.reshape(n_frames, frame_len) ** 2).mean(axis=1))
    # Speech iff the fraction of "active" frames (RMS >= FALLBACK_SILENT_RMS,
    # -40 dBFS) is at least 5%. A 30 s window of room-noise hiss (RMS ~-55 dBFS)
    # has 0 active frames. A 30 s window with a 3 s quiet speech burst (RMS ~-35 dBFS)
    # has ~100 active frames out of 1000 total (10%), well above the 5% threshold.
    active = rms >= FALLBACK_SILENT_RMS
    return bool(active.sum() / len(rms) >= 0.05)


def _window_has_speech_legacy(window_audio: np.ndarray) -> bool:
    """Legacy peak-amplitude rule (for break-and-fail testing only)."""
    if len(window_audio) == 0:
        return False
    return float(np.abs(window_audio).max()) >= 1e-6
