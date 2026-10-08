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
from collections import Counter
from collections.abc import Callable
from typing import Any

import numpy as np

from .echo_filter import filter_echo_segments
from .lang_filter import filter_language_lines
from .transcriber import TranscriptionResult

logger = logging.getLogger(__name__)

#: Peak absolute amplitude below which a 0-segment window is treated as
#: genuinely silent (skip the prompt-free retry, issue #152) rather than as
#: a lost window. A deliberately crude energy heuristic — it is a fail-safe
#: gate, not a speech detector: the real presence/absence of speech comes
#: from whisper's own decode (segments) and from the VAD slices downstream.
SILENT_PEAK_THRESHOLD = 1e-6


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


def retry_lost_windows(
    raws: list[dict[str, Any]],
    audio: np.ndarray,
    transcribe_fn: Callable[[np.ndarray, dict[str, Any]], Any],
    *,
    window_frames: int,
    window_seconds: float,
) -> list[tuple[float, float]]:
    """Prompt-free fail-safe retry for 0-segment windows with speech energy.

    A window that the energy check (peak amplitude vs
    :data:`SILENT_PEAK_THRESHOLD`) says contains speech but that returned
    0 segments is re-decoded once via *transcribe_fn* without the glossary
    prompt. The prompt is the remaining suspect: whisper can echo the
    glossary instead of transcribing, and the echo filter (post-decode) would
    then drop the only segment. A retry that returns None (a real mlx-whisper
    failure) or 0 segments records the window as lost and fails open. Returns
    the list of ``(start_s, end_s)`` tuples for windows that were lost.
    """
    lost_windows: list[tuple[float, float]] = []
    for index, raw in enumerate(raws):
        if raw.get("segments"):
            continue
        offset_samples = index * window_frames
        window_audio = audio[offset_samples : offset_samples + window_frames]
        if not _window_has_speech(window_audio):
            # Genuinely silent window: nothing to retry, not a loss.
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
            lost_windows.append((offset_s, offset_s + window_seconds))
    return lost_windows


def _window_has_speech(window_audio: np.ndarray) -> bool:
    """True if a 0-segment window carries audible signal.

    Peak-amplitude test against :data:`SILENT_PEAK_THRESHOLD`. Only used as
    the fail-safe gate for the prompt-free retry (issue #152): it decides
    whether an empty window is "nothing to decode" or a candidate for retry
    — never as a speech detector in its own right.
    """
    if len(window_audio) == 0:
        return False
    return float(np.abs(window_audio).max()) >= SILENT_PEAK_THRESHOLD
