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
from typing import Any

from .echo_filter import filter_echo_segments
from .transcriber import TranscriptionResult

logger = logging.getLogger(__name__)


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
