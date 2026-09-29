"""Consensus pipeline orchestrator (issue #34).

Chains every stage into one end-to-end run:

    ingest -> VAD -> decode A (Parakeet) -> decode B (Canary)
           -> DTW align -> disputed spans -> re-decode (Whisper)
           -> LLM adjudication -> assembled transcript

Fail-open at every stage: a stage failure degrades to the best available
result (a failed decode B skips alignment and the output falls back to
decode A's text; an unconfigured or failing LLM keeps the best non-LLM
candidate) rather than aborting the run.

VAD splits long recordings so the full decodes stay bounded; the per-VAD-slice
word timestamps are shifted back onto the full-recording timeline before
alignment, so disputed spans are valid on the full buffer that the re-decode
stage slices from. VAD is optional: on any failure the whole recording is
decoded as a single slice.

Models are loaded lazily (each transcriber's own lazy loader) and released
via ``cleanup()`` on every exit path.
"""

from __future__ import annotations

import logging
import time
from contextlib import suppress
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np

from .alignment import WordPairs, align_pairs_safe
from .audio_contract import SAMPLE_RATE
from .canary_transcriber import CanaryTranscriber
from .diarization import diarize
from .ingest import IngestError, ingest_audio
from .llm import LLMClient, LLMConfig, load_config
from .parakeet_transcriber import ParakeetTranscriber
from .progress import StageProgress, format_duration
from .redecode import WhisperReDecodeTranscriber
from .spans import Span, find_disputed_spans
from .vad import SpeechSegment, vad_segments
from .vad import load_model as load_vad_model

logger = logging.getLogger(__name__)

_DEFAULT_CONFIG_PATHS = (
    Path.home() / ".config" / "vemoizer" / "config.toml",
    Path.home() / ".vemoizer.toml",
)

Candidate = dict[str, str]  # {"source": str, "text": str}


def _load_llm_config(path: str | None) -> LLMConfig | None:
    """Load the LLM config; ``None`` (fail-open) when unconfigured."""
    if path is not None:
        return load_config(path)
    for candidate in _DEFAULT_CONFIG_PATHS:
        if candidate.is_file():
            return load_config(candidate)
    return None


def _decode(transcriber: Any, audio: np.ndarray, label: str) -> dict[str, Any] | None:
    """Run one full decode; return its result or ``None`` (fail-open).

    Frees the Metal cache after every slice so GPU memory does not accumulate
    across VAD slices.
    """
    try:
        return transcriber.transcribe(audio)
    except Exception as e:  # noqa: BLE001 - fail-open stage boundary
        logger.warning("%s failed, using best available result: %s", label, e)
        return None
    finally:
        mx.clear_cache()


def _speech_slices(audio: np.ndarray) -> list[tuple[int, np.ndarray]]:
    """VAD-split the recording into ``(offset, slice)`` pairs.

    The offset is the slice's first sample in the full recording, used to
    shift per-slice timestamps back onto the full timeline. Falls back to
    the whole recording as a single slice when VAD is unavailable or finds
    no speech.
    """
    start = time.monotonic()
    try:
        vad_model = load_vad_model()
        segments: list[SpeechSegment] = vad_segments(audio, vad_model)
    except Exception as e:  # noqa: BLE001 - fail-open stage boundary
        logger.warning("VAD unavailable, decoding full recording: %s", e)
        return [(0, audio)]
    if not segments:
        logger.info("VAD: no speech found, decoding full recording as one slice")
        return [(0, audio)]
    speech = sum(seg.end - seg.start for seg in segments) / SAMPLE_RATE
    logger.info(
        "VAD: %d speech slices (%s of speech) in %s",
        len(segments),
        format_duration(speech),
        format_duration(time.monotonic() - start),
    )
    return [(seg.start, audio[seg.start : seg.end]) for seg in segments]


def _decode_all(
    transcriber: Any, slices: list[tuple[int, np.ndarray]], label: str
) -> dict[str, Any] | None:
    """Decode every VAD slice through *transcriber* and merge the results.

    Word/segment times are shifted onto the full-recording timeline so the
    downstream alignment and re-decode stages work on one time base.
    """
    if not slices:
        return {"text": "", "words": [], "segments": []}
    merged_words: list[dict[str, Any]] = []
    merged_segments: list[dict[str, Any]] = []
    texts: list[str] = []
    audio_seconds = sum(len(s) for _, s in slices) / SAMPLE_RATE
    progress = StageProgress(label, len(slices), audio_seconds)
    for offset, slice_audio in slices:
        r = _decode(transcriber, slice_audio, label)
        progress.advance(failed=r is None)
        if r is None:
            continue
        texts.append(str(r.get("text", "")).strip())
        delta = offset / SAMPLE_RATE
        for w in r.get("words") or []:
            merged_words.append(
                {
                    **w,
                    "start": float(w.get("start", 0.0)) + delta,
                    "end": float(w.get("end", 0.0)) + delta,
                }
            )
        for s in r.get("segments") or []:
            merged_segments.append(
                {
                    **s,
                    "start": float(s.get("start", 0.0)) + delta,
                    "end": float(s.get("end", 0.0)) + delta,
                }
            )
    progress.done()
    merged = {
        "text": " ".join(t for t in texts if t),
        "words": merged_words,
        "segments": merged_segments,
    }
    logger.info(
        "%s: %d chars, %d words, %d segments",
        label,
        len(merged["text"]),
        len(merged_words),
        len(merged_segments),
    )
    return merged


def _redecode_spans(
    audio: np.ndarray, spans: list[Span]
) -> list[dict[str, Any]] | None:
    """Re-decode each disputed span; ``None`` when re-decode is unavailable."""
    redecoder = WhisperReDecodeTranscriber()
    progress = StageProgress("re-decode", len(spans), unit="spans")
    try:
        results = []
        for s in spans:
            results.append(redecoder.transcribe_span(audio, s))
            progress.advance()
        progress.done()
        return [
            {"span": r.span, "text": r.text, "words": r.words, "ok": r.ok}
            for r in results
        ]
    except Exception as e:  # noqa: BLE001 - fail-open stage boundary
        logger.warning("re-decode stage failed; skipping: %s", e)
        return None
    finally:
        redecoder.cleanup()


def _words_in_span(words: list[dict[str, Any]], span: Span) -> str:
    """The *words* falling inside *span*, joined (A-side span text)."""
    return " ".join(
        str(w.get("word", ""))
        for w in words
        if span.start <= float(w.get("start", 0.0)) < span.end
    ).strip()


def _adjudicate(
    span: Span,
    a_text: str,
    candidates: list[Candidate],
    client: LLMClient | None,
) -> str:
    """Final text for one disputed span, fail-open down the candidate list."""
    if client is not None:
        try:
            verdict = client.adjudicate(a_text, candidates)
            if verdict.strip():
                return verdict
        except Exception as e:  # noqa: BLE001 - fail-open stage boundary
            logger.warning(
                "adjudication failed for span [%0.2f, %0.2f): %s",
                span.start,
                span.end,
                e,
            )
    for candidate in reversed(candidates):  # re-decode > decode B > decode A
        if candidate["text"].strip():
            return candidate["text"]
    return a_text


def _speaker_for_segment(
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


def _assemble(
    result_a: dict[str, Any] | None,
    result_b: dict[str, Any] | None,
    redecoded: list[dict[str, Any]] | None,
    llm_config: LLMConfig | None,
    speaker_segments: list[tuple[float, float, str]] | None = None,
    pairs: WordPairs | None = None,
) -> dict[str, Any]:
    """Combine the stage outputs into the final ``{"text", "segments"}``.

    ``pairs`` is the alignment the caller already computed to pick the
    disputed spans it re-decoded. Passing it in keeps the spans here identical
    to the ones ``redecoded`` was indexed against, and avoids re-running an
    O(len(A) x len(B)) DTW that the caller has already paid for.

    ``speaker_segments`` (when given) is a list of ``(start, end, speaker)``
    triples from the diarization stage; each adjudicated segment is labelled
    with the speaker whose segment overlaps the disputed span the most. The
    ``speaker`` key is omitted when no speaker segment overlaps, so downstream
    formatters can render the label only when it is actually known.
    """
    base = result_a or result_b
    if base is None:
        return {"text": "", "segments": []}

    words = list(base.get("words") or [])
    spans = find_disputed_spans(pairs) if pairs else []
    redecoded = redecoded or []

    client = LLMClient(llm_config) if llm_config is not None else None
    segments: list[dict[str, Any]] = []
    # One LLM round-trip per span when adjudication is configured; without a
    # heartbeat this loop is the pipeline's second silent multi-minute stage.
    progress = StageProgress("adjudicate", len(spans), unit="spans")
    for i, span in enumerate(spans):
        rd = redecoded[i] if i < len(redecoded) else None
        candidates: list[Candidate] = [
            {"source": "decode A", "text": _words_in_span(words, span)},
        ]
        if result_b is not None:
            b_text = str(result_b.get("text", ""))
            candidates.append({"source": "decode B", "text": b_text})
        candidates.append({"source": "re-decode", "text": rd["text"] if rd else ""})
        verdict = _adjudicate(span, candidates[0]["text"], candidates, client)
        segment: dict[str, Any] = {
            "start": span.start,
            "end": span.end,
            "text": verdict,
        }
        if speaker_segments is not None:
            speaker = _speaker_for_segment(span.start, span.end, speaker_segments)
            if speaker is not None:
                segment["speaker"] = speaker
        segments.append(segment)
        progress.advance()

    progress.done()
    segments.sort(key=lambda s: s["start"])
    return {"text": str(base.get("text", "")).strip(), "segments": segments}


def transcribe_decode_only(path: str | Path, *, backend: str) -> dict[str, Any]:
    """Run ingest -> VAD -> one single decode; no consensus, no LLM.

    The eval harness scores each decode backend on its own so the consensus
    gain is a measured number rather than an assertion (invariant #7). The
    stage chain and fail-open behaviour mirror :func:`transcribe_file`'s
    decode stage exactly — same VAD slicing, same merge — so a backend's
    eval WER reflects what that backend contributes inside the pipeline.
    """
    backends = {"parakeet": ParakeetTranscriber, "canary": CanaryTranscriber}
    if backend not in backends:
        known = ", ".join(sorted(backends))
        raise ValueError(f"unknown backend {backend!r} (known: {known})")
    try:
        audio = ingest_audio(Path(path))
    except IngestError as e:
        logger.error("ingest failed for %s: %s", path, e)
        return {"text": "", "segments": [], "error": str(e)}
    if len(audio) == 0:
        return {"text": "", "segments": []}
    slices = _speech_slices(audio)
    transcriber: Any = None
    result: dict[str, Any] | None = None
    try:
        transcriber = backends[backend]()
        result = _decode_all(transcriber, slices, f"decode ({backend})")
    except Exception as e:  # noqa: BLE001 - fail-open stage boundary
        logger.warning("decode (%s) failed: %s", backend, e)
    finally:
        if transcriber is not None:
            with suppress(Exception):  # cleanup is best-effort (fail-open)
                transcriber.cleanup()
    if result is None:
        return {"text": "", "segments": []}
    return {
        "text": str(result.get("text", "")).strip(),
        "segments": list(result.get("segments") or []),
    }


def transcribe_file(
    path: str | Path,
    *,
    config_path: str | None = None,
    diarize: bool = False,
) -> dict:
    """Run the full consensus pipeline over one audio file.

    Args:
        path: Path to the audio file (any ffmpeg-readable container).
        config_path: Optional user config file with an ``[llm]`` section.
            When omitted, ``~/.config/vemoizer/config.toml`` and
            ``~/.vemoizer.toml`` are probed; when no config is found the LLM
            stage is skipped (fail-open).
        diarize: When True, run the pyannote speaker-diarization stage
            (opt-in, off by default) and label each disputed segment with
            the speaker whose segment overlaps it the most. Any diarization
            failure (missing token, model unavailable, inference error) is
            swallowed: the run continues without speaker labels (fail-open).

    Returns:
        ``{"text": str, "segments": list[dict]}`` — the full transcript
        (decode A preferred) plus one segment per disputed span with its
        adjudicated text. Each segment dict gains an optional ``speaker``
        key when the diarization stage was enabled and produced a matching
        speaker for that span.
    """
    run_start = time.monotonic()
    logger.info("transcribe: %s", path)
    ingest_start = time.monotonic()
    try:
        audio = ingest_audio(Path(path))
    except IngestError as e:
        logger.error("ingest failed for %s: %s", path, e)
        return {"text": "", "segments": [], "error": str(e)}
    if len(audio) == 0:
        logger.info("ingest: empty audio, nothing to transcribe")
        return {"text": "", "segments": []}
    logger.info(
        "ingest: %s of audio in %s",
        format_duration(len(audio) / SAMPLE_RATE),
        format_duration(time.monotonic() - ingest_start),
    )

    llm_config = _load_llm_config(config_path)
    logger.info(
        "LLM adjudication: %s", "configured" if llm_config is not None else "disabled"
    )
    slices = _speech_slices(audio)

    result_a: dict[str, Any] | None = None
    result_b: dict[str, Any] | None = None
    parakeet: Any = None
    canary: Any = None
    try:
        parakeet = ParakeetTranscriber()
        result_a = _decode_all(parakeet, slices, "decode A")
    except Exception as e:  # noqa: BLE001 - fail-open stage boundary
        logger.warning("decode A failed, using best available result: %s", e)
    finally:
        if parakeet is not None:
            with suppress(Exception):  # cleanup is best-effort (fail-open)
                parakeet.cleanup()
    try:
        canary = CanaryTranscriber()
        result_b = _decode_all(canary, slices, "decode B")
    except Exception as e:  # noqa: BLE001 - fail-open stage boundary
        logger.warning("decode B failed, using best available result: %s", e)
    finally:
        if canary is not None:
            with suppress(Exception):  # cleanup is best-effort (fail-open)
                canary.cleanup()

    pairs = align_pairs_safe(result_a, result_b)
    redecoded: list[dict[str, Any]] | None = None
    if pairs:
        spans = find_disputed_spans(pairs)
        logger.info("disputed spans: %d", len(spans))
        if spans:
            redecoded = _redecode_spans(audio, spans)
    else:
        logger.info("disputed spans: 0 (no alignment, re-decode skipped)")

    speaker_segments: list[tuple[float, float, str]] | None = None
    if diarize:
        logger.info("diarization: starting")
        diarize_start = time.monotonic()
        speaker_segments = _run_diarization_stage(audio)
        logger.info(
            "diarization: %s speaker segments in %s",
            len(speaker_segments) if speaker_segments is not None else "no",
            format_duration(time.monotonic() - diarize_start),
        )

    logger.info("assemble: adjudicating spans")
    result = _assemble(
        result_a, result_b, redecoded, llm_config, speaker_segments, pairs=pairs
    )
    logger.info(
        "transcribe: done in %s — %d chars, %d segments",
        format_duration(time.monotonic() - run_start),
        len(result.get("text", "")),
        len(result.get("segments", [])),
    )
    return result


def _run_diarization_stage(
    audio: np.ndarray,
) -> list[tuple[float, float, str]] | None:
    """Run the diarization stage; ``None`` (fail-open) on any failure.

    ``None`` and ``[]`` are distinct to callers: the orchestrator passes
    ``None`` when diarization was skipped or failed, and ``[]`` when the
    stage ran but found no speakers — either way the downstream overlap step
    leaves the ``speaker`` key off every segment.
    """
    try:
        result = diarize(audio)
    except Exception as e:  # noqa: BLE001 - fail-open stage boundary
        logger.warning("diarization failed, continuing without speaker labels: %s", e)
        return None
    return list(result.segments)
