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
import os
import time
from contextlib import suppress
from pathlib import Path
from typing import Any

import numpy as np

from .audio_contract import SAMPLE_RATE
from .canary_transcriber import CanaryTranscriber
from .decode_stage import decode_all
from .diarization import ATTRIBUTION as DIARIZATION_ATTRIBUTION
from .diarization import diarize, speaker_for_span
from .ingest import IngestError, ingest_audio
from .llm import LLMClient, LLMConfig, load_config
from .notes import generate_notes
from .parakeet_transcriber import ParakeetTranscriber
from .progress import StageProgress, format_duration
from .readability import paragraphs, splice_verdicts
from .redecode import WhisperReDecodeTranscriber
from .slice_align import find_disputed_slices
from .spans import (
    Span,
    apply_span_guardrails,
    span_context,
    words_in_span,
)
from .vad import SpeechSegment, vad_segments
from .vad import load_model as load_vad_model
from .whisper_transcriber import decode_meeting

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


def _b_text_in_span(result_b: dict[str, Any] | None, span: Span) -> str:
    """Decode B's slice text overlapping *span* (B has no word timestamps)."""
    if result_b is None:
        return ""
    parts: list[str] = []
    for s in result_b.get("slices") or []:
        if float(s["end_s"]) > span.start and float(s["start_s"]) < span.end:
            text = str(s.get("text", "")).strip()
            if text:
                parts.append(text)
    return " ".join(parts)


def _find_spans(
    result_a: dict[str, Any] | None, result_b: dict[str, Any] | None
) -> list[Span]:
    """Disputed spans between the decodes, guardrailed; ``[]`` = no consensus.

    The dispute unit is the VAD slice (real bounds, no synthetic
    timestamps): a slice is disputed when its normalized A/B texts diverge
    below the slice-similarity threshold. The
    ``VEMOIZER_DISABLE_CONSENSUS=1`` kill-switch and every failure path
    land on ``[]`` — the run ships decode A alone (fail-open).
    """
    if os.environ.get("VEMOIZER_DISABLE_CONSENSUS") == "1":
        logger.info("consensus disabled by VEMOIZER_DISABLE_CONSENSUS=1")
        return []
    if result_a is None or result_b is None:
        logger.info("disputed spans: 0 (a decode is missing)")
        return []
    slices_a = list(result_a.get("slices") or [])
    slices_b = list(result_b.get("slices") or [])
    spans = find_disputed_slices(slices_a, slices_b)
    if spans is None:
        logger.info("disputed spans: 0 (no comparable slices, re-decode skipped)")
        return []
    speech_seconds = sum(float(s["end_s"]) - float(s["start_s"]) for s in slices_a)
    guarded = apply_span_guardrails(spans, speech_seconds=speech_seconds)
    if guarded is None:
        logger.warning("disputed spans rejected by guardrails; shipping decode A")
        return []
    disputed_s = sum(s.end - s.start for s in guarded)
    fraction = 100.0 * disputed_s / speech_seconds if speech_seconds > 0 else 0.0
    logger.info(
        "disputed spans: %d (%s of audio, %.0f%%)",
        len(guarded),
        format_duration(disputed_s),
        fraction,
    )
    return guarded


def _redecode_spans(
    audio: np.ndarray, spans: list[Span]
) -> list[dict[str, Any]] | None:
    """Re-decode each disputed span; ``None`` when re-decode is unavailable."""
    redecoder = WhisperReDecodeTranscriber()
    progress = StageProgress("re-decode", len(spans), unit="spans")
    try:
        results = []
        for s in spans:
            results.append(redecoder.transcribe_span(audio, s, language=s.language))
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


def _adjudicate(
    span: Span,
    a_text: str,
    candidates: list[Candidate],
    client: LLMClient | None,
    context: str = "",
) -> str:
    """Final text for one disputed span, fail-open down the candidate list."""
    if client is not None:
        try:
            verdict = client.adjudicate(a_text, candidates, context)
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


def _assemble(
    result_a: dict[str, Any] | None,
    result_b: dict[str, Any] | None,
    redecoded: list[dict[str, Any]] | None,
    llm_config: LLMConfig | None,
    speaker_segments: list[tuple[float, float, str]] | None = None,
    spans: list[Span] | None = None,
) -> dict[str, Any]:
    """Combine the stage outputs into the final ``{"text", "segments"}``.

    ``spans`` are the guardrailed disputed spans the caller re-decoded —
    the exact list ``redecoded`` was indexed against, so verdicts and
    re-decode results can never drift apart.

    ``speaker_segments`` (when given) is a list of ``(start, end, speaker)``
    triples from the diarization stage; each adjudicated segment is labelled
    with the speaker whose segment overlaps the disputed span the most. The
    ``speaker`` key is omitted when no speaker segment overlaps, so downstream
    formatters can render the label only when it is actually known.

    The adjudicated verdicts are spliced INTO decode A's sentence segments
    (full coverage), so the transcript files stay whole instead of
    collapsing to the disputed fragments. With zero disputed spans the
    output text is byte-identical to decode A's.
    """
    base = result_a or result_b
    if base is None:
        return {"text": "", "segments": []}

    words = list(base.get("words") or [])
    spans = spans or []
    redecoded = redecoded or []

    client = LLMClient(llm_config) if llm_config is not None else None
    verdicts: list[dict[str, Any]] = []
    # One LLM round-trip per span when adjudication is configured; without a
    # heartbeat this loop is the pipeline's second silent multi-minute stage.
    progress = StageProgress("adjudicate", len(spans), unit="spans")
    try:
        for i, span in enumerate(spans):
            rd = redecoded[i] if i < len(redecoded) else None
            a_text = words_in_span(words, span)
            candidates: list[Candidate] = [
                {"source": "decode A", "text": a_text},
            ]
            b_text = _b_text_in_span(result_b, span)
            if b_text:
                # Span-scoped: only decode B's slice text overlapping the
                # span. The whole decode-B text as a candidate (the old
                # behaviour) fed the adjudicator the entire transcript for
                # every span.
                candidates.append({"source": "decode B", "text": b_text})
            if rd is not None and rd.get("ok"):
                candidates.append({"source": "re-decode", "text": rd["text"]})
            context = span_context(words, span)
            verdict = _adjudicate(span, a_text, candidates, client, context)
            entry: dict[str, Any] = {
                "start": span.start,
                "end": span.end,
                "text": verdict,
            }
            if speaker_segments is not None:
                speaker = speaker_for_span(span.start, span.end, speaker_segments)
                if speaker is not None:
                    entry["speaker"] = speaker
            verdicts.append(entry)
            progress.advance()
    finally:
        progress.done()
        if client is not None:
            client.close()

    verdicts.sort(key=lambda s: s["start"])
    base_text = str(base.get("text", "")).strip()
    sentences = list(base.get("segments") or [])
    if verdicts and not sentences:
        # A backend without sentence segments cannot be spliced; keep the
        # verdict list as the segments (the pre-splice contract).
        return {"text": base_text, "segments": verdicts}
    text, segments = splice_verdicts(base_text, words, sentences, verdicts)
    result: dict[str, Any] = {"text": text, "segments": segments}
    if verdicts:
        result["paragraphs"] = paragraphs(segments)
    return result


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
        result = decode_all(transcriber, slices, f"decode ({backend})")
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


#: Recording profiles: which decode A the pipeline runs. ``dictation`` is
#: the fast per-slice Parakeet path; ``meeting`` decodes the whole file with
#: whisper-large-v3-turbo, which decisively wins on far-field multi-speaker
#: audio (issue #71) and provides word timestamps.
PROFILES = ("dictation", "meeting")


def transcribe_file(
    path: str | Path,
    *,
    config_path: str | None = None,
    diarize: bool = False,
    profile: str = "dictation",
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
    if profile not in PROFILES:
        known = ", ".join(PROFILES)
        raise ValueError(f"unknown profile {profile!r} (known: {known})")
    run_start = time.monotonic()
    logger.info("transcribe: %s (profile: %s)", path, profile)
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
    if profile == "meeting":
        result_a = decode_meeting(audio, slices)
    else:
        try:
            parakeet = ParakeetTranscriber()
            result_a = decode_all(parakeet, slices, "decode A")
        except Exception as e:  # noqa: BLE001 - fail-open stage boundary
            logger.warning("decode A failed, using best available result: %s", e)
        finally:
            if parakeet is not None:
                with suppress(Exception):  # cleanup is best-effort (fail-open)
                    parakeet.cleanup()
    try:
        canary = CanaryTranscriber()
        result_b = decode_all(canary, slices, "decode B")
    except Exception as e:  # noqa: BLE001 - fail-open stage boundary
        logger.warning("decode B failed, using best available result: %s", e)
    finally:
        if canary is not None:
            with suppress(Exception):  # cleanup is best-effort (fail-open)
                canary.cleanup()

    spans = _find_spans(result_a, result_b)
    redecoded: list[dict[str, Any]] | None = None
    if spans:
        redecoded = _redecode_spans(audio, spans)

    speaker_segments: list[tuple[float, float, str]] | None = None
    diarization_ran = False
    if diarize:
        logger.info("diarization: starting")
        diarize_start = time.monotonic()
        speaker_segments = _run_diarization_stage(audio)
        diarization_ran = speaker_segments is not None
        logger.info(
            "diarization: %s speaker segments in %s",
            len(speaker_segments) if speaker_segments is not None else "no",
            format_duration(time.monotonic() - diarize_start),
        )

    logger.info("assemble: adjudicating spans")
    result = _assemble(
        result_a, result_b, redecoded, llm_config, speaker_segments, spans=spans
    )
    if diarization_ran:
        # CC-BY-4.0: the gated pyannote weights require attribution whenever
        # they actually ran; the CLI prints the warnings channel.
        result.setdefault("warnings", []).append(DIARIZATION_ATTRIBUTION)

    if llm_config is not None and result.get("text"):
        notes_start = time.monotonic()
        client = LLMClient(llm_config)
        try:
            notes = generate_notes(client, result["text"])
        finally:
            client.close()
        if notes is not None:
            result["notes"] = notes
            logger.info(
                "notes: generated in %s",
                format_duration(time.monotonic() - notes_start),
            )
        else:
            # The .md file still renders as a clean transcript document;
            # the warning tells the user why it has no summary.
            result.setdefault("warnings", []).append(
                "notes generation failed; the Markdown output has no summary"
            )
            logger.warning("notes: generation failed (transcript unaffected)")

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
