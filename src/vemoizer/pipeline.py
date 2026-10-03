"""Consensus pipeline orchestrator (issue #34).

Chains every stage into one end-to-end run:

    ingest -> VAD -> decode A (Parakeet) -> decode B (Canary)
           -> DTW align -> disputed spans -> re-decode (Whisper)
           -> LLM adjudication -> assembled transcript

Fail-open at every stage: a stage failure degrades to the best available
result rather than aborting the run. VAD splits long recordings so decodes
stay bounded (fail-open: the whole recording as one slice); per-slice
timestamps are shifted onto the full timeline. Models load lazily and are
released via ``cleanup()`` on every exit path.
"""

from __future__ import annotations

import logging
import time
from contextlib import suppress
from pathlib import Path
from typing import Any

import numpy as np

from .assembly import Candidate, _adjudicate, _b_text_in_span, _find_spans
from .audio_contract import SAMPLE_RATE
from .canary_transcriber import CanaryTranscriber
from .confidence import flag_suspect_segments
from .decode_stage import decode_all
from .diarization import ATTRIBUTION as DIARIZATION_ATTRIBUTION
from .diarization import SpeakerCount, run_diarization_stage, speaker_for_span
from .glossary import (
    apply_corrections,
    glossary_prompt,
    load_corrections,
    load_glossary,
)
from .ingest import IngestError, ingest_audio
from .llm import LLMClient, LLMConfig
from .llm_config import (
    _parse_meeting_language,
    _parse_section_language,
    load_default_config,
)
from .llm_tail import apply_llm_tail
from .notes import generate_notes  # noqa: F401
from .parakeet_transcriber import ParakeetTranscriber
from .presets import _normalize_language
from .progress import ProgressDisplay, StageProgress, format_duration
from .readability import paragraphs, splice_verdicts, tidy_paragraphs
from .redecode import WhisperReDecodeTranscriber
from .repair import repair_paragraphs  # noqa: F401
from .spans import Span, span_context, words_in_span
from .speaker_align import assign_word_speakers, split_segments_at_speaker_changes
from .vad import SpeechSegment, vad_segments
from .vad import load_model as load_vad_model
from .whisper_transcriber import decode_meeting

logger = logging.getLogger(__name__)


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
    ``speaker`` key is omitted when no speaker segment overlaps.

    The adjudicated verdicts are spliced INTO decode A's sentence segments
    (full coverage); with zero disputed spans the output text is
    byte-identical to decode A's.
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
    # Whisper segments carry avg_logprob; low-confidence regions get a
    # suspect flag that survives into paragraphs and the rendered output.
    sentences = flag_suspect_segments(list(base.get("segments") or []))
    if speaker_segments and words:
        # Word-level attribution: split whisper segments at true speaker
        # boundaries so a Q&A exchange inside one segment cannot fuse
        # under a single label (issue #71 round 2).
        labels = assign_word_speakers(words, speaker_segments)
        sentences = split_segments_at_speaker_changes(sentences, words, labels)
    if verdicts and not sentences:
        # A backend without sentence segments cannot be spliced; keep the
        # verdict list as the segments (the pre-splice contract).
        return {"text": base_text, "segments": verdicts}
    text, segments = splice_verdicts(base_text, words, sentences, verdicts)
    if speaker_segments is not None:
        # Speakers attach to EVERY segment, not only the adjudicated ones:
        # the whisper-only meeting path has zero verdicts, and diarization
        # that ran must never be thrown away (issue #71 QA regression).
        for segment in segments:
            speaker = speaker_for_span(
                float(segment["start"]), float(segment["end"]), speaker_segments
            )
            if speaker is not None:
                segment["speaker"] = speaker
    result: dict[str, Any] = {"text": text, "segments": segments}
    if segments:
        result["paragraphs"] = tidy_paragraphs(paragraphs(segments))
    return result


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
    repair: bool = False,
    glossary_path: str | None = None,
    speakers: SpeakerCount | None = None,
    display: ProgressDisplay | None = None,
    language: str | None = None,
) -> dict:
    """Run the full consensus pipeline over one audio file.

    Args:
        path: Audio file path (any ffmpeg-readable container).
        config_path: Optional ``[llm]`` config file; omitted → layered
            search (``~/.vemoizer``, legacy paths), fail-open.
        diarize: Run the pyannote diarization stage (opt-in, off by
            default) and label each disputed segment with the speaker whose
            segment overlaps it the most; any failure is swallowed
            (fail-open, no speaker labels).
        display: Optional :class:`~vemoizer.progress.ProgressDisplay`
            (issue #105) threaded from the CLI/batch layer; when given and
            the profile is ``meeting``, it is passed to ``decode_meeting``
            so the mlx-whisper tqdm shim drives the display's decode task.
            ``None`` (the default) keeps every existing call site unchanged.
        language: Optional run-level recognition-language override for the
            meeting decode (issue #108): one of ``"fi"`` / ``"en"`` pins
            every decode window; ``None`` (the default) or ``"auto"``
            leaves per-window detection on. Ignored by the dictation
            profile.

    Returns:
        ``{"text": str, "segments": list[dict]}`` — the full transcript
        (decode A preferred) plus one segment per disputed span, each with
        an optional ``speaker`` key when diarization labelled it. M6 also
        returns ``duration_s`` (decoded-audio seconds, never ffprobe) and
        ``language`` (``"fi"`` / ``"en"`` from the config layer).
    """
    if profile not in PROFILES:
        known = ", ".join(PROFILES)
        raise ValueError(f"unknown profile {profile!r} (known: {known})")

    # Inline preflight (issue #79): ~2s, fully local (no network) — ffmpeg,
    # config parse, all pinned models cached, and the HF token when the
    # meeting profile runs or diarization is requested (spec d4: the token
    # check fires when profile=="meeting" OR diarize; a meeting run with
    # --no-diarize still goes through the gated-model path).  A red check
    # aborts before a single decode, so a gated pyannote model cannot fail
    # open into a long run with no speaker labels (M1).
    from .preflight import preflight_gate

    gate = preflight_gate(
        diarize=diarize, profile=profile, echo=lambda line: logger.error(line)
    )
    if gate is not None:
        return gate

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

    # One config read per run (issue #108 review): ``load_default_config``
    # resolves the file under the correct contract for *config_path* (an
    # explicit path stays fail-open per issue #82; an omitted path runs the
    # strict project/home layer, fail-open legacy, so a malformed project
    # config still fails loud via the batch layer's pre-check) and its
    # parsed raw dict feeds the ``[meeting]`` recognition-language override
    # and the cosmetic section language below.
    llm_config, config_raw = load_default_config(config_path)
    # Issue #108, option B: an explicit run-level recognition-language
    # choice (CLI ``--language`` / presets ``RunOptions.language``),
    # else the ``[meeting] language`` key of the same config file, else
    # per-window detection (``None``). ``"auto"`` and other non-codes
    # never pin — ``transcribe_file`` is the single coercion point.
    meeting_language: str | None
    if language is not None and language != "auto":
        meeting_language = language
    else:
        if isinstance(config_raw, dict):
            meeting_section = config_raw.get("meeting")
        else:
            meeting_section = None
        configured = _parse_meeting_language(meeting_section)
        meeting_language = None if configured == "auto" else configured
    # M6 (issue #75): duration (decoded-audio, never ffprobe) and section
    # language ride on the run dict — format_md / the report read them.
    result: dict[str, Any] = {
        "duration_s": len(audio) / SAMPLE_RATE,
        "language": _normalize_language(_parse_section_language(config_raw)),
    }
    logger.info(
        "LLM adjudication: %s", "configured" if llm_config is not None else "disabled"
    )
    slices = _speech_slices(audio)

    result_a: dict[str, Any] | None = None
    result_b: dict[str, Any] | None = None
    parakeet: Any = None
    canary: Any = None
    run_consensus = True
    meeting_fallback = False
    glossary = load_glossary(glossary_path)
    corrections = load_corrections(glossary_path)
    if profile == "meeting":
        kwargs: dict[str, Any] = {
            "initial_prompt": glossary_prompt(glossary),
            "language": meeting_language,
        }
        if display is not None:
            kwargs["display"] = display
        result_a = decode_meeting(audio, slices, **kwargs)
        if result_a is not None:
            # Meeting profile skips decode B / re-decode / adjudication
            # entirely (invariant #2 allows skip-by-flag; see issue #71).
            run_consensus = False
        else:
            logger.warning("meeting decode failed; falling back to dictation path")
            meeting_fallback = True
    if result_a is None:
        try:
            parakeet = ParakeetTranscriber()
            result_a = decode_all(parakeet, slices, "decode A")
        except Exception as e:  # noqa: BLE001 - fail-open stage boundary
            logger.warning("decode A failed, using best available result: %s", e)
        finally:
            if parakeet is not None:
                with suppress(Exception):  # cleanup is best-effort (fail-open)
                    parakeet.cleanup()
    if run_consensus:
        try:
            canary = CanaryTranscriber()
            result_b = decode_all(canary, slices, "decode B")
        except Exception as e:  # noqa: BLE001 - fail-open stage boundary
            logger.warning("decode B failed, using best available result: %s", e)
        finally:
            if canary is not None:
                with suppress(Exception):  # cleanup is best-effort (fail-open)
                    canary.cleanup()

    spans = _find_spans(result_a, result_b) if run_consensus else []
    redecoded: list[dict[str, Any]] | None = None
    if spans:
        redecoded = _redecode_spans(audio, spans)

    speaker_segments: list[tuple[float, float, str]] | None = None
    diarization_ran = False
    if diarize:
        logger.info("diarization: starting")
        diarize_start = time.monotonic()
        speaker_segments = run_diarization_stage(audio, speakers)
        diarization_ran = speaker_segments is not None
        logger.info(
            "diarization: %s speaker segments in %s",
            len(speaker_segments) if speaker_segments is not None else "no",
            format_duration(time.monotonic() - diarize_start),
        )

    # Fail loud (issue #73/#78): decode A's total failure (None) must not
    # look like a successful empty transcript — an "error" key is the only
    # thing the CLI and callers key on. Partial failures (some slices dead)
    # still ship the merged slices.
    if result_a is None:
        logger.error("decode A produced no output for any of %d slices", len(slices))
        result["text"] = ""
        result["segments"] = []
        result["error"] = (
            "decode A produced no output for any of "
            + str(len(slices))
            + " slices (model may have failed to load)"
        )
        return result
    logger.info("assemble: adjudicating spans")
    assembled = _assemble(
        result_a, result_b, redecoded, llm_config, speaker_segments, spans=spans
    )
    result.update(assembled)
    if corrections and result.get("paragraphs"):
        # Deterministic known-garble replacement: "Blacksit" -> "Flagship"
        # must never depend on a model's judgment.
        result["paragraphs"] = apply_corrections(result["paragraphs"], corrections)
    if diarization_ran:
        # CC-BY-4.0: the gated pyannote weights require attribution whenever
        # they actually ran; the CLI prints the warnings channel.
        result.setdefault("warnings", []).append(DIARIZATION_ATTRIBUTION)
    if diarize and not diarization_ran:
        # Fail-open: the transcript still ships, but the user must be
        # told the labels are missing (issue #78).
        result.setdefault("warnings", []).append(
            "diarization failed; continuing without speaker labels"
        )
    if meeting_fallback:
        # The dictation decode above succeeded (a total failure already
        # returned an error), so tell the user they got the fallback
        # transcript, not the meeting-profile read (issue #78).
        result.setdefault("warnings", []).append(
            "meeting decode failed; fell back to dictation path"
        )

    # LLM tail; fn/cls args use this module's names for test monkeypatching
    apply_llm_tail(
        result,
        llm_config,
        repair=repair,
        corrections=corrections,
        glossary=glossary,
        generate_notes_fn=generate_notes,
        repair_paragraphs_fn=repair_paragraphs,
        llm_client_cls=LLMClient,
    )
    logger.info(
        "transcribe: done in %s — %d chars, %d segments",
        format_duration(time.monotonic() - run_start),
        len(result.get("text", "")),
        len(result.get("segments", [])),
    )
    return result
