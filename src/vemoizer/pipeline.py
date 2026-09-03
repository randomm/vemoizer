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
from contextlib import suppress
from pathlib import Path
from typing import Any

import numpy as np

# The consensus alignment reuses the stage's cost function and gap penalty so
# the orchestrator's pairing matches ``alignment.dtw_align`` exactly; the
# private import is intentional (same package, one cost definition).
from .alignment import GAP_PENALTY, _pair_cost  # noqa: SLF001
from .audio_contract import SAMPLE_RATE
from .canary_transcriber import CanaryTranscriber
from .ingest import IngestError, ingest_audio
from .llm import LLMClient, LLMConfig, load_config
from .parakeet_transcriber import ParakeetTranscriber
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
WordPairs = list[tuple[dict[str, Any] | None, dict[str, Any] | None]]


def _load_llm_config(path: str | None) -> LLMConfig | None:
    """Load the LLM config; ``None`` (fail-open) when unconfigured."""
    if path is not None:
        return load_config(path)
    for candidate in _DEFAULT_CONFIG_PATHS:
        if candidate.is_file():
            return load_config(candidate)
    return None


def _decode(
    transcriber: Any, audio: np.ndarray, label: str
) -> tuple[dict[str, Any] | None, str | None]:
    """Run one full decode; return ``(result, warning)`` (fail-open).

    The warning is ``None`` on success; a human-readable degraded-consensus
    message when the decode raised, so the caller can surface it to the user
    (issue #36: a silent ``logger.warning`` is not user-visible).
    """
    try:
        return transcriber.transcribe(audio), None
    except Exception as e:  # noqa: BLE001 - fail-open stage boundary
        logger.warning("%s failed, using best available result: %s", label, e)
        return None, (
            f"warning: {label} failed: {e}; "
            "consensus degraded to the best available result"
        )


def _speech_slices(audio: np.ndarray) -> list[tuple[int, np.ndarray]]:
    """VAD-split the recording into ``(offset, slice)`` pairs.

    The offset is the slice's first sample in the full recording, used to
    shift per-slice timestamps back onto the full timeline. Falls back to
    the whole recording as a single slice when VAD is unavailable or finds
    no speech.
    """
    try:
        vad_model = load_vad_model()
        segments: list[SpeechSegment] = vad_segments(audio, vad_model)
    except Exception as e:  # noqa: BLE001 - fail-open stage boundary
        logger.warning("VAD unavailable, decoding full recording: %s", e)
        return [(0, audio)]
    if not segments:
        return [(0, audio)]
    return [(seg.start, audio[seg.start : seg.end]) for seg in segments]


def _decode_all(
    transcriber: Any, slices: list[tuple[int, np.ndarray]], label: str
) -> tuple[dict[str, Any] | None, list[str]]:
    """Decode every VAD slice through *transcriber* and merge the results.

    Word/segment times are shifted onto the full-recording timeline so the
    downstream alignment and re-decode stages work on one time base.
    """
    if not slices:
        return {"text": "", "words": [], "segments": []}, []
    warnings: list[str] = []
    merged_words: list[dict[str, Any]] = []
    merged_segments: list[dict[str, Any]] = []
    texts: list[str] = []
    for offset, slice_audio in slices:
        r, warning = _decode(transcriber, slice_audio, label)
        if warning is not None:
            warnings.append(warning)
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
    return (
        {
            "text": " ".join(t for t in texts if t),
            "words": merged_words,
            "segments": merged_segments,
        },
        warnings,
    )


def _align_decodes(
    result_a: dict[str, Any], result_b: dict[str, Any]
) -> WordPairs | None:
    """DTW-align the two word streams on word onsets.

    Mirrors :func:`vemoizer.alignment.dtw_align` (same cost function and
    diagonal-first tie-breaking) but emits the original word *dicts* (with
    word and timestamps) instead of the word strings, so the disputed-span
    stage can anchor each span to the actual word times. ``None`` when either
    side produced no words or the alignment itself fails (fail-open).
    """
    words_a = result_a.get("words") or []
    words_b = result_b.get("words") or []
    if not words_a or not words_b:
        return None
    try:
        return _dtw_pairs(words_a, words_b)
    except Exception as e:  # noqa: BLE001 - fail-open stage boundary
        logger.warning("alignment failed: %s", e)
        return None


def _dtw_pairs(
    words_a: list[dict[str, Any]], words_b: list[dict[str, Any]]
) -> WordPairs:
    """DTW word-dict alignment (dict-flavoured ``alignment.dtw_align``)."""
    n, m = len(words_a), len(words_b)
    inf = float("inf")
    D: list[list[float]] = [[inf] * (m + 1) for _ in range(n + 1)]
    D[0][0] = 0.0
    for j in range(1, m + 1):
        D[0][j] = D[0][j - 1] + GAP_PENALTY
    for i in range(1, n + 1):
        D[i][0] = D[i - 1][0] + GAP_PENALTY
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            diag = D[i - 1][j - 1] + _pair_cost(words_a, words_b, i - 1, j - 1)
            up = D[i - 1][j] + GAP_PENALTY
            left = D[i][j - 1] + GAP_PENALTY
            best = diag
            if up < best:
                best = up
            if left < best:
                best = left
            D[i][j] = best

    pairs: WordPairs = []
    i, j = n, m
    while i > 0 or j > 0:
        if i == 0:
            pairs.append((None, words_b[j - 1]))
            j -= 1
            continue
        if j == 0:
            pairs.append((words_a[i - 1], None))
            i -= 1
            continue
        diag = D[i - 1][j - 1] + _pair_cost(words_a, words_b, i - 1, j - 1)
        up = D[i - 1][j] + GAP_PENALTY
        left = D[i][j - 1] + GAP_PENALTY
        if diag <= up and diag <= left:
            pairs.append((words_a[i - 1], words_b[j - 1]))
            i -= 1
            j -= 1
        elif up < left:
            pairs.append((words_a[i - 1], None))
            i -= 1
        else:
            pairs.append((None, words_b[j - 1]))
            j -= 1
    pairs.reverse()
    return pairs


def _redecode_spans(
    audio: np.ndarray, spans: list[Span]
) -> list[dict[str, Any]] | None:
    """Re-decode each disputed span; ``None`` when re-decode is unavailable."""
    redecoder = WhisperReDecodeTranscriber()
    try:
        results = [redecoder.transcribe_span(audio, s) for s in spans]
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


def _assemble(
    result_a: dict[str, Any] | None,
    result_b: dict[str, Any] | None,
    redecoded: list[dict[str, Any]] | None,
    llm_config: LLMConfig | None,
) -> dict[str, Any]:
    """Combine the stage outputs into the final ``{"text", "segments"}``."""
    base = result_a or result_b
    if base is None:
        return {"text": "", "segments": []}

    words = list(base.get("words") or [])
    pairs = (
        _align_pairs_safe(result_a, result_b)
        if result_a is not None and result_b is not None
        else None
    )
    spans = find_disputed_spans(pairs) if pairs else []
    redecoded = redecoded or []

    client = LLMClient(llm_config) if llm_config is not None else None
    segments: list[dict[str, Any]] = []
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
        segments.append({"start": span.start, "end": span.end, "text": verdict})

    segments.sort(key=lambda s: s["start"])
    return {"text": str(base.get("text", "")).strip(), "segments": segments}


def transcribe_file(path: str | Path, *, config_path: str | None = None) -> dict:
    """Run the full consensus pipeline over one audio file.

    Args:
        path: Path to the audio file (any ffmpeg-readable container).
        config_path: Optional user config file with an ``[llm]`` section.
            When omitted, ``~/.config/vemoizer/config.toml`` and
            ``~/.vemoizer.toml`` are probed; when no config is found the LLM
            stage is skipped (fail-open).

    Returns:
        ``{"text": str, "segments": list[dict]}`` — the full transcript
        (decode A preferred) plus one segment per disputed span with its
        adjudicated text.
    """
    try:
        audio = ingest_audio(Path(path))
    except IngestError as e:
        logger.error("ingest failed for %s: %s", path, e)
        return {"text": "", "segments": [], "warnings": [], "error": str(e)}
    if len(audio) == 0:
        return {"text": "", "segments": [], "warnings": []}

    llm_config = _load_llm_config(config_path)
    slices = _speech_slices(audio)

    result_a: dict[str, Any] | None = None
    result_b: dict[str, Any] | None = None
    parakeet: Any = None
    canary: Any = None
    warnings: list[str] = []
    try:
        parakeet = ParakeetTranscriber()
        result_a, a_warnings = _decode_all(parakeet, slices, "decode A")
        warnings.extend(a_warnings)
    except Exception as e:  # noqa: BLE001 - fail-open stage boundary
        logger.warning("decode A failed, using best available result: %s", e)
        warnings.append(
            f"warning: decode A (Parakeet) failed: {e}; consensus degraded "
            "to the best available result"
        )
    finally:
        if parakeet is not None:
            with suppress(Exception):  # cleanup is best-effort (fail-open)
                parakeet.cleanup()
    try:
        canary = CanaryTranscriber()
        result_b, b_warnings = _decode_all(canary, slices, "decode B")
        warnings.extend(b_warnings)
    except Exception as e:  # noqa: BLE001 - fail-open stage boundary
        logger.warning("decode B failed, using best available result: %s", e)
        warnings.append(
            f"warning: decode B (Canary) failed: {e}; consensus degraded — "
            "output is Parakeet-only"
        )
    finally:
        if canary is not None:
            with suppress(Exception):  # cleanup is best-effort (fail-open)
                canary.cleanup()

    pairs = _align_pairs_safe(result_a, result_b)
    redecoded: list[dict[str, Any]] | None = None
    if pairs:
        spans = find_disputed_spans(pairs)
        if spans:
            redecoded = _redecode_spans(audio, spans)

    result = _assemble(result_a, result_b, redecoded, llm_config)
    result["warnings"] = warnings
    return result


def _align_pairs_safe(
    result_a: dict[str, Any] | None, result_b: dict[str, Any] | None
) -> WordPairs | None:
    """Alignment wrapper that fails open to ``None``."""
    if result_a is None or result_b is None:
        return None
    try:
        return _align_decodes(result_a, result_b)
    except Exception as e:  # noqa: BLE001 - fail-open stage boundary
        logger.warning("alignment failed: %s", e)
        return None
