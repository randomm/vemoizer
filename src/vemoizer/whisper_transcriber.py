"""Whisper-large-v3-turbo transcriber — decode A for the meeting profile (issue #71).

On far-field multi-speaker meeting audio, whisper-large-v3-turbo decisively
outperforms both Parakeet and Canary (measured on the reference 4-person
meeting: it recovers "Siemensin logiikoista" and "RFID-lukijat" where both
garble) at ~23x realtime, with word timestamps.

Unlike the per-slice decoders, this transcriber feeds the WHOLE recording
to one ``mlx_whisper.transcribe`` call — mlx-whisper windows internally,
and long-window decoding is exactly where Whisper beats slice-by-slice
decoding (fewer boundary artifacts, better context). Per-VAD-slice records
for the dispute stage are derived afterwards from the word timestamps
(:func:`slice_records_from_words`).

Model loading is lazy and revision-pinned (invariant #4); a failed resolve
latches so hundreds of calls never re-attempt a broken download; language
comes from Whisper's own detection, reported per run (invariant #3 is
honored downstream, where per-slice language from decode B wins on spans).
"""

from __future__ import annotations

import logging
import threading
import time
from contextlib import suppress
from typing import Any

import mlx.core as mx
import numpy as np

from .transcriber import TranscriptionResult

logger = logging.getLogger(__name__)

#: The MLX community conversion of OpenAI's whisper-large-v3-turbo.
MODEL_ID = "mlx-community/whisper-large-v3-turbo"
MODEL_REVISION = "a4aaeec0636e6fef84abdcbe3544cb2bf7e9f6fb"

#: Audio contract (project invariant #6): 16 kHz mono float32.
SAMPLE_RATE = 16_000


class WhisperTranscriber:
    """Whisper-large-v3-turbo speech-to-text via mlx-whisper (decode A)."""

    def __init__(self, language: str | None = "fi") -> None:
        self.model: Any = None
        self._model_path: str | None = None
        self._mlx_whisper: Any = None
        self._load_failed = False
        self._load_once = threading.Lock()
        self._language = language

    def _load_model(self) -> None:
        """Resolve the revision-pinned model path once (latch on failure)."""
        with self._load_once:
            if self._model_path is not None:
                return
            if self._load_failed:
                raise RuntimeError("Whisper model failed to load (not retrying)")
            logger.info("Loading Whisper model: %s@%s", MODEL_ID, MODEL_REVISION)
            start = time.time()
            try:
                import mlx_whisper
                from huggingface_hub import snapshot_download

                # Revision-pinned: never load from the bare repo ID (invariant #4).
                self._model_path = snapshot_download(MODEL_ID, revision=MODEL_REVISION)
                self._mlx_whisper = mlx_whisper
                # Marker: the real weights live in mlx-whisper's ModelHolder
                # cache once the first transcribe runs.
                self.model = self._model_path
            except Exception as e:
                logger.error("Failed to load Whisper model: %s", e)
                self._load_failed = True
                raise RuntimeError(f"Whisper model failed to load: {e}") from e
            logger.info("Whisper model resolved in %.2fs", time.time() - start)

    def transcribe(self, audio: np.ndarray, **kwargs: Any) -> TranscriptionResult:
        """Transcribe the whole recording in one call (16 kHz mono float32)."""
        if len(audio) == 0:
            return {
                "text": "",
                "words": [],
                "segments": [],
                "transcribe_time": 0.0,
                "audio_duration": 0.0,
                "rtf": 0.0,
            }
        self._load_model()
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32)

        start = time.time()
        raw = self._mlx_whisper.transcribe(
            audio,
            path_or_hf_repo=self._model_path,
            word_timestamps=True,
            language=self._language,
            task="transcribe",
            # Deterministic; never condition across windows (the classic
            # Whisper repetition-loop trigger on long recordings).
            temperature=0.0,
            condition_on_previous_text=False,
        )
        transcribe_time = time.time() - start
        audio_duration = len(audio) / SAMPLE_RATE

        words: list[dict[str, Any]] = []
        segments: list[dict[str, Any]] = []
        for seg in raw.get("segments") or []:
            text = str(seg.get("text", "")).strip()
            if text:
                segments.append(
                    {
                        "start": float(seg.get("start", 0.0)),
                        "end": float(seg.get("end", 0.0)),
                        "text": text,
                    }
                )
            for w in seg.get("words") or []:
                word = str(w.get("word", "")).strip()
                if word:
                    words.append(
                        {
                            "word": word,
                            "start": float(w.get("start", 0.0)),
                            "end": float(w.get("end", 0.0)),
                        }
                    )

        result: TranscriptionResult = {
            "text": str(raw.get("text", "")).strip(),
            "words": words,
            "segments": segments,
            "transcribe_time": transcribe_time,
            "audio_duration": audio_duration,
            "rtf": transcribe_time / audio_duration if audio_duration > 0 else 0.0,
        }
        language = raw.get("language")
        if language:
            result["language"] = str(language)
        return result

    def cleanup(self) -> None:
        """Release the model, including mlx-whisper's own cache."""
        if self._mlx_whisper is not None:
            try:
                import importlib

                tr_mod = importlib.import_module("mlx_whisper.transcribe")
                tr_mod.ModelHolder.model = None
                tr_mod.ModelHolder.model_path = None
            except Exception:  # noqa: BLE001,S110 - best-effort cache release
                pass
        self.model = None
        self._model_path = None
        self._mlx_whisper = None
        mx.clear_cache()


def slice_records_from_words(
    words: list[dict[str, Any]],
    slices: list[tuple[int, np.ndarray]],
    *,
    language: str | None,
) -> list[dict[str, Any]]:
    """Per-VAD-slice records for the dispute stage, from whole-file words.

    The dispute detector compares per-slice texts between decode A and B.
    Decode B produces slice records natively (it decodes per slice); this
    derives decode A's from the whole-file word timestamps: a slice's text
    is the words whose start falls inside its bounds. A silent slice gets
    an empty-text record — "whisper heard nothing here" must be visible to
    the dispute stage, not indistinguishable from a missing slice.
    """
    records: list[dict[str, Any]] = []
    for index, (offset, slice_audio) in enumerate(slices):
        start_s = offset / SAMPLE_RATE
        end_s = start_s + len(slice_audio) / SAMPLE_RATE
        slice_words = [
            w for w in words if start_s <= float(w.get("start", 0.0)) < end_s
        ]
        record: dict[str, Any] = {
            "index": index,
            "start_s": start_s,
            "end_s": end_s,
            "text": " ".join(str(w.get("word", "")) for w in slice_words).strip(),
            "words": slice_words,
        }
        if language is not None:
            record["language"] = language
        records.append(record)
    return records


def decode_meeting(
    audio: np.ndarray, slices: list[tuple[int, np.ndarray]]
) -> dict[str, Any] | None:
    """Whole-file Whisper decode A for the meeting profile (fail-open).

    One transcribe call over the full recording (mlx-whisper windows
    internally; per-slice calls would forfeit Whisper's long-window
    strength and pay 1000+ fixed overheads). The per-slice records the
    dispute stage needs are derived from the word timestamps.
    """
    transcriber: WhisperTranscriber | None = None
    try:
        transcriber = WhisperTranscriber()
        # Widen from the TranscriptionResult TypedDict: the slice records are
        # a pipeline-internal extension, not part of the transcriber contract.
        result: dict[str, Any] = dict(transcriber.transcribe(audio))
        result["slices"] = slice_records_from_words(
            list(result.get("words") or []), slices, language=result.get("language")
        )
        rtf = result.get("rtf") or 0.0
        logger.info(
            "decode A (whisper): %d chars, %d words, %.1fx realtime",
            len(result.get("text", "")),
            len(result.get("words") or []),
            1.0 / rtf if rtf else 0.0,
        )
        return result
    except Exception as e:  # noqa: BLE001 - fail-open stage boundary
        logger.warning("decode A (whisper) failed, using best available: %s", e)
        return None
    finally:
        if transcriber is not None:
            with suppress(Exception):  # cleanup is best-effort (fail-open)
                transcriber.cleanup()
