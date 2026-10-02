"""Whisper-large-v3-turbo transcriber — decode A for the meeting profile (issue #71).

On far-field multi-speaker meeting audio, whisper-large-v3-turbo decisively
outperforms both Parakeet and Canary (measured on the reference 4-person
meeting: it recovers "Siemensin logiikoista" and "RFID-lukijat" where both
garble) at ~23x realtime, with word timestamps.

To keep the glossary in every decoding window, the recording is decoded in
:data:`WINDOW_SECONDS` windows: each window is its own
``mlx_whisper.transcribe`` call, so the ``initial_prompt`` seeds it from
window 1 (issue #76: a single whole-file call slides the prompt past
mlx-whisper's 223-token keep-window after ~1 minute of rolling context).
Each window is short, so its own rolling context stays inside the
keep-window. Per-VAD-slice records for the dispute stage are derived from
the word timestamps (:func:`slice_records_from_words`).

Spike (see the issue #76 spike report): the per-window loop costs ~+40%
wall-clock vs the single call (12.1 vs 8.7 min/hour measured on 15 min of
synthetic audio), which is the price of the glossary actually reaching
later windows.

Model loading is lazy and revision-pinned (invariant #4); a failed resolve
latches so hundreds of calls never re-attempt a broken download; language
comes from Whisper's own detection, reported per run (invariant #3 is
honored downstream, where per-slice language from decode B wins on spans).
"""

from __future__ import annotations

import contextlib
import logging
import threading
import time
from contextlib import suppress
from typing import TYPE_CHECKING, Any

import mlx.core as mx
import numpy as np

from .models import get_model
from .selfheal import heal
from .transcriber import TranscriptionResult

if TYPE_CHECKING:
    from .progress import ProgressDisplay

logger = logging.getLogger(__name__)

#: The MLX community conversion of OpenAI's whisper-large-v3-turbo,
#: read from the central registry so no repo/SHA pair lives in two places
#: (issue #79); the drift test in tests/test_cli_models.py pins it.
MODEL_ID = get_model("whisper-turbo").repo_id
MODEL_REVISION = get_model("whisper-turbo").revision

#: Audio contract (project invariant #6): 16 kHz mono float32.
SAMPLE_RATE = 16_000

#: Decode window length in seconds. Each window is one transcribe() call so
#: the glossary initial_prompt re-seeds it; shorter windows keep the
#: per-window rolling context inside the prompt keep-window, at the cost of
#: boundary artifacts (mitigated by whisper's own segmentation).
WINDOW_SECONDS = 30.0


class WhisperTranscriber:
    """Whisper-large-v3-turbo speech-to-text via mlx-whisper (decode A)."""

    def __init__(
        self,
        language: str | None = "fi",
        initial_prompt: str | None = None,
    ) -> None:
        self.model: Any = None
        self._model_path: str | None = None
        self._mlx_whisper: Any = None
        self._load_failed = False
        self._load_once = threading.Lock()
        self._language = language
        # Seeds every decoding window with the user's vocabulary — the fix
        # for garbled proper nouns ("FLAG-sit" for Flagship-hanke).
        self._initial_prompt = initial_prompt

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

    def transcribe(
        self,
        audio: np.ndarray,
        *,
        display: ProgressDisplay | None = None,
        **kwargs: Any,
    ) -> TranscriptionResult:
        """Transcribe the recording in :data:`WINDOW_SECONDS` windows.

        Each window is a separate ``mlx_whisper.transcribe`` call (16 kHz
        mono float32 in, recording-timeline timestamps out): a single
        whole-file call would let the rolling context slide the glossary
        out of the prompt keep-window, so the prompt must re-seed at
        every window boundary. The self-heal kwargs override still works:
        ``transcribe(chunk, condition_on_previous_text=False)`` and
        ``transcribe(chunk, initial_prompt=None)`` re-encode a single
        slice.

        ``display`` is an optional :class:`ProgressDisplay` (issue #105):
        when given, the mlx-whisper tqdm shim wraps the whole window loop
        so the bar's frame counter drives the display's decode task in
        file-level minutes. When None or the display is disabled (non-TTY),
        the shim is a pass-through and the decode result is identical.
        """
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
        # Rolling context ON by default: mlx-whisper resets the prompt only
        # on temperature > 0.5 or conditioning off (verified in 0.4.3
        # source), and the fallback ladder + thresholds are whisper's
        # designed anti-loop mechanism. Because every window is its own
        # transcribe() call, the glossary re-seeds at each boundary;
        # when the ladder still fails, the self-heal stage re-decodes the
        # wall with conditioning off via the kwargs override.
        options: dict[str, Any] = {
            "temperature": (0.0, 0.2, 0.4),
            "condition_on_previous_text": True,
            "compression_ratio_threshold": 2.4,
            "logprob_threshold": -1.0,
            "no_speech_threshold": 0.6,
            "hallucination_silence_threshold": 2.0,
            "initial_prompt": self._initial_prompt,
        }
        # In mlx-whisper 0.4.3 verbose=False ENABLES the real tqdm bar and
        # SUPPRESSES the per-segment print (inverted vs upstream whisper);
        # the contract test in tests/test_mlx_whisper_contract.py pins
        # both. When the display is active the effective verbose is forced
        # to False — even if the caller passed verbose=True or None (an
        # accidental or deliberate override must never print transcript
        # text to stdout under an active display, and the bar is owned by
        # the shim anyway). On every other path the library default
        # (verbose=None: no bar, no print) is left intact, and a caller's
        # own verbose is never overwritten.
        options.update(kwargs)
        if display is not None and not display.disable:
            options["verbose"] = False

        window_frames = int(WINDOW_SECONDS * SAMPLE_RATE)
        raws: list[dict[str, Any]] = []
        # The display (when threaded in) is driven by the shim: it patches
        # the tqdm referenced by mlx_whisper.transcribe for the duration of
        # the window loop, so every window's frame counter becomes
        # file-level minutes on the display's decode task. When display is
        # None or disabled (non-TTY) the shim is a pass-through and the
        # decode result is identical.
        from .progress_shim import with_whisper_progress

        file_total_min = (len(audio) / SAMPLE_RATE) / 60.0
        use_shim = display is not None and not display.disable
        if use_shim:
            shim_cm: Any = with_whisper_progress(
                display,
                window_seconds=WINDOW_SECONDS,
                file_total_minutes=file_total_min,
            )
            mark_window = shim_cm.mark_window
        else:
            shim_cm = contextlib.nullcontext()

            def mark_window(_offset: float) -> None:
                pass

        # Per-window protocol: declare each main-loop window so the shim's
        # factory hands its bar to the display (any other bar created before
        # the next mark is a re-entrant call and gets a no-op bar instead).
        with shim_cm:
            for index, offset in enumerate(range(0, len(audio), window_frames)):
                mark_window(offset / SAMPLE_RATE)
                raw = self._mlx_whisper.transcribe(
                    audio[offset : offset + window_frames],
                    path_or_hf_repo=self._model_path,
                    word_timestamps=True,
                    language=self._language,
                    task="transcribe",
                    **options,
                )
                if raw is None:
                    raise RuntimeError(
                        f"whisper window {index} (offset {offset / SAMPLE_RATE:.0f}s) "
                        "returned None"
                    )
                raws.append(raw)
        transcribe_time = time.time() - start
        audio_duration = len(audio) / SAMPLE_RATE

        words: list[dict[str, Any]] = []
        segments: list[dict[str, Any]] = []
        for index, raw in enumerate(raws):
            offset_s = index * WINDOW_SECONDS
            # A non-empty window that decoded to zero segments (malformed
            # payload, or the model hearing nothing) would otherwise flow
            # into the fail-open path in decode_meeting indistinguishable
            # from a model failure; make the degradation observable.
            if not raw.get("segments"):
                # Per-window, this fires once a window's worth of audio
                # produced nothing; a long mostly-silent recording would
                # spam a warning per window, so the routine case stays at
                # debug (a fully empty decode still surfaces through the
                # fail-open path in decode_meeting).
                logger.debug(
                    "whisper window %d (offset %.0fs) returned no segments; "
                    "transcript may be incomplete",
                    index,
                    offset_s,
                )
            for seg in raw.get("segments") or []:
                text = str(seg.get("text", "")).strip()
                if not text:
                    continue
                entry: dict[str, Any] = {
                    "start": float(seg.get("start", 0.0)) + offset_s,
                    "end": float(seg.get("end", 0.0)) + offset_s,
                    "text": text,
                }
                # Per-segment confidence feeds the suspect-region flagging;
                # discarding it (the old behaviour) threw away whisper's own
                # signal about hallucination and garble.
                for key in ("avg_logprob", "no_speech_prob", "compression_ratio"):
                    if seg.get(key) is not None:
                        entry[key] = float(seg[key])
                segments.append(entry)
                for w in seg.get("words") or []:
                    word = str(w.get("word", "")).strip()
                    if word:
                        words.append(
                            {
                                "word": word,
                                "start": float(w.get("start", 0.0)) + offset_s,
                                "end": float(w.get("end", 0.0)) + offset_s,
                            }
                        )

        result: TranscriptionResult = {
            "text": " ".join(str(raw["text"]).strip() for raw in raws).strip(),
            "words": words,
            "segments": segments,
            "transcribe_time": transcribe_time,
            "audio_duration": audio_duration,
            "rtf": transcribe_time / audio_duration if audio_duration > 0 else 0.0,
        }
        # Language detection is redundant across windows (same model, same
        # audio); take the first non-empty one.
        languages = {str(raw["language"]) for raw in raws if raw.get("language")}
        if len(languages) == 1:
            result["language"] = next(iter(languages))
        elif languages:
            logger.warning(
                "window language disagreement %s; not attributing a run "
                "language (per-slice language from decode B wins downstream)",
                sorted(languages),
            )
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
    audio: np.ndarray,
    slices: list[tuple[int, np.ndarray]],
    initial_prompt: str | None = None,
    display: ProgressDisplay | None = None,
) -> dict[str, Any] | None:
    """Per-window Whisper decode A for the meeting profile (fail-open).

    The recording is decoded in :data:`WINDOW_SECONDS` windows (each its
    own transcribe() call so the glossary re-seeds every window); the
    per-slice records the dispute stage needs are derived from the word
    timestamps.

    ``display`` is an optional :class:`ProgressDisplay` (issue #105)
    threaded from the CLI/batch layer: when given, the mlx-whisper tqdm
    shim wraps the whole window loop so the bar's frame counter drives the
    display's decode task in file-level minutes. When None or the display
    is disabled (non-TTY), the shim is a pass-through and the decode result
    is identical.
    """
    transcriber: WhisperTranscriber | None = None
    try:
        transcriber = WhisperTranscriber(initial_prompt=initial_prompt)
        # Widen from the TranscriptionResult TypedDict: the slice records are
        # a pipeline-internal extension, not part of the transcriber contract.
        result: dict[str, Any] = dict(transcriber.transcribe(audio, display=display))
        # Hallucination walls (context-fed repetition loops) are repaired
        # by re-decoding only the slices under them with conditioning off;
        # heal() is a no-op on a clean decode and fail-open otherwise.
        # The fallback drops the glossary prompt: when the wall is the
        # prompt itself being echoed, re-sending it reproduces the loop.
        result = heal(
            result,
            slices,
            lambda chunk: dict(
                transcriber.transcribe(chunk, condition_on_previous_text=False)
            ),
            fallback=lambda chunk: dict(
                transcriber.transcribe(
                    chunk, condition_on_previous_text=False, initial_prompt=None
                )
            ),
        )
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
