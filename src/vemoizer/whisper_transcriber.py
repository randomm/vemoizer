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

On unclear or quiet audio — and especially in English meetings — whisper
can continue the glossary ``initial_prompt`` instead of transcribing:
``Sanasto, Pia, NG-TOPI, …`` (issue #109). The post-decode
:func:`filter_echo_segments` (from :mod:`echo_filter`) drops such echo
segments while keeping real sentences that contain one or more glossary
terms, and is fail-open on any error (it never loses a real segment to a
filter bug).

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

from .echo_filter import echo_vocabulary
from .lang_filter import filter_language_lines
from .models import get_model, resolve_model_path
from .selfheal import heal
from .transcriber import TranscriptionResult
from .whisper_windows import process_window_raws, retry_lost_windows

if TYPE_CHECKING:
    from .progress import ProgressDisplay
    from .progress_shim import WhisperProgress

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
        language: str | None = None,
        initial_prompt: str | None = None,
    ) -> None:
        self.model: Any = None
        self._model_path: str | None = None
        self._mlx_whisper: Any = None
        self._load_failed = False
        self._load_once = threading.Lock()
        # Prompt-derived echo vocabulary for the post-decode filter
        # (issue #109): the glossary terms plus the former label word
        # "Sanasto", case-insensitive. ``None`` (no prompt) means the
        # filter is a no-op. The lower-cased vocabulary set is built once
        # per window inside filter_echo_segments and reused for every
        # segment, so no per-call rebuild here.
        self._echo_terms = echo_vocabulary(initial_prompt)
        # ``language=None`` (the default) lets Whisper detect the language
        # per window (invariant #3: language is a property of a span, not
        # of a file — a hard-coded ``"fi"`` pin here forced Finnish on
        # every window of every meeting, issue #108). A run-level
        # override (``decode_meeting(language="fi")`` / the
        # ``meeting --language`` flag, issue #108) still pins a language
        # explicitly when the user wants it.
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

                # Revision-pinned: never load from the bare repo ID (invariant #4).
                self._model_path = resolve_model_path(MODEL_ID, MODEL_REVISION)
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
        vad_slices: list[tuple[int, int]] | None = None,
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
        # hallucination_silence_threshold is intentionally ABSENT (issue #152):
        # with 30 s per-call windows its silence heuristics are meaningless —
        # the "surrounded by silence" test is always true by construction for
        # a window that spans ~0–30 s — and it deletes real speech. Loops are
        # handled by the temperature ladder, the compression and logprob
        # thresholds, the #109 echo filter and self-heal.
        options: dict[str, Any] = {
            "temperature": (0.0, 0.2, 0.4),
            "condition_on_previous_text": True,
            "compression_ratio_threshold": 2.4,
            "logprob_threshold": -1.0,
            "no_speech_threshold": 0.6,
            "initial_prompt": self._initial_prompt,
        }
        # verbose=False (issue #147): in mlx-whisper 0.4.3 this ENABLES the
        # tqdm bar (the shim intercepts it when a display is active; tqdm
        # auto-suppresses in non-TTY contexts) and SUPPRESSES the per-segment
        # print. It does NOT suppress the per-window "Detected language: X"
        # line (``if verbose is not None:`` is True for False), so that line
        # is filtered by :func:`filter_language_lines` around the decode loop
        # below. The contract test in tests/test_mlx_whisper_contract.py pins
        # the inverted verbose semantics.
        options.update(kwargs)
        options["verbose"] = False

        window_frames = int(WINDOW_SECONDS * SAMPLE_RATE)
        # Common kwargs for every window's transcribe() call (the retry path
        # reuses this dict with ``initial_prompt`` overridden to None, so a
        # signature or window-sizing change is kept in one place, issue #152).
        window_kwargs: dict[str, Any] = {
            "path_or_hf_repo": self._model_path,
            "word_timestamps": True,
            "language": self._language,
            "task": "transcribe",
        }
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
            shim_cm: WhisperProgress | contextlib.AbstractContextManager[None] = (
                with_whisper_progress(
                    display,
                    window_seconds=WINDOW_SECONDS,
                    file_total_minutes=file_total_min,
                )
            )
            mark_window = shim_cm.mark_window
        else:
            shim_cm = contextlib.nullcontext()

            def mark_window(_offset: float) -> None:
                pass

        # Per-window protocol: declare each main-loop window so the shim's
        # factory hands its bar to the display (any other bar created before
        # the next mark is a re-entrant call and gets a no-op bar instead).
        # filter_language_lines (issue #147) suppresses the per-window
        # "Detected language: X" print from stdout; it is scoped to the
        # decode loop and restores sys.stdout on every exit path.
        with shim_cm, filter_language_lines():
            for index, offset in enumerate(range(0, len(audio), window_frames)):
                mark_window(offset / SAMPLE_RATE)
                raw = self._mlx_whisper.transcribe(
                    audio[offset : offset + window_frames],
                    **window_kwargs,
                    **options,
                )
                if raw is None:
                    raise RuntimeError(
                        f"whisper window {index} (offset {offset / SAMPLE_RATE:.0f}s) "
                        "returned None"
                    )
                raws.append(raw)

        # Fail-safe retry (issue #152): a window that VAD says contains
        # speech (or, when VAD is unavailable, the fallback frame-RMS gate)
        # but returned 0 segments is re-decoded once without the glossary
        # prompt. The retry lives in whisper_windows.py (not in this class)
        # so a future non-whisper backend doesn't inherit a whisper-specific
        # contract; it reuses the main loop's window_kwargs + offsets and the
        # same stdout filter (the progress shim is intentionally NOT
        # re-entered — see the comment above the main loop).
        #
        # vad_slices are passed as a kwarg so the Transcriber Protocol is
        # unchanged (optional keyword, default None): other backends and
        # test callers that don't set it fall back to the RMS gate.
        lost_windows = retry_lost_windows(
            raws,
            audio,
            lambda window_audio, opts: self._mlx_whisper.transcribe(
                window_audio,
                **window_kwargs,
                **{**options, **opts},
            ),
            window_frames=window_frames,
            window_seconds=WINDOW_SECONDS,
            vad_slices=vad_slices,
        )
        transcribe_time = time.time() - start
        audio_duration = len(audio) / SAMPLE_RATE

        result: TranscriptionResult = process_window_raws(
            raws,
            offset_s_per_window=WINDOW_SECONDS,
            echo_terms=self._echo_terms,
            transcribe_time=transcribe_time,
            audio_duration=audio_duration,
        )
        if lost_windows:
            # Quality-report signal for the TranscriptionResult contract
            # (issue #152, see transcriber.py): windows that contained
            # speech but returned 0 segments even after the prompt-free
            # retry.
            result["lost_windows"] = lost_windows
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
    language: str | None = None,
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

    ``language`` (issue #108) is a run-level recognition-language
    override: ``None`` (the default) leaves per-window language detection
    on — Whisper detects per window, matching invariant #3 (language is a
    property of a span, not of a file). A non-None value (e.g. ``"fi"``)
    pins every window to that language.
    """
    transcriber: WhisperTranscriber | None = None
    try:
        transcriber = WhisperTranscriber(
            language=language, initial_prompt=initial_prompt
        )
        # Widen from the TranscriptionResult TypedDict: the slice records are
        # a pipeline-internal extension, not part of the transcriber contract.
        #
        # vad_slices: convert VAD ``slices`` (``(offset, slice_audio)``)
        # to ``(start_sample, end_sample)`` pairs so the lost-window retry
        # gate can use the real VAD signal (issue #152 FIX 1). The windows
        # are cut from the FULL audio (not the VAD-sliced audio), so the
        # sample offsets map directly onto the recording timeline.
        vad_slices = [(s, s + len(a)) for s, a in slices]
        result: dict[str, Any] = dict(
            transcriber.transcribe(audio, display=display, vad_slices=vad_slices)
        )
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
        lang_summary = result.get("language_summary")
        lang_suffix = f", languages: {lang_summary}" if lang_summary else ""
        logger.info(
            "decode A (whisper): %d chars, %d words, %.1fx realtime%s",
            len(result.get("text", "")),
            len(result.get("words") or []),
            1.0 / rtf if rtf else 0.0,
            lang_suffix,
        )
        return result
    except Exception as e:  # noqa: BLE001 - fail-open stage boundary
        logger.warning("decode A (whisper) failed, using best available: %s", e)
        return None
    finally:
        if transcriber is not None:
            with suppress(Exception):  # cleanup is best-effort (fail-open)
                transcriber.cleanup()
