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
from .models import get_model
from .selfheal import heal
from .transcriber import TranscriptionResult
from .whisper_windows import process_window_raws

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
                self._model_path = _pinned_download(MODEL_ID, MODEL_REVISION)
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
        # Always pass verbose=False (issue #147): in mlx-whisper 0.4.3
        # verbose=False SUPPRESSES the per-window "Detected language: X"
        # print (the ~30 identical lines that scroll the progress display
        # away) and ENABLES the tqdm bar (the shim intercepts it when a
        # display is active; tqdm auto-suppresses in non-TTY contexts).
        # The contract test in tests/test_mlx_whisper_contract.py pins
        # the inverted verbose semantics.
        options.update(kwargs)
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

        result: TranscriptionResult = process_window_raws(
            raws,
            offset_s_per_window=WINDOW_SECONDS,
            echo_terms=self._echo_terms,
            transcribe_time=transcribe_time,
            audio_duration=audio_duration,
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
        lang_summary = result.get("language_summary")
        if lang_summary:
            logger.info(
                "decode A (whisper): %d chars, %d words, %.1fx realtime, languages: %s",
                len(result.get("text", "")),
                len(result.get("words") or []),
                1.0 / rtf if rtf else 0.0,
                lang_summary,
            )
        else:
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


def _pinned_download(repo_id: str, revision: str) -> str:
    """Revision-pinned ``snapshot_download`` with cached-bar suppression (issue #147).

    The ``vemoizer meeting`` / ``memo`` command loads its model at run start
    (not via ``models pull``). When the snapshot is already in the HF cache
    (the routine case) ``snapshot_download`` still emits three huggingface_hub
    tqdm bars ("Fetching", "Download complete", "Reconstruction complete")
    that scroll the rich progress display away. Suppression is therefore
    scoped: when the snapshot is already local, ``disable_progress_bars()``
    silences the three bars; when a real download happens (first run, cache
    miss, ``HF_HUB_OFFLINE=1`` cache miss), the bar is left intact because a
    multi-GB silent download looks like a hang. The offline invariant
    (revision-pinned snapshot_download from a local path, invariant #4) is
    untouched.

    The cache probe (issue #147, round 2) checks the pinned snapshot's
    ``trees/<revision>.json`` tree cache, NOT ``refs/<revision>``: a full
    commit-SHA revision is treated as immutable by ``snapshot_download``
    (``REGEX_COMMIT_HASH`` match → ``commit_hash = revision``, no API call
    and no ref write — ``refs/`` is written only when ``revision !=
    commit_hash``, i.e. for branch/tag names), so ``refs/<sha>`` is never
    created for our pinned-SHA models and probing it is dead code. The tree
    cache IS written for a full SHA (``write_tree_cache`` runs unconditionally
    once the file listing resolves) and is the exact condition under which
    ``snapshot_download`` takes its fast local path (``read_tree_cache``
    hit → no download) — so it is the right cached/uncached discriminator.
    It is not the only fast path (the ``snapshots/<sha>/`` early return when
    ``local_files_only`` is true), so suppression can be missed in that
    case (a bar appears once); that is the safe direction (bar shown, never
    a silent multi-GB download).

    Fail-open: any exception from the cache check falls through to the
    bar-intact path (a download bar is the safe default); an exception from
    ``snapshot_download`` propagates to the caller's except handler.
    """
    from huggingface_hub import snapshot_download
    from huggingface_hub.utils import disable_progress_bars

    from vemoizer.models import _model_cache_name, cache_dir

    def _is_cached() -> bool:
        """Cheap local-only cache probe (no network, no download).

        True when ``snapshot_download``'s own tree cache holds the pinned
        snapshot's file listing (``trees/<revision>.json``): that is the
        condition under which it resolves the file list locally and takes
        its fast cached path. Absent tree cache → not cached (the bar stays
        on, which is safe — a real download shows a bar). Fail-open: any
        exception returns False (bar stays on).
        """
        try:
            tree = (
                cache_dir() / _model_cache_name(repo_id) / "trees" / f"{revision}.json"
            )
            return tree.is_file()
        except Exception:  # noqa: BLE001 - fail-open (bar stays on)
            return False

    if _is_cached():
        with disable_progress_bars():
            return str(snapshot_download(repo_id, revision=revision))
    return str(snapshot_download(repo_id, revision=revision))
