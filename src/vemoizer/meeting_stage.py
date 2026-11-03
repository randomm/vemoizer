"""Profile decode selection, extracted from :mod:`vemoizer.pipeline` (issue #222).

Single responsibility: choose and run the profile's decode A, and decide
whether the consensus decodes still run on top of it. The meeting profile decodes the
whole file once with whisper-large-v3-turbo (each stem separately when two
tracks exist, the mix otherwise) and then skips decode B / re-decode /
adjudication entirely (invariant #2 allows skip-by-flag, issue #71); the
dictation profile — and a failed meeting decode, which is the fallback —
gets the Parakeet (A) + Canary (B) consensus decodes.

The decode seams (``decode_meeting``, the ``track_decode`` module,
``glossary_prompt``, ``decode_all`` and the transcriber classes) are injected
by the caller from the pipeline namespace so test monkeypatching of
:mod:`vemoizer.pipeline` still governs them — the same convention as
:func:`vemoizer.llm_tail.apply_llm_tail`'s ``*_fn`` / ``llm_client_cls``
arguments.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from contextlib import suppress
from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np

from .llm_config import _parse_meeting_language

if TYPE_CHECKING:
    from .progress import ProgressDisplay
    from .tracks import TrackPaths

logger = logging.getLogger(__name__)


class ProfileDecodes(NamedTuple):
    """The four values the rest of ``transcribe_file`` needs from decoding.

    ``result_a`` is decode A (the meeting or the Parakeet decode, ``None``
    on total failure), ``result_b`` decode B (``None`` whenever the meeting
    profile succeeded and consensus was skipped), and ``meeting_fallback``
    tells the caller to warn that the user got the dictation read instead
    of the meeting one (issue #78).
    """

    result_a: dict[str, Any] | None
    result_b: dict[str, Any] | None
    run_consensus: bool
    meeting_fallback: bool


def resolve_meeting_language(
    language: str | None, config_raw: dict[str, Any] | None
) -> str | None:
    """The recognition-language pin for the meeting decode (issue #108).

    An explicit run-level choice (CLI ``--language`` / presets
    ``RunOptions.language``) wins, else the ``[meeting] language`` key of
    the config file, else ``None`` (per-window detection). ``"auto"`` and
    any other non-code never pin. This is the single coercion point.
    """
    if language is not None and language != "auto":
        return language
    meeting_section = (
        config_raw.get("meeting") if isinstance(config_raw, dict) else None
    )
    configured = _parse_meeting_language(meeting_section)
    return None if configured == "auto" else configured


def run_profile_decodes(
    audio: np.ndarray,
    slices: list[tuple[int, np.ndarray]],
    *,
    meeting: bool,
    vad_found_speech: bool,
    glossary: list[str],
    language: str | None,
    tracks: TrackPaths | None,
    display: ProgressDisplay | None,
    decode_meeting_fn: Callable[..., dict[str, Any] | None],
    track_decode_module: Any,
    glossary_prompt_fn: Callable[[list[str]], str | None],
    decode_all_fn: Callable[
        [Any, list[tuple[int, np.ndarray]], str], dict[str, Any] | None
    ],
    parakeet_cls: Callable[[], Any],
    canary_cls: Callable[[], Any],
) -> ProfileDecodes:
    """Run the profile's decode A and, unless the meeting read won, decode B.

    Every failure is fail-open: a dead stem decode falls back to the mix, a
    dead mix decode falls back to the dictation path, and a dead single
    slice only degrades the merged result (:func:`decode_all`).
    """
    result_a: dict[str, Any] | None = None
    result_b: dict[str, Any] | None = None
    run_consensus = True
    meeting_fallback = False

    if meeting:
        # The explicit VAD signal (issue #152): sample ranges of the real
        # speech spans, or None when the VAD gave no usable information.
        vad_kwargs: dict[str, Any] = (
            {"vad_slices": [(s, s + len(a)) for s, a in slices]}
            if vad_found_speech
            else {"vad_slices": None}
        )
        kwargs: dict[str, Any] = {
            "initial_prompt": glossary_prompt_fn(glossary),
            "language": language,
            **vad_kwargs,
        }
        if display is not None:
            kwargs["display"] = display
        # Two-track call: each stem holds one speaker, so decoding them
        # separately keeps each side's language; the mix decode is the fallback.
        if tracks is not None:
            result_a = track_decode_module.decode_tracks(
                tracks,
                len(audio),
                initial_prompt=kwargs["initial_prompt"],
                language=language,
                display=display,
            )
            if result_a is None:
                logger.warning("per-track meeting decode failed; decoding the mix")
        if result_a is None:
            result_a = decode_meeting_fn(audio, slices, **kwargs)
        if result_a is not None:
            # Meeting profile skips decode B / re-decode / adjudication
            # entirely (invariant #2 allows skip-by-flag; see issue #71).
            run_consensus = False
        else:
            logger.warning("meeting decode failed; falling back to dictation path")
            meeting_fallback = True

    if result_a is None:
        parakeet: Any = None
        try:
            parakeet = parakeet_cls()
            result_a = decode_all_fn(parakeet, slices, "decode A")
        except Exception as e:  # noqa: BLE001 - fail-open stage boundary
            logger.warning("decode A failed, using best available result: %s", e)
        finally:
            if parakeet is not None:
                with suppress(Exception):  # cleanup is best-effort (fail-open)
                    parakeet.cleanup()
    if run_consensus:
        canary: Any = None
        try:
            canary = canary_cls()
            result_b = decode_all_fn(canary, slices, "decode B")
        except Exception as e:  # noqa: BLE001 - fail-open stage boundary
            logger.warning("decode B failed, using best available result: %s", e)
        finally:
            if canary is not None:
                with suppress(Exception):  # cleanup is best-effort (fail-open)
                    canary.cleanup()

    return ProfileDecodes(result_a, result_b, run_consensus, meeting_fallback)
