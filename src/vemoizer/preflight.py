"""Pre-decode preflight checks (issue #79, M7).

The inline preflight runs in well under a second before any decode and is
fully local — no network, no model loading: ffmpeg present on PATH, the
LLM config parse, every pinned model present in the local HF cache, and
(an HF token check whenever diarization will run.  A red check fails the
run before a single decode is spent, so a gated pyannote model cannot
fail open into an hour-long run with no speaker labels (the silent
failure M1 exists to prevent).

:func:`run_preflight` is the seam the pipeline (and the tests) call; the
checks are module-level functions so each can be monkeypatched in
isolation.  The LLM key / 1-token ping is intentionally NOT part of the
inline preflight — it lives in :mod:`vemoizer.doctor` only.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .models import MODELS

#: A red check with its user-facing reason (printed on failure).
Check = tuple[str, str]


@dataclass(frozen=True)
class PreflightResult:
    """The outcome of the inline preflight.

    ``red``: (label, reason) pairs; non-empty means the run must abort.
    ``seconds``: wall-clock duration (logged, informational only).
    """

    red: list[Check]
    seconds: float


def ffmpeg_ok() -> bool:
    """True when an ``ffmpeg`` executable is on PATH.

    ``shutil.which`` only: the subprocess probe would cost a fork and the
    ingest stage already turns a missing binary into a clean
    ``IngestError``.
    """
    import shutil

    return shutil.which("ffmpeg") is not None


def config_parse_ok() -> bool:
    """True when the layered LLM config is absent or parses.

    The strict layered search raises ``ConfigError`` on a malformed
    project config (fail loud, issue #78) — that is the only state that
    is red here.  ``None`` (no config anywhere) is the "no LLM" state and
    is green.
    """
    from .llm_config import ConfigError, _default_search

    try:
        _default_search()
    except ConfigError:
        return False
    return True


def hf_token_present() -> bool:
    """True when ``huggingface_hub.get_token()`` returns a non-empty token.

    Covers ``HF_TOKEN`` AND the cached token file (~/.cache/huggingface) —
    the plain env-var read in diarization.py misses the latter.  May
    return ``None`` (no token) or raise (version-dependent) — both mean
    red, never a crash.
    """
    try:
        from huggingface_hub import get_token
    except ImportError:  # pragma: no cover - huggingface_hub is a hard dep
        return False
    try:
        token = get_token()
    except Exception:  # noqa: BLE001 - version-dependent raise → red
        return False
    return bool(token)


def models_cached() -> list[str]:
    """Names of pinned models missing from the local HF cache.

    Read-only: ``models.cache_size`` walks the local cache directory
    (honours ``HF_HOME``) and reports 0 for a missing model — no
    ``scan_cache_dir``/``get_model`` KeyError path, no downloads.
    """
    from .models import cache_size

    sizes = cache_size(MODELS)
    return [name for name, size in sizes.items() if size == 0]


_REASONS: dict[str, str] = {
    "ffmpeg": (
        "ffmpeg not found on PATH — install it (e.g. `brew install ffmpeg`) "
        "before transcribing"
    ),
    "config": (
        "config file failed to parse — fix or remove the "
        ".vemoizer/config.toml [llm] section"
    ),
    "hf-token": (
        "no HuggingFace token found (HF_TOKEN or the cached token file); "
        "the pyannote diarization weights are gated — accept the licence "
        "form at huggingface.co/pyannote/"
        "speaker-diarization-community-1 and set HF_TOKEN"
    ),
}


def _check(
    label: str,
    call: Callable[[], bool],
    red: list[Check],
) -> None:
    """Run one boolean preflight *call* and append a red entry when needed.

    Two failure modes become red: the check returns ``False`` (the usual
    "not present" state, with its user-facing reason), or the check itself
    raises (the "never raises" contract — a red entry naming the exception
    class, never its message, which may carry secrets).  Either way
    ``preflight_gate`` yields the clean ``preflight failed:`` message
    instead of a bare error propagating out of ``transcribe_file``.
    """
    try:
        ok = call()
    except Exception as e:  # noqa: BLE001 - per-check "never raises" guard
        red.append((label, f"{label} check failed: {type(e).__name__}"))
        return
    if not ok:
        red.append((label, _REASONS[label]))


def run_preflight(
    *,
    diarize: bool = False,
    echo=print,
) -> PreflightResult:
    """Run the inline preflight; a red check aborts before any decode.

    Checks, in order: ffmpeg, config parse, all 5 pinned models cached
    (read-only), and — only when *diarize* is requested — the HuggingFace
    token via ``get_token()``.  On any red check the reasons are printed
    via *echo* (stderr in the CLI path) and the result carries them so
    the caller can abort.  Never raises: every check is wrapped in a
    guard (``_check``) that turns an unexpected exception into a red entry,
    so a check that itself errors also counts as red.
    """
    import time

    start = time.monotonic()
    red: list[Check] = []

    _check("ffmpeg", ffmpeg_ok, red)
    _check("config", config_parse_ok, red)

    # models_cached returns the list of missing-model names; a non-empty
    # list (or a raised exception) is red, one entry per missing model.
    try:
        missing = models_cached()
    except Exception as e:  # noqa: BLE001 - per-check "never raises" guard
        red.append(("models", f"models check failed: {type(e).__name__}"))
    else:
        for name in missing:
            red.append(
                (
                    f"model {name}",
                    f"{name} is not in the local model cache — run "
                    "'vemoizer models pull' first",
                )
            )

    if diarize:
        _check("hf-token", hf_token_present, red)

    seconds = time.monotonic() - start
    for label, reason in red:
        echo(f"preflight: {label}: {reason}")
    return PreflightResult(red=red, seconds=seconds)


def preflight_gate(
    *,
    diarize: bool = False,
    profile: str = "dictation",
    echo=print,
) -> dict[str, Any] | None:
    """Run the inline preflight and return an error dict on red.

    Convenience wrapper for the pipeline: returns ``None`` when the
    preflight passes, or a ``{"text", "segments", "error"}`` dict when
    any check is red.  The *profile* argument encodes the meeting-always-
    diarize rule (issue #79 d4): the token check fires when *diarize* is
    true OR the profile is ``"meeting"``.
    """
    result = run_preflight(
        diarize=diarize or profile == "meeting",
        echo=echo,
    )
    if not result.red:
        return None
    return {
        "text": "",
        "segments": [],
        "error": "preflight failed: "
        + "; ".join(reason for _label, reason in result.red),
    }
