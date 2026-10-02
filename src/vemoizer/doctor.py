"""``vemoizer doctor`` — local health checks (issue #79, M7).

Prints a ``[ ok ]``/``[FAIL]``/``[WARN]`` line per check and exits
non-zero when ANY red check fails.  The LLM 1-token ping is the ONLY
warning-level check (it never affects the exit code); the LLM key env
var is red when a config is present but the env var is unset.

Checks (all local except the 1-token LLM ping, which is warning-only):

- ffmpeg on PATH
- HuggingFace token present (``huggingface_hub.get_token()`` — covers
  ``HF_TOKEN`` *and* the cached token file; may return ``None`` or
  raise depending on version — both mean red-with-hint, never a crash)
- pyannote licence: token presence only, no server-side verification
- each of the 5 pinned models cached (read-only local cache walk)
- LLM config parse (absent = no LLM = green; malformed = red)
- LLM key env var set (only when a config is present)
- LLM 1-token ping (warning only — the sole non-fatal check)
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import cast

from .llm import LLMConfig
from .models import MODELS
from .preflight import ffmpeg_ok, hf_token_present
from .preflight import models_cached as models_missing

#: Status constants for :class:`DoctorCheck.status`.
GREEN = "ok"
RED = "fail"
WARN = "warn"

#: Prefix markers printed per check line.
_MARKERS = {"ok": "[ ok ]", "fail": "[FAIL]", "warn": "[WARN]"}

#: Hint shown next to a red HF-token / pyannote-licence check.
HF_TOKEN_HINT = (
    "accept the pyannote licence form at "
    "huggingface.co/pyannote/speaker-diarization-community-1 and set "
    "HF_TOKEN (or store a token in the HuggingFace cache)"
)


@dataclass(frozen=True)
class DoctorCheck:
    """One doctor check result: name, status, and a hint (red/warn only)."""

    name: str
    status: str
    hint: str = ""


@dataclass
class DoctorReport:
    """All doctor check results; ``ok`` is False when any check is red."""

    checks: list[DoctorCheck] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return all(c.status != RED for c in self.checks)


def _config_load() -> tuple[LLMConfig | None, str | None]:
    """(config, parse_error) via the strict layered search.

    ``config`` is an ``LLMConfig`` or ``None`` (absent → no LLM, green).
    ``parse_error`` is the ``ConfigError`` message when the file is
    malformed (red); ``None`` otherwise.
    """
    from .llm import ConfigError, _default_search

    try:
        config = _default_search()
    except ConfigError as e:
        return None, str(e)
    return config, None


def llm_ping_ok(config: LLMConfig) -> bool:
    """1-token fail-open LLM ping; ``False`` on any failure (never raises).

    A single ``LLMClient.complete`` with ``max_tokens=1``; success is
    "returned something non-``None``".  This is the ONLY check that may
    touch the network, and it is warning-level regardless of outcome.
    """
    from .llm import LLMClient

    client = LLMClient(config)
    try:
        return client.complete("ping", "ping", max_tokens=1) is not None
    except Exception:  # noqa: BLE001 - fail-open, ping is warning-only
        return False
    finally:
        client.close()


def _call_ping(ping: object) -> bool:
    """Invoke a *ping* callable safely; ``False`` on any exception."""
    fn: Callable[[], object] = cast("Callable[[], object]", ping)
    try:
        return bool(fn())
    except Exception:  # noqa: BLE001 - fail-open, ping is warning-only
        return False


def run_doctor(
    *,
    echo: Callable[[str], None] = print,
    ping: object | None = "auto",
) -> DoctorReport:
    """Run every doctor check, printing a status line per check.

    Returns a :class:`DoctorReport`; ``report.ok`` is False when any
    check is red (the caller exits non-zero).  ``ping="auto"`` runs the
    real 1-token ping when a config is present; tests may inject a
    ``bool`` or a callable to skip the network.
    """
    report = DoctorReport()

    def add(name: str, ok: bool, *, hint: str = "") -> None:
        status = GREEN if ok else RED
        report.checks.append(DoctorCheck(name, status, hint if not ok else ""))
        line = f"{_MARKERS[status]} {name}"
        if not ok and hint:
            line += f"  — {hint}"
        echo(line)

    # 1. ffmpeg on PATH
    add("ffmpeg on PATH", ffmpeg_ok(), hint="install ffmpeg (brew install ffmpeg)")

    # 2. HF token (get_token, not only the HF_TOKEN env var)
    token_ok = hf_token_present()
    add("HuggingFace token present", token_ok, hint=HF_TOKEN_HINT)

    # 3. pyannote licence (token presence only; no server-side verify)
    add("pyannote licence (token presence only)", token_ok, hint=HF_TOKEN_HINT)

    # 4. each pinned model cached (read-only)
    missing = set(models_missing())
    for spec in MODELS:
        if spec.name in missing:
            add(
                f"model {spec.name} cached",
                False,
                hint=f"run 'vemoizer models pull' to download {spec.name}",
            )
        else:
            add(f"model {spec.name} cached", True)

    # 5. config parse
    config, parse_error = _config_load()
    if parse_error is not None:
        add("config parse", False, hint=parse_error)
    elif config is None:
        add("config parse (no LLM configured)", True)
    else:
        add("config parse", True)
        # 6. LLM key env var (only when a config is present)
        api_key_set = bool(os.environ.get(config.api_key_env, "").strip())
        add(f"LLM key env var {config.api_key_env} set", api_key_set)
        # 7. 1-token ping (warning-only — the sole non-fatal check)
        if ping is None or ping == "auto":
            ping_ok = llm_ping_ok(config)
        elif callable(ping):
            ping_ok = bool(_call_ping(ping))
        else:
            ping_ok = bool(ping)
        status = GREEN if ping_ok else WARN
        report.checks.append(
            DoctorCheck(
                "LLM ping (1 token)",
                status,
                "" if ping_ok else "LLM ping failed or returned None",
            )
        )
        line = f"{_MARKERS[status]} LLM ping (1 token)"
        if not ping_ok:
            line += "  — LLM ping failed or returned None"
        echo(line)

    return report
