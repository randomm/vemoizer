"""Per-file run log + third-party log-noise suppression (issue #111, M4c).

Every transcribed file (or group) gets a full log at
``<base_dir>/.vemoizer/logs/<NFC stem>.log`` regardless of ``-v``. A
``with file_log(stem)`` block at each seam wraps the transcribe call plus
the per-file result handling, so a file that fails immediately still
leaves a (possibly near-empty) log, and a re-run over the same input
truncates the same file.

Terminal noise control (non-verbose runs): the file handler is attached
to the ROOT logger in both modes, so an INFO record from any propagating
logger (vemoizer.*, httpx, pyannote, ...) lands in the log exactly once.
WARNING+ still reaches stderr exactly as today: in non-verbose mode via
logging's last-resort handler (the file handler writes WARNING+ to the
file but not the terminal, and does not attach a filter to itself), in
verbose mode via basicConfig's root stderr handler. ``huggingface_hub``
is the special case: when its own ``propagate`` flag is False (HF-style
configuration) its records never reach root, so the same file handler is
attached to that logger directly — never twice (that would double-write
every record), design 1.

Privacy: a :class:`_RedactingFormatter` on the file handler rewrites the
formatted message AND exception text, so HuggingFace access tokens
(``hf_…``), ``Bearer …`` header values, and the value of the LLM config's
``api_key_env`` environment variable can never land in a log (design 4).

Fail-open, mirroring :mod:`vemoizer.notify`: if the log directory or file
cannot be created/opened (read-only CWD, permissions, ENOSPC, hostile
stem) the run behaves identically to no file logging — at most ONE short
stderr notice per CLI invocation (suppressed when ``--quiet``), no
exception ever leaks, and a mid-run write failure kills the file handler
silently (design 7).
"""

from __future__ import annotations

import contextlib
import logging
import os
import re
import stat
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

__all__ = ["configure", "file_log", "reset_run_log"]

#: The log directory name under the base directory (the CWD by default).
LOG_DIR_NAME = ".vemoizer"

#: The sub-directory that holds the per-file ``<stem>.log`` files.
LOG_SUBDIR = "logs"

#: ``hf_`` + 8-64 alphanumerics — the HuggingFace access-token shape
#: (real tokens are 36 chars; the 64 cap stops a runaway match in a long
#: base64 blob from eating the rest of the line).
_HF_TOKEN_RE = re.compile(r"hf_[A-Za-z0-9]{8,64}")
#: ``Bearer <token>`` — any HTTP auth header form, case-insensitive.
_BEARER_RE = re.compile(r"Bearer\s+\S+", re.IGNORECASE)

# --- once-per-run notice state (fail-open, design 5) ----------------------
# A single short stderr notice per CLI invocation when the log cannot be
# created; the flag + reset helper keep tests able to re-arm it.
_notice_sent = False


def _reset_notice() -> None:
    global _notice_sent  # noqa: PLW0603 - module-level flag by design
    _notice_sent = False


def reset_run_log() -> None:
    """Reset the once-per-run notice flag (test helper; design 5)."""
    _reset_notice()


# --- re-entrant span guard ------------------------------------------------
# ``_open_stems`` is the set of stems that currently have an open ``file_log``
# span in this process. The guard makes the span re-entrant-safe: a nested
# ``file_log`` for the SAME stem is a no-op (the outer handler already owns
# the file — a nested ``mode="w"`` handler would truncate the log mid-run and
# the detach would strip the outer handler from every logger). Different stems
# are never nested in practice, but the guard is per-stem so a hypothetical
# same-stem nesting can never lose records.

_open_stems: set[str] = set()


# --- run context (design 5) ----------------------------------------------


class _RunContext:
    """Process-global run state set once at the CLI boundary (design 5).

    ``configure()`` is called by the transcribe/meeting/memo commands next
    to the existing ``basicConfig``/display setup; ``file_log`` reads it
    when its own ``verbose``/``quiet`` arguments are ``None``.
    """

    def __init__(self) -> None:
        self.verbose: bool | None = None
        self.quiet: bool | None = None
        self.llm_api_key_env: str | None = None


_context = _RunContext()


def configure(
    *,
    verbose: bool | None = None,
    quiet: bool | None = None,
    llm_api_key_env: str | None = None,
) -> None:
    """Set the per-invocation run context (design 5).

    Called once at the CLI boundary. Any of *verbose* / *quiet* may be
    omitted to leave the previous value (``file_log``'s explicit
    arguments win over this context; ``None`` there means "use context").
    *llm_api_key_env* names the environment variable holding the LLM API
    key; its value (when set and >= 8 chars) is scrubbed from log lines.
    """
    if verbose is not None:
        _context.verbose = verbose
    if quiet is not None:
        _context.quiet = quiet
    _context.llm_api_key_env = llm_api_key_env


def _api_key_value() -> str | None:
    """The LLM API key value to scrub, or ``None`` when not redactable."""
    name = _context.llm_api_key_env
    if not name:
        return None
    value = os.environ.get(name)
    if value is None or len(value) < 8:
        return None
    return value


# --- redaction (design 4) -------------------------------------------------


class _RedactingFormatter(logging.Formatter):
    """A file-log formatter that scrubs credentials after formatting.

    A plain ``logging.Filter`` cannot see exception text (it runs before
    ``Formatter.format`` populates ``exc_text``), so redaction is a
    ``Formatter`` subclass: ``super().format()`` produces the complete
    line (message + traceback), which is then rewritten (design 4).
    """

    def format(self, record: logging.LogRecord) -> str:
        text = super().format(record)
        text = _HF_TOKEN_RE.sub("hf_<redacted>", text)
        text = _BEARER_RE.sub("Bearer <redacted>", text)
        key = _api_key_value()
        if key:
            text = text.replace(key, "<redacted>")
        return text


# --- terminal noise filter (non-verbose, design 2/3) ----------------------


class _NoThirdPartyInfoFilter(logging.Filter):
    """Drop third-party ``INFO`` records on a *terminal* stderr handler.

    vemoizer's own loggers always pass (our INFO is always allowed to the
    terminal); only non-vemoizer loggers below WARNING are dropped, and
    only from handlers that exist (HF's own stream handlers in non-verbose
    mode). The file handler itself never carries this filter — it records
    everything at INFO.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        if record.levelno >= logging.WARNING:
            return True
        return record.name == "vemoizer" or record.name.startswith("vemoizer.")


# --- fail-open file handler (design 7) ------------------------------------


class _QuietFileHandler(logging.FileHandler):
    """A ``FileHandler`` that never lets a write error reach the caller.

    ``handle`` is overridden so a write failure (ENOSPC, EROFS, ...) is
    routed to ``handleError`` — which swallows it and disables further
    writes — instead of propagating to the caller (a mid-run write
    failure must behave identically to no file logging, design 7).
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._disabled = False

    def handle(self, record: logging.LogRecord) -> bool:  # noqa: D102
        if self._disabled:
            return False
        if self.filter(record):
            try:
                self.emit(record)
            except (KeyboardInterrupt, SystemExit):
                raise
            except Exception:
                # Fail-open: a write error (ENOSPC, EROFS, ...) must never
                # escape the handler; disable further writes and close.
                self.handleError(record)
        return True

    def handleError(self, record: logging.LogRecord) -> None:  # noqa: D102
        self._disabled = True
        self.close()


def _sanitise_stem(stem: str) -> str:
    """Replace ``/`` and NUL with ``_`` so the name cannot escape the log dir."""
    return stem.replace("/", "_").replace("\x00", "_")


def _log_path(base_dir: Path, stem: str) -> Path:
    return base_dir / LOG_DIR_NAME / LOG_SUBDIR / f"{_sanitise_stem(stem)}.log"


def _notice(message: str, quiet: bool | None) -> None:
    """Emit the once-per-run stderr notice (suppressed when quiet)."""
    global _notice_sent  # noqa: PLW0603 - module-level flag by design
    if quiet:
        return
    if _notice_sent:
        return
    _notice_sent = True
    print(message, file=sys.stderr)


def _attach(handler: logging.Handler, verbose: bool) -> list[tuple[Any, Any, Any]]:
    """Attach *handler* to the loggers that need it; return the undo list.

    The file handler is attached to the ROOT logger in both modes (so
    every propagating logger writes INFO+ to the file exactly once).
    Design 1 (runtime check, not an assumption): the ``huggingface_hub``
    logger additionally gets the handler directly ONLY when ``propagate``
    is False at entry (HF-style); when it is True (the default in this
    venv) HF records reach the file via the root handler and attaching
    here would double-write every record.

    The ROOT logger's level is raised to INFO only if it is currently
    higher (non-verbose runs default to WARNING), so the file log
    captures third-party INFO regardless of ``-v``; the raise is always
    undone (state snapshot). In non-verbose mode a ``_NoThirdPartyInfoFilter``
    is added to HF's own stream handlers (so its INFO stays off the
    terminal) and its level is raised to INFO only if it is currently 0
    (NOTSET); a level HF set explicitly is never lowered (design 2).

    Returns a list of ``(kind, target, snapshot)`` entries; ``_detach``
    restores each to its exact snapshot (handler list, level, or
    per-handler filters).
    """
    undo: list[tuple[Any, Any, Any]] = []
    root = logging.getLogger()
    hf_logger = logging.getLogger("huggingface_hub")
    attached_to_hf = not hf_logger.propagate
    targets: list[logging.Logger] = [root]
    if attached_to_hf:
        targets.append(hf_logger)
    for target in targets:
        undo.append(("state", target, (list(target.handlers), target.level)))
        target.addHandler(handler)
    # Raise the root level to INFO so the file log sees third-party INFO
    # in non-verbose runs; never lower an explicitly-set level.
    if root.level > logging.INFO:
        undo.append(("level", root, root.level))
        root.setLevel(logging.INFO)
    if not verbose:
        # Design 2: keep third-party INFO off the terminal in non-verbose
        # mode. A root stderr stream handler (basicConfig's under -v, or
        # one a third-party lib installed) gets the filter so third-party
        # INFO stays off the terminal while vemoizer INFO and WARNING+ pass.
        # The file handler itself never carries this filter — it records
        # everything at INFO.
        for h in root.handlers:
            if isinstance(h, logging.StreamHandler) and not isinstance(
                h, logging.FileHandler
            ):
                undo.append(("filters", h, list(h.filters)))
                h.addFilter(_NoThirdPartyInfoFilter())
        if attached_to_hf:
            for h in list(hf_logger.handlers):
                if isinstance(h, logging.StreamHandler):
                    undo.append(("filters", h, list(h.filters)))
                    h.addFilter(_NoThirdPartyInfoFilter())
            if hf_logger.level == 0:
                undo.append(("level", hf_logger, hf_logger.level))
                hf_logger.setLevel(logging.INFO)
    return undo


def _detach(undo: list[tuple[Any, Any, Any]]) -> None:
    """Restore every logged logger/handler to its entry snapshot."""
    for kind, target, snapshot in reversed(undo):
        if kind == "state":
            handlers, level = snapshot
            for h in list(target.handlers):
                target.removeHandler(h)
            for h in handlers:
                target.addHandler(h)
            target.setLevel(level)
        elif kind == "filters":
            for f in list(target.filters):
                target.removeFilter(f)
            for f in snapshot:
                target.addFilter(f)
        else:  # "level"
            target.setLevel(snapshot)


# --- the public context manager (design 5/6/7) ----------------------------


@contextmanager
def file_log(
    stem: str,
    *,
    base_dir: Path | None = None,
    verbose: bool | None = None,
    quiet: bool | None = None,
) -> Iterator[None]:
    """Write ``base_dir/.vemoizer/logs/<stem>.log`` for the duration.

    *stem* is the already-NFC stem (or a group's first-part label); ``/``
    and NUL are sanitised so the path cannot escape the log directory.
    *base_dir* defaults to the CWD. *verbose* / *quiet* default to the
    run context set by :func:`configure` (design 5).

    The file handler is attached to the ROOT logger at INFO level in both
    modes, and — only when ``huggingface_hub.propagate`` is False — to the
    ``huggingface_hub`` logger directly (never twice, design 1).

    Fail-open: any error creating the directory/file degrades to no file
    logging with at most one short stderr notice per invocation (design 5);
    a mid-run write error is swallowed silently (design 7). The handler is
    removed from every logger it was attached to and ``close()``d on every
    exit path (success, exception, ``KeyboardInterrupt``).
    """
    base = base_dir if base_dir is not None else Path.cwd()
    path = _log_path(base, stem)
    eff_verbose = verbose if verbose is not None else _context.verbose
    eff_quiet = quiet if quiet is not None else _context.quiet
    key = _sanitise_stem(stem)
    if key in _open_stems:
        # Re-entrant: the outer span for this stem already owns the file
        # (a nested open would truncate it mid-run). Yield a no-op.
        yield
        return
    _open_stems.add(key)
    handler: _QuietFileHandler | None = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with contextlib.suppress(OSError):
            path.parent.chmod(stat.S_IRWXU)  # 0700 (best-effort)
        handler = _QuietFileHandler(str(path), mode="w", encoding="utf-8")
        handler.setLevel(logging.INFO)
        handler.setFormatter(_RedactingFormatter())
        with contextlib.suppress(OSError):
            path.chmod(stat.S_IRUSR | stat.S_IWUSR)  # 0600 (best-effort)
    except Exception as e:  # noqa: BLE001 - fail-open: never change the run
        _open_stems.discard(key)
        _notice(f"vemoizer: could not open run log {path}: {e}", quiet=eff_quiet)
        yield
        return
    assert handler is not None
    undo = _attach(handler, bool(eff_verbose))
    try:
        yield
    finally:
        _detach(undo)
        handler.close()
        _open_stems.discard(key)
