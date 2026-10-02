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
WARNING+ still reaches stderr exactly as today. ``huggingface_hub`` is
the special case: when its ``propagate`` flag is False its records never
reach root, so the same file handler is attached directly — never twice
(design 1). When a span opens it writes one INFO ``log started for
<stem>`` line so the log is never indistinguishable from an empty file;
a nested same-stem span (a no-op) adds no second line.

Privacy: a :class:`_RedactingFormatter` on the file handler rewrites the
formatted message AND exception text, so HuggingFace access tokens
(``hf_…``), ``Bearer …`` header values, and the LLM config's ``api_key_env``
environment variable value can never land in a log (design 4).

Fail-open, mirroring :mod:`vemoizer.notify`: if the log directory or file
cannot be created/opened the run behaves identically to no file logging —
at most ONE short stderr notice per CLI invocation (suppressed when
``--quiet``), no exception ever leaks, and a mid-run write failure kills
the file handler silently (design 7).
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
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = ["configure", "file_log", "reset_run_log"]


@dataclass
class _Undo:
    """The exact logging state one span changed (relative, identity-based).

    Every field holds the *instances* this span added and the old values it
    changed; ``_detach`` reverts only those, so anything another component
    (pytest's ``caplog``, another thread, a library) added mid-span is left
    untouched. Calling ``_detach`` twice is a no-op: already-removed
    instances are gone and ``removeHandler``/``removeFilter`` by identity
    are idempotent.
    """

    _added_handlers: list[tuple[logging.Logger, logging.Handler]] = field(
        default_factory=list
    )
    _added_filters: list[tuple[logging.Handler, logging.Filter]] = field(
        default_factory=list
    )
    _old_levels: list[tuple[logging.Logger, int]] = field(default_factory=list)

    def record_level(self, logger: logging.Logger, old_level: int) -> None:
        self._old_levels.append((logger, old_level))


#: The log directory name under the base directory (the CWD by default).
LOG_DIR_NAME = ".vemoizer"

#: The sub-directory that holds the per-file ``<stem>.log`` files.
LOG_SUBDIR = "logs"

#: ``hf_`` + 8-or-more alphanumerics — the HuggingFace access-token shape
#: (real tokens are 36 chars). The unbounded quantifier still matches in
#: linear time (no alternation, no nested quantifier), and a token longer
#: than any real one is redacted in full instead of leaking its tail.
_HF_TOKEN_RE = re.compile(r"hf_[A-Za-z0-9]{8,}")
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
    """Reset the run-log module state (test helper): the once-per-run notice
    flag, the re-entrancy guard, and the sanitised-collision owner map."""
    _reset_notice()
    _open_paths.clear()
    _stem_owner.clear()


# --- re-entrant span guard ------------------------------------------------
# ``_open_paths`` is the set of LOG PATHS that currently have an open
# ``file_log`` span in this process. The guard makes the span re-entrant-
# safe: a nested ``file_log`` whose path is already owned by an open span
# is a no-op (the outer handler owns the file — a nested ``mode="w"``
# handler would truncate the log mid-run and the detach would strip the
# outer handler from every logger). Two DISTINCT raw stems are never the
# same key, so a sanitised collision (``a/b`` vs ``a_b``) can never be
# treated as one span: ``_log_path`` disambiguates the second one to
# ``<sanitised>.2.log`` (and ``.3`` ...) deterministically, so distinct
# inputs never share a log. (In practice the four CLI seams pass bare
# filename stems with no ``/``, so the disambiguation is defensive.)

_open_paths: set[str] = set()


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
        # Fail-open: a close() failure (already-closed stream, EBADF, ...)
        # must not escape the handler either — the run must behave as if
        # the log never existed (design 7).
        with contextlib.suppress(Exception):
            self.close()


def _sanitise_stem(stem: str) -> str:
    """Replace ``/`` and NUL with ``_`` so the name cannot escape the log dir."""
    return stem.replace("/", "_").replace("\x00", "_")


# Maps a sanitised name to the raw stem that owns its ``.log``/``.N`` file
# in this process, so a re-run over the SAME raw stem keeps (and truncates)
# its own file, while a DIFFERENT raw stem that sanitises to the same name
# deterministically gets ``.2.log``, ``.3.log", ...`` and never shares a
# log with the owner. (In practice the four CLI seams pass bare filename
# stems with no ``/``, so the disambiguation is defensive for hostile stems.)
_stem_owner: dict[str, str] = {}


def _log_path(base_dir: Path, stem: str) -> Path:
    """The log path for *stem*, disambiguating sanitised collisions.

    Two different raw stems that sanitise to the same name (``a/b`` and
    ``a_b``) would otherwise share one file; the second (and any later)
    colliding raw stem deterministically gets ``<sanitised>.2.log``,
    ``<sanitised>.3.log", ...`` — the first raw stem keeps the plain name,
    so a re-run over the same input still truncates its own log and distinct
    inputs never share a log (see ``_stem_owner``).
    """
    sanitised = _sanitise_stem(stem)
    base = base_dir / LOG_DIR_NAME / LOG_SUBDIR
    if _stem_owner.get(sanitised) == stem:
        # A re-run over the same raw stem keeps its own (plain) file.
        return base / f"{sanitised}.log"
    if sanitised not in _stem_owner:
        _stem_owner[sanitised] = stem
        return base / f"{sanitised}.log"
    n = 2
    while f"{sanitised}.{n}" in _stem_owner:
        n += 1
    _stem_owner[f"{sanitised}.{n}"] = stem
    return base / f"{sanitised}.{n}.log"


def _notice(message: str, quiet: bool | None) -> None:
    """Emit the once-per-run stderr notice (suppressed when quiet)."""
    global _notice_sent  # noqa: PLW0603 - module-level flag by design
    if quiet:
        return
    if _notice_sent:
        return
    _notice_sent = True
    print(message, file=sys.stderr)


def _attach(handler: logging.Handler, verbose: bool) -> _Undo:
    """Attach *handler* to the loggers that need it; return the undo state.

    The file handler is attached to the ROOT logger in both modes (so
    every propagating logger writes INFO+ to the file exactly once).
    Design 1 (runtime check, not an assumption): the ``huggingface_hub``
    logger additionally gets the handler directly ONLY when ``propagate``
    is False at entry (HF-style); when it is True (the default in this
    venv) HF records reach the file via the root handler and attaching
    here would double-write every record.

    The ROOT logger's level is raised to INFO whenever it is currently
    higher (non-verbose runs default to WARNING) so the file log captures
    third-party INFO regardless of ``-v``; the raise is always undone. A
    level that already sits at or below INFO is never touched.

    In non-verbose mode a ``_NoThirdPartyInfoFilter`` is added to the
    terminal (non-file) stream handlers so third-party INFO stays off the
    terminal while vemoizer INFO and WARNING+ pass; the file handler
    itself never carries this filter — it records everything at INFO. On
    the HF logger itself the same suppression applies to its own stream
    handlers, and its level is raised to INFO whenever it is NOTSET or
    above INFO (an explicit level above INFO would otherwise filter HF
    INFO records before they reach the file, decision 2); a level at or
    below INFO is never raised.

    Returns an :class:`_Undo` recording the exact handler and filter
    instances added and the old level values, so :func:`_detach` can revert
    only this span's changes.
    """
    root = logging.getLogger()
    hf_logger = logging.getLogger("huggingface_hub")
    attached_to_hf = not hf_logger.propagate
    targets: list[logging.Logger] = [root, hf_logger] if attached_to_hf else [root]
    undo = _Undo()
    for target in targets:
        target.addHandler(handler)
        undo._added_handlers.append((target, handler))
    # A logger level above INFO (or NOTSET, which inherits the effective
    # level) drops INFO records before any handler sees them; raise to
    # INFO only, and only when needed, remembering the exact old level.
    if root.level > logging.INFO:
        undo.record_level(root, root.level)
        root.setLevel(logging.INFO)
    if attached_to_hf and hf_logger.level not in (
        logging.NOTSET,
        logging.DEBUG,
        logging.INFO,
    ):
        # NOTSET (0) or explicitly above INFO (WARNING 30+): raise to INFO.
        undo.record_level(hf_logger, hf_logger.level)
        hf_logger.setLevel(logging.INFO)
    if not verbose:
        # Design 2: keep third-party INFO off the terminal in non-verbose
        # mode. Terminal stream handlers (basicConfig's under -v, or one a
        # third-party lib installed) get the filter; the file handler never
        # does (it records everything at INFO).
        for target in targets:
            for h in list(target.handlers):
                if isinstance(h, logging.StreamHandler) and not isinstance(
                    h, logging.FileHandler
                ):
                    f = _NoThirdPartyInfoFilter()
                    h.addFilter(f)
                    undo._added_filters.append((h, f))
    return undo


def _detach(undo: _Undo) -> None:
    """Revert exactly the handlers, filters, and levels this span changed.

    Identity-based and idempotent: a handler or filter another component
    added mid-span is never removed, a level the span did not change is
    never touched, and calling ``_detach`` twice is a no-op. Each removal
    and level restore is exception-guarded so one failed step cannot
    abort the remaining restorations.
    """
    for logger, handler in undo._added_handlers:
        with contextlib.suppress(Exception):
            logger.removeHandler(handler)
    for handler, f in undo._added_filters:
        with contextlib.suppress(Exception):
            handler.removeFilter(f)
    for logger, level in undo._old_levels:
        with contextlib.suppress(Exception):
            logger.setLevel(level)


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
    and NUL are sanitised so the path cannot escape the log directory, and
    two different raw stems that sanitise to the same name are written to
    distinct files (``<sanitised>.log`` and ``<sanitised>.2.log``).
    *base_dir* defaults to the CWD. *verbose* / *quiet* default to the
    run context set by :func:`configure` (design 5).

    Fail-open: any error creating the directory/file degrades to no file
    logging with at most one short stderr notice per invocation (design 5).
    The re-entrant guard key is added before setup and discarded in a
    ``finally`` covering the whole span, so even a ``KeyboardInterrupt``
    or ``SystemExit`` during setup cannot leak the key. The failure path
    is explicit (no ``assert``), so it behaves identically under
    ``python -O``.
    """
    base = base_dir if base_dir is not None else Path.cwd()
    eff_verbose = verbose if verbose is not None else _context.verbose
    eff_quiet = quiet if quiet is not None else _context.quiet
    # Read-only path computation (no ``_stem_owner`` mutation): a nested
    # no-op span for the same raw stem never calls the allocating
    # ``_log_path``. The read-only logic mirrors ``_log_path`` exactly.
    sanitised = _sanitise_stem(stem)
    base_logs = base / LOG_DIR_NAME / LOG_SUBDIR
    if _stem_owner.get(sanitised) == stem:
        path = base_logs / f"{sanitised}.log"  # re-run: same raw stem
    elif sanitised not in _stem_owner:
        path = base_logs / f"{sanitised}.log"  # first claimant
    else:
        n = 2
        while f"{sanitised}.{n}" in _stem_owner:
            n += 1
        path = base_logs / f"{sanitised}.{n}.log"  # collision slot
    key = str(path)
    if key in _open_paths:
        # Re-entrant: an outer span for this exact log file is already open
        # (a nested open would truncate it mid-run). Yield a no-op.
        yield
        return
    _open_paths.add(key)
    try:
        # Allocate under the guard: this mutates ``_stem_owner`` if needed.
        # For the same raw stem, the owner is already us so no mutation
        # occurs; for a new raw stem, the slot is claimed here.
        _log_path(base, stem)
        handler = _open_log_file(base, stem, key, eff_quiet)
        if handler is None:
            # Fail-open: the directory/file could not be created; degrade
            # to no file logging (explicit branch, never an AssertionError).
            yield
            return
        undo = _attach(handler, bool(eff_verbose))
        try:
            # The span's own first record: one "log started" line per open
            # span, naming the ORIGINAL stem (the sanitised path is only
            # for the file).
            logging.getLogger("vemoizer.run_log").info("log started for %s", stem)
            yield
        finally:
            _detach(undo)
            handler.close()
    finally:
        # Discard the guard key on EVERY exit path — success, exception,
        # KeyboardInterrupt, SystemExit — so a setup failure can never
        # leak the key and no-op the next span for the same stem.
        _open_paths.discard(key)


def _open_log_file(
    base: Path, stem: str, key: str, quiet: bool | None
) -> _QuietFileHandler | None:
    """Create the log directory/file and return the handler (fail-open).

    Returns ``None`` (with a once-per-run notice) when the directory or
    file cannot be created; never raises.
    """
    path = Path(key)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with contextlib.suppress(OSError):
            path.parent.chmod(stat.S_IRWXU)  # 0700 (best-effort)
        handler = _QuietFileHandler(str(path), mode="w", encoding="utf-8")
        handler.setLevel(logging.INFO)
        handler.setFormatter(_RedactingFormatter())
        with contextlib.suppress(OSError):
            path.chmod(stat.S_IRUSR | stat.S_IWUSR)  # 0600 (best-effort)
        return handler
    except Exception as e:  # noqa: BLE001 - fail-open: never change the run
        _notice(f"vemoizer: could not open run log {path}: {e}", quiet=quiet)
        return None
