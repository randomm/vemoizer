"""Per-file run-log IO: credential redaction + the secure fail-open handler.

Extracted from :mod:`vemoizer.run_log` (issue #117, headroom): the pieces
that concern the *contents* of a log line (redacting HuggingFace tokens,
``Bearer`` headers, and the configured LLM API key) and the *handler object*
that owns the log file (a ``FileHandler`` that never lets a write error reach
the caller). The run context, the re-entrancy guard, path resolution, and the
``file_log`` context manager stay in :mod:`vemoizer.run_log`, which imports
these two collaborators back (design 4 + design 7).

Privacy: a :class:`RedactingFormatter` rewrites the message and exception
text, so HuggingFace tokens (``hf_…``), ``Bearer …`` header values, and the
``api_key_env`` value can never land in a log (design 4). The API key is
looked up per record via a caller-supplied getter (the run context owns
``llm_api_key_env``), so a mid-run ``configure`` change or an environment
change within a span is honoured; this module stays stateless with respect
to the run context.

Fail-open: :class:`QuietFileHandler` swallows a mid-run write failure
(ENOSPC, EROFS, ...) and disables further writes, so the run behaves
identically to no file logging (design 7).
"""

from __future__ import annotations

import contextlib
import logging
import re
from collections.abc import Callable
from typing import Any

#: ``hf_`` + 8-or-more alphanumerics — the HuggingFace access-token shape
#: (real tokens are 36 chars). The unbounded quantifier still matches in
#: linear time (no alternation, no nested quantifier), and a token longer
#: than any real one is redacted in full instead of leaking its tail.
_HF_TOKEN_RE = re.compile(r"hf_[A-Za-z0-9]{8,}")
#: ``Bearer <token>`` — any HTTP auth header form, case-insensitive.
_BEARER_RE = re.compile(r"Bearer\s+\S+", re.IGNORECASE)

# --- redaction (design 4) -------------------------------------------------


class RedactingFormatter(logging.Formatter):
    """A file-log formatter that scrubs credentials after formatting.

    A plain ``logging.Filter`` cannot see exception text (it runs before
    ``Formatter.format`` populates ``exc_text``), so redaction is a
    ``Formatter`` subclass: ``super().format()`` produces the complete
    line (message + traceback), which is then rewritten (design 4).

    *api_key_getter* is a zero-argument callable returning the env-var value
    named by the run context's ``llm_api_key_env`` (or ``None``); it is
    called once per record inside ``format``, so a mid-run ``configure``
    change or an environment change within the span is honoured. The >= 8
    length rule and the fail-safe (``None``/empty is never redacted) live on
    the caller side — exactly where the original ``run_log._api_key_value``
    had them — keeping the run context and the environment in the caller.
    """

    def __init__(self, api_key_getter: Callable[[], str | None] = lambda: None) -> None:
        super().__init__()
        self._api_key_getter = api_key_getter

    def format(self, record: logging.LogRecord) -> str:
        text = super().format(record)
        text = _HF_TOKEN_RE.sub("hf_<redacted>", text)
        text = _BEARER_RE.sub("Bearer <redacted>", text)
        key = self._api_key_getter()
        if key:
            text = text.replace(key, "<redacted>")
        return text


# --- fail-open file handler (design 7) ------------------------------------


class QuietFileHandler(logging.FileHandler):
    """A ``FileHandler`` (file path or pre-opened *stream*) that never lets a
    write error reach the caller.

    ``handle`` is overridden so a write failure (ENOSPC, EROFS, ...) is
    routed to ``handleError`` — which swallows it and disables further
    writes — instead of propagating (a mid-run write failure must behave
    identically to no file logging, design 7).
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
                self.handleError(record)
        return True

    def handleError(self, record: logging.LogRecord) -> None:  # noqa: D102
        self._disabled = True
        # Fail-open: a close() failure (already-closed stream, EBADF, ...)
        # must not escape the handler either — the run must behave as if
        # the log never existed (design 7).
        with contextlib.suppress(Exception):
            self.close()
