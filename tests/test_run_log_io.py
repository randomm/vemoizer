"""RedactingFormatter per-record API-key getter behaviour (issue #117).

The formatter must call its ``api_key_getter`` on EVERY record (the original
``run_log_io.RedactingFormatter`` calls the getter per record), so
a mid-span ``configure`` change or environment change is honoured, and the
getter must not be invoked at construction time.
"""

from __future__ import annotations

import logging
import os
import sys

from vemoizer.run_log_io import RedactingFormatter


def _log_record(msg: str) -> logging.LogRecord:
    return logging.LogRecord(
        name="vemoizer.t",
        level=logging.INFO,
        pathname=__file__,
        lineno=0,
        msg=msg,
        args=(),
        exc_info=None,
    )


def _format(formatter: RedactingFormatter, msg: str) -> str:
    return formatter.format(_log_record(msg))


class TestApiKeyGetter:
    def test_per_record_lookup_honours_env_change(self, monkeypatch) -> None:
        """(a) Changing the env var between two records on the SAME formatter
        makes the second record redact the NEW value."""

        def getter() -> str | None:
            value = os.environ.get("IO_TEST_KEY")
            return value if value is not None and len(value) >= 8 else None

        formatter = RedactingFormatter(getter)
        monkeypatch.setenv("IO_TEST_KEY", "oldvalue1234")
        first = _format(formatter, "key oldvalue1234 present")
        assert "oldvalue1234" not in first
        assert "<redacted>" in first

        monkeypatch.setenv("IO_TEST_KEY", "newvalue5678")
        second = _format(formatter, "key newvalue5678 present")
        assert "newvalue5678" not in second
        assert "<redacted>" in second

    def test_per_record_lookup_honours_reconfigure(self) -> None:
        """(a, context flavour) Re-pointing the getter target between records:
        the second record redacts the new value, not only the first."""
        state = {"key": "firstkey9012"}

        def getter() -> str | None:
            return state["key"]

        formatter = RedactingFormatter(getter)
        first = _format(formatter, "key firstkey9012 present")
        assert "firstkey9012" not in first

        state["key"] = "secondkey3456"
        second = _format(formatter, "key secondkey3456 present")
        assert "secondkey3456" not in second
        assert "<redacted>" in second

    def test_getter_not_called_at_construction_time(self) -> None:
        """(b) The getter must not run in ``__init__`` — construction is a
        pure state setup; resolution happens at format time."""
        calls: list[int] = []

        def getter() -> str | None:
            calls.append(1)
            return "somekey1234"

        RedactingFormatter(getter)
        assert calls == []

        _format(RedactingFormatter(getter), "somekey1234 inline")
        assert calls == [1]

    def test_short_value_not_redacted(self) -> None:
        """(c) A value < 8 chars is never redacted (the >= 8 rule lives in
        the caller-side getter, fail-safe preserved)."""

        def getter() -> str | None:
            value = "short"
            return value if len(value) >= 8 else None

        formatter = RedactingFormatter(getter)
        out = _format(formatter, "value short visible")
        assert "short visible" in out
        assert "<redacted>" not in out

    def test_none_and_empty_are_not_redacted(self) -> None:
        """(c, fail-safe) ``None`` or an empty lookup result must not crash
        and must not redact."""
        for getter in (lambda: None, lambda: ""):
            out = _format(RedactingFormatter(getter), "no key here")
            assert "no key here" in out
            assert "<redacted>" not in out

    def test_hf_token_and_bearer_still_redacted(self) -> None:
        """(d) The token/Bearer regex redaction, alongside the getter."""
        formatter = RedactingFormatter()  # default getter -> None
        out = _format(
            formatter,
            "auth Bearer sk-test-1234567890 and hf_abcdef1234567890 here",
        )
        assert "sk-test-1234567890" not in out
        assert "hf_abcdef1234567890" not in out
        assert "Bearer <redacted>" in out
        assert "hf_<redacted>" in out

    def test_exception_text_redacted_per_record(self) -> None:
        """Getter + regexes apply to ``exc_text`` (the reason the redactor is
        a Formatter, not a Filter)."""
        record = logging.LogRecord(
            name="vemoizer.t",
            level=logging.ERROR,
            pathname=__file__,
            lineno=0,
            msg="op failed",
            args=(),
            exc_info=None,
        )
        # Populate exc_text as a handler would (format() populates exc_text
        # from record.exc_info).
        try:
            raise RuntimeError("Bearer sk-abc1234567890 and key exckey123456")
        except RuntimeError:
            record.exc_info = sys.exc_info()

        formatter = RedactingFormatter(lambda: "exckey12345")
        out = formatter.format(record)
        assert "sk-abc1234567890" not in out
        assert "exckey123456" not in out
        # The traceback is present (redacted), proving exc_text was covered.
        assert "Traceback (most recent call last):" in out
        assert "<redacted>" in out
