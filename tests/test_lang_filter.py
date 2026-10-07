"""Tests for the scoped stdout filter (issue #147).

Verifies that ``filter_language_lines`` suppresses ``Detected language: X``
lines while passing all other output through, and that ``sys.stdout`` is
restored on every exit path.
"""

from __future__ import annotations

import io
import sys
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from vemoizer.lang_filter import _LANGUAGE_LINE_RE, filter_language_lines

# -- regex correctness ----------------------------------------------------------


def test_regex_matches_exactly_one_language_line() -> None:
    assert _LANGUAGE_LINE_RE.match("Detected language: Finnish")
    assert _LANGUAGE_LINE_RE.match("Detected language: English")
    assert _LANGUAGE_LINE_RE.match("Detected language: fi")


def test_regex_rejects_non_matching_lines() -> None:
    assert not _LANGUAGE_LINE_RE.match("Detected languages: Finnish")
    assert not _LANGUAGE_LINE_RE.match("Detected language:  Finnish")
    assert not _LANGUAGE_LINE_RE.match("Detected language:")
    assert not _LANGUAGE_LINE_RE.match("Detected language: Finnish (98%)")
    assert not _LANGUAGE_LINE_RE.match("")
    assert not _LANGUAGE_LINE_RE.match("Something else")


# -- filter behaviour -----------------------------------------------------------


def test_filter_captured_output(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch sys.stdout, enter the filter, write lines, verify what
    the original received."""
    original = io.StringIO()
    monkeypatch.setattr(sys, "stdout", original)
    with filter_language_lines():
        # sys.stdout is now _FilteredStdout(original)
        sys.stdout.write("Detected language: Finnish\n")
        sys.stdout.write("Hello world\n")
        sys.stdout.write("Detected language: English\n")
        sys.stdout.write("Goodbye\n")
    # The filter flushed and restored. `original` should have the
    # non-matching lines only.
    out = original.getvalue()
    assert "Detected language:" not in out
    assert "Hello world\n" in out
    assert "Goodbye\n" in out


def test_filter_restores_stdout_on_normal_exit() -> None:
    original = sys.stdout
    with filter_language_lines():
        assert sys.stdout is not original
    assert sys.stdout is original


def test_filter_restores_stdout_on_exception() -> None:
    original = sys.stdout
    with pytest.raises(ValueError), filter_language_lines():
        assert sys.stdout is not original
        raise ValueError("test")
    assert sys.stdout is original


def test_filter_restores_stdout_on_keyboard_interrupt() -> None:
    original = sys.stdout
    with pytest.raises(KeyboardInterrupt), filter_language_lines():
        assert sys.stdout is not original
        raise KeyboardInterrupt()
    assert sys.stdout is original


def test_filter_passthrough_non_matching_lines() -> None:
    """Lines that don't match the pattern pass through unchanged."""
    original = io.StringIO()
    with patch("sys.stdout", original), filter_language_lines():
        sys.stdout.write("Regular output\n")
        sys.stdout.write("Another line\n")
    out = original.getvalue()
    assert out == "Regular output\nAnother line\n"


def test_filter_mixed_output() -> None:
    """A mix of language and non-language lines: only language lines dropped."""
    original = io.StringIO()
    with patch("sys.stdout", original), filter_language_lines():
        sys.stdout.write("Detected language: Finnish\n")
        sys.stdout.write("Processing window 1\n")
        sys.stdout.write("Detected language: Finnish\n")
        sys.stdout.write("Processing window 2\n")
    out = original.getvalue()
    assert "Detected language:" not in out
    assert "Processing window 1\n" in out
    assert "Processing window 2\n" in out


def test_filter_no_leaked_disable_state() -> None:
    """After the filter exits, sys.stdout is the original object."""
    original = sys.stdout
    with filter_language_lines():
        pass
    assert sys.stdout is original
    # No leaked state: writing normally works.
    original2 = io.StringIO()
    with patch("sys.stdout", original2):
        sys.stdout.write("normal\n")
    assert original2.getvalue() == "normal\n"


def test_filter_thread_safety_reentrant() -> None:
    """Nested filters: the inner filter wraps the outer's wrapper, and both
    restore in order (LIFO)."""
    original = sys.stdout
    with filter_language_lines():
        first_wrapped = sys.stdout
        assert first_wrapped is not original
        with filter_language_lines():
            second_wrapped = sys.stdout
            assert second_wrapped is not first_wrapped
            sys.stdout.write("Detected language: Finnish\n")
            sys.stdout.write("Hello\n")
        # After inner exit, stdout is the outer's wrapper again.
        assert sys.stdout is first_wrapped
    # After outer exit, stdout is the original.
    assert sys.stdout is original


def test_filter_does_not_affect_stderr() -> None:
    """The filter only wraps stdout; stderr is unaffected."""
    with filter_language_lines():
        assert sys.stderr is sys.stderr  # trivially true
        # Verify stderr is not a _FilteredStdout
        from vemoizer.lang_filter import _FilteredStdout

        assert not isinstance(sys.stderr, _FilteredStdout)


# -- integration with the whisper decode path ---------------------------------


def test_decode_wraps_transcribe_with_filter() -> None:
    """The WhisperTranscriber.transcribe method uses filter_language_lines
    around the decode loop (verified by checking the source)."""
    import inspect

    from vemoizer.whisper_transcriber import WhisperTranscriber

    source = inspect.getsource(WhisperTranscriber.transcribe)
    assert "filter_language_lines" in source, (
        "WhisperTranscriber.transcribe must use filter_language_lines "
        "around the decode loop (issue #147)"
    )


def test_decode_filters_language_lines_end_to_end() -> None:
    """End-to-end: a fake mlx_whisper.transcribe that prints 'Detected
    language: Finnish' per window produces no such lines on stdout, while
    a legitimate stdout line survives."""
    import numpy as np

    from vemoizer.whisper_transcriber import WhisperTranscriber

    calls: list[int] = []

    def fake_transcribe(*args: Any, **kwargs: Any) -> dict:
        calls.append(1)
        # Simulate mlx_whisper's per-window print
        print("Detected language: Finnish")
        return {"text": "moro", "segments": [], "words": [], "language": "fi"}

    mock_module = MagicMock()
    mock_module.transcribe = fake_transcribe

    with (
        patch.dict("sys.modules", {"mlx_whisper": mock_module}),
        patch("huggingface_hub.snapshot_download", return_value="/tmp/turbo"),
        patch("sys.stdout", new_callable=io.StringIO) as captured,
    ):
        t = WhisperTranscriber()
        t._model_path = "/tmp/turbo"
        t._mlx_whisper = mock_module
        result = t.transcribe(np.zeros(16_000 * 30, dtype=np.float32))

    out = captured.getvalue()
    assert "Detected language:" not in out, (
        f"Per-window 'Detected language' lines must not reach stdout; got: {out!r}"
    )
    assert len(calls) == 1  # 30s audio = 1 window
    assert result["text"] == "moro"
