"""Over-suppression fix: single-term segments are kept (issue #109).

An echo is a RUN of prompt terms or a segment carrying the label; a lone
glossary term ("Jira.", "Kubernetes") is a real one-word answer and must
be kept.

Also covers the multi-window text assembly in WhisperTranscriber.transcribe
(issue #109 fix): the result["text"] field must include text from ALL
windows, not just the last.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np

from vemoizer.echo_filter import echo_vocabulary, filter_echo_segments
from vemoizer.whisper_transcriber import WhisperTranscriber


def _echo_vocab(prompt):
    return echo_vocabulary(prompt)


def test_filter_keeps_single_term_segment() -> None:
    """A segment that is exactly one glossary term is kept (not an echo).

    A speaker answering just "Jira." or "Kubernetes" loses nothing: a bare
    term with no other words is more often real speech than a prompt
    continuation.
    """
    for text in ("Jira.", "Kubernetes"):
        seg = {"text": text, "start": 0.0, "end": 0.5, "words": []}
        vocab = _echo_vocab("Jira, Kubernetes, DCS.")
        segments, _ = filter_echo_segments([seg], 0.0, vocab)
        assert len(segments) == 1, f"{text!r} was incorrectly dropped"
        assert segments[0]["text"] == text


def test_filter_drops_single_label_segment() -> None:
    """Sanasto alone IS dropped (the label is never real speech)."""
    seg = {"text": "Sanasto.", "start": 0.0, "end": 0.5, "words": []}
    vocab = _echo_vocab("Pia, DCS.")
    segments, _ = filter_echo_segments([seg], 0.0, vocab)
    assert segments == []


def test_filter_drops_two_term_run() -> None:
    """A two-token run of prompt terms (Jira, DCS.) is dropped."""
    seg = {"text": "Jira, DCS.", "start": 0.0, "end": 0.8, "words": []}
    vocab = _echo_vocab("Jira, DCS.")
    segments, _ = filter_echo_segments([seg], 0.0, vocab)
    assert segments == []


def test_filter_drops_label_plus_terms() -> None:
    """Sanasto, Pia, NG-TOPI, IBC. — the canonical echo, is dropped."""
    seg = {
        "text": "Sanasto, Pia, NG-TOPI, IBC.",
        "start": 0.0,
        "end": 1.5,
        "words": [],
    }
    vocab = _echo_vocab("Pia, NG-TOPI, IBC.")
    segments, _ = filter_echo_segments([seg], 0.0, vocab)
    assert segments == []


def test_filter_keeps_real_sentence_with_one_term_new_rule() -> None:
    """A real sentence with one glossary term is kept (regression guard)."""
    seg = {
        "text": "We should use Jira for the tickets.",
        "start": 0.0,
        "end": 1.0,
        "words": [],
    }
    vocab = _echo_vocab("Jira, DCS.")
    segments, _ = filter_echo_segments([seg], 0.0, vocab)
    assert len(segments) == 1
    assert segments[0]["text"] == "We should use Jira for the tickets."


def test_filter_empty_and_punctuation_only_unaffected() -> None:
    """Empty or punctuation-only segments are neither dropped nor kept.

    Empty text is skipped by the caller (``if text and _is_echo(...)``);
    punctuation-only text has zero tokens → ``matched == 0`` → not an echo.
    Both must survive untouched.
    """
    for text in ("", ",,,", ".", " . . "):
        seg = {"text": text, "start": 0.0, "end": 0.1, "words": []}
        vocab = _echo_vocab("Jira, DCS.")
        segments, _ = filter_echo_segments([seg], 0.0, vocab)
        assert len(segments) == 1, f"{text!r} was unexpectedly dropped"


# -- Multi-window text assembly (issue #109) ---------------------------------


def _mk_raw(texts_and_words):
    """Build a raw whisper transcribe() result dict."""
    segments = []
    for text, words in texts_and_words:
        segments.append(
            {
                "text": text,
                "start": words[0]["start"] if words else 0.0,
                "end": words[-1]["end"] if words else 0.5,
                "words": words,
            }
        )
    return {
        "text": " ".join(t for t, _ in texts_and_words),
        "language": "fi",
        "segments": segments,
    }


def _mk_word(word, start, end):
    return {"word": word, "start": start, "end": end}


def test_multimindow_text_contains_all_windows() -> None:
    """result['text'] must contain text from ALL windows, not just the last.

    A 90 s recording is 3 x 30 s windows. Each window returns distinct text.
    The final result['text'] must have all three, in order.
    """
    w1 = [("alpha one", [_mk_word("alpha", 0.0, 0.5), _mk_word("one", 0.6, 1.0)])]
    w2 = [("bravo two", [_mk_word("bravo", 0.0, 0.5), _mk_word("two", 0.6, 1.0)])]
    w3 = [
        ("charlie three", [_mk_word("charlie", 0.0, 0.5), _mk_word("three", 0.6, 1.0)])
    ]

    mock = MagicMock()
    mock.transcribe = MagicMock(side_effect=[_mk_raw(w1), _mk_raw(w2), _mk_raw(w3)])
    with (
        patch.dict("sys.modules", {"mlx_whisper": mock}),
        patch("huggingface_hub.snapshot_download", return_value="/tmp/turbo"),
    ):
        t = WhisperTranscriber(initial_prompt="alpha, bravo, charlie.")
        result = t.transcribe(np.zeros(90 * 16_000, dtype=np.float32))

    assert "alpha one" in result["text"], f"window 1 text missing: {result['text']!r}"
    assert "bravo two" in result["text"], f"window 2 text missing: {result['text']!r}"
    assert "charlie three" in result["text"], (
        f"window 3 text missing: {result['text']!r}"
    )
    # Order: windows appear in recording order
    assert result["text"].index("alpha one") < result["text"].index("bravo two")
    assert result["text"].index("bravo two") < result["text"].index("charlie three")
    # segments and words cover all windows
    assert len(result["segments"]) == 3
    assert len(result["words"]) == 6


def test_multimindow_all_echo_window_not_in_text() -> None:
    """When every segment in a window is an echo, result['text'] must NOT
    contain the echo, and the surrounding windows' text must survive.

    Window 2 is the canonical echo (Sanasto, Pia, NG-TOPI, IBC.); windows 1
    and 3 have real speech. The echo must not resurrect via raw['text']."""
    w1 = [("moro vaan", [_mk_word("moro", 0.0, 0.5), _mk_word("vaan", 0.6, 1.0)])]
    w2 = [("Sanasto, Pia, NG-TOPI, IBC.", [_mk_word("Sanasto", 0.0, 1.5)])]
    w3 = [("kiitos", [_mk_word("kiitos", 0.0, 0.5)])]

    mock = MagicMock()
    mock.transcribe = MagicMock(side_effect=[_mk_raw(w1), _mk_raw(w2), _mk_raw(w3)])
    with (
        patch.dict("sys.modules", {"mlx_whisper": mock}),
        patch("huggingface_hub.snapshot_download", return_value="/tmp/turbo"),
    ):
        t = WhisperTranscriber(initial_prompt="Pia, NG-TOPI, IBC.")
        result = t.transcribe(np.zeros(90 * 16_000, dtype=np.float32))

    # The echo must NOT appear in the headline text
    assert "Sanasto" not in result["text"], (
        f"echo resurrected via raw fallback: {result['text']!r}"
    )
    # Surrounding windows survive
    assert "moro vaan" in result["text"]
    assert "kiitos" in result["text"]


def test_multimindow_zero_segments_fallback_preserved() -> None:
    """A window that returned raw with NO segments key (falsy) falls back to
    raw['text'] — the old behaviour for malformed payloads is preserved."""
    w1 = [("hello world", [_mk_word("hello", 0.0, 0.5), _mk_word("world", 0.6, 1.0)])]
    # Window 2 has "segments" as None (falsy) but has raw text
    raw2 = {"text": "some text from raw", "language": "fi", "segments": None}
    w3 = [
        (
            "goodbye friend",
            [_mk_word("goodbye", 0.0, 0.5), _mk_word("friend", 0.6, 1.0)],
        )
    ]

    mock = MagicMock()
    mock.transcribe = MagicMock(side_effect=[_mk_raw(w1), raw2, _mk_raw(w3)])
    with (
        patch.dict("sys.modules", {"mlx_whisper": mock}),
        patch("huggingface_hub.snapshot_download", return_value="/tmp/turbo"),
    ):
        t = WhisperTranscriber(initial_prompt="hello, goodbye.")
        result = t.transcribe(np.zeros(90 * 16_000, dtype=np.float32))

    # The fallback text from raw["text"] must appear for the zero-segment window
    assert "some text from raw" in result["text"]
    # Other windows also present
    assert "hello world" in result["text"]
    assert "goodbye friend" in result["text"]
