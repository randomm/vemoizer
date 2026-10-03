"""Over-suppression fix: single-term segments are kept (issue #109).

An echo is a RUN of prompt terms or a segment carrying the label; a lone
glossary term ("Jira.", "Kubernetes") is a real one-word answer and must
be kept.
"""

from __future__ import annotations

from vemoizer.echo_filter import echo_vocabulary, filter_echo_segments


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
