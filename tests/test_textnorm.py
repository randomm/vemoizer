"""Tests for text normalization (issue #11).

Pure-stdlib: no fixtures, no network, no models.
"""

from __future__ import annotations

from vemoizer.textnorm import textnorm


def test_identity_on_already_clean_input() -> None:
    assert textnorm("moro aami") == "moro aami"


def test_casefold_not_just_lower() -> None:
    # casefold handles more than lower(): ß -> ss
    assert textnorm("Straße") == "strasse"


def test_uppercase_finnish_becomes_lowercase() -> None:
    assert textnorm("MORO AAMII") == "moro aamii"


def test_strips_punctuation_to_spaces() -> None:
    assert textnorm("Moro, aami! Oke.") == "moro aami oke"


def test_collapses_whitespace_runs() -> None:
    assert textnorm("a  b\t\tc\n\nd") == "a b c d"


def test_leading_and_trailing_whitespace_stripped() -> None:
    assert textnorm("  hello  ") == "hello"


def test_empty_string_stays_empty() -> None:
    assert textnorm("") == ""


def test_punctuation_only_becomes_empty() -> None:
    assert textnorm("...,,!!") == ""


def test_finnish_special_letters_survive() -> None:
    # å ä ö are word characters and must not be treated as punctuation
    assert textnorm("ÄITI ÄÄ") == "äiti ää"


def test_dashes_and_ellipses_are_stripped() -> None:
    assert textnorm("hmm… siis, noin.") == "hmm siis noin"


def test_mixed_case_and_punctuation_combined() -> None:
    assert textnorm("Hello, WORLD!  Again...") == "hello world again"
