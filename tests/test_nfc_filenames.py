"""Tests for NFC filename normalization (issue #10, task C).

The contract under test: every filename that enters or leaves the
process is NFC, so two spellings of the same on-disk file (NFC vs NFD —
the real APFS situation) produce equal ``Path`` objects and hash
identically. That is the property a batch dedup / output-collision check
needs, and it is the bug this module closes.

The NFD fixture names are built in-test via ``unicodedata.normalize``
(as the issue's test-surface plan prescribes) so git never stores the
wrong form in the test source itself.
"""

from __future__ import annotations

import unicodedata
from pathlib import Path

import pytest

from vemoizer.output.naming import nfc, nfc_path, nfc_stem_and_suffix


def _nfd(name: str) -> str:
    """Decompose *name* to NFD — the form APFS stores filenames in."""
    return unicodedata.normalize("NFD", name)


# A Finnish name with a combining acute (NFD: ``o`` + U+0301). ``"mó"``
# is NFC. Both spell the same string for a human; to Python they are not
# equal.
NFC_MEMO = "mó"
NFD_MEMO = "mo\u0301"  # "mo" + combining acute


# ---------------------------------------------------------------------------
# nfc() — the pure string function
# ---------------------------------------------------------------------------


def test_nfc_of_nfc_is_nfc():
    assert nfc(NFC_MEMO) == NFC_MEMO
    assert nfc(NFC_MEMO) == "mó"


def test_nfc_of_nfd_is_nfc():
    assert nfc(NFD_MEMO) == NFC_MEMO
    assert nfc(_nfd("mó")) == "mó"


def test_nfc_is_idempotent():
    once = nfc(NFD_MEMO)
    twice = nfc(once)
    assert once == twice


def test_nfc_of_ascii_is_unchanged():
    assert nfc("plain-ascii-123.txt") == "plain-ascii-123.txt"


def test_nfc_preserves_no_mark_letters():
    # Characters with no decomposition pass through untouched.
    assert nfc("abcXYZ_09-") == "abcXYZ_09-"


def test_nfc_of_full_nfd_fixture():
    fixture = _nfd("mó")
    assert nfc(fixture) == "mó"
    # And the composed form is what a user would have typed in the CLI.
    assert nfc(fixture) == NFC_MEMO


# ---------------------------------------------------------------------------
# nfc_path() — the per-component path normalization
# ---------------------------------------------------------------------------


def test_nfc_path_filename_only():
    p = nfc_path(Path(NFD_MEMO) / "memo.m4a")
    assert p == Path("mó/memo.m4a")


def test_nfc_path_normalizes_all_components():
    p = nfc_path(Path(_nfd("mó")) / _nfd("tiedostoja") / _nfd("memo.m4a"))
    assert p == Path("mó/tiedostoja/memo.m4a")
    # Every part is NFC now; re-normalizing is a no-op.
    assert nfc_path(p) == p


def test_nfc_path_of_already_nfc_is_unchanged():
    p = Path("mó/tiedostoja/memo.m4a")
    assert nfc_path(p) == p


def test_nfc_path_two_spellings_of_same_disk_file_are_equal():
    # The core bug: APFS has one file "mó/memo.m4a"; a user path may
    # arrive in NFD. Both spellings must land on the same Path.
    a = nfc_path(Path(NFC_MEMO) / "memo.m4a")
    b = nfc_path(Path(NFD_MEMO) / "memo.m4a")
    assert a == b
    assert hash(a) == hash(b)


def test_nfc_path_relative_stays_relative():
    p = nfc_path(Path(NFD_MEMO) / "memo.m4a")
    assert not p.is_absolute()
    assert p.parts == ("mó", "memo.m4a")


def test_nfc_path_absolute_posix():
    p = nfc_path(Path("/" + NFD_MEMO + "/memo.m4a"))
    assert p.is_absolute()
    assert p.parts == ("/", "mó", "memo.m4a")


def test_nfc_path_accepts_str_input():
    p = nfc_path(NFD_MEMO + "/memo.m4a")
    assert p == Path("mó/memo.m4a")


def test_nfc_path_single_component_filename():
    p = nfc_path(NFD_MEMO)
    assert p == Path("mó")
    assert p.parts == ("mó",)


def test_nfc_path_dot_stem_and_suffix_intact():
    # Normalization must not eat the dot or split the suffix.
    p = nfc_path(Path(_nfd("naïve.txt")))
    assert p.name == "naïve.txt"
    assert p.suffix == ".txt"


# ---------------------------------------------------------------------------
# nfc_stem_and_suffix()
# ---------------------------------------------------------------------------


def test_stem_and_suffix_of_nfd():
    stem, suffix = nfc_stem_and_suffix(_nfd("mó") + ".m4a")
    assert stem == "mó"
    assert suffix == ".m4a"
    # The two pieces rejoin to the full NFC name.
    assert stem + suffix == "mó.m4a"


def test_stem_and_suffix_of_nfc_is_unchanged():
    stem, suffix = nfc_stem_and_suffix("mó.m4a")
    assert stem == "mó"
    assert suffix == ".m4a"


def test_stem_and_suffix_no_extension():
    stem, suffix = nfc_stem_and_suffix(NFD_MEMO)
    assert stem == "mó"
    assert suffix == ""


def test_stem_and_suffix_suffix_with_mark():
    # A legal edge: the *extension* carries a combining mark.
    p = _nfd("file.naïve")
    stem, suffix = nfc_stem_and_suffix(p)
    assert stem == "file"
    assert suffix == ".naïve"
    assert stem + suffix == nfc(p)


# ---------------------------------------------------------------------------
# The APFS scenario, end to end
# ---------------------------------------------------------------------------


def test_apfs_nfd_disk_name_vs_nfc_user_string():
    # Simulate: APFS stored "mó/memo.m4a" as NFD; the user typed the
    # same path in NFC. After normalization both refer to the same
    # output path and dedup collapses them to one.
    from_disk_nfd = nfc_path(Path(_nfd("mó")) / _nfd("memo.m4a"))
    from_user_nfc = nfc_path(Path("mó") / "memo.m4a")
    assert from_disk_nfd == from_user_nfc
    # A batch dedup keyed on the normalized path sees one entry, not two.
    assert {from_disk_nfd, from_user_nfc} == {from_user_nfc}


@pytest.mark.parametrize("raw", ["mo\u0301", "mó", "m\u0332o", "plain"])
def test_nfc_is_the_fixed_point(raw):
    """Once composed, further normalization is a no-op (NFC stability)."""
    assert nfc(nfc(raw)) == nfc(raw)
