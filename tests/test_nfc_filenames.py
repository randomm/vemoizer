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

from vemoizer.output.naming import (
    collision_free_path,
    dated_basename,
    nfc,
    nfc_path,
    nfc_stem_and_suffix,
    sanitize_title,
)


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


# ---------------------------------------------------------------------------
# sanitize_title() — LLM title -> dated output filename (issue #82)
# ---------------------------------------------------------------------------


def test_sanitize_title_plain_unchanged():
    assert sanitize_title("Q4 planning review") == "Q4 planning review"


def test_sanitize_title_removes_path_separators():
    assert sanitize_title("a/b\\c") == "abc"


def test_sanitize_title_removes_control_characters():
    assert sanitize_title("tab\there\nnewline\x00nul\x7fdel") == "tabherenewlinenuldel"


def test_sanitize_title_removes_zero_width_and_bom():
    assert sanitize_title("a\u200bb\u200ec\ufeff") == "abc"


def test_sanitize_title_strips_leading_trailing_dots_and_spaces():
    assert sanitize_title("  ..title...  ") == "title"


def test_sanitize_title_collapses_internal_whitespace_runs():
    assert sanitize_title("a\n  b\t\t c") == "a b c"


def test_sanitize_title_caps_at_80_chars():
    long = "x" * 120
    assert len(sanitize_title(long)) == 80
    assert sanitize_title(long) == "x" * 80


def test_sanitize_title_empty_for_blank_input():
    assert sanitize_title("") == ""
    assert sanitize_title("   ") == ""
    # A title made only of stripped dots and spaces.
    assert sanitize_title(". . .") == ""


def test_sanitize_title_empty_for_separators_only():
    assert sanitize_title("/\\//") == ""


def test_sanitize_title_maps_colon_to_en_dash():
    # ":" is the legacy HFS path separator on macOS (Finder shows it as
    # "/"), and invalid on Windows/SMB — map it to an en dash with
    # surrounding spaces instead of keeping it (issue #110).
    assert sanitize_title("Planning: NG Nordic & IWS") == "Planning – NG Nordic & IWS"


def test_sanitize_title_removes_windows_invalid_chars():
    # "? * " < > |" are invalid in Windows and SMB filenames (issue #110).
    assert sanitize_title('a?b*c"d<e>f|g') == "abcdefg"


def test_sanitize_title_colon_surrounding_whitespace_collapses():
    # "a:  b" -> "a –  b" -> "a – b" (whitespace-run collapse applies).
    assert sanitize_title("a: b") == "a – b"


def test_sanitize_title_nfc_output():
    # An NFD title comes back composed.
    assert sanitize_title(NFD_MEMO + " review") == "mó review"


# ---------------------------------------------------------------------------
# dated_basename() — YYYY-MM-DD <title> base
# ---------------------------------------------------------------------------


def test_dated_basename_formats_date_and_title():
    assert dated_basename("Board sync", date_str="2026-01-15") == (
        "2026-01-15 Board sync"
    )


def test_dated_basename_sanitises_the_title():
    # Separators are dropped, so "/bad/../title" -> "bad.title".
    assert dated_basename("/bad/../title", date_str="2026-01-15") == (
        "2026-01-15 bad.title"
    )


def test_dated_basename_falls_back_to_first_source_stem():
    # Blank LLM title -> deterministic fallback to the first source stem.
    assert dated_basename("   ", date_str="2026-01-15", fallback_stem="memo-001") == (
        "2026-01-15 memo-001"
    )
    assert dated_basename("", date_str="2026-01-15", fallback_stem="mó") == (
        "2026-01-15 mó"
    )


def test_dated_basename_fallback_stem_also_sanitised():
    # Separators dropped, trailing space stripped.
    assert dated_basename("", date_str="2026-01-15", fallback_stem="a/b") == (
        "2026-01-15 ab"
    )


def test_dated_basename_title_beats_fallback():
    assert dated_basename("Real title", date_str="d", fallback_stem="ignored") == (
        "d Real title"
    )


def test_dated_basename_uses_today_without_explicit_date():
    # No date_str given: today's date is used in ISO format.
    base = dated_basename("x")
    import re as _re

    assert _re.fullmatch(r"\d{4}-\d{2}-\d{2} x", base) is not None


def test_dated_basename_raises_when_both_empty():
    import pytest

    with pytest.raises(ValueError):
        dated_basename("", date_str="2026-01-15", fallback_stem="")


# ---------------------------------------------------------------------------
# collision_free_path() — " (2)" suffix, NFC, never overwrites
# ---------------------------------------------------------------------------


def test_collision_free_path_returns_plain_name_when_free(tmp_path):
    p = collision_free_path(tmp_path, "2026-01-15 Title", ".md")
    assert p == tmp_path / "2026-01-15 Title.md"
    assert not p.exists()


def test_collision_free_path_appends_suffix_2_then_3(tmp_path):
    (tmp_path / "a.md").write_text("x")
    (tmp_path / "a (2).md").write_text("y")
    p = collision_free_path(tmp_path, "a", ".md")
    assert p == tmp_path / "a (3).md"
    assert not p.exists()


def test_collision_free_path_does_not_overwrite(tmp_path):
    existing = tmp_path / "a (2).md"
    existing.write_text("keep me")
    (tmp_path / "a.md").write_text("x")
    p = collision_free_path(tmp_path, "a", ".md")
    assert p != existing
    p.write_text("new")
    assert existing.read_text() == "keep me"


def test_collision_free_path_json_suffix(tmp_path):
    (tmp_path / "a.md").write_text("x")
    (tmp_path / "a.json").write_text("y")
    p = collision_free_path(tmp_path, "a", ".json")
    assert p == tmp_path / "a (2).json"


def test_collision_free_path_nfc_normalises_candidate(tmp_path):
    # The candidate base arrives NFD (as if decoded from disk); the
    # existing file is NFC. Both must be treated as one name.
    (tmp_path / (NFC_MEMO + ".md")).write_text("x")
    p = collision_free_path(tmp_path, NFD_MEMO, ".md")
    assert p == tmp_path / "mó (2).md"


def test_collision_free_path_nfc_normalises_existing(tmp_path):
    # The existing file on disk is NFD-spelled; the candidate is NFC.
    # Both must be treated as one name (APFS folds them to one file,
    # but the probe must not miss the collision).
    (tmp_path / (NFD_MEMO + ".md")).write_text("x")
    p = collision_free_path(tmp_path, NFC_MEMO, ".md")
    assert p == tmp_path / "mó (2).md"


def test_collision_free_path_nfc_normalises_directory(tmp_path):
    sub = nfc_path(tmp_path / NFD_MEMO)
    sub.mkdir()
    (sub / "a.md").write_text("x")
    p = collision_free_path(sub, "a", ".md")
    assert p.parent == sub
    assert p.name == "a (2).md"


def test_collision_free_path_md_json_pair_never_overwrites(tmp_path):
    # The md + json pair: both colliding forces the suffix on both.
    (tmp_path / "t.md").write_text("x")
    (tmp_path / "t.json").write_text("y")
    md = collision_free_path(tmp_path, "t", ".md")
    js = collision_free_path(tmp_path, "t", ".json")
    assert md == tmp_path / "t (2).md"
    assert js == tmp_path / "t (2).json"


def test_collision_free_path_existing_md_frees_json(tmp_path):
    # Only the .md exists; the .json base is still free.
    (tmp_path / "t.md").write_text("x")
    assert collision_free_path(tmp_path, "t", ".md") == tmp_path / "t (2).md"
    assert collision_free_path(tmp_path, "t", ".json") == tmp_path / "t.json"


def test_collision_free_path_suffix_applied_before_extension(tmp_path):
    # Regression: " (2)" goes before the suffix, not into the filename.
    (tmp_path / "t.md").write_text("x")
    p = collision_free_path(tmp_path, "t", ".md")
    assert p.suffix == ".md"
    assert p.name == "t (2).md"


def test_collision_free_path_str_directory(tmp_path):
    (tmp_path / "a.md").write_text("x")
    p = collision_free_path(str(tmp_path), "a", ".md")
    assert p == tmp_path / "a (2).md"
