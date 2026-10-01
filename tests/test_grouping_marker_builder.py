"""Unit tests for ``grouping_common.with_part_markers`` (issue #77).

The batch layer must attach the ``part_markers`` sidecar without mutating
the pipeline's result dict in place. ``with_part_markers`` is the single
place that does so: it returns a NEW dict (``{**result, "part_markers":
markers}``) built from the ``PartOffset`` list, with labels in the
``— osa N (äänitys X) —`` form. Single-file groups (no offsets) get the
input dict back unchanged (no ``part_markers`` key at all).
"""

from __future__ import annotations

from vemoizer.grouping_common import PartOffset, with_part_markers


def _offsets() -> list[PartOffset]:
    return [
        PartOffset(
            part_number=1,
            source_filename="Uusi äänitys 425.m4a",
            start_offset=0.0,
        ),
        PartOffset(
            part_number=2,
            source_filename="Uusi äänitys 426.m4a",
            start_offset=2.5,
        ),
    ]


def test_with_part_markers_returns_new_dict_not_mutated() -> None:
    """A NEW dict is returned; the input dict is not mutated."""
    result = {"text": "hei", "segments": []}
    out = with_part_markers(result, _offsets())
    assert out is not result
    # The input dict was not mutated: no part_markers key added to it.
    assert "part_markers" not in result
    # The new dict carries the markers.
    assert "part_markers" in out
    # All other keys are preserved (shallow copy semantics).
    assert out["text"] == "hei"
    assert out["segments"] == []


def test_with_part_markers_exact_labels_and_offsets() -> None:
    """The label format is exactly ``— osa N (äänitys X) —`` and the
    offsets match the PartOffset start_offset values."""
    result = {"text": "hei"}
    out = with_part_markers(result, _offsets())
    markers = out["part_markers"]
    assert markers == [
        {
            "offset": 0.0,
            "label": "— osa 1 (äänitys Uusi äänitys 425.m4a)",
        },
        {
            "offset": 2.5,
            "label": "— osa 2 (äänitys Uusi äänitys 426.m4a)",
        },
    ]


def test_with_part_markers_single_part_no_key() -> None:
    """A single-part group (one offset, or no offsets) carries no
    ``part_markers`` key at all — the input dict is returned unchanged."""
    # Empty offsets list (single-file group).
    result = {"text": "hei"}
    out = with_part_markers(result, [])
    assert out is result
    assert "part_markers" not in out


def test_with_part_markers_preserves_existing_keys() -> None:
    """Pre-existing result keys (including an unrelated nested value) are
    preserved on the returned dict."""
    segments = [{"start": 0.0, "end": 1.0, "text": "a"}]
    result = {"text": "hei", "segments": segments, "language": "fi"}
    out = with_part_markers(result, _offsets())
    assert out["segments"] is segments  # shallow copy: same list object
    assert out["language"] == "fi"
    assert "part_markers" in out
