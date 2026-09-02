"""Tests for the DTW word-onset alignment (issue #7, task A).

The DTW cost function is:

    cost(i, j) = (|start_a[i] - start_b[j]|)^2 * (1 - sim(word_a[i], word_b[j]))

where ``sim`` is 1 - normalised Levenshtein distance over the
case/punctuation-stripped tokens. Insertion/deletion costs are
``GAP_PENALTY`` (default 1.0). The alignment is monotonic.

These tests are pure-stdlib: no model imports, no network, no fixtures.
"""

from __future__ import annotations

from vemoizer.alignment import GAP_PENALTY, _levenshtein_similarity, dtw_align


def _starts(words: list[dict[str, float | str]]) -> list[float]:
    return [float(w["start"]) for w in words]


# ---------------------------------------------------------------------------
# Trivial and boundary cases
# ---------------------------------------------------------------------------


def test_both_empty_inputs_return_empty() -> None:
    assert dtw_align([], []) == []


def test_a_empty_b_nonempty_returns_empty() -> None:
    words_b = [{"word": "hei", "start": 0.1}]
    assert dtw_align([], words_b) == []


def test_b_empty_a_nonempty_returns_empty() -> None:
    words_a = [{"word": "hei", "start": 0.1}]
    assert dtw_align(words_a, []) == []


def test_single_word_pair_matches_and_costs_zero_when_identical() -> None:
    a = [{"word": "hei", "start": 1.0}]
    b = [{"word": "hei", "start": 1.0}]
    result = dtw_align(a, b)
    assert len(result) == 1
    (aw,) = result
    assert aw.word_a == "hei"
    assert aw.word_b == "hei"
    assert aw.start_a == 1.0
    assert aw.start_b == 1.0
    assert aw.cost == 0.0


def test_single_word_pair_identical_text_with_time_drift_costs_zero() -> None:
    # Identical tokens, so text cost is 0; time drift does not matter.
    a = [{"word": "hei", "start": 1.0}]
    b = [{"word": "hei", "start": 2.0}]
    (aw,) = dtw_align(a, b)
    assert aw.cost == 0.0


def test_single_word_pair_case_difference_costs_zero() -> None:
    a = [{"word": "Mikko", "start": 1.0}]
    b = [{"word": "mikko", "start": 1.0}]
    (aw,) = dtw_align(a, b)
    assert aw.cost == 0.0


def test_single_word_pair_punctuation_difference_costs_zero() -> None:
    a = [{"word": "Mikko,", "start": 1.0}]
    b = [{"word": "mikko", "start": 1.0}]
    (aw,) = dtw_align(a, b)
    assert aw.cost == 0.0


def test_single_word_pair_with_text_difference_and_time_drift() -> None:
    a = [{"word": "Mikko", "start": 1.0}]
    b = [{"word": "Mikoa", "start": 2.0}]
    (aw,) = dtw_align(a, b)
    # "mikko" vs "mikoa": Levenshtein distance 2 (one substitution + one
    # deletion), max length 5, sim = 1 - 2/5 = 3/5.
    # cost = (2.0 - 1.0)^2 * (1 - 3/5) = 1 * 2/5 = 0.4
    assert abs(aw.cost - 0.4) < 1e-9


# ---------------------------------------------------------------------------
# Monotonicity: the aligned words must appear in the same temporal order
# as they did in the input lists.
# ---------------------------------------------------------------------------


def test_alignment_is_monotonic_in_a_order() -> None:
    a = [
        {"word": "hei", "start": 0.0},
        {"word": "sinä", "start": 1.0},
        {"word": "kuka", "start": 2.0},
    ]
    b = [
        {"word": "hei", "start": 0.0},
        {"word": "sinä", "start": 1.0},
        {"word": "kuka", "start": 2.0},
    ]
    result = dtw_align(a, b)
    assert len(result) == 3
    for idx, aw in enumerate(result):
        assert aw.start_a == _starts(a)[idx]
        assert aw.start_b == _starts(b)[idx]
        assert aw.word_a == a[idx]["word"]
        assert aw.word_b == b[idx]["word"]


def test_alignment_is_monotonic_with_time_drift_in_b() -> None:
    a = [
        {"word": "hei", "start": 0.0},
        {"word": "sinä", "start": 1.0},
        {"word": "kuka", "start": 2.0},
    ]
    # B's second word drifts late by 0.5s; DTW should still pair it with A's
    # second word (nearest-neighbor in time) rather than jumping.
    b = [
        {"word": "hei", "start": 0.0},
        {"word": "sinä", "start": 1.5},
        {"word": "kuka", "start": 2.5},
    ]
    result = dtw_align(a, b)
    # DTW pairs (0,0), (1,1), (2,2) — the straight path is still optimal
    # even with the drift, because there are no insertions/deletions.
    assert [aw.word_a for aw in result] == ["hei", "sinä", "kuka"]
    assert [aw.word_b for aw in result] == ["hei", "sinä", "kuka"]
    # Monotonic in both orders.
    a_starts = [aw.start_a for aw in result if aw.start_a is not None]
    b_starts = [aw.start_b for aw in result if aw.start_b is not None]
    assert a_starts == sorted(a_starts)
    assert b_starts == sorted(b_starts)


def test_alignment_with_insertion_in_b() -> None:
    # B has an extra word between two A words; DTW should pair around it.
    # The extra word is "x" (very different from "w0" and "w2") so the
    # gap cost (1.0) is cheaper than pairing it with either A word.
    a = [
        {"word": "w0", "start": 0.0},
        {"word": "w2", "start": 1.0},
    ]
    b = [
        {"word": "w0", "start": 0.0},
        {"word": "x", "start": 0.5},
        {"word": "w2", "start": 1.0},
    ]
    result = dtw_align(a, b)
    # The extra word in B is an insertion; the other two pair up.
    a_words = [aw.word_a for aw in result if aw.word_a is not None]
    b_words = [aw.word_b for aw in result if aw.word_b is not None]
    assert a_words == ["w0", "w2"]
    assert b_words == ["w0", "x", "w2"]
    gaps = [aw for aw in result if aw.word_a is None or aw.word_b is None]
    assert len(gaps) == 1
    assert gaps[0].cost == GAP_PENALTY
    a_starts = [aw.start_a for aw in result if aw.start_a is not None]
    b_starts = [aw.start_b for aw in result if aw.start_b is not None]
    assert a_starts == sorted(a_starts)
    assert b_starts == sorted(b_starts)


def test_alignment_with_deletion_in_a() -> None:
    # A has an extra word that B doesn't have; DTW should pair around it.
    # The extra word "x" is very different from "w0" and "w2" so the gap
    # cost (1.0) is cheaper than pairing it with either B word.
    a = [
        {"word": "w0", "start": 0.0},
        {"word": "x", "start": 0.5},
        {"word": "w2", "start": 1.0},
    ]
    b = [
        {"word": "w0", "start": 0.0},
        {"word": "w2", "start": 1.0},
    ]
    result = dtw_align(a, b)
    a_words = [aw.word_a for aw in result if aw.word_a is not None]
    b_words = [aw.word_b for aw in result if aw.word_b is not None]
    assert a_words == ["w0", "x", "w2"]
    assert b_words == ["w0", "w2"]
    gaps = [aw for aw in result if aw.word_a is None or aw.word_b is None]
    assert len(gaps) == 1
    assert gaps[0].cost == GAP_PENALTY
    a_starts = [aw.start_a for aw in result if aw.start_a is not None]
    b_starts = [aw.start_b for aw in result if aw.start_b is not None]
    assert a_starts == sorted(a_starts)
    assert b_starts == sorted(b_starts)


def test_alignment_with_both_sides_longer() -> None:
    # Both A and B have extra words relative to the "true" pairing.
    # A: [w0, x, w2], B: [w0, y, w2] — x and y are both gap words.
    a = [
        {"word": "w0", "start": 0.0},
        {"word": "x", "start": 0.5},
        {"word": "w2", "start": 1.0},
    ]
    b = [
        {"word": "w0", "start": 0.0},
        {"word": "y", "start": 0.5},
        {"word": "w2", "start": 1.0},
    ]
    result = dtw_align(a, b)
    # DTW pairs (w0, w0), then either (x, y) as a pair, or (x, -) and
    # (-, y) as gaps. The pair (x, y) has cost (0.5-0.5)^2 * (1-sim(x,y))
    # = 0 * (1-0) = 0, which is cheaper than two gaps (1.0 + 1.0 = 2.0).
    # So DTW pairs them. But wait — the cost is time_delta^2 * text_cost,
    # and time_delta = 0.0, so the cost is 0.0 regardless of text.
    # So the diagonal move is preferred.
    a_words = [aw.word_a for aw in result if aw.word_a is not None]
    b_words = [aw.word_b for aw in result if aw.word_b is not None]
    assert a_words == ["w0", "x", "w2"]
    assert b_words == ["w0", "y", "w2"]
    a_starts = [aw.start_a for aw in result if aw.start_a is not None]
    b_starts = [aw.start_b for aw in result if aw.start_b is not None]
    assert a_starts == sorted(a_starts)
    assert b_starts == sorted(b_starts)


def test_alignment_with_i0_branch_in_backtrace() -> None:
    # A shorter than B with all B words at very different times from A.
    # The i=0 branch is taken when all A words are consumed first (via up
    # moves), leaving B words to be emitted as gaps in the i=0 branch.
    # This requires up < left to be true at some point where i=1, which
    # means D[0][j] < D[1][j-1]. With the current cost function (time
    # penalty dominates), this is hard to achieve because pair costs are
    # large. Instead, we verify the output is monotonic and correct.
    a = [{"word": "w0", "start": 5.0}]
    b = [
        {"word": "x", "start": 0.0},
        {"word": "y", "start": 1.0},
        {"word": "z", "start": 2.0},
    ]
    result = dtw_align(a, b)
    assert len(result) == 4
    # All gaps (no pairs because times are too far apart).
    for r in result:
        assert r.cost == GAP_PENALTY
    # Monotonic in both orders.
    a_starts = [r.start_a for r in result if r.start_a is not None]
    b_starts = [r.start_b for r in result if r.start_b is not None]
    assert a_starts == sorted(a_starts)
    assert b_starts == sorted(b_starts)
    # NOTE: The i=0 branch (lines 197-208) is not covered by the test
    # suite because it requires a specific DP table shape that is hard to
    # achieve with the current cost function. It is dead code in practice
    # but kept for correctness in case the cost function changes.


def test_alignment_with_j0_branch_in_backtrace() -> None:
    # B shorter than A with all A words at very different times from B.
    # The j=0 branch is taken when all B words are consumed before A is
    # exhausted.
    a = [
        {"word": "x", "start": 0.0},
        {"word": "y", "start": 1.0},
        {"word": "z", "start": 2.0},
    ]
    b = [{"word": "w0", "start": 5.0}]
    result = dtw_align(a, b)
    # Expect: three gaps for A (up branch), then one gap for B (i=0).
    assert len(result) == 4
    # The last gap should be for B (i=0 branch).
    assert result[-1].word_a is None
    assert result[-1].word_b == "w0"
    assert result[-1].cost == GAP_PENALTY


def test_cost_function_with_empty_after_strip() -> None:
    # A word that becomes empty after stripping punctuation/case.
    # "!" strips to "", so _similarity returns 0.0.
    a = [{"word": "!", "start": 1.0}]
    b = [{"word": "w0", "start": 2.0}]
    (aw,) = dtw_align(a, b)
    # sim = 0.0 (one side empty after strip), text_cost = 1.0
    # time_delta = 1.0, time^2 = 1.0
    # cost = 1.0 * 1.0 = 1.0
    assert abs(aw.cost - 1.0) < 1e-9


def test_cost_function_both_empty_after_strip() -> None:
    # Both words become empty after stripping punctuation.
    # "!" and "." both strip to "", so _similarity returns 1.0 (both empty).
    a = [{"word": "!", "start": 1.0}]
    b = [{"word": ".", "start": 1.0}]
    (aw,) = dtw_align(a, b)
    # sim = 1.0 (both empty after strip), text_cost = 0.0
    # time_delta = 0.0, time^2 = 0.0
    # cost = 0.0 * 0.0 = 0.0
    assert aw.cost == 0.0


def test_levenshtein_similarity_both_empty_returns_one() -> None:
    # Defensive guard: both inputs empty (unreachable via _similarity).
    assert _levenshtein_similarity("", "") == 1.0


def test_levenshtein_similarity_one_empty_returns_zero() -> None:
    # Defensive guard: one input empty (unreachable via _similarity).
    assert _levenshtein_similarity("", "w0") == 0.0
    assert _levenshtein_similarity("w0", "") == 0.0


def test_alignment_with_up_branch_in_backtrace() -> None:
    # A longer than B with a very different word in A that has no good
    # match in B. The up branch is taken when consuming A[i-1] as a gap
    # is cheaper than pairing it with B[j-1].
    a = [
        {"word": "w0", "start": 0.0},
        {"word": "w1", "start": 1.0},
        {"word": "zzz", "start": 2.0},
    ]
    b = [{"word": "w0", "start": 0.0}, {"word": "w1", "start": 1.0}]
    result = dtw_align(a, b)
    # Expect: (w0,w0), (w1,w1), (zzz,None) as gap via up branch.
    assert len(result) == 3
    assert result[0].word_a == "w0" and result[0].word_b == "w0"
    assert result[1].word_a == "w1" and result[1].word_b == "w1"
    assert result[2].word_a == "zzz" and result[2].word_b is None
    assert result[2].cost == GAP_PENALTY


def test_alignment_with_left_branch_in_backtrace() -> None:
    # A shorter than B with a very different trailing word in B.
    # The left branch is taken when consuming B[j-1] as a gap is cheaper
    # than pairing it with A[i-1].
    a = [{"word": "w0", "start": 0.0}]
    b = [
        {"word": "w0", "start": 0.0},
        {"word": "zzz", "start": 1.0},
    ]
    result = dtw_align(a, b)
    # Expect: (w0,w0), (None,zzz) as gap via left branch.
    assert len(result) == 2
    assert result[0].word_a == "w0" and result[0].word_b == "w0"
    assert result[1].word_a is None and result[1].word_b == "zzz"
    assert result[1].cost == GAP_PENALTY


def test_alignment_with_repeated_words() -> None:
    # Same word appearing multiple times; DTW should pair them in order.
    a = [
        {"word": "ja", "start": 0.0},
        {"word": "ja", "start": 1.0},
        {"word": "ja", "start": 2.0},
    ]
    b = [
        {"word": "ja", "start": 0.0},
        {"word": "ja", "start": 1.0},
        {"word": "ja", "start": 2.0},
    ]
    result = dtw_align(a, b)
    assert len(result) == 3
    for idx, aw in enumerate(result):
        assert aw.word_a == "ja"
        assert aw.word_b == "ja"
        assert aw.start_a == idx * 1.0
        assert aw.start_b == idx * 1.0


def test_alignment_with_near_duplicate_onsets() -> None:
    a = [
        {"word": "a", "start": 0.0},
        {"word": "b", "start": 0.001},  # near-duplicate onset
        {"word": "c", "start": 2.0},
    ]
    b = [
        {"word": "a", "start": 0.0},
        {"word": "b", "start": 0.002},  # near-duplicate onset
        {"word": "c", "start": 2.0},
    ]
    result = dtw_align(a, b)
    assert len(result) == 3
    # Monotonic even with near-duplicates (non-decreasing).
    a_starts = [aw.start_a for aw in result if aw.start_a is not None]
    b_starts = [aw.start_b for aw in result if aw.start_b is not None]
    assert a_starts == sorted(a_starts)
    assert b_starts == sorted(b_starts)


# ---------------------------------------------------------------------------
# Cost function: verify the cost at each pair matches the formula.
# ---------------------------------------------------------------------------


def test_cost_function_with_larger_time_drift_scales_quadratically() -> None:
    a = [{"word": "Mikko", "start": 1.0}]
    b = [{"word": "Mikoa", "start": 3.0}]
    (aw,) = dtw_align(a, b)
    # (3.0 - 1.0)^2 * (1 - 3/5) = 4 * 2/5 = 1.6
    assert abs(aw.cost - 1.6) < 1e-9


def test_cost_function_identical_text_zero_cost_regardless_of_time() -> None:
    # Identical tokens: text cost is 0, so total cost is 0 even with drift.
    for delta in (0.0, 0.1, 1.0, 5.0):
        a = [{"word": "hei", "start": 1.0}]
        b = [{"word": "hei", "start": 1.0 + delta}]
        (aw,) = dtw_align(a, b)
        assert aw.cost == 0.0, f"delta={delta} gave cost {aw.cost}"


# ---------------------------------------------------------------------------
# Determinism: running the alignment twice on the same input gives the same
# result (DTW is deterministic by construction).
# ---------------------------------------------------------------------------


def test_alignment_is_deterministic() -> None:
    a = [
        {"word": "hei", "start": 0.0},
        {"word": "sinä", "start": 1.0},
        {"word": "kuka", "start": 2.0},
    ]
    b = [
        {"word": "hei", "start": 0.0},
        {"word": "sinä", "start": 1.5},
        {"word": "kuka", "start": 2.5},
    ]
    first = dtw_align(a, b)
    second = dtw_align(a, b)
    assert first == second


def test_cost_is_a_float() -> None:
    a = [{"word": "Mikko", "start": 1.0}]
    b = [{"word": "Mikoa", "start": 2.0}]
    (aw,) = dtw_align(a, b)
    assert isinstance(aw.cost, float)
