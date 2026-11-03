"""Self-healing re-decode of hallucination walls (meeting profile).

Live failure this models: whisper's rolling context turned one bad window
into ~50 consecutive "Kiitos." segments (T2) and ~95 "DCS. DCS." segments
(T5), erasing a presentation opening and a 43-minute demo. Detection and
replacement are pure functions over segment dicts; the re-decode is an
injected callable, so no models load here.
"""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
import pytest

from vemoizer import selfheal
from vemoizer.loops import find_degenerate_windows, loop_seconds
from vemoizer.selfheal import heal

SR = 16_000


def _seg(text: str, start: float, end: float) -> dict[str, Any]:
    return {"start": start, "end": end, "text": text}


def _normal_segments() -> list[dict[str, Any]]:
    return [
        _seg("Tässä on ihan tavallista puhetta kokouksesta.", 0.0, 3.0),
        _seg("Ja keskustelu jatkuu monipuolisin sanoin eteenpäin.", 3.0, 6.0),
        _seg("Kukaan ei toista itseään epäilyttävästi.", 6.0, 9.0),
    ]


# --- detection ---


def test_no_windows_on_normal_speech() -> None:
    assert find_degenerate_windows(_normal_segments()) == []


def test_kiitos_wall_is_one_window() -> None:
    wall = [_seg("Kiitos.", 10.0 + i, 11.0 + i) for i in range(8)]
    segments = _normal_segments() + wall
    assert find_degenerate_windows(segments) == [(10.0, 18.0)]


def test_drifting_spellings_still_detected() -> None:
    texts = ["DCS. DCS.", "D-css.", "DCS. DCS.", "DCS.", "D-css.", "DCS. DCS."]
    wall = [_seg(t, 5.0 + i, 6.0 + i) for i, t in enumerate(texts)]
    assert find_degenerate_windows(wall) == [(5.0, 11.0)]


def test_run_below_min_length_is_ignored() -> None:
    wall = [_seg("Kiitos.", float(i), float(i) + 1.0) for i in range(4)]
    assert find_degenerate_windows(wall) == []


def test_varied_short_backchannels_are_not_degenerate() -> None:
    texts = ["Joo.", "Aivan.", "Niin just.", "Okei.", "Kyllä.", "Selvä homma."]
    segs = [_seg(t, float(i), float(i) + 1.0) for i, t in enumerate(texts)]
    assert find_degenerate_windows(segs) == []


def test_nearby_windows_merge_across_a_real_segment() -> None:
    wall1 = [_seg("Kiitos.", float(i), float(i) + 1.0) for i in range(5)]
    real = [_seg("Yksi oikea lause tähän väliin puheesta.", 5.0, 8.0)]
    wall2 = [_seg("Kiitos.", 8.0 + i, 9.0 + i) for i in range(5)]
    assert find_degenerate_windows(wall1 + real + wall2) == [(0.0, 13.0)]


def test_distant_windows_stay_separate() -> None:
    wall1 = [_seg("Kiitos.", float(i), float(i) + 1.0) for i in range(5)]
    wall2 = [_seg("DCS.", 100.0 + i, 101.0 + i) for i in range(5)]
    assert find_degenerate_windows(wall1 + wall2) == [(0.0, 5.0), (100.0, 105.0)]


# --- healing ---


def _wall_result() -> dict[str, Any]:
    """A decode result whose 10-20s stretch is a Kiitos wall."""
    good = [_seg("Alussa on ihan oikeaa puhetta tässä näin.", 0.0, 4.0)]
    wall = [_seg("Kiitos.", 10.0 + i, 11.0 + i) for i in range(8)]
    tail = [_seg("Ja lopussa taas palataan ihan oikeaan asiaan.", 25.0, 29.0)]
    segments = good + wall + tail
    words = []
    for s in segments:
        for j, w in enumerate(str(s["text"]).split()):
            w_start = float(s["start"]) + 0.1 * j
            words.append({"word": w, "start": w_start, "end": w_start + 0.1})
    return {
        "text": " ".join(str(s["text"]) for s in segments),
        "segments": segments,
        "words": words,
    }


def _slices() -> list[tuple[int, np.ndarray]]:
    """VAD slices: [0-8s], [9-22s] (covers the wall), [24-30s]."""
    return [
        (0, np.zeros(8 * SR, dtype=np.float32)),
        (9 * SR, np.zeros(13 * SR, dtype=np.float32)),
        (24 * SR, np.zeros(6 * SR, dtype=np.float32)),
    ]


def _good_redecode(chunk: np.ndarray) -> dict[str, Any]:
    return {
        "segments": [
            _seg("Tässä kohtaa puhuttiin demosta ihan oikeasti.", 1.0, 5.0),
            _seg("Ja näytettiin lukuja ruudulta kaikille.", 5.0, 9.0),
        ],
        "words": [
            {"word": "demosta", "start": 2.0, "end": 2.4},
            {"word": "lukuja", "start": 6.0, "end": 6.4},
        ],
    }


def test_heal_replaces_wall_with_redecoded_slice() -> None:
    healed = heal(_wall_result(), _slices(), _good_redecode)
    texts = [s["text"] for s in healed["segments"]]
    assert "Kiitos." not in texts
    assert "Tässä kohtaa puhuttiin demosta ihan oikeasti." in texts
    # replacement times shifted by the slice offset (9s)
    replaced = next(s for s in healed["segments"] if "demosta" in str(s["text"]))
    assert replaced["start"] == 10.0
    # untouched regions survive on both sides
    assert texts[0] == "Alussa on ihan oikeaa puhetta tässä näin."
    assert texts[-1] == "Ja lopussa taas palataan ihan oikeaan asiaan."
    # words in the healed slice replaced, others kept
    healed_words = [w["word"] for w in healed["words"]]
    assert "demosta" in healed_words
    assert "Alussa" in healed_words
    # full text rebuilt from segments
    assert "Kiitos" not in healed["text"]


def test_heal_untouched_slices_are_not_redecoded() -> None:
    calls: list[int] = []

    def counting(chunk: np.ndarray) -> dict[str, Any]:
        calls.append(len(chunk))
        return _good_redecode(chunk)

    heal(_wall_result(), _slices(), counting)
    assert calls == [13 * SR]  # only the wall-covering slice


def test_heal_noop_without_windows() -> None:
    result = {"text": "x", "segments": _normal_segments(), "words": []}
    calls: list[int] = []

    def counting(chunk: np.ndarray) -> dict[str, Any]:
        calls.append(1)
        return {"segments": [], "words": []}

    assert heal(result, _slices(), counting) is result
    assert calls == []


def test_heal_keeps_original_when_redecode_still_loops() -> None:
    def looping(chunk: np.ndarray) -> dict[str, Any]:
        return {
            "segments": [_seg("Kiitos.", float(i), float(i) + 1.0) for i in range(6)],
            "words": [],
        }

    result = _wall_result()
    healed = heal(result, _slices(), looping)
    assert [s["text"] for s in healed["segments"]] == [
        s["text"] for s in result["segments"]
    ]


def test_heal_keeps_original_when_redecode_raises() -> None:
    def broken(chunk: np.ndarray) -> dict[str, Any]:
        raise RuntimeError("decode exploded")

    result = _wall_result()
    healed = heal(result, _slices(), broken)
    assert [s["text"] for s in healed["segments"]] == [
        s["text"] for s in result["segments"]
    ]


def test_heal_accepts_silence_as_replacement() -> None:
    """An empty re-decode erases the wall (silence beats hallucination)."""

    def silent(chunk: np.ndarray) -> dict[str, Any]:
        return {"segments": [], "words": []}

    healed = heal(_wall_result(), _slices(), silent)
    texts = [s["text"] for s in healed["segments"]]
    assert "Kiitos." not in texts
    assert len(texts) == 2  # only the good head and tail remain


# --- prompt-echo fallback + time budget (2026-09-25 live failure) ---
#
# A name-heavy glossary prompt made whisper echo "Janni." whenever the
# audio was unclear; the conditioning-off re-decode re-sent the same
# prompt and looped again. The fallback re-decodes without the prompt.


def _looping(chunk: np.ndarray) -> dict[str, Any]:
    return {
        "segments": [_seg("Janni.", float(i), float(i) + 1.0) for i in range(6)],
        "words": [],
    }


def test_fallback_used_when_primary_redecode_still_loops() -> None:
    healed = heal(_wall_result(), _slices(), _looping, fallback=_good_redecode)
    texts = [s["text"] for s in healed["segments"]]
    assert "Kiitos." not in texts
    assert "Tässä kohtaa puhuttiin demosta ihan oikeasti." in texts


def test_fallback_used_when_primary_redecode_raises() -> None:
    def broken(chunk: np.ndarray) -> dict[str, Any]:
        raise RuntimeError("decode exploded")

    healed = heal(_wall_result(), _slices(), broken, fallback=_good_redecode)
    assert "Kiitos" not in healed["text"]


def test_original_kept_when_fallback_also_loops() -> None:
    result = _wall_result()
    healed = heal(result, _slices(), _looping, fallback=_looping)
    assert [s["text"] for s in healed["segments"]] == [
        s["text"] for s in result["segments"]
    ]


def test_heal_budget_is_time_based_not_slice_count() -> None:
    """Many short slices under a wall all heal while their audio fits."""
    wall = [_seg("Kiitos.", float(i), float(i) + 1.0) for i in range(300)]
    result = {"text": "", "segments": wall, "words": []}
    slices = [(i * SR, np.zeros(SR, dtype=np.float32)) for i in range(300)]
    calls: list[int] = []

    def counting(chunk: np.ndarray) -> dict[str, Any]:
        calls.append(1)
        return {"segments": [], "words": []}

    heal(result, slices, counting)
    assert len(calls) == 300  # the old 200-slice cap would stop here


# --- one long looping segment (issue #219) ------------------------------
#
# A loop whisper emits as ONE 20 s segment ("Jumala, Jumala, …") never
# joins a run, so the run rule above cannot see it at all. The signal is
# whisper's own per-segment compression_ratio (its discard cut is 2.4),
# which the only heal caller — the whisper decode — stores. A word-
# repetition rule is NOT the signal: natural Finnish repeats itself
# ("no niin no niin …") and would be flagged as a wall.


def _long(text: str, start: float, end: float, **extra: Any) -> dict[str, Any]:
    return {"start": start, "end": end, "text": text, **extra}


LOOP_TEXT = "Jumala, " * 12
NORMAL_LONG = "Kokouksessa käytiin läpi budjetti ja seuraavan vaiheen aikataulu"


def test_long_segment_is_a_window_by_compression_ratio() -> None:
    seg = _long(NORMAL_LONG, 0.0, 20.0, compression_ratio=3.1)
    assert find_degenerate_windows([seg]) == [(0.0, 20.0)]


def test_normal_long_segment_is_not_a_window() -> None:
    seg = _long(NORMAL_LONG, 0.0, 20.0, compression_ratio=1.6)
    assert find_degenerate_windows([seg]) == []


def test_short_segment_below_token_floor_needs_a_run() -> None:
    """A few repeated words in one short segment is emphasis, not a wall."""
    seg = _long("Jumala, Jumala, Jumala", 0.0, 2.0, compression_ratio=3.5)
    assert find_degenerate_windows([seg]) == []


@pytest.mark.parametrize(
    "text",
    [
        # Backchannel chains and quoted speech: repeated words, no wall.
        "no niin no niin mennään sitten seuraavaan kohtaan",
        "minä sanon että minä sanon että se on hyvä asia",
        "mä luulen että mä luulen että tämä menee hyvin",
    ],
)
def test_natural_repetition_in_a_long_segment_is_not_a_window(text: str) -> None:
    """Word repetition at a normal compression ratio is Finnish, not a loop."""
    seg = _long(text, 0.0, 20.0, compression_ratio=1.3)
    assert find_degenerate_windows([seg]) == []


def test_long_segment_window_merges_with_a_nearby_run_wall() -> None:
    seg = _long(LOOP_TEXT, 10.0, 14.0, compression_ratio=3.1)
    wall = [_seg("Kiitos.", 15.0 + i, 16.0 + i) for i in range(5)]
    assert find_degenerate_windows([seg] + wall) == [(10.0, 20.0)]


# --- loop_seconds (shared with `vemoizer accept`) -----------------------


def test_loop_seconds_sums_the_windows() -> None:
    wall = [_seg("Kiitos.", float(i), float(i) + 1.0) for i in range(6)]
    distant = _long(LOOP_TEXT, 100.0, 112.5, compression_ratio=3.1)
    assert loop_seconds(wall + [distant]) == 18.5


def test_loop_seconds_is_zero_on_clean_segments() -> None:
    assert loop_seconds(_normal_segments()) == 0.0


# --- walls that survive the heal are flagged, not kept silently ---------


def test_unhealed_wall_is_flagged_and_warned() -> None:
    result = _wall_result()
    healed = heal(result, _slices(), _looping, fallback=_looping)
    wall = [s for s in healed["segments"] if s["text"] == "Kiitos."]
    assert len(wall) == 8  # the wall is still there (fail-open)...
    assert all(s["suspect"] == "loop" for s in wall)  # ...but it is labelled
    assert not any("suspect" in s for s in healed["segments"] if "Alussa" in s["text"])
    assert len(healed["warnings"]) == 1
    assert "8s" in healed["warnings"][0]
    assert "loop" in healed["warnings"][0].lower()


def test_healed_wall_gets_no_flag_and_no_warning() -> None:
    healed = heal(_wall_result(), _slices(), _good_redecode)
    assert not any("suspect" in s for s in healed["segments"])
    assert "warnings" not in healed


def test_partially_healed_wall_flags_only_the_unhealed_half() -> None:
    """A healed replacement is never a loop; only unhealed time is counted.

    10 x "Kiitos." over 0-10 s; [0,5) heals into three "Hyvä asia."
    segments, [5,10) still loops. The replacements must not be stamped
    suspect, and the warning must name the 5 s still looping, not 10 s.
    """
    wall = [_seg("Kiitos.", float(i), float(i) + 1.0) for i in range(10)]
    result = {"text": "", "segments": wall, "words": []}
    slices = [
        (0, np.zeros(5 * SR, dtype=np.float32)),
        (5 * SR, np.zeros(5 * SR, dtype=np.float32)),
    ]
    calls: list[int] = []

    def half_heals(chunk: np.ndarray) -> dict[str, Any]:
        calls.append(len(chunk))
        if len(calls) == 1:
            return {
                "segments": [
                    _seg("Hyvä asia.", 0.5, 1.5),
                    _seg("Hyvä asia.", 1.5, 2.5),
                    _seg("Hyvä asia.", 2.5, 3.5),
                ],
                "words": [],
            }
        return {
            "segments": [_seg("Kiitos.", float(i), float(i) + 1.0) for i in range(5)],
            "words": [],
        }

    healed = heal(result, slices, half_heals)
    replacements = [s for s in healed["segments"] if s["text"] == "Hyvä asia."]
    kept = [s for s in healed["segments"] if s["text"] == "Kiitos."]
    assert len(replacements) == 3
    assert not any("suspect" in s for s in replacements)
    assert len(kept) == 5  # the [5,10) wall survived the fail-open heal
    assert all(s["suspect"] == "loop" for s in kept)
    assert all(float(s["start"]) >= 5.0 for s in kept)
    assert len(healed["warnings"]) == 1
    assert "5s" in healed["warnings"][0]
    assert "10s" not in healed["warnings"][0]


def test_residual_ignores_a_healed_slice_far_from_the_remaining_wall() -> None:
    """A healed slice elsewhere must not stretch the residual window."""
    wall_a = [_seg("Kiitos.", float(i), float(i) + 1.0) for i in range(6)]
    wall_b = [_seg("Joo.", 100.0 + i, 101.0 + i) for i in range(6)]
    result = {"text": "", "segments": wall_a + wall_b, "words": []}
    slices = [
        (0, np.zeros(6 * SR, dtype=np.float32)),
        (100 * SR, np.zeros(6 * SR, dtype=np.float32)),
    ]
    calls: list[int] = []

    def second_only(chunk: np.ndarray) -> dict[str, Any]:
        calls.append(len(chunk))
        if len(calls) == 1:
            return {
                "segments": [
                    _seg("Kiitos.", float(i), float(i) + 1.0) for i in range(6)
                ],
                "words": [],
            }
        return {
            "segments": [_seg("Tässä on oikeaa puhetta demosta.", 0.0, 5.0)],
            "words": [],
        }

    healed = heal(result, slices, second_only)
    assert len(healed["warnings"]) == 1
    assert "6s" in healed["warnings"][0]  # wall A only, not 0-100 s
    assert all(
        s.get("suspect") == "loop" for s in healed["segments"] if "Kiitos" in s["text"]
    )
    assert not any("suspect" in s for s in healed["segments"] if "demosta" in s["text"])


def test_healed_slice_inside_a_window_leaves_both_flanks_residual() -> None:
    """Healing the middle of a wall flags and counts only the flanks."""
    wall = [_seg("Kiitos.", float(i), float(i) + 1.0) for i in range(10)]
    result = {"text": "", "segments": wall, "words": []}
    slices = [
        (0, np.zeros(3 * SR, dtype=np.float32)),
        (3 * SR, np.zeros(4 * SR, dtype=np.float32)),
        (7 * SR, np.zeros(3 * SR, dtype=np.float32)),
    ]
    calls: list[int] = []

    def middle_only(chunk: np.ndarray) -> dict[str, Any]:
        calls.append(len(chunk))
        if len(calls) == 2:
            return {
                "segments": [
                    _seg("Hyvä asia.", float(i), float(i) + 1.0) for i in range(4)
                ],
                "words": [],
            }
        return {  # still looping -> rejected, slice kept as-is
            "segments": [_seg("Kiitos.", float(i), float(i) + 1.0) for i in range(6)],
            "words": [],
        }

    healed = heal(result, slices, middle_only)
    flagged = [
        float(s["start"]) for s in healed["segments"] if s.get("suspect") == "loop"
    ]
    assert flagged == [0.0, 1.0, 2.0, 7.0, 8.0, 9.0]  # the flanks, not 3-7
    assert len(healed["warnings"]) == 1
    assert "6s" in healed["warnings"][0]  # 3 s + 3 s, not the whole 10 s


def test_budget_skipped_wall_is_flagged(monkeypatch: pytest.MonkeyPatch) -> None:
    """The MAX_HEAL_SECONDS backstop must not swallow the remaining wall."""
    monkeypatch.setattr(selfheal, "MAX_HEAL_SECONDS", 5.0)
    wall = [_seg("Kiitos.", float(i), float(i) + 1.0) for i in range(40)]
    result = {"text": "", "segments": wall, "words": []}
    slices = [(i * SR, np.zeros(SR, dtype=np.float32)) for i in range(40)]

    def quiet(chunk: np.ndarray) -> dict[str, Any]:
        seg = _seg("Tässä on oikeaa puhetta demosta.", 0.0, 1.0)
        return {"segments": [seg], "words": []}

    healed = heal(result, slices, quiet)
    flagged = [s for s in healed["segments"] if s.get("suspect") == "loop"]
    assert len(flagged) == 35  # 5 slices healed, the rest kept as-is
    assert any("loop" in w.lower() for w in healed["warnings"])


def test_heal_does_not_mutate_the_input_segments() -> None:
    result = _wall_result()
    before = copy.deepcopy(result["segments"])
    heal(result, _slices(), _looping, fallback=_looping)
    assert result["segments"] == before
    assert all("suspect" not in s for s in result["segments"])
    assert "warnings" not in result
