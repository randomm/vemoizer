"""Self-healing re-decode of hallucination walls (meeting profile).

Live failure this models: whisper's rolling context turned one bad window
into ~50 consecutive "Kiitos." segments (T2) and ~95 "DCS. DCS." segments
(T5), erasing a presentation opening and a 43-minute demo. Detection and
replacement are pure functions over segment dicts; the re-decode is an
injected callable, so no models load here.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from vemoizer.selfheal import find_degenerate_windows, heal

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
