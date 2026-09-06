"""Full-coverage splice + paragraph grouping (issue #53).

Pure functions over word/segment dicts — no models, no pipeline. The
splice is what keeps txt/srt/vtt whole once consensus activates: without
it, disputed-span segments would replace the entire rendered transcript
instead of patching into it.
"""

from __future__ import annotations

from vemoizer.readability import paragraphs, splice_verdicts, tidy_paragraphs


def _w(word: str, start: float, end: float) -> dict:
    return {"word": word, "start": start, "end": end}


def _seg(text: str, start: float, end: float, **extra) -> dict:
    return {"start": start, "end": end, "text": text, **extra}


WORDS = [
    _w("hei", 0.0, 0.4),
    _w("maailma", 0.5, 1.0),
    _w("tämä", 1.5, 1.9),
    _w("on", 2.0, 2.2),
    _w("testi", 2.3, 2.8),
]
SENTENCES = [
    _seg("hei maailma", 0.0, 1.0),
    _seg("tämä on testi", 1.5, 2.8),
]


# -- splice_verdicts -----------------------------------------------------


def test_no_verdicts_is_identity() -> None:
    text, segments = splice_verdicts("hei maailma tämä on testi", WORDS, SENTENCES, [])
    assert text == "hei maailma tämä on testi"  # byte-identical
    assert segments == SENTENCES


def test_mid_sentence_splice_replaces_only_span_words() -> None:
    verdicts = [_seg("upea", 0.5, 1.0)]  # replaces "maailma"
    text, segments = splice_verdicts("irrelevant", WORDS, SENTENCES, verdicts)
    assert segments[0]["text"] == "hei upea"
    assert segments[1]["text"] == "tämä on testi"
    assert text == "hei upea tämä on testi"


def test_full_coverage_is_preserved() -> None:
    """Every sentence stays present; nothing collapses to just the disputes."""
    verdicts = [_seg("koe", 2.3, 2.8)]  # replaces "testi"
    _text, segments = splice_verdicts("x", WORDS, SENTENCES, verdicts)
    assert len(segments) == len(SENTENCES)
    assert [s["start"] for s in segments] == [0.0, 1.5]
    assert segments[1]["text"] == "tämä on koe"


def test_span_crossing_sentence_boundary_anchors_once() -> None:
    """A verdict spanning two sentences lands once, in its anchor sentence."""
    verdicts = [_seg("kaikki muuttui", 0.5, 2.2)]  # "maailma tämä on" dropped
    text, segments = splice_verdicts("x", WORDS, SENTENCES, verdicts)
    assert segments[0]["text"] == "hei kaikki muuttui"
    assert segments[1]["text"] == "testi"
    assert "kaikki muuttui" in text
    assert text.count("kaikki muuttui") == 1


def test_verdict_speaker_is_carried_onto_the_segment() -> None:
    verdicts = [_seg("upea", 0.5, 1.0, speaker="S1")]
    _text, segments = splice_verdicts("x", WORDS, SENTENCES, verdicts)
    assert segments[0].get("speaker") == "S1"


def test_empty_verdict_text_still_drops_span_words() -> None:
    """An adjudicated deletion removes the disputed words entirely."""
    verdicts = [_seg("", 0.5, 1.0)]
    text, segments = splice_verdicts("x", WORDS, SENTENCES, verdicts)
    assert segments[0]["text"] == "hei"
    assert "maailma" not in text


# -- paragraphs ----------------------------------------------------------


def test_paragraphs_split_on_silence_gap() -> None:
    segments = [
        _seg("eka lause", 0.0, 1.0),
        _seg("heti perään", 1.2, 2.0),
        _seg("pitkän tauon jälkeen", 4.0, 5.0),  # 2.0s gap >= 1.5s
    ]
    paras = paragraphs(segments)
    assert len(paras) == 2
    assert paras[0]["text"] == "eka lause heti perään"
    assert paras[1]["text"] == "pitkän tauon jälkeen"
    assert paras[0]["start"] == 0.0
    assert paras[0]["end"] == 2.0


def test_paragraphs_split_on_speaker_change() -> None:
    segments = [
        _seg("moi", 0.0, 1.0, speaker="S1"),
        _seg("no moi", 1.1, 2.0, speaker="S2"),
    ]
    paras = paragraphs(segments)
    assert len(paras) == 2
    assert paras[0]["speaker"] == "S1"
    assert paras[1]["speaker"] == "S2"


def test_paragraphs_keep_same_speaker_together() -> None:
    segments = [
        _seg("moi", 0.0, 1.0, speaker="S1"),
        _seg("jatkuu", 1.1, 2.0, speaker="S1"),
    ]
    paras = paragraphs(segments)
    assert len(paras) == 1
    assert paras[0]["text"] == "moi jatkuu"


def test_paragraphs_empty_input() -> None:
    assert paragraphs([]) == []


def test_paragraphs_unlabelled_segments_have_no_speaker_key() -> None:
    paras = paragraphs([_seg("moi", 0.0, 1.0)])
    assert "speaker" not in paras[0]


# -- tidy_paragraphs (issue #71 forensics) -------------------------------
#
# Deterministic hygiene between assembly and repair: whisper repetition
# loops ("Janni, " x74), duplicate adjacent paragraphs, digit-only noise
# and monologue walls are mechanical defects — an LLM is the wrong tool
# (its no-invention guard rightly vetoes a 74->1 collapse).


def test_repetition_loop_collapses() -> None:
    paras = [_seg("Janni, " * 74 + "aloitetaan", 0.0, 10.0)]
    out = tidy_paragraphs(paras)
    text = out[0]["text"]
    assert text.count("Janni") == 1
    assert "aloitetaan" in text
    assert "…" in text  # the collapse is marked, not hidden


def test_ngram_loops_collapse_too() -> None:
    paras = [_seg("se on hyvä " * 5 + "idea", 0.0, 5.0)]
    out = tidy_paragraphs(paras)
    assert out[0]["text"].count("se on hyvä") == 1


def test_moderate_repetition_is_left_alone() -> None:
    """Two repeats are normal speech ('joo joo'); three+ is a loop."""
    paras = [_seg("joo joo mennään", 0.0, 2.0)]
    assert tidy_paragraphs(paras)[0]["text"] == "joo joo mennään"


def test_identical_adjacent_paragraphs_dedupe() -> None:
    paras = [
        _seg("Tilauksen luominen ERP:ään.", 0.0, 2.0, speaker="S1"),
        _seg("Tilauksen luominen ERP:ään.", 2.0, 4.0, speaker="S2"),
        _seg("eri asia", 4.0, 5.0),
    ]
    out = tidy_paragraphs(paras)
    assert [p["text"] for p in out] == ["Tilauksen luominen ERP:ään.", "eri asia"]


def test_non_alphabetic_paragraphs_drop() -> None:
    paras = [_seg("2708202", 0.0, 1.0), _seg("oikea lause", 1.0, 2.0)]
    out = tidy_paragraphs(paras)
    assert [p["text"] for p in out] == ["oikea lause"]


def test_monologue_walls_split_at_sentences() -> None:
    sentence = "Tässä on yksi kokonainen virke joka kertoo asioista. "
    paras = [_seg((sentence * 40).strip(), 0.0, 100.0, speaker="S1")]
    out = tidy_paragraphs(paras)
    assert len(out) >= 2
    for p in out:
        assert len(p["text"]) <= 1200
        assert p["speaker"] == "S1"
    # timing stays monotonic and covers the original span
    assert out[0]["start"] == 0.0
    assert out[-1]["end"] == 100.0
    for a, b in zip(out, out[1:], strict=False):
        assert a["end"] <= b["start"] + 1e-6


def test_tidy_empty_input() -> None:
    assert tidy_paragraphs([]) == []
