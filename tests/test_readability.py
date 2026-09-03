"""Full-coverage splice + paragraph grouping (issue #53).

Pure functions over word/segment dicts — no models, no pipeline. The
splice is what keeps txt/srt/vtt whole once consensus activates: without
it, disputed-span segments would replace the entire rendered transcript
instead of patching into it.
"""

from __future__ import annotations

from vemoizer.readability import paragraphs, splice_verdicts


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
