"""Word-level speaker attribution (issue #71 round 2).

The visible structural weakness: speakers were attached per whisper
sentence segment (midpoint overlap), so a Q&A exchange inside one segment
fused under one label. We already hold whisper word timestamps AND
pyannote turns — intersecting them is pure geometry. Pitfalls handled per
the research recipe: timestamp jitter (tolerance), zero-overlap words
(nearest turn, capped), single-word flicker (smoothing), backchannels
(legitimate one-word turns).
"""

from __future__ import annotations

from vemoizer.speaker_align import (
    assign_word_speakers,
    split_segments_at_speaker_changes,
)


def _w(word, start, end):
    return {"word": word, "start": start, "end": end}


TURNS = [(0.0, 5.0, "S1"), (5.0, 8.0, "S2"), (8.0, 12.0, "S1")]


def test_words_take_the_max_overlap_turn() -> None:
    words = [_w("kysymys", 4.0, 4.8), _w("vastaus", 5.2, 6.0)]
    labels = assign_word_speakers(words, TURNS)
    assert labels == ["S1", "S2"]


def test_jittered_word_at_boundary_does_not_flip() -> None:
    """A word straddling a boundary goes to the side holding most of it."""
    words = [_w("sana", 4.7, 5.4)]  # 0.3s in S1, 0.4s in S2
    assert assign_word_speakers(words, TURNS) == ["S2"]


def test_zero_overlap_word_takes_nearest_turn() -> None:
    words = [_w("irrallaan", 12.5, 12.9)]  # after all turns, 0.5s from S1
    assert assign_word_speakers(words, TURNS) == ["S1"]


def test_far_from_any_turn_inherits_previous() -> None:
    words = [_w("eka", 4.0, 4.5), _w("kaukana", 20.0, 20.5)]
    assert assign_word_speakers(words, TURNS) == ["S1", "S1"]


def test_single_word_island_is_smoothed_away() -> None:
    """One word of S2 inside a run of S1 is diarization flicker."""
    words = [
        _w("puhun", 1.0, 1.4),
        _w("tässä", 1.5, 1.9),
        _w("koko", 2.0, 2.4),
        _w("ajan", 2.5, 2.9),
    ]
    turns = [(0.0, 1.95, "S1"), (1.95, 2.05, "S2"), (2.05, 5.0, "S1")]
    labels = assign_word_speakers(words, turns)
    assert labels == ["S1", "S1", "S1", "S1"]


def test_backchannel_single_word_survives() -> None:
    """'Joo' fully inside the other speaker's turn is a real interjection."""
    words = [
        _w("kerron", 1.0, 1.5),
        _w("joo", 2.0, 2.4),
        _w("asiasta", 3.0, 3.5),
    ]
    turns = [(0.0, 1.9, "S1"), (1.95, 2.5, "S2"), (2.6, 5.0, "S1")]
    labels = assign_word_speakers(words, turns)
    assert labels == ["S1", "S2", "S1"]


def test_fused_qa_segment_splits_into_turns() -> None:
    """The headline case: question and answer inside ONE whisper segment."""
    segment = {
        "start": 0.0,
        "end": 8.0,
        "text": "mitä mieltä olet siitä minusta se toimii hyvin",
    }
    words = [
        _w("mitä", 0.5, 0.9),
        _w("mieltä", 1.0, 1.4),
        _w("olet", 1.5, 1.9),
        _w("siitä", 2.0, 2.4),
        _w("minusta", 5.2, 5.7),
        _w("se", 5.8, 6.0),
        _w("toimii", 6.1, 6.5),
        _w("hyvin", 6.6, 7.0),
    ]
    labels: list[str | None] = ["S1", "S1", "S1", "S1", "S2", "S2", "S2", "S2"]
    out = split_segments_at_speaker_changes([segment], words, labels)
    assert len(out) == 2
    assert out[0]["text"] == "mitä mieltä olet siitä"
    assert out[0]["speaker"] == "S1"
    assert out[1]["text"] == "minusta se toimii hyvin"
    assert out[1]["speaker"] == "S2"
    assert out[0]["end"] <= out[1]["start"]


def test_segment_without_speaker_change_is_untouched() -> None:
    segment = {
        "start": 0.0,
        "end": 2.0,
        "text": "yksi puhuja vain",
        "suspect": "garble",
    }
    words = [_w("yksi", 0.1, 0.4), _w("puhuja", 0.5, 0.9), _w("vain", 1.0, 1.3)]
    labels_single: list[str | None] = ["S1", "S1", "S1"]
    out = split_segments_at_speaker_changes([segment], words, labels_single)
    assert len(out) == 1
    assert out[0]["speaker"] == "S1"
    assert out[0]["suspect"] == "garble"  # metadata preserved
    assert out[0]["text"] == "yksi puhuja vain"


def test_segment_with_no_words_keeps_no_speaker() -> None:
    segment = {"start": 50.0, "end": 51.0, "text": "sanaton"}
    out = split_segments_at_speaker_changes([segment], [], [])
    assert out == [segment]


# -- round 3: turn shrapnel, question boundaries, backchannel phrases ----
#
# Persona-coherence QA over five real meetings found: 2-3 word flicker
# runs shredding sentences across labels (phantom personas), askers'
# turns swallowing short answers ("Pääseekö Topiinkin? Pääsee."), and
# multi-word acknowledgements ("Mä arvostan.") absorbed into monologues.


def test_short_flicker_run_merges_into_surrounding_label() -> None:
    """'05: ...Mä / 04: olen / 01: samalta...' -> one label owns it all."""
    words = [
        _w("mä", 1.0, 1.2),
        _w("olen", 1.3, 1.5),
        _w("tehnyt", 1.6, 1.9),
        _w("tämän", 2.0, 2.3),
        _w("aiemmin", 2.4, 2.8),
    ]
    turns = [(0.0, 1.25, "S5"), (1.25, 1.55, "S4"), (1.55, 5.0, "S5")]
    labels = assign_word_speakers(words, turns)
    assert labels == ["S5", "S5", "S5", "S5", "S5"]


def test_two_word_island_between_different_labels_takes_bigger_neighbor() -> None:
    words = [
        _w("puhun", 0.2, 0.6),
        _w("pitkään", 0.7, 1.1),
        _w("ajaa", 1.2, 1.4),
        _w("teissä", 1.5, 1.7),
        _w("vastaan", 2.2, 2.6),
        _w("tulee", 2.7, 3.1),
    ]
    turns = [(0.0, 1.15, "S1"), (1.15, 1.8, "S2"), (2.0, 5.0, "S3")]
    labels = assign_word_speakers(words, turns)
    # the 2-word 0.5s S2 flicker is absorbed; sentence integrity wins
    assert labels[2] == labels[3]
    assert labels[2] in ("S1", "S3")


def test_genuine_short_answer_after_question_is_kept() -> None:
    """'Pääseekö Topiinkin?' (A) 'Pääsee.' (B): the ? is a hard boundary."""
    words = [
        _w("pääseekö", 1.0, 1.5),
        _w("Topiinkin?", 1.6, 2.1),
        _w("pääsee", 2.3, 2.7),
    ]
    turns = [(0.0, 2.15, "S1"), (2.2, 3.0, "S2")]
    labels = assign_word_speakers(words, turns)
    assert labels == ["S1", "S1", "S2"]


def test_question_mark_without_competing_turn_changes_nothing() -> None:
    words = [_w("mitä?", 1.0, 1.4), _w("niin", 1.5, 1.8)]
    turns = [(0.0, 5.0, "S1")]
    assert assign_word_speakers(words, turns) == ["S1", "S1"]


def test_backchannel_phrase_island_survives() -> None:
    """'Mä arvostan' (2 words) inside the other speaker's turn is real."""
    words = [
        _w("kerron", 0.5, 1.0),
        _w("lisää", 1.1, 1.5),
        _w("mä", 2.0, 2.2),
        _w("arvostan", 2.3, 2.8),
        _w("jatketaan", 3.5, 4.0),
        _w("tästä", 4.1, 4.5),
    ]
    turns = [(0.0, 1.9, "S1"), (1.95, 2.9, "S2"), (3.0, 6.0, "S1")]
    labels = assign_word_speakers(words, turns)
    assert labels == ["S1", "S1", "S2", "S2", "S1", "S1"]


def test_single_word_answer_after_question_survives_smoothing() -> None:
    """'Tehdäänkö?' (S1) 'Tehdään.' (S2) 'Hyvä.' (S1): the answer is a
    legitimate one-word turn, not flicker — the ? guards the flip."""
    words = [
        _w("tehdäänkö?", 1.0, 1.5),
        _w("tehdään", 1.7, 2.1),
        _w("hyvä", 2.4, 2.8),
    ]
    turns = [(0.0, 1.6, "S1"), (1.65, 2.2, "S2"), (2.3, 4.0, "S1")]
    labels = assign_word_speakers(words, turns)
    assert labels == ["S1", "S2", "S1"]
