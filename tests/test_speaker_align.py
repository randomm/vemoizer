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
    assign_track_labels,
    assign_word_speakers,
    label_segments_by_track,
    split_segments_at_speaker_changes,
    turns_for_track,
)


def _w(word, start, end):
    return {"word": word, "start": start, "end": end}


def _tw(word, start, end, track):
    return {"word": word, "start": start, "end": end, "track": track}


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


def test_trusted_owner_word_between_remote_runs_is_not_smoothed() -> None:
    """Trusted (owner) and untrusted (remote) labels never flip across a boundary."""
    words = [
        _w("kerron", 0.0, 0.5),
        _w("tässä", 0.6, 1.0),
        _w("Selvä.", 1.1, 1.5),
        _w("nyt", 1.6, 2.0),
        _w("eteenpäin", 2.1, 2.5),
    ]
    turns = [(0.0, 1.05, "R"), (1.05, 1.55, "O"), (1.55, 3.0, "R")]
    assert assign_word_speakers(words, turns, trusted=frozenset({"O"})) == [
        "R",
        "R",
        "O",
        "R",
        "R",
    ]
    assert assign_word_speakers(words, turns) == ["R"] * 5


def test_two_word_owner_run_on_short_turn_is_not_merged_into_remote() -> None:
    words = [
        _w("kerron", 0.0, 1.0),
        _w("tuo", 1.1, 1.4),
        _w("on", 1.5, 1.8),
        _w("asia", 2.0, 3.0),
    ]
    turns = [(0.0, 1.05, "R"), (1.05, 1.85, "O"), (1.85, 4.0, "R")]
    assert assign_word_speakers(words, turns, trusted=frozenset({"O"})) == [
        "R",
        "O",
        "O",
        "R",
    ]
    assert assign_word_speakers(words, turns) == ["R"] * 4


def test_remote_island_between_owner_runs_is_not_flipped() -> None:
    words = [
        _w("puhun", 0.0, 0.5),
        _w("tässä", 0.6, 1.0),
        _w("eli", 1.3, 1.6),
        _w("koko", 1.8, 2.2),
        _w("ajan", 2.3, 2.7),
    ]
    turns = [(0.0, 1.2, "O"), (1.2, 1.7, "R"), (1.7, 3.0, "O")]
    assert assign_word_speakers(words, turns, trusted=frozenset({"O"})) == [
        "O",
        "O",
        "R",
        "O",
        "O",
    ]
    assert assign_word_speakers(words, turns) == ["O"] * 5


# -- per-track attribution (issue #207) ------------------------------------
#
# A two-track call merges the mic ("me") and system ("them") decodes onto
# one timeline; the segments then OVERLAP in time. Membership and labelling
# must respect the track, or remote words get pulled into the owner's
# sentence and the owner's segment can take the remote speaker.


def test_overlapping_track_segments_keep_their_own_track_words() -> None:
    """Time-only membership interleaved and duplicated both segments."""
    me_seg = {"start": 0.0, "end": 1.4, "text": "tämä on suomeksi", "track": "me"}
    them_seg = {"start": 0.3, "end": 1.4, "text": "if you're happy", "track": "them"}
    words = [
        _tw("tämä", 0.0, 0.4, "me"),
        _tw("if", 0.3, 0.5, "them"),
        _tw("on", 0.5, 0.7, "me"),
        _tw("you're", 0.6, 0.9, "them"),
        _tw("suomeksi", 0.8, 1.3, "me"),
        _tw("happy", 1.0, 1.4, "them"),
    ]
    labels: list[str | None] = ["A", "B", "A", "B", "A", "B"]
    out = split_segments_at_speaker_changes([me_seg, them_seg], words, labels)
    assert len(out) == 2, "no interleaved pieces, no duplication"
    assert [(s["text"], s["speaker"]) for s in out] == [
        ("tämä on suomeksi", "A"),
        ("if you're happy", "B"),
    ]


def test_turns_for_track_filters_by_the_trusted_line() -> None:
    turns = [(0.0, 1.0, "O"), (2.0, 3.0, "R"), (4.0, 5.0, "O")]
    trusted = frozenset({"O"})
    assert turns_for_track("me", turns, trusted) == [turns[0], turns[2]]
    assert turns_for_track("them", turns, trusted) == [turns[1]]
    assert turns_for_track(None, turns, trusted) == turns
    assert turns_for_track("me", turns, frozenset()) == turns  # fail-open


def test_me_word_takes_the_owner_label_despite_a_bigger_remote_turn() -> None:
    """An owner word must not be stolen by a longer-overlapping remote turn."""
    turns = [(0.1, 0.7, "O"), (0.0, 4.0, "R")]  # R overlaps the word more
    words = [_tw("tämä", 0.2, 0.6, "me"), _tw("hello", 0.2, 0.6, "them")]
    labels = assign_track_labels(words, turns, trusted=frozenset({"O"}))
    assert labels == ["O", "R"]


def test_assign_track_labels_without_track_keys_matches_plain_attribution() -> None:
    """Single-file runs (no track keys) are byte-identical to before."""
    words = [_w("kysymys", 4.0, 4.8), _w("vastaus", 5.2, 6.0)]
    assert assign_track_labels(words, TURNS) == assign_word_speakers(words, TURNS)


def test_label_segments_by_track_uses_the_segment_own_turns() -> None:
    segments = [
        {"start": 0.0, "end": 0.8, "text": "moi", "track": "me"},
        {"start": 0.0, "end": 0.8, "text": "hi", "track": "them"},
        {"start": 0.0, "end": 0.8, "text": "plain"},
    ]
    turns = [(0.0, 0.4, "O"), (0.0, 4.0, "R")]
    label_segments_by_track(segments, turns, trusted=frozenset({"O"}))
    # The untracked segment keeps the all-turns pick (R overlaps more).
    assert [s["speaker"] for s in segments] == ["O", "R", "R"]


def test_empty_trusted_set_keeps_untrusted_smoothing() -> None:
    """Regression pin: no trusted labels means the pre-#170 behaviour."""
    words = [
        _w("puhun", 0.0, 0.5),
        _w("tässä", 0.6, 1.0),
        _w("eli", 1.3, 1.6),
        _w("koko", 1.8, 2.2),
        _w("ajan", 2.3, 2.7),
    ]
    turns = [(0.0, 1.2, "O"), (1.2, 1.7, "R"), (1.7, 3.0, "O")]
    assert assign_word_speakers(words, turns, trusted=frozenset()) == ["O"] * 5
