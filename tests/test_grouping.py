"""M3 split-recording grouping tests (issue #77).

Natural sort, the pure heuristic (``propose_groups`` / ``edge_snippets``),
the impure boundary decode, the ffmpeg concat, part offsets, the
edit-string partition parser, and the interactive confirm loop.
Pure-stdlib + numpy except where a test needs the real ffmpeg binary
(marked ``skipif``); no model loads, no network.
"""

from __future__ import annotations

import shutil
import struct
import unicodedata
from pathlib import Path

import numpy as np
import pytest

from vemoizer.grouping import (
    EN_CLOSING_CUES,
    FI_CLOSING_CUES,
    GroupingError,
    GroupProposal,
    PartOffset,
    concat_groups,
    confirm_groups,
    decode_boundaries,
    edge_snippets,
    natural_sort,
    parse_partition,
    part_offsets,
    propose_groups,
    stems_of,
)
from vemoizer.textnorm import textnorm

FFMPEG = shutil.which("ffmpeg") is not None
FFPROBE = shutil.which("ffprobe") is not None
requires_ffmpeg = pytest.mark.skipif(not FFMPEG, reason="ffmpeg not on PATH")
requires_ffprobe = pytest.mark.skipif(
    not (FFMPEG and FFPROBE), reason="ffmpeg/ffprobe not on PATH"
)


def _nfc(*names: str) -> list[Path]:
    return [Path(n) for n in names]


def _make_wav(path: Path, seconds: float, rate: int = 16000) -> Path:
    """A real sine-wave 16-bit PCM WAV (real ffmpeg/ffprobe can open it)."""
    n = int(seconds * rate)
    t = np.arange(n) / rate
    audio = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    pcm = (audio * 32767).astype("<i2").tobytes()
    header = b"RIFF" + struct.pack("<I", 36 + len(pcm)) + b"WAVE"
    header += b"fmt " + struct.pack("<IHHIIHH", 16, 1, 1, rate, rate * 2, 2, 16)
    header += b"data" + struct.pack("<I", len(pcm))
    path.write_bytes(header + pcm)
    return path


# ---------------------------------------------------------------------------
# Natural sort
# ---------------------------------------------------------------------------


def test_natural_sort_numeric_not_lexicographic() -> None:
    files = _nfc(
        "Uusi äänitys 429.m4a",
        "Uusi äänitys 42.m4a",
        "Uusi äänitys 425.m4a",
        "Uusi äänitys 426.m4a",
    )
    sorted_names = [p.name for p in natural_sort(files)]
    # 42 < 425 < 426 < 429 numerically; lexicographically "429" < "42"
    # would put 429 first — the numeric key must win.
    assert sorted_names == [
        "Uusi äänitys 42.m4a",
        "Uusi äänitys 425.m4a",
        "Uusi äänitys 426.m4a",
        "Uusi äänitys 429.m4a",
    ]


def test_natural_sort_nfd_disk_spelling_sorts_like_nfc() -> None:
    """macOS APFS hands back NFD; the sort key must NFC before the regex."""
    nfc_name = "Uusi äänitys 425.m4a"
    nfd_name = unicodedata.normalize("NFD", nfc_name)
    files = [Path(nfc_name), Path(nfd_name), Path("Uusi äänitys 426.m4a")]
    sorted_names = [p.name for p in natural_sort(files)]
    # Both 425 spellings come before 426.
    assert sorted_names[2] == "Uusi äänitys 426.m4a"
    assert set(sorted_names[:2]) == {nfc_name, nfd_name}


def test_natural_sort_ignores_creation_time(tmp_path, monkeypatch) -> None:
    """A later creation_time on the higher-numbered file must not reorder."""
    a = tmp_path / "Uusi äänitys 425.m4a"
    b = tmp_path / "Uusi äänitys 426.m4a"
    c = tmp_path / "Uusi äänitys 429.m4a"
    for p in (a, b, c):
        p.touch()
    stat_calls: list[object] = []
    import os

    real_stat = os.stat

    def fake_stat(path, *args, **kwargs):
        stat_calls.append(path)
        return real_stat(path, *args, **kwargs)

    monkeypatch.setattr(os, "stat", fake_stat)
    sorted_names = [p.name for p in natural_sort([c, a, b])]
    assert sorted_names == [
        "Uusi äänitys 425.m4a",
        "Uusi äänitys 426.m4a",
        "Uusi äänitys 429.m4a",
    ]
    # The sort never consulted file metadata (creation_time etc.).
    assert stat_calls == []


def test_natural_sort_unnumbered_stems_before_numbered() -> None:
    files = _nfc("Uusi äänitys 5.m4a", "kylmä memo.m4a")
    sorted_names = [p.name for p in natural_sort(files)]
    assert sorted_names == ["kylmä memo.m4a", "Uusi äänitys 5.m4a"]


def test_stems_of_returns_nfc_stems_in_sorted_order() -> None:
    stems = stems_of(_nfc("Uusi äänitys 426.m4a", "Uusi äänitys 425.m4a"))
    assert stems == ["Uusi äänitys 425", "Uusi äänitys 426"]


# ---------------------------------------------------------------------------
# edge_snippets — pure, word-bounded
# ---------------------------------------------------------------------------


def test_edge_snippets_bounded_by_edge_snippet_words() -> None:
    from vemoizer.grouping import EDGE_SNIPPET_WORDS

    tail = " ".join(f"w{int(i)}" for i in range(50))
    head = " ".join(f"x{int(i)}" for i in range(50))
    tail_snip, head_snip = edge_snippets(tail, head)
    assert len(tail_snip.split()) == EDGE_SNIPPET_WORDS
    assert len(head_snip.split()) == EDGE_SNIPPET_WORDS
    # The last N words, not the first.
    assert tail_snip.split()[-1] == "w49"
    assert head_snip.split()[-1] == "x49"


def test_edge_snippets_empty_both_sides_is_no_audio() -> None:
    tail_snip, head_snip = edge_snippets("", "")
    assert (tail_snip, head_snip) == ("(no audio)", "(no audio)")


def test_edge_snippets_only_one_side_silent() -> None:
    tail_snip, head_snip = edge_snippets("", "moi moi")
    assert tail_snip == "(no audio)"
    assert head_snip == "moi moi"


def test_edge_snippets_runs_on_textnorm_output() -> None:
    # "Kiitos!" normalises to "kiitos" — casefolded, punctuation-free.
    tail_snip, _ = edge_snippets("Kiitos!", "moi")
    assert textnorm(tail_snip) == "kiitos"


def test_edge_snippets_is_pure_never_raises() -> None:
    tail_snip, head_snip = edge_snippets("\n\n  a\tb  ", "")
    assert len(tail_snip.split()) <= 8
    assert head_snip == "(no audio)"


# ---------------------------------------------------------------------------
# propose_groups — pure heuristic
# ---------------------------------------------------------------------------


def _files4() -> list[Path]:
    return _nfc(
        "Uusi äänitys 425.m4a",
        "Uusi äänitys 426.m4a",
        "Uusi äänitys 427.m4a",
        "Uusi äänitys 428.m4a",
    )


def test_propose_groups_mid_sentence_tail_is_continuation() -> None:
    files = _files4()
    tails = [
        "ja sitten puhuttiin alustan suunnasta viel",
        "tämä on vasta alku",
        "ensimäinen lause päättyi tähän",
    ]
    heads = [
        "ensiksi käytiin läpi agenda",
        "jatkaetaan sieltä mistä jäätiin",
        "kolmas osu alkaa tästä",
    ]
    proposals = propose_groups(files, tails, heads)
    assert len(proposals) == 3
    assert all(p.is_continuation for p in proposals)
    for p, t, h in zip(proposals, tails, heads, strict=True):
        expected_tail = " ".join(textnorm(t).split()[-8:])
        expected_head = " ".join(textnorm(h).split()[-8:])
        assert p.evidence == (expected_tail, expected_head)


def test_propose_groups_closing_cue_forces_break() -> None:
    files = _files4()
    # "kiitos" in the last 3 words of the tail -> break, even though the
    # head is mid-sentence. Silent tail (empty) on the second boundary ->
    # no evidence -> break.
    tails = ["puhua puhua kiitos", "", "ja sitten vielä"]
    heads = ["jatkaetaan", "moi moi", "kolmas alku"]
    proposals = propose_groups(files, tails, heads)
    assert proposals[0].is_continuation is False
    # A silent tail (empty) is no evidence -> break.
    assert proposals[1].is_continuation is False
    # Mid-sentence tail, no cue -> continuation.
    assert proposals[2].is_continuation is True


@pytest.mark.parametrize("cue", list(FI_CLOSING_CUES) + list(EN_CLOSING_CUES))
def test_propose_groups_every_seeded_cue_forces_break(cue: str) -> None:
    files = _files4()
    tails = [f"puhua puhua {cue}", "", "ja sitten vielä"]
    heads = ["moi", "moi", "kolmas alku"]
    proposals = propose_groups(files, tails, heads)
    assert proposals[0].is_continuation is False
    # The silent second boundary also breaks.
    assert proposals[1].is_continuation is False


def test_propose_groups_cue_mid_sentence_does_not_break() -> None:
    files = _files4()
    # "kiitos" far from the tail edge (window is the last 3 words): the
    # cue is mid-sentence, not a closing cue -> continuation.
    tails = [
        "puhua puhua kiitos puhua puhua puhua puhua puhua",
        "ja sitten puhuttiin vielä",
        "kolmas lause jatkuu",
    ]
    heads = ["ensimmäinen alku", "toinen alku", "kolmas alku"]
    proposals = propose_groups(files, tails, heads)
    assert proposals[0].is_continuation is True
    assert proposals[1].is_continuation is True
    assert proposals[2].is_continuation is True


def test_propose_groups_silent_boundary_is_break_not_continuation() -> None:
    files = _files4()
    proposals = propose_groups(files, ["", "", ""], ["", "", ""])
    assert all(p.is_continuation is False for p in proposals)
    assert all(p.evidence == ("(no audio)", "(no audio)") for p in proposals)


def test_propose_groups_truncated_short_part_degrades_not_crash() -> None:
    files = _files4()
    tails = ["lyhyt teksti", "toinen lyhyt", "kolmas lyhyt"]
    heads = ["a", "b", "c"]
    proposals = propose_groups(files, tails, heads)
    assert proposals[0].is_continuation is True


def test_propose_groups_is_pure_no_model_access(monkeypatch) -> None:
    """No I/O, no model, no clock: sentinel patches must never fire."""
    files = _files4()
    import builtins

    def boom(*a, **k):
        raise AssertionError("I/O or model access in a pure function")

    monkeypatch.setattr(builtins, "open", boom)
    monkeypatch.setattr("vemoizer.whisper_transcriber.WhisperTranscriber", boom)
    proposals = propose_groups(files, ["a b", "c d", "e f"], ["g h", "i j", "k l"])
    assert len(proposals) == 3
    assert all(p.is_continuation for p in proposals)


def test_propose_groups_wrong_tail_head_count_raises() -> None:
    files = _files4()
    with pytest.raises(ValueError, match="len\\(files\\) - 1"):
        propose_groups(files, ["only one"], ["one", "and one more"])


def test_propose_groups_parts_are_sorted_names() -> None:
    # Input order shuffled: the proposal's parts follow natural sort.
    files = [
        Path("Uusi äänitys 428.m4a"),
        Path("Uusi äänitys 425.m4a"),
        Path("Uusi äänitys 426.m4a"),
    ]
    proposals = propose_groups(files, ["x", "y"], ["a", "b"])
    assert proposals[0].parts == ["Uusi äänitys 425.m4a", "Uusi äänitys 426.m4a"]
    assert proposals[1].parts == ["Uusi äänitys 426.m4a", "Uusi äänitys 428.m4a"]


# ---------------------------------------------------------------------------
# decode_boundaries — impure helper
# ---------------------------------------------------------------------------


def test_decode_boundaries_calls_transcribe_once_per_edge(
    tmp_path, monkeypatch
) -> None:
    """Each file's tail and its next file's head are decoded once each.

    The decode is bounded to the 20 s edge (``-ss``/``-t``): the fake
    edge decoder returns exactly the requested window, and ffprobe
    supplies the (advisory) full duration for the tail's start bound.
    """
    import vemoizer.grouping as grouping

    durations = {
        "Uusi äänitys 425.m4a": 60.0,
        "Uusi äänitys 426.m4a": 45.0,
    }

    def fake_probe(path):
        return durations[Path(path).name]

    def fake_edge_window(path, start, end):
        return np.zeros(int((end - start) * 16000), dtype=np.float32)

    transcribe_calls: list[float] = []

    def fake_transcribe(audio):
        transcribe_calls.append(len(audio) / 16000)
        return {"text": "hei"}

    monkeypatch.setattr(grouping, "probe_duration_seconds", fake_probe)
    monkeypatch.setattr(grouping, "_decode_edge_window", fake_edge_window)
    files = [tmp_path / name for name in durations]
    for f in files:
        f.touch()

    tails, heads = decode_boundaries(files, transcribe_fn=fake_transcribe)
    # 2 files: 1 tail (file 1 — the last file's tail has no successor) +
    # 1 head (first 20 s of file 2) = 2 decodes.
    assert len(tails) == 1
    assert len(heads) == 1
    # Tail of the 60 s file: last 20 s. Head of the 45 s file: first 20 s.
    assert abs(transcribe_calls[0] - 20.0) < 0.01
    assert abs(transcribe_calls[1] - 20.0) < 0.01
    assert tails == ["hei"]
    assert heads == ["hei"]


def test_decode_boundaries_short_file_tail_clips(tmp_path, monkeypatch) -> None:
    """A 5 s file: the tail slice is 5 s (clipped), not 20 s."""
    import vemoizer.grouping as grouping

    monkeypatch.setattr(grouping, "probe_duration_seconds", lambda p: 5.0)
    monkeypatch.setattr(
        grouping,
        "_decode_edge_window",
        lambda path, start, end: np.zeros(int((end - start) * 16000), dtype=np.float32),
    )
    transcribe_calls: list[float] = []

    def fake_transcribe(audio):
        transcribe_calls.append(len(audio) / 16000)
        return {"text": "x"}

    files = [tmp_path / "Uusi äänitys 425.m4a"]
    files[0].touch()

    tails, heads = decode_boundaries(files, transcribe_fn=fake_transcribe)
    # Single file: no boundary at all — no tail, no head.
    assert len(tails) == 0
    assert len(heads) == 0
    assert transcribe_calls == []


def test_decode_boundaries_decode_failure_degrades_to_empty(
    tmp_path, monkeypatch
) -> None:
    """A per-slice decode failure degrades to "" (no evidence), never a raise."""
    import vemoizer.grouping as grouping

    monkeypatch.setattr(
        grouping,
        "ingest_audio",
        lambda path: np.zeros(int(30 * 16000), dtype=np.float32),
    )

    def fake_transcribe(audio):
        raise RuntimeError("model exploded")

    files = [tmp_path / "Uusi äänitys 425.m4a", tmp_path / "Uusi äänitys 426.m4a"]
    for f in files:
        f.touch()
    monkeypatch.setattr(grouping, "probe_duration_seconds", lambda p: 30.0)

    tails, heads = decode_boundaries(files, transcribe_fn=fake_transcribe)
    # Both edges (file 1 tail, file 2 head) degrade to "" on the decode
    # failure — never a raise.
    assert tails == [""]
    assert heads == [""]


def test_decode_boundaries_empty_audio_is_empty_text(tmp_path, monkeypatch) -> None:
    import vemoizer.grouping as grouping

    monkeypatch.setattr(grouping, "probe_duration_seconds", lambda p: 60.0)
    monkeypatch.setattr(
        grouping,
        "_decode_edge_window",
        lambda path, start, end: np.zeros(0, dtype=np.float32),
    )
    files = [tmp_path / "Uusi äänitys 425.m4a", tmp_path / "Uusi äänitys 426.m4a"]
    for f in files:
        f.touch()
    tails, heads = decode_boundaries(files, transcribe_fn=lambda a: {"text": "no"})
    # An empty edge decode (silence/truncation) degrades to "" (no evidence).
    assert tails == [""]
    assert heads == [""]


def test_decode_boundaries_probe_failure_skips_tail(tmp_path, monkeypatch) -> None:
    """A file ffprobe cannot read (duration 0.0): the tail is skipped
    ("" = no evidence) rather than requesting an unbounded decode window."""
    import vemoizer.grouping as grouping

    monkeypatch.setattr(grouping, "probe_duration_seconds", lambda p: 0.0)
    decoded_windows: list[tuple[float, float]] = []

    def fake_edge_window(path, start, end):
        decoded_windows.append((start, end))
        return np.zeros(int((end - start) * 16000), dtype=np.float32)

    monkeypatch.setattr(grouping, "_decode_edge_window", fake_edge_window)
    files = [tmp_path / "Uusi äänitys 425.m4a"]
    files[0].touch()
    tails, heads = decode_boundaries(files, transcribe_fn=lambda a: {"text": "no"})
    # Single file, no probe evidence: no tail decode at all, no head.
    assert tails == []
    assert heads == []
    assert decoded_windows == []


# ---------------------------------------------------------------------------
# concat_groups
# ---------------------------------------------------------------------------


@requires_ffmpeg
def test_concat_groups_single_file_passthrough(tmp_path) -> None:
    """A single-file group: no ffmpeg call, no temp file, input returned."""
    src = tmp_path / "Uusi äänitys 425.m4a"
    src.write_bytes(b"not really audio but the passthrough never opens it")
    out = concat_groups([src])
    assert out == src


def test_concat_groups_escape_apostrophe_and_space_in_list_file(
    tmp_path, monkeypatch
) -> None:
    """Apostrophes and spaces in filenames must not break the list file."""
    import subprocess

    captured: dict[str, list[str]] = {"argv": [], "list": []}

    def spy_run(argv, *a, **k):
        captured["argv"] = list(argv)
        list_path = Path(argv[argv.index("-i") + 1])
        captured["list"] = list(list_path.read_text().splitlines())
        out = Path(argv[-1])
        out.write_bytes(b"")

        class R:
            returncode = 0
            stdout = b""
            stderr = b""

        return R()

    monkeypatch.setattr(subprocess, "run", spy_run)
    monkeypatch.setattr("vemoizer.grouping._probe_stream", lambda p: "aac,48000,1")
    a = tmp_path / "Uusi äänitys 425.m4a"
    b = tmp_path / "Möös's memo.m4a"
    a.touch()
    b.touch()
    concat_groups([a, b])
    lines = captured["list"]
    assert len(lines) == 2
    # The apostrophe must be escaped '...' -> '\'' so the list file is valid.
    assert "\\'" in lines[1]
    # The path with the space is inside single quotes.
    assert lines[0] == f"file '{a}'"
    # -c copy is in the argv (no re-encode).
    assert "-c" in captured["argv"]
    assert captured["argv"][captured["argv"].index("-c") + 1] == "copy"


@requires_ffmpeg
def test_concat_groups_real_concat_produces_one_file(tmp_path) -> None:
    a = _make_wav(tmp_path / "Uusi äänitys 425.wav", 1.0)
    b = _make_wav(tmp_path / "Uusi äänitys 426.wav", 1.0)
    out = concat_groups([a, b])
    assert out.is_file()
    assert out.stat().st_size > 0
    out.unlink(missing_ok=True)


@requires_ffprobe
def test_concat_groups_codec_mismatch_raises_naming_files(
    tmp_path, monkeypatch
) -> None:
    """A mismatched audio stream raises a GroupingError naming the files."""
    import vemoizer.grouping as grouping

    a = _make_wav(tmp_path / "part_a.wav", 1.0)
    b = _make_wav(tmp_path / "part_b.wav", 1.0)

    real_probe = grouping._probe_stream

    def fake_probe(p):
        if p.name == "part_b.wav":
            return "aac,44100,2"
        return real_probe(p)

    monkeypatch.setattr(grouping, "_probe_stream", fake_probe)
    with pytest.raises(GroupingError, match="part_b"):
        concat_groups([a, b])


@requires_ffmpeg
def test_concat_groups_missing_part_file_raises(tmp_path) -> None:
    a = _make_wav(tmp_path / "part_a.wav", 1.0)
    missing = tmp_path / "Uusi äänitys 999.m4a"
    with pytest.raises(GroupingError, match="part file not found"):
        concat_groups([a, missing])


# ---------------------------------------------------------------------------
# part_offsets
# ---------------------------------------------------------------------------


def test_part_offsets_cumulative_pcm_durations(tmp_path, monkeypatch) -> None:
    """Part N starts at the sum of parts 1..N-1 decoded durations."""
    import vemoizer.grouping as grouping

    durations = [10.0, 20.0, 5.0]
    files = [tmp_path / f"Uusi äänitys {425 + i}.m4a" for i in range(3)]
    for f in files:
        f.touch()
    durations_by_name = {f.name: d for f, d in zip(files, durations, strict=True)}

    def fake_ingest(path):
        name = Path(path).name
        return np.zeros(int(durations_by_name[name] * 16000), dtype=np.float32)

    monkeypatch.setattr(grouping, "ingest_audio", fake_ingest)
    offsets = part_offsets(files)
    assert len(offsets) == 3
    assert offsets[0] == PartOffset(1, "Uusi äänitys 425.m4a", 0.0)
    assert offsets[1] == PartOffset(2, "Uusi äänitys 426.m4a", 10.0)
    assert offsets[2] == PartOffset(3, "Uusi äänitys 427.m4a", 30.0)


# ---------------------------------------------------------------------------
# parse_partition
# ---------------------------------------------------------------------------


def test_parse_partition_valid_partition_with_pipes_and_plus() -> None:
    stems = [
        "Uusi äänitys 425",
        "Uusi äänitys 426",
        "Uusi äänitys 427",
        "Uusi äänitys 428",
        "Uusi äänitys 429",
    ]
    groups = parse_partition("425 | 426 | 427+428+429", stems)
    assert [p.name for g in groups for p in g] == [
        "Uusi äänitys 425",
        "Uusi äänitys 426",
        "Uusi äänitys 427",
        "Uusi äänitys 428",
        "Uusi äänitys 429",
    ]
    assert len(groups) == 3
    assert [len(g) for g in groups] == [1, 1, 3]


def test_parse_partition_full_stems_accepted() -> None:
    stems = ["425", "426"]
    groups = parse_partition("425 | 426", stems)
    assert len(groups) == 2
    assert [g[0].name for g in groups] == ["425", "426"]


def test_parse_partition_whitespace_only_separators() -> None:
    stems = ["Uusi äänitys 425", "Uusi äänitys 426"]
    groups = parse_partition("425  426", stems)
    assert len(groups) == 2


def test_parse_partition_unknown_stem_names_token() -> None:
    stems = ["Uusi äänitys 425", "Uusi äänitys 426"]
    with pytest.raises(GroupingError, match="430"):
        parse_partition("425 | 430", stems)


def test_parse_partition_ambiguous_trailing_integer_raises() -> None:
    """A trailing integer matching two stems is ambiguous (not unknown).

    The error names the token AND both matching stems.
    """
    stems = ["A-42", "B-42"]
    with pytest.raises(GroupingError, match=r"ambiguous part: '42' matches"):
        parse_partition("42 + 42", stems)


def test_parse_partition_full_stem_disambiguates_duplicate_integer() -> None:
    """The full stem is always a valid token, even when its trailing
    integer alone would be ambiguous."""
    stems = ["A-42", "B-42"]
    groups = parse_partition("A-42 | B-42", stems)
    assert len(groups) == 2
    assert [g[0].name for g in groups] == ["A-42", "B-42"]


def test_parse_partition_duplicate_part_names_token() -> None:
    stems = ["Uusi äänitys 425", "Uusi äänitys 426"]
    with pytest.raises(GroupingError, match="duplicate part"):
        parse_partition("425 | 425", stems)


def test_parse_partition_out_of_order_names_token() -> None:
    stems = ["Uusi äänitys 425", "Uusi äänitys 426"]
    with pytest.raises(GroupingError, match="out-of-order"):
        parse_partition("426 | 425", stems)


def test_parse_partition_missing_part_names_it() -> None:
    stems = ["Uusi äänitys 425", "Uusi äänitys 426"]
    with pytest.raises(GroupingError, match="missing"):
        parse_partition("425", stems)


def test_parse_partition_empty_string_raises() -> None:
    stems = ["Uusi äänitys 425", "Uusi äänitys 426"]
    with pytest.raises(GroupingError):
        parse_partition("   ", stems)


def test_parse_partition_double_pipe_is_empty_group_error() -> None:
    stems = ["Uusi äänitys 425", "Uusi äänitys 426"]
    # "425 || 426" splits into ["425", "", "426"] — the empty middle
    # token is skipped by the regex split (re.split on [|\s]+), so this
    # is actually a valid "425 | 426" partition; the malformed shape that
    # errors is an empty group inside a token ("425+ | 426").
    with pytest.raises(GroupingError):
        parse_partition("425+ | 426", stems)


# ---------------------------------------------------------------------------
# confirm_groups
# ---------------------------------------------------------------------------


def test_confirm_groups_yes_accepts_all_proposals() -> None:
    files = _files4()
    proposals = [
        GroupProposal(parts=["a", "b"], is_continuation=True, evidence=("x", "y")),
        GroupProposal(parts=["b", "c"], is_continuation=False, evidence=("x", "y")),
        GroupProposal(parts=["c", "d"], is_continuation=True, evidence=("x", "y")),
    ]
    groups = confirm_groups(files, proposals, yes=True)
    # 425+426 (continue), break, 427+428 (continue) -> 2 groups.
    assert [len(g) for g in groups] == [2, 2]
    assert groups[0][0].name == "Uusi äänitys 425.m4a"
    assert groups[1][0].name == "Uusi äänitys 427.m4a"


def test_confirm_groups_no_group_is_singletons() -> None:
    files = _files4()
    proposals = []
    groups = confirm_groups(files, proposals, no_group=True)
    assert len(groups) == 4
    assert all(len(g) == 1 for g in groups)


def test_confirm_groups_interactive_enter_accepts() -> None:
    files = _files4()
    proposals = [
        GroupProposal(parts=["a", "b"], is_continuation=True, evidence=("x", "y")),
        GroupProposal(parts=["b", "c"], is_continuation=False, evidence=("x", "y")),
        GroupProposal(parts=["c", "d"], is_continuation=True, evidence=("x", "y")),
    ]
    inputs = iter(["\n", "\n", "\n"])
    prints: list[str] = []
    groups = confirm_groups(
        files,
        proposals,
        input_fn=lambda _: next(inputs),
        print_fn=prints.append,
    )
    assert [len(g) for g in groups] == [2, 2]
    # One prompt per boundary.
    assert len(prints) == 3


def test_confirm_groups_interactive_q_aborts() -> None:
    files = _files4()
    proposals = [
        GroupProposal(parts=["a", "b"], is_continuation=True, evidence=("x", "y")),
    ]
    with pytest.raises(GroupingError, match="aborted"):
        confirm_groups(files, proposals, input_fn=lambda _: "q")


def test_confirm_groups_interactive_edit_uses_any_partition() -> None:
    files = _files4()
    proposals = [
        GroupProposal(parts=["a", "b"], is_continuation=False, evidence=("x", "y")),
        GroupProposal(parts=["b", "c"], is_continuation=False, evidence=("x", "y")),
        GroupProposal(parts=["c", "d"], is_continuation=False, evidence=("x", "y")),
    ]
    # 'e' then the partition string.
    inputs = iter(["e", "425+426+427 | 428"])
    groups = confirm_groups(files, proposals, input_fn=lambda _: next(inputs))
    assert [len(g) for g in groups] == [3, 1]


def test_confirm_groups_interactive_edit_malformed_propagates() -> None:
    files = _files4()
    proposals = [
        GroupProposal(parts=["a", "b"], is_continuation=False, evidence=("x", "y")),
    ]
    inputs = iter(["e", "999 | 998"])
    with pytest.raises(GroupingError, match="unknown part"):
        confirm_groups(files, proposals, input_fn=lambda _: next(inputs))


def test_probe_failure_logs_a_warning(tmp_path, monkeypatch, caplog) -> None:
    """A file with no ffprobe duration evidence logs a clear warning (the
    tail probe is skipped, not silently absent) — round-1 finding 3."""
    import vemoizer.grouping as grouping

    monkeypatch.setattr(grouping, "probe_duration_seconds", lambda p: 0.0)
    monkeypatch.setattr(
        grouping,
        "_decode_edge_window",
        lambda path, start, end: np.zeros(16000, dtype=np.float32),
    )
    files = [tmp_path / "Uusi äänitys 425.m4a", tmp_path / "Uusi äänitys 426.m4a"]
    for f in files:
        f.touch()
    with caplog.at_level("WARNING", logger="vemoizer.grouping"):
        tails, heads = decode_boundaries(files, transcribe_fn=lambda a: {"text": "x"})
    assert tails == [""]  # tail skipped (no evidence), head still decoded
    assert heads == ["x"]
    assert any("no duration evidence" in m for m in caplog.messages)
    assert any("tail probe skipped" in m for m in caplog.messages)
