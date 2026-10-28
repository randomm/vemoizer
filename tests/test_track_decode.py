"""Per-track meeting decode (issue #207).

The owner's mic stem and the remote system stem are decoded separately so
each speaker keeps their own language; the mix decode is only the fallback.
All models, VAD and ingest are faked — no weights, no network, no ffmpeg.
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from test_pipeline import (  # noqa: F401 - shared orchestrator fixtures
    _patch_ingest,
    _patch_preflight_pass,
    _patch_vad,
)

import vemoizer.pipeline as pipeline
import vemoizer.track_decode as track_decode
from vemoizer.pipeline import transcribe_file
from vemoizer.progress import ProgressDisplay
from vemoizer.track_decode import (
    decode_tracks,
    mask_outside,
    merge_track_results,
)
from vemoizer.tracks import TrackPaths
from vemoizer.vad import SpeechSegment

SR = 16_000
OWNER = Path("call-me.wav")
THEM = Path("call-them.wav")


# -- mask_outside ---------------------------------------------------------


def test_mask_outside_keeps_inside_and_zeroes_outside() -> None:
    audio = np.full(SR * 2, 0.5, dtype=np.float32)
    masked = mask_outside(audio, [{"start": 0.25, "end": 0.75}], pad_s=0.0)
    assert masked is not audio
    assert masked[round(0.30 * SR) : round(0.70 * SR)] == pytest.approx(0.5)
    assert not masked[: round(0.25 * SR)].any()
    assert not masked[round(0.75 * SR) :].any()
    assert audio == pytest.approx(0.5), "input must not be mutated"


def test_mask_outside_pads_spans_by_default() -> None:
    audio = np.ones(SR, dtype=np.float32)
    masked = mask_outside(audio, [{"start": 0.5, "end": 0.6}])
    assert masked[round(0.30 * SR) : round(0.75 * SR)] == pytest.approx(1.0)
    assert not masked[: round(0.20 * SR)].any()
    assert not masked[round(0.85 * SR) :].any()


def test_mask_outside_clamps_spans_to_the_samples() -> None:
    audio = np.ones(SR, dtype=np.float32)
    masked = mask_outside(audio, [{"start": -5.0, "end": 30.0}], pad_s=0.0)
    assert masked == pytest.approx(1.0)


def test_mask_outside_with_no_spans_silences_everything() -> None:
    audio = np.ones(SR, dtype=np.float32)
    assert not mask_outside(audio, []).any()


# -- merge_track_results --------------------------------------------------


def _side(
    language: str,
    word_starts: list[float],
    segment_texts: list[str],
) -> dict[str, Any]:
    """One decode result shaped like ``decode_meeting``'s."""
    return {
        "text": " ".join(segment_texts),
        "words": [
            {"word": f"w{i}", "start": s, "end": s + 0.2}
            for i, s in enumerate(word_starts)
        ],
        "segments": [
            {"start": s, "end": s + 0.5, "text": t}
            for s, t in zip(word_starts, segment_texts, strict=True)
        ],
        "slices": [
            {"index": i, "start_s": s, "end_s": s + 0.5, "text": t}
            for i, (s, t) in enumerate(zip(word_starts, segment_texts, strict=True))
        ],
        "language": language,
    }


def test_merge_track_results_none_cases() -> None:
    assert merge_track_results(None, None) is None


def test_merge_track_results_sorts_by_start() -> None:
    me = _side("fi", [0.0, 3.0], ["aa", "cc"])
    them = _side("en", [1.5], ["bb"])
    merged = merge_track_results(me, them)
    assert merged is not None
    assert [w["start"] for w in merged["words"]] == [0.0, 1.5, 3.0]
    assert [s["start"] for s in merged["segments"]] == [0.0, 1.5, 3.0]
    # Slice records carry "start_s" (slice_records_from_words), not "start".
    assert [s["start_s"] for s in merged["slices"]] == [0.0, 1.5, 3.0]


def test_merge_track_results_text_follows_sorted_segments() -> None:
    me = _side("fi", [3.0], ["cc"])
    them = _side("en", [1.5], ["bb"])
    merged = merge_track_results(me, them)
    assert merged is not None
    assert merged["text"] == "bb cc"


def test_merge_track_results_language_comes_from_the_word_majority() -> None:
    me = _side("fi", [0.0, 0.5, 1.0], ["a", "b", "c"])
    them = _side("en", [2.0, 2.5], ["x", "y"])
    merged = merge_track_results(me, them)
    assert merged is not None
    assert merged["language"] == "fi"
    them_more = _side("en", [2.0, 2.5, 3.0, 3.5], ["w", "x", "y", "z"])
    merged_en = merge_track_results(me, them_more)
    assert merged_en is not None
    assert merged_en["language"] == "en"


def test_merge_track_results_combines_string_language_summaries() -> None:
    """language_summary is the producer's string ("fi 29/30, en 1/30"),
    not a dict: counts and totals add, format matches whisper_windows."""
    me = _side("fi", [0.0], ["moi"])
    me["language_summary"] = "fi 2/2"
    them = _side("en", [1.0], ["hi"])
    them["language_summary"] = "en 1/1, fi 1/1"
    merged = merge_track_results(me, them)
    assert merged is not None
    assert merged["language_summary"] == "fi 3/3, en 1/3"


def test_merge_track_results_language_summary_tie_order_matches_producer() -> None:
    me = _side("fi", [0.0], ["moi"])
    me["language_summary"] = "fi 1/1"
    them = _side("en", [1.0], ["hi"])
    them["language_summary"] = "en 1/1"
    merged = merge_track_results(me, them)
    assert merged is not None
    # Count tie -> the existing order is kept (stable count-desc sort):
    # the me side's languages first, each side in the order the producer
    # (process_window_raws) wrote them. "fi 1/1" + "en 1/1" -> "fi 1/2, en 1/2".
    assert merged["language_summary"] == "fi 1/2, en 1/2"


def test_merge_track_results_language_summary_unparseable_falls_back() -> None:
    me = _side("fi", [0.0], ["moi"])
    me["language_summary"] = "fi 1/1"
    them = _side("en", [1.0], ["hi"])
    them["language_summary"] = "gibberish"
    merged = merge_track_results(me, them)
    assert merged is not None
    assert merged["language_summary"] == "fi 1/1"


def test_merge_track_results_language_summary_both_unparseable_omits() -> None:
    me = _side("fi", [0.0], ["moi"])
    me["language_summary"] = "nope"
    them = _side("en", [1.0], ["hi"])
    them["language_summary"] = "also nope"
    merged = merge_track_results(me, them)
    assert merged is not None
    assert "language_summary" not in merged


def test_merge_track_results_language_summary_absent_when_neither_has_one() -> None:
    plain = merge_track_results(_side("fi", [0.0], ["a"]), _side("en", [1.0], ["b"]))
    assert plain is not None
    assert "language_summary" not in plain


def test_merge_track_results_carries_lost_windows_sorted() -> None:
    """issue #152 entries ((start_s, end_s) pairs) must survive the merge."""
    me = _side("fi", [0.0], ["moi"])
    me["lost_windows"] = [(4.0, 5.0)]
    them = _side("en", [1.0], ["hi"])
    them["lost_windows"] = [[1.0, 2.0], (2.5, 3.0)]
    merged = merge_track_results(me, them)
    assert merged is not None
    assert merged["lost_windows"] == [(1.0, 2.0), (2.5, 3.0), (4.0, 5.0)]


def test_merge_track_results_no_lost_windows_key_when_neither_lost() -> None:
    merged = merge_track_results(_side("fi", [0.0], ["a"]), _side("en", [1.0], ["b"]))
    assert merged is not None
    assert "lost_windows" not in merged


def test_merge_track_results_tags_the_source_track_of_words_and_segments() -> None:
    me = _side("fi", [0.0, 3.0], ["aa", "cc"])
    them = _side("en", [1.5], ["bb"])
    merged = merge_track_results(me, them)
    assert merged is not None
    assert [w["track"] for w in merged["words"]] == ["me", "them", "me"]
    assert [s["track"] for s in merged["segments"]] == ["me", "them", "me"]
    assert all("track" not in s for s in merged["slices"])


def test_merge_track_results_single_side_tags_the_surviving_stem() -> None:
    """A legitimately skipped stem (no speech) must not lose attribution:
    the surviving side is tagged on a copy so labelling does not fall
    back to all turns."""
    me = _side("fi", [0.0], ["moi"])
    merged_me = merge_track_results(me, None)
    assert merged_me is not None and merged_me is not me
    assert all(w["track"] == "me" for w in merged_me["words"])
    assert all(s["track"] == "me" for s in merged_me["segments"])
    assert "track" not in me["words"][0], "input must not be mutated"

    them = _side("en", [1.0], ["hi"])
    merged_them = merge_track_results(None, them)
    assert merged_them is not None and merged_them is not them
    assert all(w["track"] == "them" for w in merged_them["words"])
    assert all(s["track"] == "them" for s in merged_them["segments"])
    assert "track" not in them["words"][0], "input must not be mutated"


def test_merge_track_results_single_side_keeps_lost_windows() -> None:
    me = _side("fi", [0.0], ["moi"])
    me["lost_windows"] = [(1.0, 2.0)]
    merged = merge_track_results(me, None)
    assert merged is not None
    assert merged["lost_windows"] == [(1.0, 2.0)]


def test_merge_track_results_does_not_mutate_inputs() -> None:
    me = _side("fi", [3.0], ["cc"])
    them = _side("en", [1.0], ["bb"])
    before = (copy.deepcopy(me), copy.deepcopy(them))
    merge_track_results(me, them)
    assert (me, them) == before


# -- decode_tracks --------------------------------------------------------


def _stems(mic_val: float, them_val: float) -> tuple[np.ndarray, np.ndarray]:
    return (
        np.full(SR * 2, mic_val, dtype=np.float32),
        np.full(SR * 2, them_val, dtype=np.float32),
    )


def _patch_stem_vad(monkeypatch: pytest.MonkeyPatch) -> None:
    """Mic loud at 0-1 s, them loud at 1-2 s (amplitude says which stem)."""
    monkeypatch.setattr(track_decode, "load_vad_model", lambda: object())

    def fake_vad(audio, model, **kw):
        if not audio.any():
            return []
        if float(audio[0]) < 1.5:
            return [SpeechSegment(start=0, end=SR)]  # mic: owner at 0-1 s
        if float(audio[SR]) >= 1.0:
            return [SpeechSegment(start=SR, end=SR * 2)]  # them: remote at 1-2 s
        return []

    monkeypatch.setattr(track_decode, "vad_segments", fake_vad)


def _fake_decode_meeting(calls: list[dict[str, Any]], fail: str | None = None):
    def fake(audio, slices, initial_prompt=None, language=None, vad_slices=None, **kw):
        if fail in ("me", "them") and (float(audio[0]) < 1.5) == (fail == "me"):
            calls.append({"audio": audio.copy(), "slices": slices, "failed": True})
            return None  # decode_meeting fails open with None
        calls.append(
            {
                "audio": audio.copy(),
                "slices": slices,
                "vad_slices": vad_slices,
                "initial_prompt": initial_prompt,
                "language": language,
                "display": kw.get("display"),
            }
        )
        start = slices[0][0] / SR if slices else 0.0
        # The stems differ in level: 1.0 is the mic, 2.0 the system track.
        word = "moi" if float(audio[0]) < 1.5 else "hi"
        return {
            "text": word,
            "words": [{"word": word, "start": start, "end": start + 0.3}],
            "segments": [{"start": start, "end": start + 0.5, "text": word}],
            "slices": [
                {"index": 0, "start_s": start, "end_s": start + 0.5, "text": word}
            ],
            "language": "fi" if word == "moi" else "en",
        }

    return fake


def _patch_stems(
    monkeypatch: pytest.MonkeyPatch, mic: np.ndarray, them: np.ndarray
) -> None:
    monkeypatch.setattr(
        track_decode, "ingest_stems", lambda tracks, mix_len, **kw: (mic, them)
    )


def test_decode_tracks_decodes_each_stem_with_the_right_audio(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mic, them = _stems(1.0, 2.0)
    _patch_stems(monkeypatch, mic, them)
    _patch_stem_vad(monkeypatch)
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(track_decode, "decode_meeting", _fake_decode_meeting(calls))

    result = decode_tracks(
        TrackPaths(me=OWNER, them=THEM),
        SR * 2,
        initial_prompt="glossary",
        language=None,
    )

    assert result is not None
    assert len(calls) == 2
    assert all(c["initial_prompt"] == "glossary" for c in calls)
    # The mic stem arrives masked: owner speech (plus the mask padding)
    # kept, everything beyond it silent.
    mic_call = calls[0]
    assert mic_call["audio"][:SR] == pytest.approx(1.0)
    assert mic_call["audio"][SR : round(1.2 * SR)] == pytest.approx(1.0)
    assert not mic_call["audio"][round(1.3 * SR) :].any()
    # The them stem is decoded unchanged.
    them_call = calls[1]
    assert them_call["audio"] == pytest.approx(them)
    # Slices and vad_slices cover only the stem's own speech span.
    assert mic_call["slices"][0][0] == 0
    assert them_call["slices"][0][0] == SR
    assert mic_call["vad_slices"] == [(0, SR)]
    assert them_call["vad_slices"] == [(SR, SR * 2)]
    # Both sides came out on one timeline, each with its own language.
    assert [w["word"] for w in result["words"]] == ["moi", "hi"]


def test_decode_tracks_forwards_the_display_to_every_stem(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_stems(monkeypatch, *_stems(1.0, 2.0))
    _patch_stem_vad(monkeypatch)
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(track_decode, "decode_meeting", _fake_decode_meeting(calls))
    display = ProgressDisplay(verbose=False)

    decode_tracks(
        TrackPaths(me=OWNER, them=THEM),
        SR * 2,
        initial_prompt=None,
        language=None,
        display=display,
    )

    assert len(calls) == 2
    assert all(c["display"] is display for c in calls)


def test_decode_tracks_skips_a_stem_with_no_speech(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mic, them = _stems(1.0, 0.0)  # them is silent: VAD finds nothing
    _patch_stems(monkeypatch, mic, them)
    _patch_stem_vad(monkeypatch)
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(track_decode, "decode_meeting", _fake_decode_meeting(calls))

    result = decode_tracks(
        TrackPaths(me=OWNER, them=THEM), SR * 2, initial_prompt=None, language="fi"
    )

    assert result is not None
    assert len(calls) == 1
    assert calls[0]["language"] == "fi"
    assert result["language"] == "fi"
    # The skipped stem is legitimate silence, but the survivor is still tagged.
    assert all(w["track"] == "me" for w in result["words"])


def test_decode_tracks_no_speech_anywhere_returns_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_stems(monkeypatch, *_stems(0.0, 0.0))
    _patch_stem_vad(monkeypatch)
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(track_decode, "decode_meeting", _fake_decode_meeting(calls))
    assert (
        decode_tracks(
            TrackPaths(me=OWNER, them=THEM), SR * 2, initial_prompt=None, language=None
        )
        is None
    )
    assert calls == []


@pytest.mark.parametrize("failed_side", ["me", "them"])
def test_decode_tracks_stem_decode_failure_falls_back_to_mix(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    failed_side: str,
) -> None:
    """A stem with speech whose decode_meeting returned None is a failure,
    not "no speech": warn and return None so the mix is decoded instead
    (otherwise the remote side would vanish silently, issue #207)."""
    _patch_stems(monkeypatch, *_stems(1.0, 2.0))
    _patch_stem_vad(monkeypatch)
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr(
        track_decode, "decode_meeting", _fake_decode_meeting(calls, fail=failed_side)
    )

    with caplog.at_level("WARNING"):
        result = decode_tracks(
            TrackPaths(me=OWNER, them=THEM),
            SR * 2,
            initial_prompt=None,
            language=None,
        )

    assert result is None, "a failed stem decode must fall back to the mix decode"
    assert any(f"{failed_side} stem" in r.message for r in caplog.records)


def test_decode_tracks_fails_open_on_any_exception(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    def boom(tracks, mix_len, **kw):
        raise RuntimeError("stem ingest exploded")

    monkeypatch.setattr(track_decode, "ingest_stems", boom)
    with caplog.at_level("WARNING"):
        assert (
            decode_tracks(
                TrackPaths(me=OWNER, them=THEM),
                SR * 2,
                initial_prompt=None,
                language=None,
            )
            is None
        )
    assert any("stem ingest exploded" in r.message for r in caplog.records)


# -- pipeline wiring -------------------------------------------------------


def _meeting_result(text: str, start: float = 0.0) -> dict[str, Any]:
    words = [{"word": text, "start": start, "end": start + 0.3}]
    return {
        "text": text,
        "words": words,
        "segments": [{"start": start, "end": start + 0.5, "text": text}],
        "slices": [{"index": 0, "start_s": start, "end_s": start + 0.5, "text": text}],
        "language": "fi",
    }


def test_meeting_with_tracks_prefers_the_per_track_decode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_preflight_pass(monkeypatch)
    _patch_ingest(monkeypatch)
    _patch_vad(monkeypatch)
    monkeypatch.setattr(
        pipeline, "run_speaker_stage", lambda *a, **kw: _empty_speaker_stage()
    )
    tracks = TrackPaths(me=OWNER, them=THEM)
    seen: dict[str, Any] = {}

    def fake_decode_tracks(tracks_arg, mix_len, **kw):
        seen["tracks"] = tracks_arg
        seen["mix_len"] = mix_len
        seen.update(kw)
        return _meeting_result("erilliset")

    def fake_decode_meeting(audio, slices, **kw):
        raise AssertionError("the mix must not be decoded when tracks work")

    monkeypatch.setattr(track_decode, "decode_tracks", fake_decode_tracks)
    monkeypatch.setattr(pipeline, "decode_meeting", fake_decode_meeting)
    monkeypatch.setattr(pipeline, "track_decode", track_decode)

    result = transcribe_file(
        "/nonexistent.m4a",
        config_path=str(tmp_path / "none.toml"),
        profile="meeting",
        tracks=tracks,
    )

    assert seen["tracks"] is tracks
    assert result["text"] == "erilliset"
    assert "error" not in result


def test_meeting_falls_back_to_the_mix_when_track_decode_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_preflight_pass(monkeypatch)
    _patch_ingest(monkeypatch)
    _patch_vad(monkeypatch)
    monkeypatch.setattr(
        pipeline, "run_speaker_stage", lambda *a, **kw: _empty_speaker_stage()
    )
    track_calls: list[tuple[Any, ...]] = []

    def _no_tracks(tracks_arg, mix_len, **kw):
        track_calls.append((tracks_arg, mix_len))
        return None

    monkeypatch.setattr(track_decode, "decode_tracks", _no_tracks)
    mix_calls: list[np.ndarray] = []

    def fake_decode_meeting(audio, slices, **kw):
        mix_calls.append(audio)
        return _meeting_result("seos")

    monkeypatch.setattr(pipeline, "decode_meeting", fake_decode_meeting)
    monkeypatch.setattr(pipeline, "track_decode", track_decode)

    result = transcribe_file(
        "/nonexistent.m4a",
        config_path=str(tmp_path / "none.toml"),
        profile="meeting",
        tracks=TrackPaths(me=OWNER, them=THEM),
    )

    assert len(mix_calls) == 1
    assert track_calls, "the per-track decode must be tried first"
    assert track_calls[0][0] == TrackPaths(me=OWNER, them=THEM)
    assert result["text"] == "seos"
    assert "error" not in result


def test_meeting_without_tracks_never_touches_track_decode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_preflight_pass(monkeypatch)
    _patch_ingest(monkeypatch)
    _patch_vad(monkeypatch)
    monkeypatch.setattr(
        pipeline, "run_speaker_stage", lambda *a, **kw: _empty_speaker_stage()
    )

    def _must_not_run(*a, **kw):
        raise AssertionError("per-track decode ran without tracks")

    monkeypatch.setattr(pipeline, "track_decode", track_decode)
    monkeypatch.setattr(track_decode, "decode_tracks", _must_not_run)
    monkeypatch.setattr(
        pipeline, "decode_meeting", lambda audio, slices, **kw: _meeting_result("seos")
    )

    result = transcribe_file(
        "/nonexistent.m4a",
        config_path=str(tmp_path / "none.toml"),
        profile="meeting",
    )
    assert result["text"] == "seos"


def test_assemble_keeps_overlapping_tracks_apart() -> None:
    """Real-call regression: a remote turn overlapping the owner's segment
    must not pull English words into the Finnish sentence nor steal its
    speaker label (issue #207 word-attribution bug)."""
    result_a = merge_track_results(
        {
            "text": "tämä on suomeksi",
            "words": [
                {"word": "tämä", "start": 0.0, "end": 0.4},
                {"word": "on", "start": 0.5, "end": 0.7},
                {"word": "suomeksi", "start": 0.8, "end": 1.3},
            ],
            "segments": [{"start": 0.0, "end": 1.3, "text": "tämä on suomeksi"}],
            "slices": [],
            "language": "fi",
        },
        {
            "text": "if you're happy",
            "words": [
                {"word": "if", "start": 0.3, "end": 0.5},
                {"word": "you're", "start": 0.6, "end": 0.9},
                {"word": "happy", "start": 1.0, "end": 1.4},
            ],
            "segments": [{"start": 0.3, "end": 1.4, "text": "if you're happy"}],
            "slices": [],
            "language": "en",
        },
    )
    assert result_a is not None
    # The remote diarization turn swallows the whole owner segment.
    turns = [(0.0, 1.5, "SPK_1"), (0.0, 4.0, "SPK_2")]
    run = pipeline._assemble(
        result_a,
        None,
        [],
        None,
        turns,
        spans=[],
        trusted_labels=frozenset({"SPK_1"}),
    )
    owners = [s for s in run["segments"] if s.get("speaker") == "SPK_1"]
    remote = [s for s in run["segments"] if s.get("speaker") == "SPK_2"]
    assert len(owners) == 1
    assert owners[0]["text"] == "tämä on suomeksi"
    assert len(remote) == 1
    assert remote[0]["text"] == "if you're happy"


def _empty_speaker_stage() -> Any:
    from vemoizer.speaker_stage import SpeakerStage

    return SpeakerStage(turns=None, ran=False, warnings=[])
