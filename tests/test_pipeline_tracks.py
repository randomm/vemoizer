"""Pipeline wiring for two-track meetings (issue #171).

The decodes, VAD and ingest are faked (no models, no ffmpeg). The mix is
decoded as usual; the owner's words must come out labelled from the mic
stem, asserted on paragraphs because ``_assemble`` re-labels per segment.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from _cli_helpers import isolate_home
from test_names_cli import _sidecar, _write_sidecar

import vemoizer.diarization as diarization_mod
import vemoizer.pipeline as pipeline
import vemoizer.preflight as preflight
import vemoizer.speaker_stage as speaker_stage
from vemoizer.names_cli import run_names
from vemoizer.pipeline import _assemble, transcribe_file
from vemoizer.tracks import REMOTE_LABEL, TrackPaths
from vemoizer.vad import SpeechSegment

SR = 16_000
WORDS = [
    {"word": "moi", "start": 0.0, "end": 0.5},
    {"word": "hei", "start": 0.5, "end": 1.0},
    {"word": "terve", "start": 2.5, "end": 3.0},
    {"word": "sinulle", "start": 3.0, "end": 3.5},
]
SEGMENTS = [
    {"start": 0.0, "end": 3.5, "text": "moi hei terve sinulle", "avg_logprob": -0.1}
]
TEXT = "moi hei terve sinulle"


@pytest.fixture(autouse=True)
def _preflight_ok(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(preflight, "ffmpeg_ok", lambda: True)
    monkeypatch.setattr(preflight, "config_parse_ok", lambda: True)
    monkeypatch.setattr(preflight, "models_cached", lambda: [])
    monkeypatch.setattr(preflight, "hf_token_present", lambda: True)


def _decoder(text: str, words: list[dict], segments: list[dict]):
    class _T:
        def __init__(self) -> None:
            self.model = "fake"

        def _ensure_loaded(self) -> None:
            pass

        def transcribe(self, audio, **kw) -> dict:
            return {"text": text, "words": words, "segments": segments}

        def cleanup(self) -> None:
            self.model = None

    return _T


def _patch_decodes(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        pipeline, "ParakeetTranscriber", _decoder(TEXT, WORDS, SEGMENTS)
    )
    monkeypatch.setattr(pipeline, "CanaryTranscriber", _decoder(TEXT, [], []))
    monkeypatch.setattr(pipeline, "load_vad_model", lambda: None)
    monkeypatch.setattr(
        pipeline,
        "vad_segments",
        lambda audio, model, **kw: [SpeechSegment(0, len(audio))],
    )
    monkeypatch.setattr(diarization_mod, "diarize", _no_pyannote)


def _no_pyannote(audio, **kw):
    raise AssertionError("pyannote must not run for a 1:1 call")


def _stems(monkeypatch: pytest.MonkeyPatch, ingest_calls: list[str]) -> None:
    """Mix, mic and remote stems share one length; mic and remote differ."""

    def ingest(path, *, preprocess=None):
        if Path(path).name.startswith("call-"):
            ingest_calls.append(Path(path).name)
        return np.full(
            SR * 4, 1.0 if "me" in Path(path).name else 2.0, dtype=np.float32
        )

    monkeypatch.setattr(pipeline, "ingest_audio", ingest)
    monkeypatch.setattr(speaker_stage, "ingest_audio", ingest)
    monkeypatch.setattr(speaker_stage, "load_vad_model", lambda: None)

    def stem_vad(audio, model, **kw):
        if float(audio[0]) == 1.0:
            return [SpeechSegment(start=0, end=SR)]  # owner: 0-1 s
        return [SpeechSegment(start=SR * 2, end=SR * 3)]  # remote: 2-3 s

    monkeypatch.setattr(speaker_stage, "vad_segments", stem_vad)


def _speaker_per_paragraph(result: dict) -> list[tuple[str, str]]:
    return [(p["speaker"], p["text"]) for p in result["paragraphs"]]


def test_owner_words_are_labelled_from_the_mic_stem(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_decodes(monkeypatch)
    ingested: list[str] = []
    _stems(monkeypatch, ingested)
    tracks = TrackPaths(me=Path("call-me.wav"), them=Path("call-them.wav"))

    result = transcribe_file(
        "/nonexistent.m4a",
        config_path=str(tmp_path / "none.toml"),
        diarize=True,
        speakers=2,
        tracks=tracks,
        owner="Janni",
    )

    labelled = _speaker_per_paragraph(result)
    assert labelled[0][0] == "Janni"
    assert labelled[-1][0] == REMOTE_LABEL
    assert "Janni" not in {s for s, t in labelled if "terve" in t}
    assert ingested.count("call-me.wav") == 1
    assert ingested.count("call-them.wav") == 1


def test_untracked_run_ingests_the_mix_once_and_no_stems(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_decodes(monkeypatch)
    ingested: list[str] = []
    _stems(monkeypatch, ingested)
    mix_calls: list[str] = []
    real_ingest = pipeline.ingest_audio

    def counting(path, **kw):
        mix_calls.append(Path(path).name)
        return real_ingest(path, **kw)

    monkeypatch.setattr(pipeline, "ingest_audio", counting)

    result = transcribe_file(
        "/nonexistent.m4a",
        config_path=str(tmp_path / "none.toml"),
        diarize=False,
    )

    assert mix_calls == ["nonexistent.m4a"]
    assert ingested == []
    assert result["text"] == TEXT


def test_assemble_passes_trusted_labels_to_word_attribution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: dict[str, object] = {}
    real_assign = pipeline.assign_track_labels

    def spy(words, turns, **kwargs):
        seen.update(kwargs)
        return real_assign(words, turns, **kwargs)

    monkeypatch.setattr(pipeline, "assign_track_labels", spy)
    result_a = {"text": TEXT, "words": WORDS, "segments": SEGMENTS}
    turns = [(0.0, 1.0, "Janni"), (2.5, 3.5, REMOTE_LABEL)]

    _assemble(
        result_a,
        None,
        None,
        None,
        turns,
        spans=[],
        trusted_labels=frozenset({"Janni"}),
    )

    assert seen == {"trusted": frozenset({"Janni"})}


def test_assemble_drops_them_segments_inside_echo_spans() -> None:
    """Issue #238: the them stem re-transcribes the owner; those segments
    drop before labelling, me-track and out-of-span segments stay."""
    segments = [
        {"start": 0.0, "end": 1.0, "text": "moi", "track": "them"},
        {"start": 0.0, "end": 1.0, "text": "moi", "track": "me"},
        {"start": 2.5, "end": 3.5, "text": "terve", "track": "them"},
    ]
    result_a = {"text": TEXT, "words": [], "segments": segments}
    turns = [(0.0, 1.0, "Janni"), (2.5, 3.5, REMOTE_LABEL)]

    result = _assemble(
        result_a,
        None,
        None,
        None,
        turns,
        spans=[],
        trusted_labels=frozenset({"Janni"}),
        echo_spans=[(0.0, 1.0)],
    )

    assert [(s["track"], s["speaker"]) for s in result["segments"]] == [
        ("me", "Janni"),
        ("them", REMOTE_LABEL),
    ]
    # The text loses the echo too, or the render/notes fallbacks that read
    # it would bring the dropped words back (PR #240 review).
    assert result["text"] == "moi terve"


def test_assemble_sidecar_words_follow_only_the_kept_segments() -> None:
    """Issue #263: the result's ``words`` list is the kept segments' words.

    The sidecar word timings come from decode A's words but are filtered
    against the FINAL segments, so a word of an echo-dropped segment and a
    word falling outside every segment are both absent, and only the
    sidecar keys (``word``/``start``/``end``/``p``, plus ``speaker`` when
    the word already had one) are mirrored.
    """
    segments = [
        {"start": 0.0, "end": 1.0, "text": "moi", "track": "them"},
        {"start": 2.5, "end": 3.5, "text": "terve", "track": "them"},
        {"start": 6.0, "end": 7.0, "text": "sinulle", "track": "me"},
    ]
    words = [
        {"word": "moi", "start": 0.0, "end": 0.5, "p": 0.42, "probability": 0.42},
        {"word": "hei", "start": 0.5, "end": 1.0, "track": "them"},
        {"word": "terve", "start": 2.5, "end": 3.0, "p": 0.99, "speaker": REMOTE_LABEL},
        {"word": "kaiku", "start": 4.0, "end": 4.5, "p": 0.1},
        {"word": "sinulle", "start": 6.5, "end": 7.0, "p": 0.7},
    ]
    result_a = {"text": TEXT, "words": words, "segments": segments}
    turns = [(0.0, 1.0, "Janni"), (2.5, 3.5, REMOTE_LABEL)]

    result = _assemble(
        result_a,
        None,
        None,
        None,
        turns,
        spans=[],
        trusted_labels=frozenset({"Janni"}),
        echo_spans=[(0.0, 1.0), (4.0, 4.5)],
    )

    assert result["words"] == [
        {"word": "terve", "start": 2.5, "end": 3.0, "p": 0.99, "speaker": REMOTE_LABEL},
        {"word": "sinulle", "start": 6.5, "end": 7.0, "p": 0.7},
    ]
    # The words key is a sidecar addition: the transcript itself is unchanged
    # (the echo text was already dropped by issue #238, not by this change).
    assert result["text"] == "terve sinulle"


def test_assemble_without_segments_has_no_sidecar_words() -> None:
    """No kept segments means no words, so the sidecar omits the key."""
    words = [{"word": "moi", "start": 0.0, "end": 0.5, "p": 0.5}]
    turns = [(0.0, 1.0, "Janni")]
    result = _assemble(
        {"text": "moi", "words": words, "segments": []},
        None,
        None,
        None,
        turns,
        spans=[],
        trusted_labels=frozenset({"Janni"}),
    )

    assert result["words"] == []


def test_two_track_run_seeds_no_speaker_names(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Decision (issue #171): no ``{owner: owner}`` seeding in ``speaker_names``."""
    _patch_decodes(monkeypatch)
    _stems(monkeypatch, [])
    tracks = TrackPaths(me=Path("call-me.wav"), them=Path("call-them.wav"))

    result = transcribe_file(
        "/nonexistent.m4a",
        config_path=str(tmp_path / "none.toml"),
        tracks=tracks,
        owner="Janni",
    )

    assert "speaker_names" not in result


def test_owner_label_is_still_prompted_by_names(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without seeding, ``names`` still asks for the owner's label like any other."""
    isolate_home(monkeypatch, tmp_path, tmp_path)
    data = _sidecar()
    data["paragraphs"][0]["speaker"] = "Janni"
    data["paragraphs"][1]["speaker"] = REMOTE_LABEL
    sidecar = _write_sidecar(tmp_path, data)
    prompts: list[str] = []

    def record(prompt: str) -> str:
        prompts.append(prompt)
        return ""

    rc = run_names(sidecar, no_play=True, input_fn=record, tty_isatty=lambda: True)

    assert rc == 0
    assert any("Name for Janni" in p for p in prompts)
