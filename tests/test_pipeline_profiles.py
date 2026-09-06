"""Pipeline wiring for the LLM notes, repair, and profile stages.

Split from ``test_pipeline.py`` (800-line test-file limit): these cover
the orchestrator's stage wiring for notes (issue #57), the meeting
profile (issue #71), and the repair pass (issue #68). All stages are
mocked — no models, no network, no ffmpeg.
"""

from __future__ import annotations

import pytest
from test_pipeline import (  # noqa: F401 - shared orchestrator fixtures
    _consensus_setup,
    _llm_config,
    _patch_decoders,
    _patch_diarize,
    _patch_ingest,
    _patch_redecode,
    _patch_vad,
)

import vemoizer.pipeline as pipeline
from vemoizer.pipeline import transcribe_file

# -- notes stage wiring (issue #57) --------------------------------------


def test_notes_failure_lands_in_warnings_not_errors(tmp_path, monkeypatch) -> None:
    """A failed notes stage warns and ships the transcript untouched."""
    _consensus_setup(monkeypatch)
    _patch_redecode(monkeypatch, "moikka")
    monkeypatch.setattr(
        pipeline,
        "generate_notes",
        lambda client, text, paragraphs=None, glossary=None: None,
    )

    class _Client:
        def adjudicate(self, a_text, candidates, context=""):
            return "moikka"

        def close(self) -> None:
            pass

    monkeypatch.setattr(pipeline, "LLMClient", lambda cfg: _Client())
    cfg = _llm_config(tmp_path)
    result = transcribe_file("/nonexistent.m4a", config_path=str(cfg))

    assert result["text"]  # transcript unaffected
    assert "notes" not in result
    assert any("notes" in w for w in result.get("warnings", []))


def test_notes_attach_when_generated(tmp_path, monkeypatch) -> None:
    _consensus_setup(monkeypatch)
    _patch_redecode(monkeypatch, "moikka")
    fake_notes = {"title": "T", "summary": "S", "key_points": [], "action_items": []}
    monkeypatch.setattr(
        pipeline,
        "generate_notes",
        lambda client, text, paragraphs=None, glossary=None: fake_notes,
    )

    class _Client:
        def adjudicate(self, a_text, candidates, context=""):
            return "moikka"

        def close(self) -> None:
            pass

    monkeypatch.setattr(pipeline, "LLMClient", lambda cfg: _Client())
    cfg = _llm_config(tmp_path)
    result = transcribe_file("/nonexistent.m4a", config_path=str(cfg))
    assert result["notes"] == fake_notes


def test_no_llm_config_skips_notes_silently(tmp_path, monkeypatch) -> None:
    _consensus_setup(monkeypatch)
    _patch_redecode(monkeypatch, "moikka")

    def _must_not_run(client, text, paragraphs=None, glossary=None):
        raise AssertionError("notes stage ran without an LLM config")

    monkeypatch.setattr(pipeline, "generate_notes", _must_not_run)
    result = transcribe_file(
        "/nonexistent.m4a", config_path=str(tmp_path / "none.toml")
    )
    assert "notes" not in result
    assert "warnings" not in result


def test_diarization_run_appends_cc_by_attribution(tmp_path, monkeypatch) -> None:
    """CC-BY-4.0 requires attribution whenever the gated weights ran."""
    _consensus_setup(monkeypatch)
    _patch_redecode(monkeypatch, "moikka")
    _patch_diarize(monkeypatch, segments=[(0.0, 2.0, "SPEAKER_00")])
    result = transcribe_file(
        "/nonexistent.m4a", config_path=str(tmp_path / "none.toml"), diarize=True
    )
    from vemoizer.diarization import ATTRIBUTION

    assert ATTRIBUTION in result.get("warnings", [])


def test_no_attribution_without_diarization(tmp_path, monkeypatch) -> None:
    _consensus_setup(monkeypatch)
    _patch_redecode(monkeypatch, "moikka")
    result = transcribe_file(
        "/nonexistent.m4a", config_path=str(tmp_path / "none.toml")
    )
    from vemoizer.diarization import ATTRIBUTION

    assert ATTRIBUTION not in result.get("warnings", [])


# -- meeting profile (issue #71) -----------------------------------------


def _patch_whisper_a(monkeypatch, text="hei maailma"):
    from vemoizer.whisper_transcriber import slice_records_from_words

    words = [
        {"word": "hei", "start": 0.0, "end": 0.4},
        {"word": "maailma", "start": 0.5, "end": 1.0},
    ]

    def fake_decode_meeting(audio, slices, initial_prompt=None):
        return {
            "text": text,
            "words": words,
            "segments": [{"start": 0.0, "end": 1.0, "text": text}],
            "language": "fi",
            "slices": slice_records_from_words(words, slices, language="fi"),
        }

    monkeypatch.setattr(pipeline, "decode_meeting", fake_decode_meeting)


def test_meeting_profile_is_whisper_only_no_consensus(tmp_path, monkeypatch) -> None:
    """Measured (issue #71): consensus rewriting on top of the whole-file
    Whisper read injects noise. The meeting profile runs NO other decoder."""
    _patch_ingest(monkeypatch)
    _patch_vad(monkeypatch)
    _patch_whisper_a(monkeypatch)

    def _must_not_construct(*a, **kw):
        raise AssertionError("a consensus decoder ran under the meeting profile")

    monkeypatch.setattr(pipeline, "ParakeetTranscriber", _must_not_construct)
    monkeypatch.setattr(pipeline, "CanaryTranscriber", _must_not_construct)
    monkeypatch.setattr(pipeline, "WhisperReDecodeTranscriber", _must_not_construct)
    result = transcribe_file(
        "/nonexistent.m4a",
        config_path=str(tmp_path / "none.toml"),
        profile="meeting",
    )
    assert result["text"] == "hei maailma"  # whisper's read, unrewritten
    assert result["segments"][0]["text"] == "hei maailma"


def test_meeting_profile_whisper_failure_fails_open_to_empty(
    tmp_path, monkeypatch
) -> None:
    _patch_ingest(monkeypatch)
    _patch_vad(monkeypatch)

    # decode_meeting fails open to None internally; simulate that outcome.
    monkeypatch.setattr(
        pipeline, "decode_meeting", lambda audio, slices, initial_prompt=None: None
    )
    _patch_decoders(
        monkeypatch,
        {"text": "parakeet varalla", "words": []},
        {"text": "parakeet varalla", "words": []},
    )
    result = transcribe_file(
        "/nonexistent.m4a",
        config_path=str(tmp_path / "none.toml"),
        profile="meeting",
    )
    # whisper failed: the run falls open INTO the dictation pipeline
    assert result["text"] == "parakeet varalla"


def test_unknown_profile_raises() -> None:
    with pytest.raises(ValueError, match="unknown profile"):
        transcribe_file("/nonexistent.m4a", profile="podcast")


def test_dictation_profile_never_touches_whisper(tmp_path, monkeypatch) -> None:
    _patch_ingest(monkeypatch)
    _patch_vad(monkeypatch)

    def _must_not_run(audio, slices, initial_prompt=None):
        raise AssertionError("decode_meeting ran under dictation profile")

    monkeypatch.setattr(pipeline, "decode_meeting", _must_not_run)
    _patch_decoders(
        monkeypatch,
        {"text": "vain parakeet", "words": []},
        {"text": "vain parakeet", "words": []},
    )
    result = transcribe_file(
        "/nonexistent.m4a", config_path=str(tmp_path / "none.toml")
    )
    assert result["text"] == "vain parakeet"


def test_repair_pass_updates_paragraphs_only(tmp_path, monkeypatch) -> None:
    _consensus_setup(monkeypatch)
    _patch_redecode(monkeypatch, "moikka")

    class _Client:
        def adjudicate(self, a_text, candidates, context=""):
            return "moikka"

        def complete(self, system, user, max_tokens=2048):
            return user.replace("moikka", "moikka!")  # a visible "repair"

        def close(self) -> None:
            pass

    monkeypatch.setattr(pipeline, "LLMClient", lambda cfg: _Client())
    monkeypatch.setattr(
        pipeline,
        "generate_notes",
        lambda client, text, paragraphs=None, glossary=None: None,
    )
    cfg = _llm_config(tmp_path)
    result = transcribe_file("/nonexistent.m4a", config_path=str(cfg), repair=True)
    assert result["paragraphs"][0]["text"] == "moikka!"
    # segments stay the verbatim record
    assert result["segments"][0]["text"] == "moikka"


def test_repair_off_by_default(tmp_path, monkeypatch) -> None:
    _consensus_setup(monkeypatch)
    _patch_redecode(monkeypatch, "moikka")
    monkeypatch.setattr(
        pipeline,
        "repair_paragraphs",
        lambda client, paras: (_ for _ in ()).throw(AssertionError("repair ran")),
    )
    result = transcribe_file(
        "/nonexistent.m4a", config_path=str(tmp_path / "none.toml")
    )
    assert result["text"]


def test_speakers_attach_to_all_segments_not_just_verdicts(
    tmp_path, monkeypatch
) -> None:
    """Regression (issue #71 QA): in the whisper-only meeting path there are
    zero verdicts, so speaker labels attached only to verdicts meant
    diarization ran and was then thrown away."""
    _patch_ingest(monkeypatch)
    _patch_vad(monkeypatch)
    _patch_whisper_a(monkeypatch)
    _patch_diarize(monkeypatch, segments=[(0.0, 2.0, "SPEAKER_00")])
    result = transcribe_file(
        "/nonexistent.m4a",
        config_path=str(tmp_path / "none.toml"),
        profile="meeting",
        diarize=True,
    )
    assert result["segments"][0]["speaker"] == "SPEAKER_00"
    assert result["paragraphs"], "paragraphs must exist without any disputes"
    assert result["paragraphs"][0]["speaker"] == "SPEAKER_00"


def test_paragraphs_exist_even_without_disputes(tmp_path, monkeypatch) -> None:
    _patch_ingest(monkeypatch)
    _patch_vad(monkeypatch)
    _patch_whisper_a(monkeypatch)
    result = transcribe_file(
        "/nonexistent.m4a",
        config_path=str(tmp_path / "none.toml"),
        profile="meeting",
    )
    assert result["paragraphs"]
    assert result["paragraphs"][0]["text"] == "hei maailma"


def test_glossary_reaches_whisper_and_notes(tmp_path, monkeypatch) -> None:
    """The glossary must reach the recognizer (initial_prompt) and the
    notes stage (canonical spellings) — issue #71 QA."""
    _patch_ingest(monkeypatch)
    _patch_vad(monkeypatch)
    gl = tmp_path / "glossary.txt"
    gl.write_text("Flagship-hanke\nRiihimäki\n", encoding="utf-8")
    seen: dict = {}

    def fake_decode_meeting(audio, slices, initial_prompt=None):
        seen["initial_prompt"] = initial_prompt
        return {
            "text": "hei maailma",
            "words": [{"word": "hei", "start": 0.0, "end": 0.4}],
            "segments": [{"start": 0.0, "end": 1.0, "text": "hei maailma"}],
            "slices": [],
        }

    monkeypatch.setattr(pipeline, "decode_meeting", fake_decode_meeting)

    def fake_notes(client, text, paragraphs=None, glossary=None):
        seen["notes_glossary"] = glossary
        seen["notes_paragraphs"] = paragraphs
        return None

    monkeypatch.setattr(pipeline, "generate_notes", fake_notes)

    class _Client:
        def close(self) -> None:
            pass

    monkeypatch.setattr(pipeline, "LLMClient", lambda cfg: _Client())
    cfg = _llm_config(tmp_path)
    transcribe_file(
        "/nonexistent.m4a",
        config_path=str(cfg),
        profile="meeting",
        glossary_path=str(gl),
    )
    assert "Flagship-hanke" in (seen["initial_prompt"] or "")
    assert seen["notes_glossary"] == ["Flagship-hanke", "Riihimäki"]
    # notes get the speaker-labelled paragraphs, not just raw text
    assert seen["notes_paragraphs"] is not None


def test_speakers_hint_reaches_diarization(tmp_path, monkeypatch) -> None:
    """--speakers pins pyannote clustering (5 labels for 4 people was a
    real forensics finding)."""
    _patch_ingest(monkeypatch)
    _patch_vad(monkeypatch)
    _patch_whisper_a(monkeypatch)
    seen = {}

    def fake_diarize(audio, num_speakers=None, **kw):
        from vemoizer.diarization import DiarizationResult

        seen["num_speakers"] = num_speakers
        return DiarizationResult(segments=[(0.0, 2.0, "SPEAKER_00")])

    monkeypatch.setattr(pipeline, "diarize", fake_diarize)
    transcribe_file(
        "/nonexistent.m4a",
        config_path=str(tmp_path / "none.toml"),
        profile="meeting",
        diarize=True,
        speakers=4,
    )
    assert seen["num_speakers"] == 4


def test_paragraph_hygiene_runs_in_the_pipeline(tmp_path, monkeypatch) -> None:
    """A recognizer repetition loop must not survive into the output."""
    _patch_ingest(monkeypatch)
    _patch_vad(monkeypatch)

    def fake_decode_meeting(audio, slices, initial_prompt=None):
        loop = "Janni, " * 10 + "aloitetaan"
        return {
            "text": loop,
            "words": [{"word": "Janni,", "start": 0.0, "end": 0.1}],
            "segments": [{"start": 0.0, "end": 2.0, "text": loop}],
            "slices": [],
        }

    monkeypatch.setattr(pipeline, "decode_meeting", fake_decode_meeting)
    result = transcribe_file(
        "/nonexistent.m4a",
        config_path=str(tmp_path / "none.toml"),
        profile="meeting",
    )
    assert result["paragraphs"][0]["text"].count("Janni") == 1


def test_glossary_corrections_apply_deterministically(tmp_path, monkeypatch) -> None:
    _patch_ingest(monkeypatch)
    _patch_vad(monkeypatch)
    gl = tmp_path / "glossary.txt"
    gl.write_text("Blacksit => Flagship\n", encoding="utf-8")

    def fake_decode_meeting(audio, slices, initial_prompt=None):
        return {
            "text": "he kutsuvat Blacksit-hankkeiksi",
            "words": [],
            "segments": [
                {"start": 0.0, "end": 2.0, "text": "he kutsuvat Blacksit-hankkeiksi"}
            ],
            "slices": [],
        }

    monkeypatch.setattr(pipeline, "decode_meeting", fake_decode_meeting)
    result = transcribe_file(
        "/nonexistent.m4a",
        config_path=str(tmp_path / "none.toml"),
        profile="meeting",
        glossary_path=str(gl),
    )
    assert result["paragraphs"][0]["text"] == "he kutsuvat Flagship-hankkeiksi"
