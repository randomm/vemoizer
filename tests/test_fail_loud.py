"""Fail-loud pipeline tests (issue #78, extends #73).

A total decode failure must not look like a successful empty transcript:
``decode_all`` returns ``None`` when a non-empty slice list yields no
successful slice, and ``transcribe_file`` then reports ``result["error"]``
instead of an empty ``{"text": "", "segments": []}`` that the CLI would
have happily written to disk and printed a success line for.

All stages are mocked — no models, no network, no ffmpeg (``np.zeros``
audio via the ``_audio`` helper). Every test passes an explicit temp
``config_path`` so the developer's live LLM config is never loaded.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from test_pipeline import (
    _consensus_setup,
    _llm_config,
    _patch_decoders,
    _patch_diarize,
    _patch_ingest,
    _patch_preflight_pass,
    _patch_redecode,
    _patch_vad,
)

import vemoizer.eval_cli as eval_cli
import vemoizer.llm_tail as llm_tail
import vemoizer.pipeline as pipeline
from vemoizer.decode_stage import decode_all
from vemoizer.pipeline import transcribe_file


def _audio(seconds: float = 2.0) -> np.ndarray:
    return np.zeros(int(seconds * 16_000), dtype=np.float32)


def _n_slice_vad(monkeypatch, n: int) -> None:
    """VAD variant of ``_patch_vad`` returning *n* equal slices."""

    class _Model:
        def reset_states(self) -> None:
            pass

        def __call__(self, window: np.ndarray, sample_rate: int) -> np.ndarray:
            return np.zeros(len(window), dtype=np.float32)

    def _segments(audio: np.ndarray, model, **kw):
        size = len(audio) // n
        return [
            pipeline.SpeechSegment(i * size, min((i + 1) * size, len(audio)))
            for i in range(n)
        ]

    monkeypatch.setattr(pipeline, "load_vad_model", lambda: _Model())
    monkeypatch.setattr(pipeline, "vad_segments", _segments)


def _no_config(tmp_path: Path) -> str:
    """Explicit nonexistent config path: no [llm], no live-config pickup."""
    return str(tmp_path / "none.toml")


# -- decode stage contract (#73) -----------------------------------------


def test_decode_all_returns_none_on_total_failure(monkeypatch) -> None:
    """A non-empty slice list where every slice raises returns ``None`` —
    the total-failure contract the pipeline's ``error`` check relies on."""

    class _Latch:
        def transcribe(self, audio, **kw):
            raise RuntimeError("Parakeet model failed to load")

    slices = [(0, _audio(1.0)), (16_000, _audio(1.0))]
    assert decode_all(_Latch(), slices, "decode A") is None


def test_decode_all_partial_failure_still_merges() -> None:
    """One failed slice among successes: the good slices still merge and the
    result is a dict (the partial-failure path is not a total failure)."""

    class _Partial:
        def __init__(self) -> None:
            self._calls = 0

        def transcribe(self, audio, **kw):
            self._calls += 1
            if self._calls == 1:
                raise RuntimeError("slice 1 failed")
            return {"text": "toinen onnistui", "words": [], "segments": []}

    slices = [(0, _audio(1.0)), (16_000, _audio(1.0))]
    result = decode_all(_Partial(), slices, "decode A")
    assert result is not None
    assert result["text"] == "toinen onnistui"
    assert len(result["slices"]) == 1


def test_decode_all_empty_input_keeps_four_key_contract() -> None:
    """An empty slice list is NOT a failure: the 4-key empty dict is
    preserved so callers can read the fields unconditionally."""
    result = decode_all(object(), [], "decode A")
    assert result is not None
    assert result == {"text": "", "words": [], "segments": [], "slices": []}


def test_eval_decode_only_still_green_on_total_failure(monkeypatch) -> None:
    """``eval_cli.transcribe_decode_only`` already tolerates the ``None``
    contract from #73 and must stay green when decode_all starts returning
    it (no eval change, no eval error key)."""
    _patch_ingest(monkeypatch)
    _patch_vad(monkeypatch)
    _patch_decoders(monkeypatch, None, None, latched_a=True)
    result = eval_cli.transcribe_decode_only("/nonexistent.m4a", backend="parakeet")
    assert result["text"] == ""
    assert "error" not in result


# -- pipeline fail-loud: total decode failure -----------------------------


def _latch_both(monkeypatch) -> None:
    """Patch decode A to the latched shape (ctor ok, transcribe raises) and
    decode B to a healthy text (consensus runs but A's total failure wins)."""
    _patch_decoders(monkeypatch, {"text": "varalla"}, {"text": "varalla"})

    class _LatchA:
        def __init__(self) -> None:
            pass

        def transcribe(self, audio, **kw):
            raise RuntimeError("Parakeet model failed to load")

        def cleanup(self) -> None:
            pass

    monkeypatch.setattr(pipeline, "ParakeetTranscriber", _LatchA)


def test_latched_decode_total_failure_fails_loud(tmp_path, monkeypatch) -> None:
    """Latched-load total failure (ctor succeeds, transcribe raises) must
    produce ``result["error"]`` with the exact #73 wording — not a truthy
    empty dict that looks like a successful empty transcript."""
    _patch_preflight_pass(monkeypatch)
    _patch_ingest(monkeypatch)
    _patch_vad(monkeypatch)
    _latch_both(monkeypatch)
    result = transcribe_file("/nonexistent.m4a", config_path=_no_config(tmp_path))
    assert "error" in result
    assert result["error"] == (
        "decode A produced no output for any of 1 slices "
        "(model may have failed to load)"
    )
    assert result["text"] == ""
    assert result["segments"] == []


def test_latched_decode_two_slices_counts_slices_in_error(
    tmp_path, monkeypatch
) -> None:
    """The slice count in the error message reflects the VAD slice count."""
    _patch_preflight_pass(monkeypatch)
    _patch_ingest(monkeypatch)
    _n_slice_vad(monkeypatch, 2)
    _latch_both(monkeypatch)
    result = transcribe_file("/nonexistent.m4a", config_path=_no_config(tmp_path))
    assert "error" in result
    assert "any of 2 slices" in result["error"]


def test_silent_audio_still_ships_empty_without_error(tmp_path, monkeypatch) -> None:
    """Positive control: a decode that SUCCEEDS with empty text (silent
    speech) is not a total failure — no error key, empty output."""
    _patch_preflight_pass(monkeypatch)
    _patch_ingest(monkeypatch)
    _patch_vad(monkeypatch)
    _patch_decoders(monkeypatch, {"text": ""}, None)
    result = transcribe_file("/nonexistent.m4a", config_path=_no_config(tmp_path))
    assert "error" not in result
    assert result["text"] == ""
    assert result["segments"] == []


def test_latched_decode_b_failure_still_fails_open(tmp_path, monkeypatch) -> None:
    """Only decode B latched: consensus degrades to decode A (fail-open,
    not fail-loud) — the #73 error is exclusively decode A's total failure."""
    _consensus_setup(monkeypatch)
    _patch_redecode(monkeypatch, "moikka")

    class _LatchB:
        def __init__(self) -> None:
            pass

        def transcribe(self, audio, **kw):
            raise RuntimeError("canary model failed to load")

        def cleanup(self) -> None:
            pass

    monkeypatch.setattr(pipeline, "CanaryTranscriber", _LatchB)
    result = transcribe_file("/nonexistent.m4a", config_path=_no_config(tmp_path))
    assert "error" not in result
    assert result["text"] == "hei maailma"


# -- meeting profile fallback (issue #78) ---------------------------------


def test_meeting_fallback_appends_exactly_one_warning(tmp_path, monkeypatch) -> None:
    """decode_meeting returns None and the dictation path succeeds: exactly
    the fallback warning lands in ``result["warnings"]``."""
    _patch_preflight_pass(monkeypatch)
    _patch_ingest(monkeypatch)
    _patch_vad(monkeypatch)
    monkeypatch.setattr(
        pipeline, "decode_meeting", lambda audio, slices, initial_prompt=None: None
    )
    _patch_decoders(
        monkeypatch,
        {"text": "parakeet varalla", "words": []},
        {"text": "parakeet varalla", "words": []},
    )
    result = transcribe_file(
        "/nonexistent.m4a", config_path=_no_config(tmp_path), profile="meeting"
    )
    assert result["text"] == "parakeet varalla"
    assert "error" not in result
    warnings = result.get("warnings", [])
    assert warnings.count("meeting decode failed; fell back to dictation path") == 1


def test_meeting_fallback_total_failure_is_error_not_warning(
    tmp_path, monkeypatch
) -> None:
    """decode_meeting fails AND the dictation fallback totally fails: the
    result is the #73 error. The fallback warning must NOT be appended
    (error takes precedence over warning-plus-empty-success)."""
    _patch_preflight_pass(monkeypatch)
    _patch_ingest(monkeypatch)
    _patch_vad(monkeypatch)
    monkeypatch.setattr(
        pipeline, "decode_meeting", lambda audio, slices, initial_prompt=None: None
    )
    _patch_decoders(monkeypatch, None, None, latched_a=True)
    result = transcribe_file(
        "/nonexistent.m4a", config_path=_no_config(tmp_path), profile="meeting"
    )
    assert "error" in result
    assert "decode A produced no output for any of" in result["error"]
    assert "meeting decode failed; fell back to dictation path" not in result.get(
        "warnings", []
    )


def test_successful_meeting_decode_appends_no_warning(tmp_path, monkeypatch) -> None:
    """A successful meeting decode must add NOTHING to the warnings channel
    (catches an unconditional warnings-append regression)."""
    _patch_preflight_pass(monkeypatch)
    _patch_ingest(monkeypatch)
    _patch_vad(monkeypatch)
    _patch_meeting_success(monkeypatch)
    result = transcribe_file(
        "/nonexistent.m4a", config_path=_no_config(tmp_path), profile="meeting"
    )
    assert result["text"] == "hei maailma"
    assert "warnings" not in result


def _patch_meeting_success(monkeypatch) -> None:
    def fake_decode_meeting(audio, slices, initial_prompt=None):
        return {
            "text": "hei maailma",
            "words": [{"word": "hei", "start": 0.0, "end": 0.4}],
            "segments": [{"start": 0.0, "end": 1.0, "text": "hei maailma"}],
            "language": "fi",
            "slices": [],
        }

    monkeypatch.setattr(pipeline, "decode_meeting", fake_decode_meeting)


# -- diarization failure (issue #78) --------------------------------------


def test_diarize_failure_appends_exact_warning_and_stays_fail_open(
    tmp_path, monkeypatch
) -> None:
    """A diarization exception: the transcript still ships (fail-open), no
    error key, and the exact failure warning is in the warnings channel."""
    _consensus_setup(monkeypatch)
    _patch_redecode(monkeypatch, "moikka")
    _patch_diarize(monkeypatch, segments=RuntimeError("pyannote crashed"))
    result = transcribe_file(
        "/nonexistent.m4a", config_path=_no_config(tmp_path), diarize=True
    )
    # Fail-open: the transcript still ships (the re-decode verdict splices
    # into the disputed span, so the text is "moikka" — same as the
    # existing test_diarize_failure_fails_open).
    assert result["text"] == "moikka"  # fail-open transcript survives
    assert len(result["segments"]) == 1
    assert "speaker" not in result["segments"][0]
    assert "error" not in result
    assert "diarization failed; continuing without speaker labels" in result["warnings"]


def test_diarize_success_has_no_failure_warning(tmp_path, monkeypatch) -> None:
    """A diarization that RUNS (successfully) carries the CC-BY attribution
    but must NOT carry the failure warning."""
    _consensus_setup(monkeypatch)
    _patch_redecode(monkeypatch, "moikka")
    _patch_diarize(monkeypatch, segments=[(0.0, 2.0, "SPEAKER_00")])
    result = transcribe_file(
        "/nonexistent.m4a", config_path=_no_config(tmp_path), diarize=True
    )
    warnings = result.get("warnings", [])
    assert "diarization failed; continuing without speaker labels" not in warnings
    from vemoizer.diarization import ATTRIBUTION

    assert ATTRIBUTION in warnings


# -- LLM tail relocation (issue #78) ---------------------------------------
#
# The repair + notes stages moved from pipeline.py to llm_tail.py (the
# 500-line cap). These pin that the call sites still route through the
# pipeline namespace so existing monkeypatch targets keep working.


def test_notes_failure_warning_still_lands(tmp_path, monkeypatch) -> None:
    """Notes generation failure still warns through the same channel
    (llm_tail.NOTES_FAILURE_WARNING), unchanged wording."""
    _consensus_setup(monkeypatch)
    _patch_redecode(monkeypatch, "moikka")
    monkeypatch.setattr(
        llm_tail,
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
    assert any("notes generation failed" in w for w in result.get("warnings", []))


def test_repair_pass_still_updates_paragraphs_only(tmp_path, monkeypatch) -> None:
    """The repair stage (now in llm_tail) still rewrites paragraphs only;
    segments stay the verbatim record."""
    _consensus_setup(monkeypatch)
    _patch_redecode(monkeypatch, "moikka")

    class _Client:
        def adjudicate(self, a_text, candidates, context=""):
            return "moikka"

        def complete(self, system, user, max_tokens=2048):
            # A high-similarity fix (extra letter, same word) — the repair
            # guard rejects low-similarity or over-grown candidates.
            return user.replace("moikka", "moikkaa")

        def close(self) -> None:
            pass

    monkeypatch.setattr(pipeline, "LLMClient", lambda cfg: _Client())
    cfg = _llm_config(tmp_path)
    result = transcribe_file("/nonexistent.m4a", config_path=str(cfg), repair=True)
    assert result["paragraphs"][0]["text"] == "moikkaa"
    assert result["segments"][0]["text"] == "moikka"


# -- CLI fail-loud (issue #78) -------------------------------------------


def test_cli_latched_total_failure_exits_nonzero_and_writes_nothing(
    tmp_path, monkeypatch
) -> None:
    """End-to-end latched failure through the CLI: exit_code 1, the error
    text on stderr, and NO output files on disk."""
    from typer.testing import CliRunner

    from vemoizer.cli import app as cli_app

    _patch_preflight_pass(monkeypatch)
    _patch_ingest(monkeypatch)
    _patch_vad(monkeypatch)
    _patch_decoders(monkeypatch, None, None, latched_a=True)
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    result = runner.invoke(cli_app, ["transcribe", "memo.m4a"])
    assert result.exit_code == 1
    assert "decode A produced no output for any of 1 slices" in result.stderr
    # The check must run BEFORE _write_output: no file of any format exists.
    for ext in (".txt", ".json", ".srt", ".vtt", ".md"):
        assert not (tmp_path / f"memo{ext}").exists(), f"memo{ext} was written"


def test_cli_empty_transcript_exits_nonzero_and_writes_nothing(
    tmp_path, monkeypatch
) -> None:
    """The empty-transcript rule: no text AND no segments AND no "error" key
    exits non-zero and writes NOTHING — the check runs before the
    file-writing loop (a silently-succeeded run must not ship empty files).
    Legitimately silent audio hits this rule by design (committed simple
    rule per the ticket: no pipeline marker distinguishes silence)."""
    from typer.testing import CliRunner

    import vemoizer.pipeline as pipeline_module
    from vemoizer.cli import app as cli_app

    def fake_transcribe_file(path, **kwargs):
        # No "error" key: the empty-transcript rule, not the error branch.
        return {"text": "", "segments": []}

    monkeypatch.setattr(pipeline_module, "transcribe_file", fake_transcribe_file)
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    result = runner.invoke(cli_app, ["transcribe", "silent.m4a"])
    assert result.exit_code == 1
    assert "empty transcript" in result.stderr
    # The rule runs BEFORE _write_output: no output file of any format.
    for ext in (".txt", ".json", ".srt", ".vtt", ".md"):
        assert not (tmp_path / f"silent{ext}").exists(), f"silent{ext} was written"


def test_cli_diarize_no_labels_exits_nonzero(tmp_path, monkeypatch) -> None:
    """The diarize-no-labels rule: --diarize with segments that carry no
    "speaker" key exits non-zero (even though the transcript is non-empty,
    so the empty-transcript rule does NOT fire — no double-report)."""
    from typer.testing import CliRunner

    import vemoizer.pipeline as pipeline_module
    from vemoizer.cli import app as cli_app

    def fake_transcribe_file(path, **kwargs):
        assert kwargs.get("diarize") is True
        return {
            "text": "moikka",
            "segments": [{"start": 0.0, "end": 1.0, "text": "moikka"}],
        }

    monkeypatch.setattr(pipeline_module, "transcribe_file", fake_transcribe_file)
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    result = runner.invoke(cli_app, ["transcribe", "a.m4a", "--diarize"])
    assert result.exit_code == 1
    assert "no speaker labels" in result.stderr
    # Exactly one error line: the diarize rule, not the empty-transcript rule.
    assert "empty transcript" not in result.stderr
    assert result.stderr.count("error:") == 1


def test_cli_diarize_with_labels_exits_zero(tmp_path, monkeypatch) -> None:
    """Control: --diarize with at least one labeled segment exits 0 and
    writes the transcript — the diarize-no-labels rule only fires when NO
    segment carries a "speaker" key."""
    from typer.testing import CliRunner

    import vemoizer.pipeline as pipeline_module
    from vemoizer.cli import app as cli_app

    def fake_transcribe_file(path, **kwargs):
        return {
            "text": "moikka",
            "segments": [
                {"start": 0.0, "end": 1.0, "text": "moikka", "speaker": "SPEAKER_00"}
            ],
        }

    monkeypatch.setattr(pipeline_module, "transcribe_file", fake_transcribe_file)
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    result = runner.invoke(
        cli_app, ["transcribe", "a.m4a", "--diarize", "--format", "txt"]
    )
    assert result.exit_code == 0
    assert (tmp_path / "a.txt").is_file()


def test_cli_batch_continues_after_empty_transcript(tmp_path, monkeypatch) -> None:
    """Batch continuation: the first file yields an empty transcript
    (exit_code 1 + continue, no files for it) and the second succeeds and
    IS written; the final exit code is 1."""
    from typer.testing import CliRunner

    import vemoizer.grouping as grouping
    import vemoizer.pipeline as pipeline_module
    from vemoizer.cli import app as cli_app

    def fake_transcribe_file(path, **kwargs):
        if str(path).endswith("silent.m4a"):
            return {"text": "", "segments": []}
        return {"text": "moikka", "segments": []}

    def fake_decode(files, transcribe_fn=None):
        return ["kiitos ja moi"], ["a"]

    monkeypatch.setattr(pipeline_module, "transcribe_file", fake_transcribe_file)
    monkeypatch.setattr(grouping, "decode_boundaries", fake_decode)
    monkeypatch.setattr(grouping, "concat_groups", lambda files: files[0])
    monkeypatch.setattr(grouping, "part_offsets", lambda files: [])
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    result = runner.invoke(
        cli_app,
        ["transcribe", "silent.m4a", "good.m4a", "--format", "txt", "--yes"],
    )
    assert result.exit_code == 1
    assert not (tmp_path / "silent.txt").exists()
    assert (tmp_path / "good.txt").is_file()
    assert "wrote transcript for good.m4a" in result.stdout
