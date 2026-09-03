"""End-to-end pipeline orchestrator tests (issue #34).

All stages are mocked — no models, no network, no ffmpeg. The orchestrator's
job is to chain ingest -> VAD -> decode A -> decode B -> align -> disputed
spans -> re-decode -> LLM adjudicate, and to fail open at every stage.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import vemoizer.pipeline as pipeline
import vemoizer.progress as progress
from vemoizer.pipeline import transcribe_file


def _audio(seconds: float = 2.0) -> np.ndarray:
    return np.zeros(int(seconds * 16_000), dtype=np.float32)


def _patch_ingest(monkeypatch) -> None:
    monkeypatch.setattr(pipeline, "ingest_audio", lambda path: _audio())


def _patch_vad(monkeypatch) -> None:
    class _Model:
        def reset_states(self) -> None:
            pass

        def __call__(self, window: np.ndarray, sample_rate: int) -> np.ndarray:
            return np.zeros(len(window), dtype=np.float32)

    monkeypatch.setattr(pipeline, "load_vad_model", lambda: _Model())
    monkeypatch.setattr(
        pipeline,
        "vad_segments",
        lambda audio, model, **kw: [pipeline.SpeechSegment(0, len(audio))],
    )


def _transcriber(
    text: str,
    words: list[dict] | None = None,
    segments: list[dict] | None = None,
):
    class _T:
        def __init__(self) -> None:
            self.model = "fake"
            self._loaded = False

        def _ensure_loaded(self) -> None:
            self._loaded = True

        def transcribe(self, audio: np.ndarray, **kw) -> dict:
            return {"text": text, "words": words or [], "segments": segments or []}

        def cleanup(self) -> None:
            self.model = None

    return _T()


def _patch_decoders(monkeypatch, a: dict | None, b: dict | None) -> None:
    ta = _transcriber(
        (a or {}).get("text", ""), (a or {}).get("words"), (a or {}).get("segments")
    )
    tb = (
        _transcriber(b.get("text", ""), b.get("words"), b.get("segments"))
        if b is not None
        else None
    )

    class _A:
        def __init__(self) -> None:
            if a is None:
                raise RuntimeError("parakeet unavailable")
            self._t = ta

        def transcribe(self, audio, **kw):
            assert ta is not None
            return ta.transcribe(audio)

        def cleanup(self) -> None:
            ta.cleanup()

    class _B:
        def __init__(self) -> None:
            if tb is None:
                raise RuntimeError("canary unavailable")
            self._t = tb

        def transcribe(self, audio, **kw):
            assert tb is not None
            return tb.transcribe(audio)

        def cleanup(self) -> None:
            if tb is not None:
                tb.cleanup()

    monkeypatch.setattr(pipeline, "ParakeetTranscriber", _A)
    monkeypatch.setattr(pipeline, "CanaryTranscriber", _B)


def _patch_diarize(
    monkeypatch,
    segments: list[tuple[float, float, str]] | Exception | None = None,
) -> None:
    """Patch ``pipeline.diarize`` to return the given speaker segments.

    ``segments=None`` (default) means "use a default single speaker segment";
    ``segments=<Exception instance>`` means "raise that exception".
    """
    default = [(0.0, 0.5, "SPEAKER_00")]

    def _fake_diarize(audio: np.ndarray, **kw):
        if isinstance(segments, Exception):
            raise segments
        from vemoizer.diarization import DiarizationResult

        segs = default if segments is None else segments
        return DiarizationResult(segments=list(segs))

    monkeypatch.setattr(pipeline, "diarize", _fake_diarize)


def _patch_redecode(monkeypatch, text: str = "moottori") -> None:
    class _R:
        def __init__(self) -> None:
            self.model = "fake"
            self._loaded = False

        def _ensure_loaded(self) -> None:
            self._loaded = True

        def transcribe_span(self, audio, span, language=None):
            from vemoizer.redecode import ReDecodeResult

            return ReDecodeResult(span=span, text=text, words=[], ok=True)

        def cleanup(self) -> None:
            self.model = None

    monkeypatch.setattr(pipeline, "WhisperReDecodeTranscriber", _R)


def _llm_config(tmp_path: Path) -> Path:
    cfg = tmp_path / "config.toml"
    toml_text = (
        "[llm]\n"
        'base_url = "http://localhost"\n'
        'model = "m"\n'
        'api_key_env = "K"\n'
        "timeout_seconds = 1\n"
    )
    cfg.write_text(toml_text, encoding="utf-8")
    return cfg


def test_full_pipeline_assembles_text_and_spans(tmp_path, monkeypatch) -> None:
    _patch_ingest(monkeypatch)
    _patch_vad(monkeypatch)
    words_a = [
        {"word": "hei", "start": 0.0, "end": 0.4},
        {"word": "maailma", "start": 0.5, "end": 1.0},
    ]
    words_b: list[dict] = []  # decode B has no word timestamps
    _patch_decoders(
        monkeypatch,
        {"text": "hei maailma", "words": words_a},
        {"text": "nyt puhutaan aivan muusta", "words": words_b},
    )
    _patch_redecode(monkeypatch, "moikka")

    # Point config at a nonexistent file so no [llm] is found: the LLM stage
    # is skipped (fail-open) and the re-decode text wins the disputed span.
    cfg_path = str(tmp_path / "none.toml")
    result = transcribe_file("/nonexistent.m4a", config_path=cfg_path)

    assert result["text"] == "hei maailma"
    assert len(result["segments"]) == 1
    seg = result["segments"][0]
    assert seg["text"] == "moikka"  # re-decode fallback (no LLM configured)
    assert 0.0 <= seg["start"] < seg["end"]


def test_fail_open_when_decode_b_missing(tmp_path, monkeypatch) -> None:
    _patch_ingest(monkeypatch)
    _patch_vad(monkeypatch)
    _patch_decoders(
        monkeypatch,
        {"text": "vain A", "words": [{"word": "vain", "start": 0.0, "end": 0.3}]},
        None,  # decode B fails
    )
    result = transcribe_file("/nonexistent.m4a")
    assert result["text"] == "vain A"
    assert result["segments"] == []


def test_fail_open_when_both_decodes_fail(tmp_path, monkeypatch) -> None:
    _patch_ingest(monkeypatch)
    _patch_vad(monkeypatch)
    _patch_decoders(monkeypatch, None, None)
    result = transcribe_file("/nonexistent.m4a")
    assert result["text"] == ""
    assert result["segments"] == []


def test_fail_open_on_ingest_error(monkeypatch) -> None:
    from vemoizer.ingest import IngestError

    def _boom(path):
        raise IngestError("corrupt")

    monkeypatch.setattr(pipeline, "ingest_audio", _boom)
    result = transcribe_file("/nonexistent.m4a")
    assert result["text"] == ""
    assert result["segments"] == []


def test_empty_audio_short_circuits(monkeypatch) -> None:
    monkeypatch.setattr(pipeline, "ingest_audio", lambda path: _audio(0.0))
    result = transcribe_file("/nonexistent.m4a")
    assert result["text"] == ""
    assert result["segments"] == []


def test_diarize_off_by_default_skips_diarization(tmp_path, monkeypatch) -> None:
    """Without ``diarize=True`` the diarization stage must never be called."""
    _patch_ingest(monkeypatch)
    _patch_vad(monkeypatch)
    _patch_decoders(
        monkeypatch,
        {"text": "hei maailma", "words": [{"word": "hei", "start": 0.0, "end": 0.4}]},
        None,
    )

    def _boom(audio, **kw):
        raise AssertionError("diarize() must not be called when diarize=False")

    monkeypatch.setattr(pipeline, "diarize", _boom)
    result = transcribe_file("/nonexistent.m4a")
    assert result["segments"] == []


def test_diarize_on_overlaps_speaker_labels(tmp_path, monkeypatch) -> None:
    """``diarize=True`` overlaps each segment's time range with the
    speaker segments and attaches the speaker with the most overlap."""
    _patch_ingest(monkeypatch)
    _patch_vad(monkeypatch)
    words_a = [
        {"word": "hei", "start": 0.0, "end": 0.4},
        {"word": "maailma", "start": 0.5, "end": 1.0},
    ]
    words_b: list[dict] = []  # decode B has no word timestamps
    _patch_decoders(
        monkeypatch,
        {"text": "hei maailma", "words": words_a},
        {"text": "nyt puhutaan aivan muusta", "words": words_b},
    )
    _patch_redecode(monkeypatch, "moikka")
    _patch_diarize(
        monkeypatch,
        segments=[
            (0.0, 1.5, "SPEAKER_00"),  # dominates the [0, 2.0) disputed slice
            (1.5, 2.0, "SPEAKER_01"),
        ],
    )

    cfg_path = str(tmp_path / "none.toml")
    result = transcribe_file("/nonexistent.m4a", config_path=cfg_path, diarize=True)

    assert len(result["segments"]) == 1
    seg = result["segments"][0]
    assert seg["text"] == "moikka"
    assert seg["speaker"] == "SPEAKER_00"


def test_diarize_on_no_overlap_no_speaker_key(tmp_path, monkeypatch) -> None:
    """A segment with no overlapping speaker segment gets no ``speaker`` key."""
    _patch_ingest(monkeypatch)
    _patch_vad(monkeypatch)
    words_a = [
        {"word": "hei", "start": 0.0, "end": 0.4},
        {"word": "maailma", "start": 0.5, "end": 1.0},
    ]
    words_b: list[dict] = []  # decode B has no word timestamps
    _patch_decoders(
        monkeypatch,
        {"text": "hei maailma", "words": words_a},
        {"text": "nyt puhutaan aivan muusta", "words": words_b},
    )
    _patch_redecode(monkeypatch, "moikka")
    _patch_diarize(
        monkeypatch,
        segments=[
            (3.0, 3.5, "SPEAKER_00"),  # entirely outside the [0, 2.0) span
        ],
    )

    cfg_path = str(tmp_path / "none.toml")
    result = transcribe_file("/nonexistent.m4a", config_path=cfg_path, diarize=True)

    assert len(result["segments"]) == 1
    assert "speaker" not in result["segments"][0]


def test_diarize_failure_fails_open(tmp_path, monkeypatch) -> None:
    """A diarization exception must not abort the run; segments are still
    produced, just without speaker labels (fail-open)."""
    _patch_ingest(monkeypatch)
    _patch_vad(monkeypatch)
    words_a = [
        {"word": "hei", "start": 0.0, "end": 0.4},
        {"word": "maailma", "start": 0.5, "end": 1.0},
    ]
    words_b: list[dict] = []  # decode B has no word timestamps
    _patch_decoders(
        monkeypatch,
        {"text": "hei maailma", "words": words_a},
        {"text": "nyt puhutaan aivan muusta", "words": words_b},
    )
    _patch_redecode(monkeypatch, "moikka")
    _patch_diarize(monkeypatch, segments=RuntimeError("pyannote crashed"))

    cfg_path = str(tmp_path / "none.toml")
    result = transcribe_file("/nonexistent.m4a", config_path=cfg_path, diarize=True)

    assert result["text"] == "hei maailma"
    assert len(result["segments"]) == 1
    assert "speaker" not in result["segments"][0]
    assert "error" not in result


def test_diarize_on_empty_diarization_segments_no_speaker_keys(
    tmp_path, monkeypatch
) -> None:
    """Empty diarization result (no speaker segments) leaves every segment
    without a speaker key."""
    _patch_ingest(monkeypatch)
    _patch_vad(monkeypatch)
    words_a = [
        {"word": "hei", "start": 0.0, "end": 0.4},
        {"word": "maailma", "start": 0.5, "end": 1.0},
    ]
    words_b: list[dict] = []  # decode B has no word timestamps
    _patch_decoders(
        monkeypatch,
        {"text": "hei maailma", "words": words_a},
        {"text": "nyt puhutaan aivan muusta", "words": words_b},
    )
    _patch_redecode(monkeypatch, "moikka")
    _patch_diarize(monkeypatch, segments=[])

    cfg_path = str(tmp_path / "none.toml")
    result = transcribe_file("/nonexistent.m4a", config_path=cfg_path, diarize=True)

    assert len(result["segments"]) == 1
    assert "speaker" not in result["segments"][0]


def test_diarize_on_multiple_speakers_picks_max_overlap(tmp_path, monkeypatch) -> None:
    """When several speaker segments overlap a disputed span, the one with
    the largest overlap length wins."""
    _patch_ingest(monkeypatch)
    _patch_vad(monkeypatch)
    words_a = [
        {"word": "hei", "start": 0.0, "end": 0.4},
        {"word": "maailma", "start": 0.5, "end": 1.0},
    ]
    words_b: list[dict] = []  # decode B has no word timestamps
    _patch_decoders(
        monkeypatch,
        {"text": "hei maailma", "words": words_a},
        {"text": "nyt puhutaan aivan muusta", "words": words_b},
    )
    _patch_redecode(monkeypatch, "moikka")
    _patch_diarize(
        monkeypatch,
        segments=[
            (0.3, 0.4, "SPEAKER_00"),  # overlap 0.1s
            (0.0, 0.4, "SPEAKER_01"),  # overlap 0.4s (should win)
        ],
    )

    cfg_path = str(tmp_path / "none.toml")
    result = transcribe_file("/nonexistent.m4a", config_path=cfg_path, diarize=True)

    assert len(result["segments"]) == 1
    assert result["segments"][0]["speaker"] == "SPEAKER_01"


# -- progress logging ----------------------------------------------------
#
# A 64-minute memo splits into ~1100 VAD slices and the decode stages run one
# model call per slice. Before these lines the stages logged nothing between
# their first and last call, so a slow decode was indistinguishable from a
# deadlock. These tests pin the heartbeat down so it cannot silently regress.


def test_format_duration_scales_by_magnitude() -> None:
    assert progress.format_duration(9.7) == "9.7s"
    assert progress.format_duration(219.0) == "3m39s"
    assert progress.format_duration(3720.0) == "1h02m"


def test_stage_progress_logs_start_and_summary(caplog) -> None:
    caplog.set_level("INFO", logger="vemoizer")
    stage = progress.StageProgress("decode A", 3, audio_seconds=6.0)
    for _ in range(3):
        stage.advance()
    stage.done()

    messages = [r.getMessage() for r in caplog.records]
    assert "decode A: starting over 3 slices (6.0s of audio)" in messages
    assert any(
        m.startswith("decode A: finished 3/3 slices in") and "x realtime" in m
        for m in messages
    )


def test_stage_progress_throttles_heartbeats(caplog, monkeypatch) -> None:
    """Only one heartbeat per interval, no matter how many items land."""
    caplog.set_level("INFO", logger="vemoizer")
    monkeypatch.setattr(progress, "PROGRESS_INTERVAL_S", 0.0)  # every advance
    stage = progress.StageProgress("decode B", 4)
    for _ in range(4):
        stage.advance()
    heartbeats = [r.getMessage() for r in caplog.records if "eta" in r.getMessage()]
    assert len(heartbeats) == 4

    caplog.clear()
    monkeypatch.setattr(progress, "PROGRESS_INTERVAL_S", 3600.0)  # never due
    quiet = progress.StageProgress("decode B", 4)
    for _ in range(4):
        quiet.advance()
    assert not [r for r in caplog.records if "eta" in r.getMessage()]


def test_stage_progress_reports_failures(caplog) -> None:
    caplog.set_level("INFO", logger="vemoizer")
    stage = progress.StageProgress("decode B", 2)
    stage.advance()
    stage.advance(failed=True)
    stage.done()
    assert any(
        "decode B: finished 1/2 slices" in r.getMessage()
        and "(1 failed)" in r.getMessage()
        for r in caplog.records
    )


def test_pipeline_logs_every_stage(tmp_path, monkeypatch, caplog) -> None:
    caplog.set_level("INFO", logger="vemoizer")
    _patch_ingest(monkeypatch)
    _patch_vad(monkeypatch)
    _patch_decoders(
        monkeypatch,
        {
            "text": "hei maailma",
            "words": [
                {"word": "hei", "start": 0.0, "end": 0.4},
                {"word": "maailma", "start": 0.5, "end": 1.0},
            ],
        },
        {"text": "nyt puhutaan aivan muusta", "words": []},
    )
    _patch_redecode(monkeypatch, "moikka")
    transcribe_file("/nonexistent.m4a", config_path=str(tmp_path / "none.toml"))

    log = "\n".join(r.getMessage() for r in caplog.records)
    for expected in (
        "transcribe: /nonexistent.m4a",
        "ingest: 2.0s of audio in",
        "LLM adjudication: disabled",
        "VAD: 1 speech slices",
        "decode A: starting over 1 slices",
        "decode A: finished 1/1 slices",
        "decode B: starting over 1 slices",
        "slice dispute: 1/1 slices disputed",
        "disputed spans: 1",
        "re-decode: starting over 1 spans",
        "adjudicate: starting over 1 spans",
        "transcribe: done in",
    ):
        assert expected in log, f"missing progress line: {expected!r}\n---\n{log}"


def test_wordless_decode_b_still_aligns_via_slices(
    tmp_path, monkeypatch, caplog
) -> None:
    """Decode B has no word timestamps — the per-slice alignment is what
    makes consensus possible anyway (issue #55).

    B's words are synthesized from its slice text over the slice bounds;
    disputes anchor to decode A's real times. This is the whole activation
    in miniature: a text-only decode B produces a real disputed span.
    """
    caplog.set_level("INFO", logger="vemoizer")
    _patch_ingest(monkeypatch)
    _patch_vad(monkeypatch)
    _patch_decoders(
        monkeypatch,
        {
            "text": "hei maailma",
            "words": [
                {"word": "hei", "start": 0.0, "end": 0.4},
                {"word": "maailma", "start": 0.5, "end": 1.0},
            ],
        },
        {"text": "nyt puhutaan aivan muusta", "words": []},  # text-only decode B
    )
    _patch_redecode(monkeypatch, "moikka")
    result = transcribe_file(
        "/nonexistent.m4a", config_path=str(tmp_path / "none.toml")
    )

    log = "\n".join(r.getMessage() for r in caplog.records)
    assert "disputed spans: 1" in log
    assert len(result["segments"]) == 1
    assert result["segments"][0]["text"] == "moikka"  # re-decode won the span


def test_decode_only_parakeet_returns_single_decode(tmp_path, monkeypatch) -> None:
    _patch_ingest(monkeypatch)
    _patch_vad(monkeypatch)
    _patch_decoders(
        monkeypatch,
        {"text": "vain parakeet", "words": []},
        {"text": "vain canary", "words": []},
    )
    result = pipeline.transcribe_decode_only("/nonexistent.m4a", backend="parakeet")
    assert result["text"] == "vain parakeet"


def test_decode_only_canary_returns_single_decode(tmp_path, monkeypatch) -> None:
    _patch_ingest(monkeypatch)
    _patch_vad(monkeypatch)
    _patch_decoders(
        monkeypatch,
        {"text": "vain parakeet", "words": []},
        {"text": "vain canary", "words": []},
    )
    result = pipeline.transcribe_decode_only("/nonexistent.m4a", backend="canary")
    assert result["text"] == "vain canary"


def test_decode_only_unknown_backend_raises() -> None:
    with pytest.raises(ValueError, match="unknown backend"):
        pipeline.transcribe_decode_only("/nonexistent.m4a", backend="whisperx")


def test_decode_only_ingest_error_fails_open(monkeypatch) -> None:
    from vemoizer.ingest import IngestError

    def _boom(path):
        raise IngestError("corrupt")

    monkeypatch.setattr(pipeline, "ingest_audio", _boom)
    result = pipeline.transcribe_decode_only("/nonexistent.m4a", backend="parakeet")
    assert result["text"] == ""
    assert "error" in result


def test_decode_only_backend_failure_fails_open(monkeypatch) -> None:
    _patch_ingest(monkeypatch)
    _patch_vad(monkeypatch)
    _patch_decoders(monkeypatch, None, None)  # constructors raise
    result = pipeline.transcribe_decode_only("/nonexistent.m4a", backend="parakeet")
    assert result["text"] == ""


# -- assembly fixes (issue #53) ------------------------------------------


def _consensus_setup(monkeypatch, *, b_words=None):
    _patch_ingest(monkeypatch)
    _patch_vad(monkeypatch)
    words_a = [
        {"word": "hei", "start": 0.0, "end": 0.4},
        {"word": "maailma", "start": 0.5, "end": 1.0},
    ]
    a = {
        "text": "hei maailma",
        "words": words_a,
        "segments": [{"start": 0.0, "end": 1.0, "text": "hei maailma"}],
    }
    b = {
        "text": "nyt puhutaan aivan muusta",
        "words": b_words if b_words is not None else [],
    }
    _patch_decoders(monkeypatch, a, b)


def test_decode_b_candidate_is_span_scoped_not_whole_text(
    tmp_path, monkeypatch
) -> None:
    """The old code sent decode B's ENTIRE text as the candidate for every
    span (52K chars on a real memo)."""
    _consensus_setup(monkeypatch)
    _patch_redecode(monkeypatch, "moikka")
    seen: list[list[dict]] = []

    def spy_adjudicate(span, a_text, candidates, client, context=""):
        seen.append(candidates)
        return "moikka"

    monkeypatch.setattr(pipeline, "_adjudicate", spy_adjudicate)
    cfg = _llm_config(tmp_path)
    transcribe_file("/nonexistent.m4a", config_path=str(cfg))

    assert seen, "no span reached adjudication"
    b_cands = [c for cands in seen for c in cands if c["source"] == "decode B"]
    assert b_cands, "decode B candidate missing"
    for cand in b_cands:
        # span-scoped: the disputed slice's text, never the full transcript
        assert cand["text"] == "nyt puhutaan aivan muusta"


def test_failed_redecode_result_is_not_a_candidate(tmp_path, monkeypatch) -> None:
    _consensus_setup(monkeypatch)

    class _BadRedecoder:
        def __init__(self) -> None:
            self.model = "fake"

        def transcribe_span(self, audio, span, language=None):
            from vemoizer.redecode import ReDecodeResult as SpanResult

            return SpanResult(span=span, text="roskaa", words=[], ok=False)

        def cleanup(self) -> None:
            pass

    monkeypatch.setattr(pipeline, "WhisperReDecodeTranscriber", _BadRedecoder)
    seen: list[list[dict]] = []

    def spy_adjudicate(span, a_text, candidates, client, context=""):
        seen.append(candidates)
        return "x"

    monkeypatch.setattr(pipeline, "_adjudicate", spy_adjudicate)
    transcribe_file("/nonexistent.m4a", config_path=str(tmp_path / "none.toml"))

    assert seen
    sources = [c["source"] for c in seen[0]]
    assert "re-decode" not in sources  # ok=False result must not be offered


def test_adjudicator_receives_surrounding_context(tmp_path, monkeypatch) -> None:
    _consensus_setup(monkeypatch)
    _patch_redecode(monkeypatch, "moikka")
    # Pin the span to the first word only, so "maailma" is context.
    from vemoizer.spans import Span

    monkeypatch.setattr(pipeline, "_find_spans", lambda a, b: [Span(0.0, 0.45)])
    contexts: list[str] = []

    class _SpyClient:
        def adjudicate(self, a_text, candidates, context=""):
            contexts.append(context)
            return "moikka"

        def close(self) -> None:
            pass

    monkeypatch.setattr(pipeline, "LLMClient", lambda cfg: _SpyClient())
    cfg = _llm_config(tmp_path)
    transcribe_file("/nonexistent.m4a", config_path=str(cfg))

    assert contexts, "LLM adjudicator was not called"
    # +-10s of decode-A words around the span: "maailma" is within 10s
    assert "maailma" in contexts[0]


def test_assembled_output_is_full_coverage_with_paragraphs(
    tmp_path, monkeypatch
) -> None:
    _consensus_setup(monkeypatch)
    _patch_redecode(monkeypatch, "moikka")
    # Explicit nonexistent config: never pick up the developer's real
    # ~/.config/vemoizer/config.toml (a live LLM would adjudicate).
    result = transcribe_file(
        "/nonexistent.m4a", config_path=str(tmp_path / "none.toml")
    )

    # The whole disputed slice is replaced by the verdict, spliced into the
    # sentence segment: full coverage, sentence bounds preserved.
    assert len(result["segments"]) == 1
    assert result["segments"][0]["text"] == "moikka"
    assert result["text"] == "moikka"
    assert result["paragraphs"] == [{"start": 0.0, "end": 1.0, "text": "moikka"}]


# -- notes stage wiring (issue #57) --------------------------------------


def test_notes_failure_lands_in_warnings_not_errors(tmp_path, monkeypatch) -> None:
    """A failed notes stage warns and ships the transcript untouched."""
    _consensus_setup(monkeypatch)
    _patch_redecode(monkeypatch, "moikka")
    monkeypatch.setattr(pipeline, "generate_notes", lambda client, text: None)

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
    monkeypatch.setattr(pipeline, "generate_notes", lambda client, text: fake_notes)

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

    def _must_not_run(client, text):
        raise AssertionError("notes stage ran without an LLM config")

    monkeypatch.setattr(pipeline, "generate_notes", _must_not_run)
    result = transcribe_file(
        "/nonexistent.m4a", config_path=str(tmp_path / "none.toml")
    )
    assert "notes" not in result
    assert "warnings" not in result
