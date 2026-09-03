"""End-to-end pipeline orchestrator tests (issue #34).

All stages are mocked — no models, no network, no ffmpeg. The orchestrator's
job is to chain ingest -> VAD -> decode A -> decode B -> align -> disputed
spans -> re-decode -> LLM adjudicate, and to fail open at every stage.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

import vemoizer.pipeline as pipeline
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


def _transcriber(text: str, words: list[dict] | None = None):
    class _T:
        def __init__(self) -> None:
            self.model = "fake"
            self._loaded = False

        def _ensure_loaded(self) -> None:
            self._loaded = True

        def transcribe(self, audio: np.ndarray, **kw) -> dict:
            return {"text": text, "words": words or [], "segments": []}

        def cleanup(self) -> None:
            self.model = None

    return _T()


def _patch_decoders(monkeypatch, a: dict | None, b: dict | None) -> None:
    ta = _transcriber((a or {}).get("text", ""), (a or {}).get("words"))
    tb = _transcriber(b.get("text", ""), b.get("words")) if b is not None else None

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


def _patch_redecode(monkeypatch, text: str = "moottori") -> None:
    class _R:
        def __init__(self) -> None:
            self.model = "fake"
            self._loaded = False

        def _ensure_loaded(self) -> None:
            self._loaded = True

        def transcribe_span(self, audio, span):
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
    words_a = [{"word": "hei", "start": 0.0, "end": 0.4}]
    words_b = [{"word": "moi", "start": 0.1, "end": 0.5}]  # disputed vs "hei"
    _patch_decoders(
        monkeypatch,
        {"text": "hei maailma", "words": words_a},
        {"text": "moi maailma", "words": words_b},
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
    # Fail-loud: a degraded consensus (decode B down) must surface to the
    # user, not just a logger.warning (issue #36).
    assert "decode B" in result["warnings"][0]
    assert "consensus" in result["warnings"][0]


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


def test_per_slice_warning_propagates(tmp_path, monkeypatch) -> None:
    # A per-slice decode failure (the _decode fail-open path) must surface
    # as a user-visible warning, not just a logger.warning.
    _patch_ingest(monkeypatch)
    _patch_vad(monkeypatch)

    class _Failing:
        def __init__(self) -> None:
            self._t = _transcriber("x")

        def transcribe(self, audio, **kw):
            raise RuntimeError("slice boom")

        def cleanup(self) -> None:
            pass

    _good = _transcriber("moikka")

    class _B:
        def __init__(self) -> None:
            self._t = _good

        def transcribe(self, audio, **kw):
            return _good.transcribe(audio)

        def cleanup(self) -> None:
            pass

    monkeypatch.setattr(pipeline, "ParakeetTranscriber", _Failing)
    monkeypatch.setattr(pipeline, "CanaryTranscriber", _B)
    result = transcribe_file("/nonexistent.m4a")
    assert result["text"] == ""
    assert any("decode A" in w and "slice boom" in w for w in result["warnings"])


def test_no_warnings_when_both_decodes_succeed(tmp_path, monkeypatch) -> None:
    _patch_ingest(monkeypatch)
    _patch_vad(monkeypatch)
    _patch_decoders(
        monkeypatch,
        {"text": "hei", "words": [{"word": "hei", "start": 0.0, "end": 0.3}]},
        {"text": "hei", "words": [{"word": "hei", "start": 0.0, "end": 0.3}]},
    )
    result = transcribe_file("/nonexistent.m4a")
    assert result.get("warnings", []) == []
