"""Tests for the informational far-field eval arm (issue #154).

No models, no network: backends are fakes and the loudnorm step is a
counting stub. Asserts loudnorm runs exactly once per processed clip and
only in the loudnorm arm, that a failed loudnorm pass never falls back to
the plain decode, and that the gate's corpus is left alone.
"""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

import vemoizer.eval_cli as eval_cli
import vemoizer.ingest as ingest
import vemoizer.loudnorm as loudnorm
from vemoizer.audio_contract import SAMPLE_RATE
from vemoizer.degrade import (
    DegradeError,
    build_variants,
    loudnorm_corpus,
    run_degrade_eval,
    write_wav,
)
from vemoizer.ingest import IngestError


def _corpus(root: Path, stems: tuple[str, ...] = ("a", "b")) -> Path:
    corpus = root / "corpus"
    corpus.mkdir()
    for stem in stems:
        x = (0.3 * np.sin(np.arange(SAMPLE_RATE // 2) / 10.0)).astype(np.float32)
        write_wav(corpus / f"{stem}.wav", x)
        (corpus / f"{stem}.txt").write_text(f"hello {stem}\n", encoding="utf-8")
    return corpus


def _ingest(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as w:
        raw = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
    return raw.astype(np.float32) / 32767.0


def test_build_variants_writes_only_into_work_dir(tmp_path: Path) -> None:
    corpus = _corpus(tmp_path)
    before = sorted(p.name for p in corpus.iterdir())
    variants = build_variants(corpus, "far", tmp_path / "work", _ingest)
    assert sorted(p.name for p in corpus.iterdir()) == before
    assert {p.stem for p in variants} == {"a", "b"}
    assert all(p.is_file() for p in variants)
    txt = tmp_path / "work" / "far" / "a.txt"
    assert txt.read_text(encoding="utf-8") == "hello a\n"


def test_unpaired_wav_is_skipped_with_warning(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    corpus = _corpus(tmp_path)
    write_wav(corpus / "orphan.wav", _ingest(corpus / "a.wav"))
    with caplog.at_level("WARNING"):
        variants = build_variants(corpus, "far", tmp_path / "work", _ingest)
    assert {p.stem for p in variants} == {"a", "b"}
    assert "orphan.wav" in caplog.text


def test_loudnorm_applied_exactly_once_per_clip_and_only_in_its_arm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    corpus = _corpus(tmp_path)
    calls: list[str] = []

    def fake_decode_measured(path, measurement):
        calls.append(f"{Path(path).name}:loudnorm")
        return _ingest(Path(path))

    monkeypatch.setattr(loudnorm, "measurement_for", lambda path: object())
    monkeypatch.setattr(loudnorm, "decode_measured", fake_decode_measured)

    seen: list[str] = []

    def backend(wav: Path) -> str:
        seen.append(wav.parent.name)
        return "hello"

    rows, clean_rows = run_degrade_eval(
        corpus, "noisy", {"fake": backend}, tmp_path / "work", _ingest
    )

    # One loudnorm call per clip per arm: clean (2 clips) + noisy (2 clips).
    assert sorted(calls) == sorted(
        ["a.wav:loudnorm", "b.wav:loudnorm", "a.wav:loudnorm", "b.wav:loudnorm"]
    )
    assert "noisy" in seen and "noisy-loudnorm" in seen
    assert len(rows) == 1 and len(clean_rows) == 1


def test_failed_measurement_raises_instead_of_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    corpus = _corpus(tmp_path)
    monkeypatch.setattr(loudnorm, "measurement_for", lambda path: None)
    monkeypatch.setattr(
        loudnorm,
        "decode_measured",
        lambda *a, **k: pytest.fail("fallback decode must not run"),
    )
    with pytest.raises(DegradeError, match="a.wav"):
        loudnorm_corpus(corpus, tmp_path / "out")


def test_failed_pass2_decode_raises_instead_of_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    corpus = _corpus(tmp_path)
    monkeypatch.setattr(loudnorm, "measurement_for", lambda path: object())

    def pass2_fails(path, measurement):
        raise IngestError("ffmpeg failed on pass 2")

    monkeypatch.setattr(loudnorm, "decode_measured", pass2_fails)
    with pytest.raises(IngestError, match="pass 2"):
        loudnorm_corpus(corpus, tmp_path / "out")


def test_cli_exits_2_when_loudnorm_pass2_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from vemoizer.cli import app

    corpus = _corpus(tmp_path)
    monkeypatch.setattr(ingest, "ingest_audio", _ingest)
    monkeypatch.setattr(eval_cli, "BACKENDS", {"fake": lambda wav: "hello"})
    monkeypatch.setattr(loudnorm, "measurement_for", lambda path: object())

    def pass2_fails(path, measurement):
        raise IngestError("ffmpeg failed on pass 2")

    monkeypatch.setattr(loudnorm, "decode_measured", pass2_fails)
    result = CliRunner().invoke(
        app,
        ["eval", "--corpus", str(corpus), "--backend", "fake", "--degrade", "far"],
    )
    assert result.exit_code == 2


def test_cli_rejects_empty_degrade_profile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from vemoizer.cli import app

    corpus = _corpus(tmp_path)
    monkeypatch.setattr(eval_cli, "BACKENDS", {"fake": lambda wav: "hello"})
    result = CliRunner().invoke(
        app, ["eval", "--corpus", str(corpus), "--backend", "fake", "--degrade", ""]
    )
    assert result.exit_code == 2


def test_empty_clean_transcript_refuses_instead_of_scoring(tmp_path: Path) -> None:
    corpus = _corpus(tmp_path)
    with pytest.raises(DegradeError, match="empty transcript"):
        run_degrade_eval(
            corpus, "noisy", {"silent": lambda wav: ""}, tmp_path / "work", _ingest
        )


def test_crashing_backend_refuses_instead_of_scoring_wer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    corpus = _corpus(tmp_path)
    monkeypatch.setattr(loudnorm, "measurement_for", lambda path: object())
    monkeypatch.setattr(
        loudnorm, "decode_measured", lambda path, measurement: _ingest(Path(path))
    )

    def crashing(wav: Path) -> str:
        if wav.parent.name.endswith("loudnorm"):
            raise RuntimeError("boom on processed audio")
        return "hello"

    with pytest.raises(DegradeError, match="backend failed"):
        run_degrade_eval(corpus, "noisy", {"bad": crashing}, tmp_path / "work", _ingest)
