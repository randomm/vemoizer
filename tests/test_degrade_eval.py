"""Tests for the informational far-field eval arm (issue #154).

No models, no network: backends are fakes and the loudnorm step is a
counting stub. Asserts loudnorm runs exactly once per processed clip and
only in the loudnorm arm, and that the gate's corpus is left alone.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import vemoizer.loudnorm as loudnorm
from vemoizer.audio_contract import SAMPLE_RATE
from vemoizer.degrade import (
    LoudnormMeasurementError,
    build_variants,
    loudnorm_corpus,
    run_degrade_eval,
    write_wav,
)


def _corpus(root: Path, stems: tuple[str, ...] = ("a", "b")) -> Path:
    corpus = root / "corpus"
    corpus.mkdir()
    for stem in stems:
        x = (0.3 * np.sin(np.arange(SAMPLE_RATE // 2) / 10.0)).astype(np.float32)
        write_wav(corpus / f"{stem}.wav", x)
        (corpus / f"{stem}.txt").write_text(f"hello {stem}\n", encoding="utf-8")
    return corpus


def _ingest(path: Path) -> np.ndarray:
    import wave

    with wave.open(str(path), "rb") as w:
        raw = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
    return raw.astype(np.float32) / 32767.0


def test_build_variants_writes_only_into_work_dir(tmp_path: Path) -> None:
    corpus = _corpus(tmp_path)
    before = sorted(p.name for p in corpus.iterdir())
    variants = build_variants(corpus, "far", tmp_path / "work", _ingest)
    assert sorted(p.name for p in corpus.iterdir()) == before
    assert {v.stem for v in variants} == {"a", "b"}
    assert all(v.path.is_file() for v in variants)
    txt = tmp_path / "work" / "far" / "a.txt"
    assert txt.read_text(encoding="utf-8") == "hello a\n"


def test_loudnorm_applied_exactly_once_per_clip_and_only_in_its_arm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    corpus = _corpus(tmp_path)
    calls: list[str] = []

    def fake_preprocess(path, preprocess=None):
        calls.append(f"{Path(path).name}:{preprocess}")
        return _ingest(Path(path))

    monkeypatch.setattr(loudnorm, "preprocess_audio", fake_preprocess)
    monkeypatch.setattr(loudnorm, "measurement_for", lambda path: object())

    seen: list[str] = []

    def backend(wav: Path) -> str:
        seen.append(wav.parent.name)
        return "hello"

    rows, clean_rows = run_degrade_eval(
        corpus, "noisy", {"fake": backend}, tmp_path / "work", _ingest
    )

    # One loudnorm call per clip per arm: clean (2 clips) + noisy (2 clips).
    assert len(calls) == 4
    assert all(c.endswith(":loudnorm") for c in calls)
    assert sorted(calls) == sorted(
        ["a.wav:loudnorm", "b.wav:loudnorm", "a.wav:loudnorm", "b.wav:loudnorm"]
    )
    # Plain arms never touch loudnorm: their decodes read the plain dirs.
    assert "noisy" in seen and "noisy-loudnorm" in seen
    assert len(rows) == 1 and len(clean_rows) == 1


def test_failed_measurement_raises_instead_of_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    corpus = _corpus(tmp_path)
    monkeypatch.setattr(loudnorm, "measurement_for", lambda path: None)
    monkeypatch.setattr(
        loudnorm,
        "preprocess_audio",
        lambda *a, **k: pytest.fail("fallback decode must not run"),
    )
    with pytest.raises(LoudnormMeasurementError, match="a.wav"):
        loudnorm_corpus(corpus, tmp_path / "out")


def test_gate_default_path_has_no_preprocess(tmp_path: Path) -> None:
    # The gate's ingest kwargs stay empty; the degrade arm never reaches it.
    import inspect

    from vemoizer import eval_harness

    assert "preprocess" not in inspect.signature(eval_harness.run_eval).parameters


def test_crashing_backend_refuses_instead_of_scoring_wer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from vemoizer.degrade import DegradeError

    corpus = _corpus(tmp_path)
    monkeypatch.setattr(loudnorm, "measurement_for", lambda path: object())
    monkeypatch.setattr(
        loudnorm, "preprocess_audio", lambda path, preprocess=None: _ingest(Path(path))
    )

    def crashing(wav: Path) -> str:
        if wav.parent.name.endswith("loudnorm"):
            raise RuntimeError("boom on processed audio")
        return "hello"

    with pytest.raises(DegradeError, match="backend failed"):
        run_degrade_eval(corpus, "noisy", {"bad": crashing}, tmp_path / "work", _ingest)
