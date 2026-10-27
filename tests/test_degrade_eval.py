"""Tests for the informational far-field eval arm (issue #154).

No models, no network, no ffmpeg: backends are fakes and the loudnorm
measurement and pass-2 decode are stubbed at the private seams that
``preprocess_audio`` calls. Asserts loudnorm runs exactly once per processed
clip and only in the loudnorm arm, that a failed loudnorm pass never falls
back to the plain decode, and that the gate's corpus is left alone.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from _degrade_helpers import make_corpus, read_wav_float, rms
from typer.testing import CliRunner, Result

import vemoizer.degrade as degrade
import vemoizer.eval_backends as eval_backends
import vemoizer.ingest as ingest
import vemoizer.loudnorm as loudnorm
from vemoizer.audio_contract import write_wav
from vemoizer.degrade import (
    DegradeError,
    build_variants,
    gain_corpus,
    loudnorm_corpus,
    run_degrade_eval,
)
from vemoizer.ingest import IngestError
from vemoizer.loudnorm import LoudnormMeasurement

_MEAS = LoudnormMeasurement(
    input_i=-20.0, input_tp=-3.0, input_lra=5.0, input_thresh=-30.0, offset=0.0
)


def _no_measurement(path: Path) -> None:
    return None


def _pass2_fails(path: Path, measurement: LoudnormMeasurement) -> np.ndarray:
    raise IngestError("ffmpeg failed on pass 2")


@pytest.fixture(autouse=True)
def _fake_loudnorm(monkeypatch: pytest.MonkeyPatch) -> None:
    """Hermetic loudnorm: a fixed measurement and a plain re-read as pass 2."""

    def measure(path: Path) -> LoudnormMeasurement:
        return _MEAS

    def pass2(path: Path, measurement: LoudnormMeasurement) -> np.ndarray:
        return read_wav_float(Path(path))

    monkeypatch.setattr(loudnorm, "_measurement_for", measure)
    monkeypatch.setattr(loudnorm, "_decode_pass2", pass2)


def test_build_variants_writes_only_into_work_dir(tmp_path: Path) -> None:
    corpus = make_corpus(tmp_path)
    before = sorted(p.name for p in corpus.iterdir())
    variants = build_variants(corpus, "far", tmp_path / "work", read_wav_float)
    assert sorted(p.name for p in corpus.iterdir()) == before
    assert {p.stem for p in variants} == {"a", "b"}
    assert all(p.is_file() for p in variants)
    txt = tmp_path / "work" / "far" / "a.txt"
    assert txt.read_text(encoding="utf-8") == "hello a\n"


def test_gain_control_restores_clean_rms(tmp_path: Path) -> None:
    corpus = make_corpus(tmp_path)
    build_variants(corpus, "far", tmp_path / "work", read_wav_float)
    out = gain_corpus(
        corpus, tmp_path / "work" / "far", tmp_path / "gain", read_wav_float
    )
    clean = read_wav_float(corpus / "a.wav").astype(np.float64)
    got = read_wav_float(out / "a.wav").astype(np.float64)
    assert rms(got) == pytest.approx(rms(clean), rel=0.01)
    assert (out / "a.txt").read_text(encoding="utf-8") == "hello a\n"


def test_cli_prints_reproducibility_header_and_gain_column(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    corpus = make_corpus(tmp_path)
    monkeypatch.setattr(ingest, "ingest_audio", read_wav_float)
    monkeypatch.setattr(eval_backends, "BACKENDS", {"fake": lambda wav: "hello"})
    result = _cli(corpus)
    assert result.exit_code == 0
    assert "base_seed=154" in result.output
    assert "ffmpeg=" in result.output
    assert (
        "parakeet=mlx-community/parakeet-tdt-0.6b-v3@"
        "ed2b7e8c15f9aaa0b5772e2efb986255eaef7e15" in result.output
    )
    assert "profile\tbackend\tplain\tgain\tloudnorm\tdelta" in result.output


def test_unpaired_wav_refuses_instead_of_skipping(tmp_path: Path) -> None:
    corpus = make_corpus(tmp_path)
    write_wav(corpus / "orphan.wav", read_wav_float(corpus / "a.wav"))
    with pytest.raises(DegradeError, match="orphan.wav"):
        build_variants(corpus, "far", tmp_path / "work", read_wav_float)


def test_loudnorm_applied_exactly_once_per_clip_and_only_in_its_arm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    corpus = make_corpus(tmp_path)
    calls: list[str] = []

    def counting_pass2(path: Path, measurement: LoudnormMeasurement) -> np.ndarray:
        calls.append(f"{Path(path).name}:loudnorm")
        return read_wav_float(Path(path))

    monkeypatch.setattr(loudnorm, "_decode_pass2", counting_pass2)

    seen: list[str] = []

    def backend(wav: Path) -> str:
        seen.append(wav.parent.name)
        return "hello"

    rows = run_degrade_eval(
        corpus, "noisy", {"fake": backend}, tmp_path / "work", read_wav_float
    )

    # One loudnorm call per clip per arm: clean (2 clips) + noisy (2 clips).
    assert sorted(calls) == sorted(
        ["a.wav:loudnorm", "b.wav:loudnorm", "a.wav:loudnorm", "b.wav:loudnorm"]
    )
    assert "noisy" in seen and "noisy-loudnorm" in seen
    assert [r.profile for r in rows] == ["clean", "noisy"]


def test_failed_measurement_raises_instead_of_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    corpus = make_corpus(tmp_path)
    monkeypatch.setattr(loudnorm, "_measurement_for", _no_measurement)
    monkeypatch.setattr(
        loudnorm,
        "_decode_pass2",
        lambda *a, **k: pytest.fail("fallback decode must not run"),
    )
    with pytest.raises(IngestError, match="a.wav"):
        loudnorm_corpus(corpus, tmp_path / "out")


def test_failed_pass2_decode_raises_instead_of_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    corpus = make_corpus(tmp_path)
    monkeypatch.setattr(loudnorm, "_decode_pass2", _pass2_fails)
    with pytest.raises(IngestError, match="pass 2"):
        loudnorm_corpus(corpus, tmp_path / "out")


def test_cli_exits_2_when_loudnorm_pass2_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from vemoizer.cli import app

    corpus = make_corpus(tmp_path)
    monkeypatch.setattr(ingest, "ingest_audio", read_wav_float)
    monkeypatch.setattr(eval_backends, "BACKENDS", {"fake": lambda wav: "hello"})
    monkeypatch.setattr(loudnorm, "_decode_pass2", _pass2_fails)
    result = CliRunner().invoke(
        app,
        ["eval", "--corpus", str(corpus), "--backend", "fake", "--degrade", "far"],
    )
    assert result.exit_code == 2


def test_cli_rejects_empty_degrade_profile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from vemoizer.cli import app

    corpus = make_corpus(tmp_path)
    monkeypatch.setattr(eval_backends, "BACKENDS", {"fake": lambda wav: "hello"})
    result = CliRunner().invoke(
        app, ["eval", "--corpus", str(corpus), "--backend", "fake", "--degrade", ""]
    )
    assert result.exit_code == 2


def test_cli_rejects_llm_with_degrade(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from vemoizer.cli import app

    corpus = make_corpus(tmp_path)
    monkeypatch.setattr(eval_backends, "BACKENDS", {"fake": lambda wav: "hello"})
    result = CliRunner().invoke(
        app,
        [
            "eval",
            "--corpus",
            str(corpus),
            "--backend",
            "fake",
            "--degrade",
            "far",
            "--llm",
        ],
    )
    assert result.exit_code == 2


def test_clean_plain_and_clean_loudnorm_consume_identical_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    corpus = make_corpus(tmp_path)
    plain_bytes: dict[str, bytes] = {}
    loudnorm_input_bytes: dict[str, bytes] = {}

    def measure(path: Path) -> LoudnormMeasurement:
        p = Path(path)
        if p.parent.name == "clean":
            loudnorm_input_bytes[p.name] = p.read_bytes()
        return _MEAS

    def plain(wav: Path) -> str:
        if wav.parent.name == "clean":
            plain_bytes[wav.name] = wav.read_bytes()
        return "hello"

    monkeypatch.setattr(loudnorm, "_measurement_for", measure)
    run_degrade_eval(corpus, "far", {"fake": plain}, tmp_path / "work", read_wav_float)

    assert sorted(plain_bytes) == ["a.wav", "b.wav"]
    assert plain_bytes == loudnorm_input_bytes
    # Both arms see the 16-bit write path of the decoded clip, not the
    # committed file's bytes.
    ref = write_wav(tmp_path / "ref.wav", read_wav_float(corpus / "a.wav"))
    assert plain_bytes["a.wav"] == ref.read_bytes()


def test_empty_clean_transcript_refuses_instead_of_scoring(tmp_path: Path) -> None:
    corpus = make_corpus(tmp_path)
    with pytest.raises(DegradeError, match="empty transcript"):
        run_degrade_eval(
            corpus,
            "noisy",
            {"silent": lambda wav: ""},
            tmp_path / "work",
            read_wav_float,
        )


def test_crashing_backend_refuses_instead_of_scoring_wer(tmp_path: Path) -> None:
    corpus = make_corpus(tmp_path)

    def crashing(wav: Path) -> str:
        if wav.parent.name.endswith("loudnorm"):
            raise RuntimeError("boom on processed audio")
        return "hello"

    with pytest.raises(DegradeError, match="backend failed"):
        run_degrade_eval(
            corpus, "noisy", {"bad": crashing}, tmp_path / "work", read_wav_float
        )


def test_unknown_profile_refuses_before_writing_variants(tmp_path: Path) -> None:
    corpus = make_corpus(tmp_path)
    with pytest.raises(DegradeError, match="unknown degrade profile"):
        run_degrade_eval(
            corpus, "", {"fake": lambda wav: "hello"}, tmp_path / "w", read_wav_float
        )
    assert not (tmp_path / "w").exists()


def _cli(corpus: Path, *extra: str) -> Result:
    from vemoizer.cli import app

    args = ["eval", "--corpus", str(corpus), "--backend", "fake", "--degrade", "far"]
    return CliRunner().invoke(app, [*args, *extra])


def test_scoring_bug_is_not_relabelled_as_degrade_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    corpus = make_corpus(tmp_path)

    def buggy_run_eval(*args: object, **kwargs: object) -> dict[str, float]:
        raise ValueError("scoring bug")

    monkeypatch.setattr(degrade, "run_eval", buggy_run_eval)
    with pytest.raises(ValueError, match="scoring bug"):
        run_degrade_eval(
            corpus,
            "noisy",
            {"fake": lambda wav: "hello"},
            tmp_path / "w",
            read_wav_float,
        )


def test_cli_lets_programming_errors_raise(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    corpus = make_corpus(tmp_path)
    monkeypatch.setattr(eval_backends, "BACKENDS", {"fake": lambda wav: "hello"})

    def bug(*args: object, **kwargs: object) -> list[object]:
        raise ValueError("programming bug")

    monkeypatch.setattr(degrade, "run_degrade_eval", bug)
    result = _cli(corpus)
    assert isinstance(result.exception, ValueError)


@pytest.mark.parametrize(
    "flag", ["--check", "--update-baseline", "--agreement", "--llm"]
)
def test_cli_refuses_gate_flags_with_degrade(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, flag: str
) -> None:
    corpus = make_corpus(tmp_path)
    monkeypatch.setattr(eval_backends, "BACKENDS", {"fake": lambda wav: "hello"})
    assert _cli(corpus, flag).exit_code == 2


def test_cli_exits_2_without_traceback_when_ffmpeg_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    corpus = make_corpus(tmp_path)
    monkeypatch.setattr(eval_backends, "BACKENDS", {"fake": lambda wav: "hello"})

    def no_ffmpeg(path: Path) -> np.ndarray:
        raise IngestError("ffmpeg not found on PATH")

    monkeypatch.setattr(ingest, "ingest_audio", no_ffmpeg)
    result = _cli(corpus)
    assert result.exit_code == 2
    assert isinstance(result.exception, SystemExit)


def test_cli_exits_2_without_traceback_when_variant_write_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    corpus = make_corpus(tmp_path)
    monkeypatch.setattr(eval_backends, "BACKENDS", {"fake": lambda wav: "hello"})
    monkeypatch.setattr(ingest, "ingest_audio", read_wav_float)

    def disk_full(path: Path, x: np.ndarray) -> Path:
        raise OSError("No space left on device")

    monkeypatch.setattr(degrade, "write_wav", disk_full)
    result = _cli(corpus)
    assert result.exit_code == 2
    assert isinstance(result.exception, SystemExit)


def test_cli_exits_2_without_traceback_when_backend_crashes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    corpus = make_corpus(tmp_path)
    monkeypatch.setattr(ingest, "ingest_audio", read_wav_float)

    def model_missing(wav: Path) -> str:
        raise RuntimeError("model not cached")

    monkeypatch.setattr(eval_backends, "BACKENDS", {"fake": model_missing})
    result = _cli(corpus)
    assert result.exit_code == 2
    assert isinstance(result.exception, SystemExit)
