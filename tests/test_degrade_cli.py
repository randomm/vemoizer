"""CLI and header tests for the far-field eval arm (issue #154).

No models and no network. Backends are fakes and ingest is a plain WAV read;
the header test uses the real ``ffmpeg -version`` seam (or a spy on it).
Covers: the reproducibility header (seeds, every profile parameter, ffmpeg,
pinned model revisions), temp-dir failures exiting 2 with one line, and
temp-dir cleanup on interrupt.
"""

from __future__ import annotations

import os
import signal
import tempfile
import wave
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner, Result

import vemoizer.degrade as degrade
import vemoizer.eval_cli as eval_cli
import vemoizer.ingest as ingest
from vemoizer.audio_contract import SAMPLE_RATE, write_wav
from vemoizer.cli import app
from vemoizer.degrade_signal import (
    ATTENUATION_DB,
    BASE_SEED,
    NOISE_SNR_DB,
    REVERB_LEN_S,
    REVERB_T60_S,
)
from vemoizer.models import MODELS, ModelSpec


def _corpus(root: Path) -> Path:
    corpus = root / "corpus"
    corpus.mkdir()
    for stem in ("a", "b"):
        x = (0.3 * np.sin(np.arange(SAMPLE_RATE // 2) / 10.0)).astype(np.float32)
        write_wav(corpus / f"{stem}.wav", x)
        (corpus / f"{stem}.txt").write_text(f"hello {stem}\n", encoding="utf-8")
    return corpus


def _ingest(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as w:
        raw = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
    return raw.astype(np.float32) / 32767.0


def _invoke(corpus: Path, profile: str = "far") -> Result:
    return CliRunner().invoke(
        app,
        ["eval", "--corpus", str(corpus), "--backend", "fake", "--degrade", profile],
    )


@pytest.fixture(autouse=True)
def _hermetic(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(ingest, "ingest_audio", _ingest)
    monkeypatch.setattr(eval_cli, "BACKENDS", {"fake": lambda wav: "hello"})
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(scratch))


def test_header_states_seeds_parameters_steps_ffmpeg_and_models(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def fake_ffmpeg() -> str:
        calls.append("ffmpeg")
        return "ffmpeg version FAKE-9.9"

    fake_models = (ModelSpec("fakeasr", "org/fake-repo", "a" * 40),)
    monkeypatch.setattr(degrade, "_ffmpeg_version", fake_ffmpeg)
    monkeypatch.setattr(degrade, "MODELS", fake_models)

    header = degrade.run_header("far-noisy")

    assert calls == ["ffmpeg"]
    text = "\n".join(header)
    assert all(line.startswith("#") for line in header)
    assert f"base_seed={BASE_SEED}" in text
    assert "seed=clip_seed(stem, profile)" in text
    assert f"attenuation_db={ATTENUATION_DB}" in text
    assert f"reverb_t60_s={REVERB_T60_S}" in text
    assert f"reverb_len_s={REVERB_LEN_S}" in text
    assert f"snr_db={NOISE_SNR_DB}" in text
    assert "noise=white" in text
    assert "steps=far->noisy" in text
    assert "ffmpeg=ffmpeg version FAKE-9.9" in text
    assert "fakeasr=org/fake-repo@" + "a" * 40 in text


def test_header_lists_every_pinned_model_revision_from_registry() -> None:
    text = "\n".join(degrade.run_header("noisy"))
    for spec in MODELS:
        assert f"{spec.name}={spec.repo_id}@{spec.revision}" in text


def test_cli_prints_header_then_table_on_stdout(tmp_path: Path) -> None:
    result = _invoke(_corpus(tmp_path))
    assert result.exit_code == 0
    lines = result.output.splitlines()
    table = lines.index("profile\tbackend\tplain\tgain\tloudnorm\tdelta")
    assert lines[0].startswith("# degrade=far base_seed=154")
    assert all(line.startswith("#") for line in lines[:table])
    assert any(line.startswith("# ffmpeg=") for line in lines[:table])


def test_unusable_tempdir_override_exits_2_with_one_line(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # An explicit tempfile.tempdir override is used as-is: no fallback.
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path / "missing"))
    result = _invoke(_corpus(tmp_path))
    _assert_one_line_exit_2(result)


def test_unusable_tmpdir_env_var_falls_back_to_system_tempdir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The real TMPDIR path: tempfile skips a missing $TMPDIR and falls back,
    # so the run completes (documented in docs/pipeline-spec.md).
    monkeypatch.setattr(tempfile, "tempdir", None)
    monkeypatch.setenv("TMPDIR", str(tmp_path / "missing"))
    result = _invoke(_corpus(tmp_path))
    assert result.exit_code == 0
    assert "Traceback" not in result.output


def test_readonly_tmpdir_exits_2_with_one_line(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    readonly = tmp_path / "readonly"
    readonly.mkdir()
    readonly.chmod(0o500)
    try:
        monkeypatch.setattr(tempfile, "tempdir", str(readonly))
        result = _invoke(_corpus(tmp_path))
        _assert_one_line_exit_2(result)
    finally:
        readonly.chmod(0o700)


def _assert_one_line_exit_2(result: Result) -> None:
    assert result.exit_code == 2
    assert "Traceback" not in result.output
    lines = result.output.strip().splitlines()
    assert len(lines) == 1
    assert lines[0].startswith("error:")


def test_sigterm_during_run_removes_temp_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[Path] = []

    def terminated(*args: object, **kwargs: object) -> None:
        work = args[3]
        assert isinstance(work, Path)
        seen.append(work)
        os.kill(os.getpid(), signal.SIGTERM)

    monkeypatch.setattr(degrade, "run_degrade_eval", terminated)
    result = _invoke(_corpus(tmp_path))
    assert result.exit_code == 143
    assert len(seen) == 1
    assert not seen[0].exists()


def test_interrupt_after_creating_temp_dir_removes_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[Path] = []

    def interrupted(*args: object, **kwargs: object) -> None:
        work = args[3]
        assert isinstance(work, Path) and work.is_dir()
        seen.append(work)
        raise KeyboardInterrupt

    monkeypatch.setattr(degrade, "run_degrade_eval", interrupted)
    result = _invoke(_corpus(tmp_path))
    assert result.exit_code != 0
    assert len(seen) == 1
    assert not seen[0].exists()
