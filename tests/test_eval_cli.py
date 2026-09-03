"""``vemoizer eval`` CLI: backends, baseline gate, exit codes (issue #51).

All backends are monkeypatched fakes — no models, no network. The CLI's
job is registry dispatch, baseline bookkeeping, and exit codes; scoring
itself is pinned by ``tests/test_eval_harness.py``.
"""

from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

import vemoizer.eval_cli as eval_cli
from vemoizer.cli import app

runner = CliRunner()


def _corpus(tmp_path: Path) -> Path:
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "one.wav").write_bytes(b"RIFF0000WAVE")
    (corpus / "one.txt").write_text("moro maailma", encoding="utf-8")
    (corpus / "two.wav").write_bytes(b"RIFF1111WAVE")
    (corpus / "two.txt").write_text("toinen testi", encoding="utf-8")
    return corpus


def _patch_backends(monkeypatch, hypotheses: dict[str, str]) -> None:
    """Every registered backend returns the same canned hypothesis map."""

    def _make(name: str):
        def _transcribe(wav: Path) -> str:
            return hypotheses.get(wav.stem, "")

        return _transcribe

    monkeypatch.setattr(
        eval_cli, "BACKENDS", {name: _make(name) for name in eval_cli.BACKENDS}
    )


def test_eval_scores_one_backend(tmp_path, monkeypatch) -> None:
    corpus = _corpus(tmp_path)
    _patch_backends(monkeypatch, {"one": "moro maailma", "two": "väärä teksti"})
    result = runner.invoke(
        app, ["eval", "--corpus", str(corpus), "--backend", "parakeet"]
    )
    assert result.exit_code == 0
    assert "parakeet" in result.stdout
    assert "one\t0.0000" in result.stdout
    assert "two\t1.0000" in result.stdout
    assert "aggregate\t0.5000" in result.stdout


def test_eval_backend_all_runs_every_backend(tmp_path, monkeypatch) -> None:
    corpus = _corpus(tmp_path)
    _patch_backends(monkeypatch, {"one": "moro maailma", "two": "toinen testi"})
    result = runner.invoke(app, ["eval", "--corpus", str(corpus), "--backend", "all"])
    assert result.exit_code == 0
    for name in ("parakeet", "canary", "consensus"):
        assert name in result.stdout


def test_eval_unknown_backend_rejected(tmp_path, monkeypatch) -> None:
    corpus = _corpus(tmp_path)
    result = runner.invoke(
        app, ["eval", "--corpus", str(corpus), "--backend", "whisperx"]
    )
    assert result.exit_code == 2
    assert "whisperx" in result.stderr


def test_eval_update_baseline_writes_fingerprinted_file(tmp_path, monkeypatch) -> None:
    corpus = _corpus(tmp_path)
    baseline_path = tmp_path / "wer_baseline.json"
    _patch_backends(monkeypatch, {"one": "moro maailma", "two": "toinen testi"})
    result = runner.invoke(
        app,
        [
            "eval",
            "--corpus",
            str(corpus),
            "--backend",
            "all",
            "--baseline",
            str(baseline_path),
            "--update-baseline",
        ],
    )
    assert result.exit_code == 0
    data = json.loads(baseline_path.read_text(encoding="utf-8"))
    assert data["corpus_fingerprint"]
    assert data["tolerance"] > 0
    assert data["backends"]["parakeet"]["aggregate"] == 0.0
    assert data["backends"]["consensus"]["one"] == 0.0


def test_eval_check_passes_against_matching_baseline(tmp_path, monkeypatch) -> None:
    corpus = _corpus(tmp_path)
    baseline_path = tmp_path / "wer_baseline.json"
    _patch_backends(monkeypatch, {"one": "moro maailma", "two": "toinen testi"})
    args = ["eval", "--corpus", str(corpus), "--backend", "all"]
    assert (
        runner.invoke(
            app, [*args, "--baseline", str(baseline_path), "--update-baseline"]
        ).exit_code
        == 0
    )
    result = runner.invoke(app, [*args, "--baseline", str(baseline_path), "--check"])
    assert result.exit_code == 0
    assert "regression" not in result.stdout.lower()


def test_eval_check_fails_on_regression(tmp_path, monkeypatch) -> None:
    corpus = _corpus(tmp_path)
    baseline_path = tmp_path / "wer_baseline.json"
    args = ["eval", "--corpus", str(corpus), "--backend", "parakeet"]
    _patch_backends(monkeypatch, {"one": "moro maailma", "two": "toinen testi"})
    assert (
        runner.invoke(
            app, [*args, "--baseline", str(baseline_path), "--update-baseline"]
        ).exit_code
        == 0
    )
    # The backend got worse: sample "two" now transcribes wrong.
    _patch_backends(monkeypatch, {"one": "moro maailma", "two": "aivan väärin"})
    result = runner.invoke(app, [*args, "--baseline", str(baseline_path), "--check"])
    assert result.exit_code == 2
    assert "two" in result.stderr


def test_eval_check_refuses_a_changed_corpus(tmp_path, monkeypatch) -> None:
    """Baseline numbers are only comparable on the corpus they measured."""
    corpus = _corpus(tmp_path)
    baseline_path = tmp_path / "wer_baseline.json"
    args = ["eval", "--corpus", str(corpus), "--backend", "parakeet"]
    _patch_backends(monkeypatch, {"one": "moro maailma", "two": "toinen testi"})
    assert (
        runner.invoke(
            app, [*args, "--baseline", str(baseline_path), "--update-baseline"]
        ).exit_code
        == 0
    )
    (corpus / "one.wav").write_bytes(b"RIFF2222WAVE")  # corpus drift
    result = runner.invoke(app, [*args, "--baseline", str(baseline_path), "--check"])
    assert result.exit_code == 2
    assert "corpus" in result.stderr.lower()


def test_eval_check_without_baseline_file_fails_actionably(
    tmp_path, monkeypatch
) -> None:
    corpus = _corpus(tmp_path)
    _patch_backends(monkeypatch, {})
    result = runner.invoke(
        app,
        [
            "eval",
            "--corpus",
            str(corpus),
            "--baseline",
            str(tmp_path / "missing.json"),
            "--check",
        ],
    )
    assert result.exit_code == 2
    assert "update-baseline" in result.stderr


def test_eval_missing_corpus_exits_one(tmp_path) -> None:
    result = runner.invoke(app, ["eval", "--corpus", str(tmp_path / "nope")])
    assert result.exit_code == 1
