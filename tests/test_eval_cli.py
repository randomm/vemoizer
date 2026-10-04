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
import vemoizer.pipeline
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


def test_eval_agreement_emits_metric_when_two_backends_scored(
    tmp_path, monkeypatch
) -> None:
    """--agreement emits the informational metric when 2+ backends are scored.

    The backends return *different* hypotheses per backend so that pairing
    the wrong two backends would change the printed value (the adversarial
    reviewer's non-vacuity requirement). Here parakeet says the right
    thing for "one" and the wrong thing for "two"; canary says the wrong
    thing for "one" and the right thing for "two". The consensus backend
    says the right thing for both (it is excluded from the pair).

    parakeet x canary pair:
      "one": parakeet="moro maailma" (correct), canary="x y z" (wrong)
             -> not both wrong -> False
      "two": parakeet="väärä teksti" (wrong), canary="toinen testi" (correct)
             -> not both wrong -> False
    So 0/2 = 0.0.
    """
    corpus = _corpus(tmp_path)
    parakeet_hyps = {"one": "moro maailma", "two": "väärä teksti"}
    canary_hyps = {"one": "x y z", "two": "toinen testi"}
    consensus_hyps = {"one": "moro maailma", "two": "toinen testi"}

    def _make(name: str, hyps: dict[str, str]):
        def _transcribe(wav: Path) -> str:
            return hyps.get(wav.stem, "")

        return _transcribe

    monkeypatch.setattr(
        eval_cli,
        "BACKENDS",
        {
            "parakeet": _make("parakeet", parakeet_hyps),
            "canary": _make("canary", canary_hyps),
            "consensus": _make("consensus", consensus_hyps),
        },
    )
    result = runner.invoke(
        app,
        ["eval", "--corpus", str(corpus), "--backend", "all", "--agreement"],
    )
    assert result.exit_code == 0
    assert "[agreement_on_wrong]" in result.stdout
    # Neither sample is "both wrong" (one hypothesis is always correct).
    assert "[agreement_on_wrong]\t0.0000" in result.stdout


def test_eval_agreement_pin_by_name_uses_parakeet_canary(tmp_path, monkeypatch) -> None:
    """The pair is pinned by name (parakeet + canary), not by dict order.

    If the code picked "the first two independent backends by dict order"
    and the dict were re-ordered, the pair would change. This test verifies
    that the parakeet × canary pair is used specifically by making the
    consensus backend return a hypothesis that would change the result if
    it were paired instead.

    parakeet: "one"->"x y z" (wrong), "two"->"toinen testi" (correct)
    canary:   "one"->"x y z" (wrong), "two"->"toinen testi" (correct)
    consensus:"one"->"moro maailma" (correct), "two"->"väärä teksti" (wrong)

    parakeet x canary: "one" both wrong -> True; "two" both correct -> False. 1/2 = 0.5.
    parakeet x consensus: "one" parakeet wrong, consensus correct -> False;
                          "two" parakeet correct, consensus wrong -> False. 0/2 = 0.0.
    canary x consensus: same as parakeet x consensus -> 0.0.
    """
    corpus = _corpus(tmp_path)
    parakeet_hyps = {"one": "x y z", "two": "toinen testi"}
    canary_hyps = {"one": "x y z", "two": "toinen testi"}
    consensus_hyps = {"one": "moro maailma", "two": "väärä teksti"}

    def _make(name: str, hyps: dict[str, str]):
        def _transcribe(wav: Path) -> str:
            return hyps.get(wav.stem, "")

        return _transcribe

    monkeypatch.setattr(
        eval_cli,
        "BACKENDS",
        {
            "parakeet": _make("parakeet", parakeet_hyps),
            "canary": _make("canary", canary_hyps),
            "consensus": _make("consensus", consensus_hyps),
        },
    )
    result = runner.invoke(
        app,
        ["eval", "--corpus", str(corpus), "--backend", "all", "--agreement"],
    )
    assert result.exit_code == 0
    # The pinned pair is parakeet x canary: "one" both say "x y z" (wrong),
    # "two" both say "toinen testi" (correct) -> 1/2 = 0.5.
    # If the code had paired parakeet x consensus (or canary x consensus),
    # the result would be 0.0 (neither sample has both hypotheses wrong).
    assert "[agreement_on_wrong]\t0.5000" in result.stdout


def test_eval_agreement_not_emitted_for_single_backend(tmp_path, monkeypatch) -> None:
    """--agreement with a single backend produces no agreement line.

    The test asserts the skip-note text (not just the absence of the
    metric) so that a future change that prints the metric anyway — or
    changes the skip note — is caught here rather than silently passing.
    """
    corpus = _corpus(tmp_path)
    _patch_backends(monkeypatch, {"one": "moro maailma", "two": "toinen testi"})
    result = runner.invoke(
        app,
        ["eval", "--corpus", str(corpus), "--backend", "parakeet", "--agreement"],
    )
    assert result.exit_code == 0
    assert "[agreement_on_wrong]" not in result.stdout
    # The skip note is printed to stderr, naming the missing backend.
    assert "skipping agreement_on_wrong" in result.stderr
    assert "canary" in result.stderr


def test_eval_agreement_emitted_when_both_backends_present(
    tmp_path, monkeypatch
) -> None:
    """--agreement with exactly parakeet + canary (no consensus) prints the metric.

    This is the non-vacant single-pair case: both pinned backends are
    present, so the skip-note path is not taken and the metric is computed
    and printed. The existing test
    ``test_eval_agreement_emits_metric_when_two_backends_scored`` covers
    the three-backend case (consensus excluded); this test isolates the
    two-backend case to make clear the metric fires when *both* pinned
    names are in ``per_backend_hyps``.
    """
    corpus = _corpus(tmp_path)
    parakeet_hyps = {"one": "moro maailma", "two": "väärä teksti"}
    canary_hyps = {"one": "x y z", "two": "toinen testi"}

    def _make(name: str, hyps: dict[str, str]):
        def _transcribe(wav: Path) -> str:
            return hyps.get(wav.stem, "")

        return _transcribe

    monkeypatch.setattr(
        eval_cli,
        "BACKENDS",
        {
            "parakeet": _make("parakeet", parakeet_hyps),
            "canary": _make("canary", canary_hyps),
        },
    )
    result = runner.invoke(
        app,
        ["eval", "--corpus", str(corpus), "--backend", "all", "--agreement"],
    )
    assert result.exit_code == 0
    # Both pinned backends are present, so the metric is printed (no skip note).
    assert "[agreement_on_wrong]" in result.stdout
    # Neither sample has both hypotheses wrong -> 0.0.
    assert "[agreement_on_wrong]\t0.0000" in result.stdout
    # No skip note in stderr.
    assert "skipping agreement_on_wrong" not in result.stderr


def test_eval_decodes_without_preprocess(tmp_path, monkeypatch) -> None:
    """The eval harness never applies ``--preprocess``: the ingest argv stays default.

    ``transcribe_decode_only`` calls ``ingest_audio(path)`` with no preprocess
    keyword, so the regression gate (``vemoizer eval --backend all --check``,
    issue #135) scores unprocessed audio and the eval argv is byte-identical
    to the pre-loudnorm baseline. ``ingest_audio`` resolves through the
    pipeline module here, so we pin it where it actually resolves.
    """
    corpus = _corpus(tmp_path)
    calls: list[tuple[object, dict]] = []

    def _recorded_ingest(path: Path, **kwargs: object) -> object:
        calls.append((path, kwargs))
        return []  # empty audio short-circuits transcribe_decode_only

    monkeypatch.setattr(vemoizer.pipeline, "ingest_audio", _recorded_ingest)
    result = runner.invoke(
        app, ["eval", "--corpus", str(corpus), "--backend", "parakeet"]
    )
    assert result.exit_code == 0
    assert calls  # ingest ran for the corpus samples
    for _path, kwargs in calls:
        assert kwargs == {}  # no preprocess keyword — plain unprocessed decode


def test_eval_agreement_does_not_affect_wer_gate(tmp_path, monkeypatch) -> None:
    """The agreement metric is informational: --check behaviour is unchanged."""
    corpus = _corpus(tmp_path)
    baseline_path = tmp_path / "wer_baseline.json"
    _patch_backends(monkeypatch, {"one": "moro maailma", "two": "toinen testi"})
    assert (
        runner.invoke(
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
        ).exit_code
        == 0
    )
    result = runner.invoke(
        app,
        [
            "eval",
            "--corpus",
            str(corpus),
            "--backend",
            "all",
            "--agreement",
            "--baseline",
            str(baseline_path),
            "--check",
        ],
    )
    assert result.exit_code == 0
    assert "baseline check passed" in result.stdout
