"""Model-backed WER integration tests (opt-in: ``uv run pytest -m models``).

These run the real decode backends over the committed Piper corpus and gate
against ``tests/fixtures/wer_baseline.json`` — the same check as
``vemoizer eval --backend all --check``, wired into pytest so a warm-cache
machine can run it as part of the suite. Unit tests never download models
(AGENTS.md); everything here is behind the ``models`` marker.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from vemoizer.eval_harness import (
    AGGREGATE_KEY,
    compare_to_baseline,
    corpus_fingerprint,
    run_eval,
)

CORPUS = Path(__file__).resolve().parent / "fixtures" / "corpus"
BASELINE = Path(__file__).resolve().parent / "fixtures" / "wer_baseline.json"

pytestmark = pytest.mark.models


@pytest.fixture(scope="module")
def baseline() -> dict:
    assert BASELINE.is_file(), "no committed baseline; run eval --update-baseline"
    return json.loads(BASELINE.read_text(encoding="utf-8"))


def test_corpus_matches_the_committed_baseline_fingerprint(baseline: dict) -> None:
    assert corpus_fingerprint(CORPUS) == baseline["corpus_fingerprint"], (
        "corpus drifted; re-measure with `vemoizer eval --update-baseline` "
        "in a dedicated commit"
    )


@pytest.mark.parametrize("backend", ["parakeet", "canary", "consensus"])
def test_backend_wer_does_not_regress(backend: str, baseline: dict) -> None:
    from vemoizer.eval_cli import BACKENDS

    results = run_eval(CORPUS, BACKENDS[backend])
    regressions = compare_to_baseline(
        results,
        baseline["backends"][backend],
        tolerance=float(baseline["tolerance"]),
    )
    assert regressions == [], f"{backend} regressed: {regressions}"
    # The number the whole project is built around: once consensus goes
    # live, it must not be worse than decode A alone.
    if backend == "consensus":
        assert results[AGGREGATE_KEY] <= baseline["backends"]["parakeet"][
            AGGREGATE_KEY
        ] + float(baseline["tolerance"])
