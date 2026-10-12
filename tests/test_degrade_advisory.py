"""Advisory text and reference-path tests for the far-field arm (issue #154).

No models, no network, no ffmpeg. Asserts that the printed header states what
the gain column is (an oracle level control), what loudnorm's far benefit
means, what is not covered (whisper), and that no default changes; and that
the gain arm's RMS reference is read through the same write path as the arms.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from _degrade_helpers import make_corpus, read_wav_float

import vemoizer.loudnorm as loudnorm
from vemoizer.degrade import run_degrade_eval, run_header
from vemoizer.loudnorm import LoudnormMeasurement


@pytest.fixture(autouse=True)
def _fake_loudnorm(monkeypatch: pytest.MonkeyPatch) -> None:
    meas = LoudnormMeasurement(
        input_i=-20.0, input_tp=-3.0, input_lra=5.0, input_thresh=-30.0, offset=0.0
    )
    monkeypatch.setattr(loudnorm, "_measurement_for", lambda path: meas)
    monkeypatch.setattr(
        loudnorm, "_decode_pass2", lambda path, m: read_wav_float(Path(path))
    )


def test_header_labels_gain_as_oracle_control() -> None:
    text = "\n".join(run_header("far"))
    assert "gain=oracle level control" in text
    assert "clean RMS" in text
    assert "not a proven upper bound" in text


def test_header_is_neutral_and_carries_no_fixed_measured_claim() -> None:
    text = "\n".join(run_header("far"))
    assert "compare the gain column with loudnorm" in text
    assert "docs/pipeline-spec.md" in text
    # The measured result is a dated single run and lives in the spec only.
    assert "level restoration" not in text
    assert "0.4723" not in text


def test_header_states_whisper_is_not_covered_and_no_default_changes() -> None:
    text = "\n".join(run_header("noisy"))
    assert "whisper not covered" in text
    assert "no default changes" in text


def test_gain_reference_is_read_through_the_same_write_path(tmp_path: Path) -> None:
    corpus = make_corpus(tmp_path)
    work = tmp_path / "work"
    reads: list[Path] = []

    def spy(path: Path) -> np.ndarray:
        reads.append(Path(path))
        return read_wav_float(Path(path))

    run_degrade_eval(corpus, "far", {"fake": lambda wav: "hello"}, work, spy)
    assert work / "clean" / "a.wav" in reads
