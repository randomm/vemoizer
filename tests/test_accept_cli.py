"""``vemoizer accept``: flags, bench discovery, the report and the gate (#229).

The command's own surface: the anonymised report that is safe to paste into a
PR body, the draft refusals, the fingerprint and baseline handling, and the
gate that fails closed. The per-excerpt pipeline-argument contract and the
scoring of a fresh decode live in ``tests/test_accept_run.py``; the shared
synthetic bench and the pipeline fakes are in ``tests/_accept_helpers.py``.

No models, no network, no audio, and no real transcript text: every assertion
about the printed output is ALSO a privacy assertion, so the real excerpt ids,
the reference text and the hypothesis text must never appear.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from _accept_helpers import (
    GLOSSARY_TERM,
    HYP_TOKEN,
    MIX_ID,
    REF_TOKEN,
    TITLE_TOKEN,
    TRACK_ID,
    _invoke,
    _mark_reviewed,
    _metric_keys,
    _result,
)

# The shared fixtures are imported into THIS module's namespace, where pytest
# looks them up; the redundant alias is the re-export form ruff accepts.
from _accept_helpers import bench as bench
from _accept_helpers import fake_glossary as fake_glossary
from _accept_helpers import fake_pipeline as fake_pipeline
from _accept_helpers import isolated_home as isolated_home

import vemoizer.pipeline as pipeline_module
from vemoizer import batch_preset


def test_preview_run_marks_drafts_and_prints_every_metric(
    tmp_path: Path, bench: Path, fake_pipeline: Any, fake_glossary: list[Any]
) -> None:
    res = _invoke(tmp_path, bench=bench)

    assert res.exit_code == 0, res.output
    lines = [line for line in res.output.splitlines() if line.startswith("E")]
    assert len(lines) == 2
    assert lines[0].startswith("E1\t(draft)")
    assert lines[1].startswith("E2\t(draft)")
    # Draft text is drift only: no WER and no flag keys (issue #243).
    unreviewed_keys = [
        key
        for key in _metric_keys()
        if key not in ("wer", "unflagged_error_s", "flag_precision", "flag_recall")
    ]
    for key in [*unreviewed_keys, "text_drift"]:
        assert f"{key}=" in res.output
    assert "wer=" not in res.output
    # Scoring sees no terms through load_glossary(None) any more (that was
    # the #229 bug: it returned [] and zeroed glossary_insertions); the
    # terms come from the layered options instead.
    assert fake_glossary == []


def test_check_refuses_draft_references(
    tmp_path: Path, bench: Path, fake_pipeline: Any, fake_glossary: list[Any]
) -> None:
    res = _invoke(tmp_path, "--check", bench=bench)

    assert res.exit_code == 2
    assert "unreviewed" in res.output
    assert "E1" in res.output and "E2" in res.output
    # Anonymised only: the real excerpt ids never leave the bench.
    assert MIX_ID not in res.output and TRACK_ID not in res.output
    assert fake_pipeline.calls == []  # refused before any decode


def test_update_baseline_then_check_passes(
    tmp_path: Path, bench: Path, fake_pipeline: Any, fake_glossary: list[Any]
) -> None:
    _mark_reviewed(bench, MIX_ID, TRACK_ID)
    baseline = tmp_path / "accept_baseline.json"

    written = _invoke(tmp_path, "--update-baseline", bench=bench, baseline=baseline)
    assert written.exit_code == 0, written.output
    assert "baseline updated" in written.output

    data = json.loads(baseline.read_text(encoding="utf-8"))
    assert sorted(data["excerpts"]) == ["E1", "E2"]
    assert set(data) == {"fingerprint", "tolerance", "excerpts", "aggregate"}
    # Numbers (and "no signal" nulls) only — never text of any kind.
    assert all(
        v is None or isinstance(v, (int, float)) for v in data["aggregate"].values()
    )
    assert len(data["fingerprint"]) == 12

    checked = _invoke(tmp_path, "--check", bench=bench, baseline=baseline)
    assert checked.exit_code == 0, checked.output
    assert "accept check passed" in checked.output


def test_check_reports_regressed_key(
    tmp_path: Path, bench: Path, fake_pipeline: Any, fake_glossary: list[Any]
) -> None:
    _mark_reviewed(bench, MIX_ID, TRACK_ID)
    baseline = tmp_path / "accept_baseline.json"
    fake_pipeline.state["default"] = _result(f"{REF_TOKEN} eilen toinen osa")
    assert (
        _invoke(tmp_path, "--update-baseline", bench=bench, baseline=baseline).exit_code
        == 0
    )

    # A worse transcript on the same audio: the gate must name the key.
    fake_pipeline.state["default"] = _result(f"{REF_TOKEN} aivan väärin tässä")
    res = _invoke(tmp_path, "--check", bench=bench, baseline=baseline)

    assert res.exit_code == 2
    assert "wer" in res.output
    assert "regression" in res.output


def test_check_fingerprint_drift_exits_2(
    tmp_path: Path, bench: Path, fake_pipeline: Any, fake_glossary: list[Any]
) -> None:
    _mark_reviewed(bench, MIX_ID, TRACK_ID)
    baseline = tmp_path / "accept_baseline.json"
    assert (
        _invoke(tmp_path, "--update-baseline", bench=bench, baseline=baseline).exit_code
        == 0
    )

    ref = bench / "reference" / f"{MIX_ID}.json"
    obj = json.loads(ref.read_text(encoding="utf-8"))
    obj["reviewed_on"] = "2026-03-03"
    ref.write_text(json.dumps(obj), encoding="utf-8")

    decoded_before = len(fake_pipeline.calls)
    res = _invoke(tmp_path, "--check", bench=bench, baseline=baseline)
    assert res.exit_code == 2
    assert "fingerprint" in res.output
    # Drift is refused before a single decode.
    assert len(fake_pipeline.calls) == decoded_before


def test_check_missing_baseline_exits_2(
    tmp_path: Path, bench: Path, fake_pipeline: Any, fake_glossary: list[Any]
) -> None:
    _mark_reviewed(bench, MIX_ID, TRACK_ID)
    res = _invoke(
        tmp_path, "--check", bench=bench, baseline=tmp_path / "nope" / "absent.json"
    )

    assert res.exit_code == 2
    assert "baseline" in res.output


def test_failed_excerpt_is_reported_and_gated(
    tmp_path: Path, bench: Path, fake_pipeline: Any, fake_glossary: list[Any]
) -> None:
    _mark_reviewed(bench, MIX_ID, TRACK_ID)
    baseline = tmp_path / "accept_baseline.json"
    assert (
        _invoke(tmp_path, "--update-baseline", bench=bench, baseline=baseline).exit_code
        == 0
    )
    fake_pipeline.state["per_excerpt"][MIX_ID] = _result("", error="ingest failed")

    preview = _invoke(tmp_path, bench=bench, baseline=baseline)
    assert preview.exit_code == 0
    assert "E1\tFAILED" in preview.output

    res = _invoke(tmp_path, "--check", bench=bench, baseline=baseline)
    assert res.exit_code == 2
    assert "E1" in res.output


def test_output_leaks_no_ids_names_or_text(
    tmp_path: Path, bench: Path, fake_pipeline: Any, fake_glossary: list[Any]
) -> None:
    _mark_reviewed(bench, MIX_ID, TRACK_ID)
    fake_pipeline.state["default"] = _result(f"{HYP_TOKEN} {REF_TOKEN}")

    res = _invoke(tmp_path, bench=bench)

    assert res.exit_code == 0, res.output
    for secret in (MIX_ID, TRACK_ID, REF_TOKEN, HYP_TOKEN, TITLE_TOKEN, "Maija"):
        assert secret not in res.output
    assert str(bench) not in res.output


def test_merge_notices_go_to_stderr_not_the_report(
    tmp_path: Path,
    bench: Path,
    fake_pipeline: Any,
    fake_glossary: list[Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The shared composition prints dropped-term notices to stderr, so
    stdout stays the anonymised report and nothing else.
    """
    notice = "glossary: dropped 'vanha' (token budget 150 exceeded)"
    # Patched where it is looked up: batch_preset owns the composition.
    monkeypatch.setattr(
        batch_preset, "merge", lambda *a, **kw: ([GLOSSARY_TERM], {}, [notice])
    )

    res = _invoke(tmp_path, bench=bench)

    assert res.exit_code == 0, res.output
    assert notice in res.stderr
    # The report itself (stdout) carries no notices: only rows and the
    # fingerprint.
    for line in res.stdout.splitlines():
        assert line.startswith(("E1", "E2", "aggregate", "fingerprint"))


def test_aggregate_means_and_sums(
    tmp_path: Path, bench: Path, fake_pipeline: Any, fake_glossary: list[Any]
) -> None:
    fake_pipeline.state["per_excerpt"][TRACK_ID] = _result(
        HYP_TOKEN,
        segments=[
            {"start": 0.0, "end": 3.0, "text": "a " * 10, "compression_ratio": 3.0}
        ],
    )
    _mark_reviewed(bench, MIX_ID, TRACK_ID)  # wer is reported for reviewed text

    res = _invoke(tmp_path, bench=bench)

    line = next(
        line for line in res.output.splitlines() if line.startswith("aggregate")
    )
    values = dict(part.split("=", 1) for part in line.split("\t")[1:])
    # Mean of the rates over the two excerpts, sum of the seconds.
    assert values["wer"] == "1.0000"
    assert values["loop_seconds"] == "3.0000"
    assert values["split_count"] == "0"


def test_missing_bench_exits_2(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fake_pipeline: Any
) -> None:
    monkeypatch.chdir(tmp_path)  # no .vemoizer/bench above the cwd
    res = _invoke(tmp_path)

    assert res.exit_code == 2
    assert "bench" in res.output


def test_bench_discovered_from_project_layer(
    tmp_path: Path, bench: Path, fake_pipeline: Any, fake_glossary: list[Any]
) -> None:
    res = _invoke(tmp_path)  # cwd is <tmp>/proj/sub (set by the bench fixture)

    assert res.exit_code == 0, res.output
    assert str(bench) not in res.output  # found, and never printed
    assert len(fake_pipeline.calls) == 2


def test_bundle_error_exits_2_with_message(
    tmp_path: Path, bench: Path, fake_pipeline: Any, fake_glossary: list[Any]
) -> None:
    (bench / "audio" / MIX_ID / "mix.flac").write_bytes(b"tampered")

    res = _invoke(tmp_path, bench=bench)

    assert res.exit_code == 2
    assert "sha256" in res.output


# ---------------------------------------------------------------------------
# The report is stdout-clean (no absolute paths), the gate fails closed, and
# a crashing excerpt is a FAILED row rather than a traceback.
# ---------------------------------------------------------------------------


def test_stdout_carries_no_absolute_or_home_path(
    tmp_path: Path,
    bench: Path,
    fake_pipeline: Any,
    fake_glossary: list[Any],
    isolated_home: Path,
) -> None:
    """The pasted-into-a-PR-body report must never leak the owner's paths.

    The baseline is printed exactly as the owner passed it (the relative
    default reads best in a PR body anyway); the absolute path exists only
    for IO, because the run happens with the cwd at the bench's project.
    """
    _mark_reviewed(bench, MIX_ID, TRACK_ID)

    res = _invoke(
        tmp_path,
        "--update-baseline",
        bench=bench,
        baseline=Path("accept_baseline.json"),
    )

    assert res.exit_code == 0, res.output
    assert "baseline updated: accept_baseline.json" in res.stdout
    # The file itself still landed (relative to the project layer's cwd).
    assert (bench.parent.parent / "sub" / "accept_baseline.json").is_file()
    assert str(tmp_path) not in res.stdout
    assert str(isolated_home) not in res.stdout
    assert str(bench) not in res.stdout


def test_update_baseline_refuses_when_an_excerpt_failed(
    tmp_path: Path, bench: Path, fake_pipeline: Any, fake_glossary: list[Any]
) -> None:
    """A failed excerpt must never be silently missing from a new baseline.

    Writing a baseline over a run that had a crash would record only the
    excerpts that survived — the gate would quietly shrink, so the run
    writes nothing at all, not even over an existing file.
    """
    _mark_reviewed(bench, MIX_ID, TRACK_ID)
    missing = tmp_path / "never-written.json"
    fake_pipeline.state["per_excerpt"][MIX_ID] = _result("", error="ingest failed")

    res = _invoke(tmp_path, "--update-baseline", bench=bench, baseline=missing)

    assert res.exit_code == 2
    assert "baseline" in res.stderr
    assert not missing.exists()

    # An existing baseline keeps its old numbers, untouched.
    good = tmp_path / "accept_baseline.json"
    fake_pipeline.state["per_excerpt"] = {}
    assert (
        _invoke(tmp_path, "--update-baseline", bench=bench, baseline=good).exit_code
        == 0
    )
    before = good.read_text(encoding="utf-8")
    fake_pipeline.state["per_excerpt"][TRACK_ID] = _result("", error="ingest failed")

    res = _invoke(tmp_path, "--update-baseline", bench=bench, baseline=good)

    assert res.exit_code == 2
    assert "E2" in res.stderr
    assert good.read_text(encoding="utf-8") == before


def test_pipeline_exception_counts_as_a_regression_under_check(
    tmp_path: Path,
    bench: Path,
    fake_pipeline: Any,
    fake_glossary: list[Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _mark_reviewed(bench, MIX_ID, TRACK_ID)
    baseline = tmp_path / "accept_baseline.json"
    assert (
        _invoke(tmp_path, "--update-baseline", bench=bench, baseline=baseline).exit_code
        == 0
    )

    def boom(path: str | Path, **kwargs: Any) -> dict[str, Any]:
        raise RuntimeError("decode exploded")

    monkeypatch.setattr(pipeline_module, "transcribe_file", boom)

    res = _invoke(tmp_path, "--check", bench=bench, baseline=baseline)

    assert res.exit_code == 2
    assert "E1" in res.stderr and "E2" in res.stderr
    assert "Traceback" not in res.stderr


def _baseline_dict(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_good_baseline(tmp_path: Path, bench: Path, fake_pipeline: Any) -> Path:
    _mark_reviewed(bench, MIX_ID, TRACK_ID)
    baseline = tmp_path / "accept_baseline.json"
    assert (
        _invoke(tmp_path, "--update-baseline", bench=bench, baseline=baseline).exit_code
        == 0
    )
    return baseline


def test_check_fails_closed_on_a_missing_aggregate_block(
    tmp_path: Path, bench: Path, fake_pipeline: Any, fake_glossary: list[Any]
) -> None:
    """A baseline without the aggregate block is not a pass (issue #229 review).

    The old code compared an empty dict against the run's aggregate, and
    every "no signal" skip made the gate agree with itself: a baseline that
    measures nothing must be an error, never a green check.
    """
    baseline = _write_good_baseline(tmp_path, bench, fake_pipeline)
    data = _baseline_dict(baseline)
    del data["aggregate"]
    baseline.write_text(json.dumps(data), encoding="utf-8")

    res = _invoke(tmp_path, "--check", bench=bench, baseline=baseline)

    assert res.exit_code == 2
    assert "aggregate" in res.stderr
    assert "update-baseline" in res.stderr
    assert "accept check passed" not in res.output


def test_check_fails_closed_on_a_missing_metric_key(
    tmp_path: Path, bench: Path, fake_pipeline: Any, fake_glossary: list[Any]
) -> None:
    baseline = _write_good_baseline(tmp_path, bench, fake_pipeline)
    data = _baseline_dict(baseline)
    del data["excerpts"]["E1"]["wer"]
    baseline.write_text(json.dumps(data), encoding="utf-8")

    res = _invoke(tmp_path, "--check", bench=bench, baseline=baseline)

    assert res.exit_code == 2
    assert "wer" in res.stderr
    assert "E1" in res.stderr
    assert "accept check passed" not in res.output


def test_check_fails_closed_on_a_missing_excerpt(
    tmp_path: Path, bench: Path, fake_pipeline: Any, fake_glossary: list[Any]
) -> None:
    baseline = _write_good_baseline(tmp_path, bench, fake_pipeline)
    data = _baseline_dict(baseline)
    del data["excerpts"]["E2"]
    baseline.write_text(json.dumps(data), encoding="utf-8")

    res = _invoke(tmp_path, "--check", bench=bench, baseline=baseline)

    assert res.exit_code == 2
    assert "E2" in res.stderr
    assert "accept check passed" not in res.output


def test_bench_outside_a_project_layer_exits_2(
    tmp_path: Path, bench: Path, fake_pipeline: Any, fake_glossary: list[Any]
) -> None:
    """The bench must live at ``<project>/.vemoizer/bench`` (issue #229 review).

    ``project_root`` is derived as ``bench.parent.parent``, which is only the
    project that drives the glossary and the config when the two directory
    names are exactly ``.vemoizer/bench``; anywhere else the run would
    measure some other layer, so it is refused before any decode.
    """
    misplaced = tmp_path / "data" / "bench"
    misplaced.parent.mkdir(parents=True)
    bench.rename(misplaced)

    res = _invoke(tmp_path, bench=misplaced)

    assert res.exit_code == 2
    assert ".vemoizer/bench" in res.stderr
    assert fake_pipeline.calls == []

    wrongly_named = tmp_path / "data2" / "notbench"
    wrongly_named.parent.mkdir(parents=True)
    misplaced.rename(wrongly_named)
    res = _invoke(tmp_path, bench=wrongly_named / "notbench")
    assert res.exit_code == 2
    assert ".vemoizer/bench" in res.stderr


# --- --import-answers (issue #243) ------------------------------------------


def _answers_file(tmp_path: Path) -> Path:
    obj = {
        "version": 1,
        "answered_on": "2026-11-04",
        "questions": [
            {
                "id": f"{MIX_ID}/s1",
                "excerpt": MIX_ID,
                "kind": "speaker",
                "start": 0.1,
                "end": 0.6,
                "options": ["Anna", "Ben"],
            }
        ],
        "answers": {f"{MIX_ID}/s1": "Someone else"},
    }
    path = tmp_path / "bench-answers.json"
    path.write_text(json.dumps(obj), encoding="utf-8")
    return path


def _ref_obj(bench: Path, excerpt_id: str) -> dict[str, Any]:
    path = bench / "reference" / f"{excerpt_id}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def test_import_answers_writes_probes_and_skips_the_bench_run(
    bench: Path, tmp_path: Path, fake_pipeline: Any
) -> None:
    path = _answers_file(tmp_path)
    res = _invoke(tmp_path, "--import-answers", str(path), "--by", "Janni", bench=bench)
    assert res.exit_code == 0, res.output
    assert "E1\tprobes=1\tterm_probes=0" in res.output
    assert fake_pipeline.calls == []  # the bench never ran
    ref = _ref_obj(bench, MIX_ID)
    assert ref["probes"][0]["answer"] == "other"
    assert ref["probes_reviewed_by"] == "Janni"


def test_import_answers_owner_comes_from_the_project_config(
    bench: Path, tmp_path: Path
) -> None:
    project = bench.parent.parent
    (project / ".vemoizer" / "config.toml").write_text(
        '[meeting]\nowner = "Maija"\n', encoding="utf-8"
    )
    res = _invoke(
        tmp_path, "--import-answers", str(_answers_file(tmp_path)), bench=bench
    )
    assert res.exit_code == 0, res.output
    assert _ref_obj(bench, MIX_ID)["probes_reviewed_by"] == "Maija"


def test_import_answers_without_an_owner_exits_2(bench: Path, tmp_path: Path) -> None:
    res = _invoke(
        tmp_path, "--import-answers", str(_answers_file(tmp_path)), bench=bench
    )
    assert res.exit_code == 2
    assert "no owner" in res.output
    assert _ref_obj(bench, MIX_ID)["probes"] == []


def test_import_answers_unknown_excerpt_exits_2_and_writes_nothing(
    bench: Path, tmp_path: Path
) -> None:
    path = _answers_file(tmp_path)
    obj = json.loads(path.read_text(encoding="utf-8"))
    obj["questions"][0]["excerpt"] = "no-such-excerpt"
    path.write_text(json.dumps(obj), encoding="utf-8")
    before = (bench / "reference" / f"{MIX_ID}.json").read_bytes()
    res = _invoke(tmp_path, "--import-answers", str(path), "--by", "Janni", bench=bench)
    assert res.exit_code == 2
    assert "unknown excerpt" in res.output
    assert (bench / "reference" / f"{MIX_ID}.json").read_bytes() == before


def test_import_answers_refuses_the_gate_flags(bench: Path, tmp_path: Path) -> None:
    res = _invoke(
        tmp_path,
        "--import-answers",
        str(_answers_file(tmp_path)),
        "--by",
        "Janni",
        "--check",
        bench=bench,
    )
    assert res.exit_code == 2
    assert "runs alone" in res.output
