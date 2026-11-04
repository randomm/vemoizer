"""Tests for importing the owner's quiz answers into the bench (issue #243).

The bench is the synthetic two-excerpt fixture from ``_accept_helpers``: no
audio is cut for real, no model runs, and nothing private is read.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
from _accept_helpers import MIX_ID, TRACK_ID
from _accept_helpers import bench as bench

from vemoizer import accept_bundle
from vemoizer.accept_answers import AnswersError, import_answers


def _questions() -> list[dict[str, Any]]:
    return [
        {
            "id": f"{MIX_ID}/s1",
            "excerpt": MIX_ID,
            "kind": "speaker",
            "start": 0.1,
            "end": 0.6,
            "options": ["Anna", "Ben"],
        },
        {
            "id": f"{MIX_ID}/s2",
            "excerpt": MIX_ID,
            "kind": "speaker",
            "start": 1.1,
            "end": 1.6,
            "options": ["Anna", "Ben"],
        },
        {
            "id": f"{MIX_ID}/t1",
            "excerpt": MIX_ID,
            "kind": "term",
            "start": 0.2,
            "end": 0.9,
            "prompt": "Which name is said at the start of this clip?",
            "options": ["IWS", "IBS", "IBC", "Something else"],
        },
        {
            "id": f"{TRACK_ID}/s1",
            "excerpt": TRACK_ID,
            "kind": "speaker",
            "start": 0.0,
            "end": 0.9,
            "options": ["Anna", "Ben"],
        },
    ]


def _write_answers(
    path: Path,
    answers: dict[str, Any],
    questions: list[dict[str, Any]] | None = None,
    *,
    version: int = 1,
) -> Path:
    obj = {
        "version": version,
        "answered_on": "2026-11-04",
        "questions": _questions() if questions is None else questions,
        "answers": answers,
    }
    path.write_text(json.dumps(obj), encoding="utf-8")
    return path


def _ref(bench_dir: Path, excerpt_id: str) -> dict[str, Any]:
    path = bench_dir / "reference" / f"{excerpt_id}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _snapshot(bench_dir: Path) -> dict[str, str]:
    return {
        str(p.relative_to(bench_dir)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(bench_dir.rglob("*"))
        if p.is_file()
    }


def test_import_writes_normalised_probes_and_counts_them(
    bench: Path, tmp_path: Path
) -> None:
    answers = {
        f"{MIX_ID}/s1": "Someone else",
        f"{MIX_ID}/s2": "Two at once",
        f"{MIX_ID}/t1": "Something else",
        f"{TRACK_ID}/s1": "Ben",
    }
    path = _write_answers(tmp_path / "bench-answers.json", answers)
    counts = import_answers(bench, path, "Janni")
    assert counts == {MIX_ID: (2, 1), TRACK_ID: (1, 0)}

    ref = _ref(bench, MIX_ID)
    assert [p["answer"] for p in ref["probes"]] == ["other", "overlap"]
    assert ref["probes"][0] == {
        "id": f"{MIX_ID}/s1",
        "start": 0.1,
        "end": 0.6,
        "answer": "other",
    }
    assert ref["term_probes"] == [
        {
            "id": f"{MIX_ID}/t1",
            "start": 0.2,
            "end": 0.9,
            "options": ["IWS", "IBS", "IBC", "Something else"],
            "answer": "other",
        }
    ]
    assert ref["probes_reviewed_by"] == "Janni"
    assert ref["probes_reviewed_on"] == "2026-11-04"
    assert [p["answer"] for p in _ref(bench, TRACK_ID)["probes"]] == ["Ben"]


def test_cant_tell_normalises_to_unclear_for_both_kinds(
    bench: Path, tmp_path: Path
) -> None:
    answers = {f"{MIX_ID}/s1": "Can't tell", f"{MIX_ID}/t1": "Can't tell"}
    import_answers(bench, _write_answers(tmp_path / "a.json", answers), "Janni")
    ref = _ref(bench, MIX_ID)
    assert ref["probes"][0]["answer"] == "unclear"
    assert ref["term_probes"][0]["answer"] == "unclear"


def test_import_keeps_every_other_reference_key(bench: Path, tmp_path: Path) -> None:
    before = _ref(bench, MIX_ID)
    path = _write_answers(tmp_path / "a.json", {f"{MIX_ID}/s1": "Anna"})
    import_answers(bench, path, "Janni")
    after = _ref(bench, MIX_ID)
    for key in ("id", "segments", "title_expect", "reviewed_by", "reviewed_on"):
        assert after[key] == before[key]
    assert after["reviewed_by"] is None  # the quiz does not review the text


def test_unanswered_questions_are_skipped(bench: Path, tmp_path: Path) -> None:
    path = _write_answers(
        tmp_path / "a.json", {f"{MIX_ID}/s2": "Ben", f"{MIX_ID}/t1": None}
    )
    counts = import_answers(bench, path, "Janni")
    assert counts[MIX_ID] == (1, 0)
    assert [p["id"] for p in _ref(bench, MIX_ID)["probes"]] == [f"{MIX_ID}/s2"]


def test_reimport_replaces_probes(bench: Path, tmp_path: Path) -> None:
    first = _write_answers(
        tmp_path / "1.json", {f"{MIX_ID}/s1": "Anna", f"{MIX_ID}/s2": "Ben"}
    )
    import_answers(bench, first, "Janni")
    second = _write_answers(tmp_path / "2.json", {f"{MIX_ID}/s2": "Anna"})
    import_answers(bench, second, "Janni")
    probes = _ref(bench, MIX_ID)["probes"]
    assert [(p["id"], p["answer"]) for p in probes] == [(f"{MIX_ID}/s2", "Anna")]


def test_round_trip_loads_and_marks_the_quiz_reviewed(
    bench: Path, tmp_path: Path
) -> None:
    path = _write_answers(tmp_path / "a.json", {f"{MIX_ID}/s1": "Anna"})
    import_answers(bench, path, "Janni")
    _, refs = accept_bundle.load_bundle(bench)
    assert refs[MIX_ID].probes[0].answer == "Anna"
    assert refs[MIX_ID].term_probes == ()
    assert accept_bundle.unreviewed(refs) == [TRACK_ID]


def test_unknown_excerpt_exits_before_any_write(bench: Path, tmp_path: Path) -> None:
    questions = _questions() + [
        {
            "id": "nope/s1",
            "excerpt": "no-such-excerpt",
            "kind": "speaker",
            "start": 0.0,
            "end": 0.5,
            "options": ["Anna"],
        }
    ]
    before = _snapshot(bench)
    path = _write_answers(tmp_path / "a.json", {f"{MIX_ID}/s1": "Anna"}, questions)
    with pytest.raises(AnswersError, match="unknown excerpt"):
        import_answers(bench, path, "Janni")
    assert _snapshot(bench) == before


def test_answer_outside_the_options_writes_nothing(bench: Path, tmp_path: Path) -> None:
    before = _snapshot(bench)
    path = _write_answers(
        tmp_path / "a.json", {f"{MIX_ID}/s1": "Anna", f"{MIX_ID}/t1": "IXX"}
    )
    with pytest.raises(AnswersError, match="not an option"):
        import_answers(bench, path, "Janni")
    assert _snapshot(bench) == before


def test_answer_for_an_unknown_question_is_rejected(
    bench: Path, tmp_path: Path
) -> None:
    path = _write_answers(tmp_path / "a.json", {"ghost/s9": "Anna"})
    with pytest.raises(AnswersError, match="unknown questions"):
        import_answers(bench, path, "Janni")


def test_wrong_answers_version_is_rejected(bench: Path, tmp_path: Path) -> None:
    path = _write_answers(tmp_path / "a.json", {}, version=2)
    with pytest.raises(AnswersError, match="version"):
        import_answers(bench, path, "Janni")


def test_unreadable_answers_file_is_a_value_error(bench: Path, tmp_path: Path) -> None:
    path = tmp_path / "broken.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(ValueError, match="cannot read JSON"):
        import_answers(bench, path, "Janni")


@pytest.mark.parametrize(
    "mutate",
    [
        lambda d: d.pop("answered_on"),
        lambda d: d.update(questions="nope"),
        lambda d: d.update(answers=["nope"]),
        lambda d: d["questions"].append("not-an-object"),
        lambda d: d["questions"].append({"excerpt": MIX_ID, "id": 3}),
        lambda d: d["questions"][0].update(kind="guess"),
        lambda d: d["questions"][0].update(options="Anna"),
    ],
)
def test_malformed_answers_file_writes_nothing(
    bench: Path, tmp_path: Path, mutate: Any
) -> None:
    path = _write_answers(tmp_path / "a.json", {f"{MIX_ID}/s1": "Anna"})
    obj = json.loads(path.read_text(encoding="utf-8"))
    mutate(obj)
    path.write_text(json.dumps(obj), encoding="utf-8")
    before = _snapshot(bench)
    with pytest.raises(ValueError):
        import_answers(bench, path, "Janni")
    assert _snapshot(bench) == before


def test_duplicate_question_id_is_rejected(bench: Path, tmp_path: Path) -> None:
    """PR #244 review: one answer must never fan out to a second window."""
    questions = _questions()
    questions.append({**questions[0], "start": 1.1, "end": 1.6})
    path = _write_answers(tmp_path / "a.json", {}, questions)
    with pytest.raises(AnswersError, match="duplicate question id"):
        import_answers(bench, path, "Janni")
