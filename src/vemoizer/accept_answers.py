"""Import the owner's quiz answers into the acceptance bench (issue #243).

The owner answers a local quiz page; it downloads ``bench-answers.json``.
This module turns the answered questions into probes and writes them into
each excerpt's ``reference/<id>.json``, leaving every other key alone. The
page's answer labels are normalised here, once, so the metrics only ever see
``other`` / ``overlap`` / ``unclear`` or a name.

Validate-then-write: every excerpt's new reference is validated by the bench's
own loader before the first file is written, so a bad answer writes nothing.
Each file is written atomically; an I/O failure part-way can still leave the
earlier excerpts updated, and re-running the import repairs that.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from . import accept_bundle

ANSWERS_VERSION = 1  # the answers-file schema version the quiz page writes

# The quiz page's labels for the non-name answers, per question kind.
_SPEAKER_LABELS = {
    "Someone else": accept_bundle.OTHER,
    "Two at once": accept_bundle.OVERLAP,
    "Can't tell": accept_bundle.UNCLEAR,
}
_TERM_LABELS = {
    "Can't tell": accept_bundle.UNCLEAR,
    "Something else": accept_bundle.OTHER,
}


class AnswersError(ValueError):
    """The answers file cannot be imported; no bench file was written."""


def import_answers(
    bench_dir: Path, answers_path: Path, owner: str
) -> dict[str, tuple[int, int]]:
    """Write the answered probes into each excerpt's reference.

    Returns ``{excerpt id: (speaker probes, term probes)}`` for every excerpt
    the questions name. Re-importing replaces the probes; an unanswered
    question is skipped. Raises :class:`AnswersError` (or the bench's own
    ``ValueError`` / ``BundleError``) before anything is written.
    """
    data = _read_answers(answers_path)
    excerpts, _refs = accept_bundle.load_bundle(bench_dir)
    grouped = _group_answers(data, {e.id for e in excerpts})
    updated: dict[str, dict[str, Any]] = {}
    for excerpt_id, (speaker, term) in grouped.items():
        ref_path = _ref_path(bench_dir, excerpt_id)
        obj = accept_bundle.load_json(ref_path)
        obj["probes"] = speaker
        obj["term_probes"] = term
        # An excerpt with no answered question is not quiz-reviewed.
        answered = bool(speaker or term)
        obj["probes_reviewed_by"] = owner if answered else None
        obj["probes_reviewed_on"] = data["answered_on"] if answered else None
        accept_bundle.reference_from_json(obj, excerpt_id)  # validate, no write yet
        updated[excerpt_id] = obj
    for excerpt_id, obj in updated.items():
        accept_bundle.atomic_write_json(_ref_path(bench_dir, excerpt_id), obj)
    return {eid: (len(s), len(t)) for eid, (s, t) in grouped.items()}


def _ref_path(bench_dir: Path, excerpt_id: str) -> Path:
    return Path(bench_dir) / "reference" / f"{excerpt_id}.json"


def _read_answers(path: Path) -> dict[str, Any]:
    data = accept_bundle.load_json(Path(path))
    if not isinstance(data, dict) or data.get("version") != ANSWERS_VERSION:
        raise AnswersError(f"answers file must be version {ANSWERS_VERSION}")
    if not isinstance(data.get("answered_on"), str):
        raise AnswersError("answers file: answered_on must be a date string")
    if not isinstance(data.get("questions"), list) or not isinstance(
        data.get("answers"), dict
    ):
        raise AnswersError("answers file: questions must be a list, answers an object")
    return data


def _group_answers(
    data: dict[str, Any], known: set[str]
) -> dict[str, tuple[list[dict[str, Any]], list[dict[str, Any]]]]:
    """Per excerpt: (speaker probes, term probes) for every answered question."""
    answers: dict[str, Any] = data["answers"]
    grouped: dict[str, tuple[list[dict[str, Any]], list[dict[str, Any]]]] = {}
    question_ids: set[str] = set()
    for q in data["questions"]:
        if not isinstance(q, dict):
            raise AnswersError("question must be an object")
        qid, excerpt = q.get("id"), q.get("excerpt")
        if not isinstance(qid, str) or not isinstance(excerpt, str):
            raise AnswersError("question needs a string id and excerpt")
        if excerpt not in known:
            raise AnswersError(f"unknown excerpt id {excerpt!r} in question {qid!r}")
        if qid in question_ids:
            raise AnswersError(f"duplicate question id {qid!r}")
        question_ids.add(qid)
        speaker, term = grouped.setdefault(excerpt, ([], []))
        answer = answers.get(qid)
        if answer is None or answer == "":
            continue  # unanswered: skipped, not an error
        options = q.get("options")
        if not isinstance(answer, str) or not isinstance(options, list):
            raise AnswersError(
                f"question {qid!r}: needs an option list and a string answer"
            )
        base = {"id": qid, "start": q.get("start"), "end": q.get("end")}
        if q.get("kind") == "speaker":
            value = _normalise(qid, answer, options, _SPEAKER_LABELS)
            speaker.append({**base, "answer": value})
        elif q.get("kind") == "term":
            value = _normalise(qid, answer, options, _TERM_LABELS)
            term.append({**base, "options": options, "answer": value})
        else:
            raise AnswersError(f"question {qid!r}: kind must be speaker or term")
    stray = sorted(set(answers) - question_ids)
    if stray:
        raise AnswersError(f"answers for unknown questions: {', '.join(stray)}")
    return grouped


def _normalise(
    qid: str, answer: str, options: list[Any], labels: dict[str, str]
) -> str:
    if answer in labels:
        return labels[answer]
    if answer in options:
        return answer
    raise AnswersError(f"question {qid!r}: answer {answer!r} is not an option")
