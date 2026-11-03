"""The ``vemoizer accept`` baseline file: fingerprint, aggregate, gate (#229).

Single responsibility — everything about the numbers-only baseline
``tests/fixtures/accept_baseline.json``: the bench fingerprint, the aggregate
row, and the direction-aware comparison. No audio, no models, no CLI: the
command that uses this lives in :mod:`vemoizer.accept_cli`.

The baseline holds numbers and ANONYMISED ids (``E1``..``Ek``) only, never
transcript text, reviewer names or paths, so it can be committed while the
bench itself stays out of git (epic #220: the bench is the owner's private
real-meeting audio).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

#: The one baseline scope that is not an excerpt.
AGGREGATE_SCOPE = "aggregate"

#: Keys where a HIGHER value is better. Every other metric key is read as
#: lower-is-better (WER, walls, unflagged error seconds, invented glossary
#: terms, split/merge counts, title misses and forbid hits).
HIGHER_IS_BETTER = frozenset(
    {"speaker_name_acc", "speaker_mapped_acc", "flag_precision", "flag_recall"}
)

#: Keys averaged (not summed) into the aggregate row: shares, not quantities.
RATE_KEYS = HIGHER_IS_BETTER | {"wer"}

#: Keys measured in SECONDS, so their tolerance is a wall-clock slack.
SECOND_KEYS = frozenset({"loop_seconds", "unflagged_error_s"})

#: Per-key-kind tolerances until #230 calibrates them over repeated runs.
#: Rates get the run-to-run spread of a diarization/LLM pipeline; seconds
#: get a whole-heartbeat of slack; a count is exact (one extra invented
#: glossary term IS the thing the gate is for).
RATE_TOLERANCE = 0.03
SECOND_TOLERANCE = 5.0
COUNT_TOLERANCE = 0.0

Value = float | int | None


def default_tolerance(key: str) -> float:
    """The gate slack for *key* when the baseline file carries no override."""
    if key in RATE_KEYS:
        return RATE_TOLERANCE
    if key in SECOND_KEYS:
        return SECOND_TOLERANCE
    return COUNT_TOLERANCE


def _default_tolerances(keys: Any) -> dict[str, float]:
    return {key: default_tolerance(key) for key in keys}


def fingerprint(bench_dir: Path, excerpt_ids: list[str]) -> str:
    """12 hex chars over the manifest bytes + every reference file's bytes.

    Reference files are hashed in sorted id order, so the digest depends on
    the bench CONTENT, not on dict or filesystem order. A mismatch means the
    bench drifted (an excerpt was re-frozen or a reference edited) and the
    recorded numbers no longer describe what is being measured.
    """
    manifest = Path(bench_dir) / "manifest.json"
    if not manifest.is_file():
        raise ValueError(f"bench manifest not found: {manifest}")
    digest = hashlib.sha256(manifest.read_bytes())
    for excerpt_id in sorted(excerpt_ids):
        reference = Path(bench_dir) / "reference" / f"{excerpt_id}.json"
        if not reference.is_file():
            raise ValueError(f"reference file not found for {excerpt_id}")
        digest.update(reference.read_bytes())
    return digest.hexdigest()[:12]


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """One aggregate row over the scored excerpts' metric dicts.

    Rates and accuracies are MEANS over the excerpts where they exist (None
    means "no signal", e.g. no flag to score — averaging it in as 0 would
    punish a run that flagged nothing); counts and seconds are SUMS, so a
    bench that grows cannot hide a regression by diluting it.
    """
    out: dict[str, Any] = {}
    for key in dict.fromkeys(k for row in rows for k in row):
        values = [row[key] for row in rows if row.get(key) is not None]
        if key in RATE_KEYS:
            out[key] = (sum(values) / len(values)) if values else None
        else:
            out[key] = sum(values)
    return out


def read_baseline(path: Path) -> dict[str, Any] | None:
    """The baseline dict, or ``None`` when absent or unreadable."""
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def write_baseline(
    path: Path,
    *,
    bench_fingerprint: str,
    excerpts: dict[str, dict[str, Any]],
    aggregate_row: dict[str, Any],
) -> None:
    """Write the numbers-only baseline, keeping a calibrated tolerance block.

    The tolerance is the operator's (or #230's) measurement of run-to-run
    noise; re-measuring the numbers must never silently reset it.
    """
    existing = read_baseline(path) or {}
    payload = {
        "fingerprint": bench_fingerprint,
        "tolerance": existing.get("tolerance") or _default_tolerances(aggregate_row),
        "excerpts": excerpts,
        "aggregate": aggregate_row,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(
        json.dumps(payload, indent=1, sort_keys=True) + "\n", encoding="utf-8"
    )


class BaselineDrift(Exception):
    """The baseline does not describe the current run (missing block/key/excerpt).

    Raised by :func:`compare` under ``--check``. A baseline the run cannot
    compare against is bench drift, never a pass: an absent block used to
    compare as "no signal" against every key and print ``accept check
    passed`` over a gate that measured nothing.
    """


@dataclass(frozen=True)
class Regression:
    """One metric that moved the wrong way beyond its tolerance."""

    scope: str  # "E1".."Ek" or "aggregate"
    key: str
    baseline: Value
    measured: Value

    def line(self) -> str:
        """The PR-body-safe one-liner naming this regression."""
        was = fmt(self.baseline)
        return (
            f"regression [{self.scope}] {self.key}: baseline {was} -> "
            f"measured {fmt(self.measured)}"
        )


def fmt(value: Value) -> str:
    """How one metric value reads in a row or a regression line.

    ``-`` for "no signal" (a null is never a number), 4 decimals for
    rates/seconds, plain digits for counts (never ``7.0``).
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return "-"
    return f"{value:.4f}" if isinstance(value, float) else str(value)


def _scope_sort_key(scope: str) -> tuple[int, int]:
    """Excerpt scopes numerically (E1, E2, … E10), the aggregate row last.

    Lexicographic order would print E10 between E1 and E2, which reads as a
    shuffled report in a PR body.
    """
    if scope.startswith("E") and scope[1:].isdigit():
        return (0, int(scope[1:]))
    return (1, 0)


def compare(
    measured: dict[str, dict[str, Any]], baseline: dict[str, Any]
) -> list[Regression]:
    """Direction-aware regressions of *measured* scopes against *baseline*.

    *measured* maps a scope (``E1``..``Ek``, plus :data:`AGGREGATE_SCOPE`) to
    its metric dict; *baseline* is the baseline file's dict. A scope, a key or
    the aggregate block that the current run has and the baseline lacks raises
    :class:`BaselineDrift` — that is bench drift, and it must be a deliberate
    ``--update-baseline``, never a silent pass (the same rule as the WER
    gate). A key whose value is ``None`` on either side is NOT compared:
    "no signal" is not a measurement to regress against, and flagging
    0.8 -> None (the flags simply were never raised) would make the gate flake
    on the pipeline's least stable stage.
    """
    tolerance = baseline.get("tolerance") or {}
    expected: dict[str, Any] = dict(baseline.get("excerpts") or {})
    aggregate = baseline.get("aggregate")
    if not isinstance(aggregate, dict):
        raise BaselineDrift(
            "baseline has no aggregate block — re-measure with "
            "--update-baseline in a dedicated commit"
        )
    expected[AGGREGATE_SCOPE] = aggregate
    regressions: list[Regression] = []
    for scope in sorted(measured, key=_scope_sort_key):
        row = expected.get(scope)
        if not isinstance(row, dict):
            raise BaselineDrift(
                f"baseline has no {scope} block — the bench or the baseline "
                "drifted; re-measure with --update-baseline in a dedicated commit"
            )
        for key, value in measured[scope].items():
            if key not in row:
                raise BaselineDrift(
                    f"baseline has no {key} for {scope} — re-measure with "
                    "--update-baseline in a dedicated commit"
                )
            reference = row[key]
            if value is None or reference is None:
                continue
            slack = float(tolerance.get(key, default_tolerance(key)))
            higher = key in HIGHER_IS_BETTER
            if (value < reference - slack) if higher else (value > reference + slack):
                regressions.append(Regression(scope, key, reference, value))
    return regressions
