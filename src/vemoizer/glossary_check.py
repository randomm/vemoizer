"""``vemoizer glossary check`` — glossary file audit (issue #79, M7).

Audits one glossary file (or the merged ``.vemoizer`` layers when no
file is given) and prints:

- the merged term and correction layers (explicit file, or the
  ``glossary_layers`` merge of project + home layers);
- the exact whisper prompt string from :func:`glossary.glossary_prompt`
  with its token count (whisper tokenizer from M0) and any dropped
  terms (over the 150-token budget);
- conflicting spellings (same wrong side mapped to two different
  rights, detected from the RAW lines before dict collapse);
- chained pairs (a=>b, b=>c) and self-referential pairs (x=>x);
- ``\\b``-unmatchable wrong sides (the compiled whole-word/prefix
  pattern matches no probe text, so the pair can never fire);
- ignored/malformed lines (no ``=>``, empty side, duplicate wrong side).

Fail-open: a missing file reports empty (never an error).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from . import glossary as _glossary_mod
from .glossary import glossary_prompt

#: Separator between the wrong and right sides of a correction line.
_ARROW = "=>"


@dataclass
class CheckReport:
    """The output of one ``glossary check`` run."""

    terms: list[str] = field(default_factory=list)
    corrections: dict[str, str] = field(default_factory=dict)
    prompt: str | None = None
    prompt_tokens: int | None = None
    dropped_terms: list[str] = field(default_factory=list)
    conflicts: list[str] = field(default_factory=list)
    chained: list[str] = field(default_factory=list)
    self_ref: list[str] = field(default_factory=list)
    unmatchable: list[str] = field(default_factory=list)
    ignored: list[str] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return not (
            self.terms
            or self.corrections
            or self.conflicts
            or self.chained
            or self.self_ref
            or self.unmatchable
            or self.ignored
            or self.dropped_terms
            or self.prompt is not None
        )


def _read_raw_lines(path: Path | str | None) -> list[str]:
    """The non-comment, non-blank lines of *path*; ``[]`` when absent."""
    if path is None:
        return []
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError:
        return []
    return [
        line.strip()
        for line in text.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def _parse_raw(
    lines: list[str],
) -> tuple[list[str], list[tuple[str, str]], list[str]]:
    """(terms, raw correction pairs, ignored lines) from raw *lines*.

    The raw pairs list keeps EVERY ``wrong => right`` pair in file order
    (duplicates included) so the conflict/chained/self-ref detection can
    see what the collapsed :func:`glossary.load_corrections` dict hides.
    """
    terms: list[str] = []
    pairs: list[tuple[str, str]] = []
    ignored: list[str] = []
    for line in lines:
        if _ARROW not in line:
            terms.append(line)
            continue
        wrong, _, right = line.partition(_ARROW)
        wrong, right = wrong.strip(), right.strip()
        if wrong and right:
            pairs.append((wrong, right))
        else:
            ignored.append(line)
    return terms, pairs, ignored


def _compile_whole_word(wrong: str) -> re.Pattern[str]:
    """The whole-word pattern ``_compile_corrections`` uses for *wrong*."""
    if wrong.endswith("*"):
        stem = re.escape(wrong[:-1])
        return re.compile(rf"\b{stem}(\w*)", re.IGNORECASE)
    return re.compile(rf"\b{re.escape(wrong)}\b", re.IGNORECASE)


def _is_unmatchable(wrong: str) -> bool:
    """True when the compiled pattern matches no probe string.

    Probes: the bare wrong side, and the same string embedded in word and
    hyphen contexts.  A wrong side starting with a non-word char that is
    not a digit (e.g. ``-foo``, ``*foo*``) cannot match at a ``\\b``
    boundary and is unmatchable (the pair can never fire).  Digits ARE
    word chars in Python's ``re`` (``\\w``), so a numeric wrong side such
    as ``3.14`` CAN match its own ``\\b3\\.14\\b`` pattern and is not
    unmatchable.
    """
    pattern = _compile_whole_word(wrong)
    probes = (wrong, f" {wrong} ", f"x-{wrong} y", f"x-{wrong}", f"-{wrong}x")
    return all(pattern.search(p) is None for p in probes)


def _find_chained_and_self_ref(
    pairs: list[tuple[str, str]],
) -> tuple[list[str], list[str]]:
    """(chained, self_ref) from the raw *pairs* (dict-collapse would hide
    conflicts; the graph traversal needs every pair)."""
    wrong_set = {w for w, _ in pairs}
    chained: list[str] = []
    self_ref: list[str] = []
    seen_chained: set[str] = set()
    seen_self: set[str] = set()
    for wrong, right in pairs:
        if wrong == right and wrong not in seen_self:
            self_ref.append(wrong)
            seen_self.add(wrong)
        if right in wrong_set and right != wrong:
            label = f"{wrong} => {right} (right side is also a wrong side)"
            if label not in seen_chained:
                chained.append(label)
                seen_chained.add(label)
    return chained, self_ref


def _find_conflicts(pairs: list[tuple[str, str]]) -> list[str]:
    """Wrong sides mapped to two or more different rights (from raw lines)."""
    by_wrong: dict[str, set[str]] = {}
    for wrong, right in pairs:
        by_wrong.setdefault(wrong, set()).add(right)
    conflicts: list[str] = []
    for wrong in sorted(by_wrong):
        rights = by_wrong[wrong]
        if len(rights) > 1:
            conflicts.append(
                f"{wrong} => {' / '.join(sorted(rights))} (conflicting rights)"
            )
    return conflicts


def _whisper_token_count(prompt: str) -> int | None:
    """Token count of *prompt* via the whisper tokenizer (M0); ``None``
    when the tokenizer is unavailable (fail-open — count reported as
    ``n/a`` by the caller, the prompt still prints)."""
    tokenizer = _glossary_mod._whisper_tokenizer()
    if tokenizer is None:
        return None
    try:
        ids = tokenizer.encode(prompt)
        return len(list(ids))
    except Exception:  # noqa: BLE001 - fail-open: no count, prompt still shown
        return None


def _dropped_terms(
    terms: list[str],
    prompt: str | None,
) -> list[str]:
    """Terms excluded from the built prompt (over budget or @-only).

    Uses substring containment, not a naive ``split(", ")``, because a
    glossary term may itself contain ``", "`` (e.g.
    ``Riihimäki, Nurmijärvi``).  A term is reported dropped only when the
    prompt body does not contain it at all.  ``glossary_prompt`` builds
    the body as ``prefix + ", ".join(kept) + "."`` in file order, so a
    term that survived budgeting always appears verbatim in the body.
    """
    if prompt is None:
        return []
    prompt_body = prompt
    if prompt_body.startswith("Sanasto: "):
        prompt_body = prompt_body[len("Sanasto: ") :]
    if prompt_body.endswith("."):
        prompt_body = prompt_body[:-1]
    return [t for t in terms if t not in prompt_body]


def _layer_lines() -> list[str]:
    """Raw lines of the merged ``.vemoizer`` layers (project first).

    Reads the two layers via ``glossary_layers.load_layers`` (the same seam
    the presets use) so the audit sees exactly what the pipeline's whisper
    prompt would; the ``merge`` semantics (case-insensitive dedupe, project
    wins for a conflicting wrong side) apply here too.  The raw-line list
    keeps the correction pairs in file order so conflict/chained detection
    still works.  A missing layer contributes nothing (fail-open).
    """
    from . import glossary_layers as gl

    home_terms, home_corr, project_terms, project_corr = gl.load_layers()
    _, merged_corr, _ = gl.merge(
        project_terms,
        project_corr,
        home_terms,
        home_corr,
        tokenizer=None,  # budgeting is applied by glossary_prompt, not here
    )
    lines: list[str] = []
    lines.extend(project_terms)
    lines.extend(home_terms)
    lines.extend(f"{w} => {r}" for w, r in merged_corr.items())
    return lines


def check_file(path: Path | str | None = None) -> CheckReport:
    """Audit *path*; ``None`` reads the merged ``.vemoizer`` layers.

    Fail-open: a missing or unreadable file reports empty (never an
    error).  The merged-layer path uses the two ``.vemoizer`` glossary
    files (project first); an explicit path replaces both layers.
    """
    lines = _read_raw_lines(path) if path is not None else _layer_lines()

    terms, pairs, ignored = _parse_raw(lines)
    corrections: dict[str, str] = {}
    for wrong, right in pairs:
        corrections[wrong] = right

    prompt = glossary_prompt(terms)
    prompt_tokens = _whisper_token_count(prompt) if prompt else None
    dropped = _dropped_terms(terms, prompt)
    conflicts = _find_conflicts(pairs)
    chained, self_ref = _find_chained_and_self_ref(pairs)
    unmatchable = [
        f"{wrong} => {right}" for wrong, right in pairs if _is_unmatchable(wrong)
    ]

    return CheckReport(
        terms=terms,
        corrections=corrections,
        prompt=prompt,
        prompt_tokens=prompt_tokens,
        dropped_terms=dropped,
        conflicts=conflicts,
        chained=chained,
        self_ref=self_ref,
        unmatchable=unmatchable,
        ignored=ignored,
    )


def render_report(report: CheckReport) -> str:
    """Human-readable multi-line summary of *report* (stdout-safe)."""
    lines: list[str] = []
    if report.is_empty:
        return "(glossary is empty — nothing to check)"

    if report.terms:
        lines.append("Terms:")
        lines.extend(f"  {t}" for t in report.terms)
    else:
        lines.append("Terms: (none)")

    if report.corrections:
        lines.append("Corrections:")
        for wrong, right in report.corrections.items():
            lines.append(f"  {wrong} => {right}")
    else:
        lines.append("Corrections: (none)")

    lines.append("")
    if report.prompt is not None:
        lines.append(f"Prompt: {report.prompt}")
        if report.prompt_tokens is not None:
            lines.append(f"Prompt token count: {report.prompt_tokens}")
        else:
            lines.append("Prompt token count: n/a (tokenizer unavailable)")
    else:
        lines.append("Prompt: (none — no non-@ terms or tokenizer unavailable)")
    if report.dropped_terms:
        lines.append("Dropped terms (over budget): " + ", ".join(report.dropped_terms))

    if report.conflicts:
        lines.append("Conflicting spellings:")
        lines.extend(f"  {c}" for c in report.conflicts)
    if report.chained:
        lines.append("Chained pairs (right side is also a wrong side):")
        lines.extend(f"  {c}" for c in report.chained)
    if report.self_ref:
        lines.append("Self-referential pairs:")
        lines.extend(f"  {x} => {x}" for x in report.self_ref)
    if report.unmatchable:
        lines.append("\\b-unmatchable wrong sides (pair can never fire):")
        lines.extend(f"  {u}" for u in report.unmatchable)
    if report.ignored:
        lines.append("Ignored/malformed lines:")
        lines.extend(f"  {i}" for i in report.ignored)
    return "\n".join(lines) if lines else "(glossary is empty — nothing to check)"
