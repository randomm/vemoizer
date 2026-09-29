"""Layered glossary: home + project ``.vemoizer/glossary.txt`` layers (issue #82).

``load_layers`` reads the two layer files (``~/.vemoizer/glossary.txt`` and
the nearest ``./.vemoizer/glossary.txt`` walking up from CWD); ``merge`` is
pure — it combines the two parsed layers and applies the M0 token budget,
returning ``(terms, corrections, notices)``.

Merge rules
-----------
- Prompt terms: project first, case-insensitive dedupe keeping project
  spelling; the M0 token budget is applied after the merge and the
  lowest-priority (earliest-listed) terms are dropped first, each named
  in ``notices``.
- Correction pairs: union of both layers; for the same ``wrong``-side key
  the project's ``right`` side wins.  Correction lines never contribute
  prompt terms.

``merge`` never prints or logs.  ``batch.py`` prints the notices to stderr.
"""

from __future__ import annotations

from pathlib import Path

from vemoizer.glossary import (
    GLOSSARY_PROMPT_TOKEN_BUDGET,
    Tokenizer,
    _as_token_ids,
    _read_lines,
)

# ---------------------------------------------------------------------------
# Layer I/O
# ---------------------------------------------------------------------------


def _home_glossary_path() -> Path:
    return Path.home() / ".vemoizer" / "glossary.txt"


def _nearest_project_glossary(start: Path | None = None) -> Path | None:
    """The nearest ``./.vemoizer/glossary.txt`` walking up from *start*.

    Walks from *start* (or CWD) up to the filesystem root.  Returns ``None``
    if no project glossary exists.  Symlink loops are prevented by tracking
    the real-path of each directory visited.
    """
    current = start or Path.cwd()
    seen: set[str] = set()
    while True:
        real = str(current.resolve())
        if real in seen:
            return None
        seen.add(real)
        candidate = current / ".vemoizer" / "glossary.txt"
        if candidate.is_file():
            return candidate
        parent = current.parent
        if parent == current:  # reached filesystem root
            return None
        current = parent


def _parse_glossary(
    path: Path | str | None,
) -> tuple[list[str], dict[str, str]]:
    """Read one glossary file into ``(terms, corrections)``.

    ``terms`` preserves ``@``-prefixed LLM-only terms (no stripping —
    ``glossary_prompt`` and the pipeline handle that).  ``corrections`` maps
    wrong-side key → right-side value.
    """
    if path is None:
        return [], {}
    terms: list[str] = []
    corrections: dict[str, str] = {}
    for line in _read_lines(path):
        if "=>" in line:
            wrong, _, right = line.partition("=>")
            wrong, right = wrong.strip(), right.strip()
            if wrong and right:
                corrections[wrong] = right
        else:
            terms.append(line)
    return terms, corrections


def load_layers(
    home_path: Path | None = None,
    project_path: Path | None = None,
) -> tuple[
    list[str], dict[str, str], list[str], dict[str, str]
]:
    """Read both glossary layers and return ``(home_terms, home_corrections,
    project_terms, project_corrections)``.

    ``home_path`` defaults to ``~/.vemoizer/glossary.txt``.
    ``project_path`` defaults to the nearest ``./.vemoizer/glossary.txt``
    walking up from CWD (or ``None`` if none found).

    Fail-open: a missing or unreadable file contributes ``([], {})``.
    """
    hp = home_path if home_path is not None else _home_glossary_path()
    pp = project_path if project_path is not None else _nearest_project_glossary()
    ht, hc = _parse_glossary(hp)
    pt, pc = _parse_glossary(pp)
    return ht, hc, pt, pc


# ---------------------------------------------------------------------------
# Pure merge
# ---------------------------------------------------------------------------


def _token_cost(tokenizer: Tokenizer, text: str) -> int:
    return len(_as_token_ids(tokenizer, text))


def _budget_terms(
    terms: list[str],
    tokenizer: Tokenizer | None,
    budget: int = GLOSSARY_PROMPT_TOKEN_BUDGET,
) -> tuple[list[str], list[str]]:
    """Apply the M0 token budget to *terms*, returning ``(kept, dropped)``.

    Priority follows the ``glossary_prompt`` contract: the LAST-listed term
    is highest-priority; lowest-priority (earliest-listed) terms are dropped
    first.  ``@``-prefixed LLM-only terms are excluded from the budget
    entirely — they never enter the whisper prompt and therefore occupy no
    tokens.

    ``tokenizer=None`` means budget cannot be computed (fail-open): all
    non-``@`` terms are kept and no drop notice is generated.
    """
    asr_terms = [t for t in terms if not t.startswith("@")]
    llm_only_terms = [t for t in terms if t.startswith("@")]

    if tokenizer is None:
        # Fail-open: no tokenizer means we can't measure cost; keep everything.
        return list(terms), []

    # Import locally to avoid circular import at module level; glossary.py
    # does not import glossary_layers.py so a top-level import would also
    # work, but keeping the cost helpers local makes the budget logic self-
    # contained.
    prefix_cost = _token_cost(tokenizer, "Sanasto: ")
    period_cost = _token_cost(tokenizer, ".")
    sep_cost = _token_cost(tokenizer, ", ")
    overhead = prefix_cost + period_cost
    if overhead > budget:
        # Degenerate: even an empty glossary overflows — ship nothing.
        return llm_only_terms, list(asr_terms)

    remaining = budget - overhead
    kept: list[str] = []
    dropped: list[str] = []
    used = 0
    for term in reversed(asr_terms):
        cost = _token_cost(tokenizer, term)
        additional = cost if not kept else cost + sep_cost
        if used + additional > remaining:
            dropped.append(term)
            continue
        kept.append(term)
        used += additional
    kept.reverse()
    dropped.reverse()
    return kept + llm_only_terms, dropped


def merge(
    project_terms: list[str],
    project_corrections: dict[str, str],
    home_terms: list[str],
    home_corrections: dict[str, str],
    tokenizer: Tokenizer | None = None,
    budget: int = GLOSSARY_PROMPT_TOKEN_BUDGET,
) -> tuple[list[str], dict[str, str], list[str]]:
    """Merge the project and home glossary layers (pure — no I/O, no prints).

    Parameters
    ----------
    project_terms, project_corrections:
        Parsed from the project (``./.vemozer``) layer.
    home_terms, home_corrections:
        Parsed from the home (``~/.vemozer``) layer.
    tokenizer:
        The mlx-whisper tokenizer; ``None`` disables budgeting (fail-open).
    budget:
        Token budget for the merged whisper prompt (default: M0 budget).

    Returns
    -------
    (merged_terms, merged_corrections, notices)

    * ``merged_terms`` — case-insensitive dedupe keeping the project
      spelling; project terms first; token budget applied after merge;
      ``@``-prefixed LLM-only terms are preserved.
    * ``merged_corrections`` — union of both layers; for the same
      wrong-side key the project's right side wins.
    * ``notices`` — one string per dropped term (budget overflows), naming
      the dropped terms in lowest-priority order.  Empty when nothing is
      dropped.  ``merge`` itself never prints; ``batch.py`` prints them.
    """
    # --- case-insensitive dedupe, project first, keep project spelling ---
    seen: set[str] = set()
    deduped: list[str] = []
    for term in project_terms + home_terms:
        key = term.lower()
        if key not in seen:
            seen.add(key)
            deduped.append(term)

    # --- token budget applied after merge ---
    kept, dropped = _budget_terms(deduped, tokenizer, budget)

    # --- correction union, project right-side wins ---
    merged_corrections: dict[str, str] = {}
    merged_corrections.update(home_corrections)
    merged_corrections.update(project_corrections)

    # --- notices: one per dropped term, lowest-priority first ---
    notices: list[str] = [
        f"glossary: dropped '{t}' (token budget {budget} exceeded)"
        for t in dropped
    ]

    return kept, merged_corrections, notices
