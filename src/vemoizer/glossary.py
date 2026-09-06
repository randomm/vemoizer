"""User glossary: domain terms for ASR and LLM prompts (issue #71).

Real-meeting QA showed the systematic failure this fixes: whisper garbles
vocabulary it has no context for ("FLAG-sit" for Flagship-hanke,
"Riihimäärä" for Riihimäki, "Nurdea" for Nordea). Whisper's
``initial_prompt`` conditions every decoding window on preceding "text",
and seeding it with the user's terms measurably biases recognition toward
them; the same list grounds the notes and repair prompts so the LLM stages
spell names the way the user does.

The glossary is a plain text file, one term per line, ``#`` comments
allowed. Fail-open: missing or unreadable file = empty glossary.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

#: Whisper's prompt window is small (~224 tokens); the glossary must not
#: overflow it. Terms beyond the budget are dropped with a warning.
_MAX_PROMPT_CHARS = 1000


def load_glossary(path: str | Path | None) -> list[str]:
    """Terms from the glossary file; ``[]`` when absent (fail-open)."""
    if path is None:
        return []
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError:
        logger.warning("glossary not readable: %s (continuing without)", path)
        return []
    terms = []
    for line in text.splitlines():
        term = line.strip()
        if term and not term.startswith("#"):
            terms.append(term)
    if terms:
        logger.info("glossary: %d terms from %s", len(terms), path)
    return terms


def glossary_prompt(terms: list[str]) -> str | None:
    """The whisper ``initial_prompt`` seeding recognition with *terms*.

    Styled as preceding transcript text (that is what initial_prompt is),
    so the terms read as vocabulary already in use, not as an instruction.
    """
    if not terms:
        return None
    prefix = "Sanasto: "
    budget = _MAX_PROMPT_CHARS - len(prefix)
    kept: list[str] = []
    used = 0
    for term in terms:
        cost = len(term) + 2
        if used + cost > budget:
            logger.warning(
                "glossary: %d terms exceed the prompt budget; using first %d",
                len(terms),
                len(kept),
            )
            break
        kept.append(term)
        used += cost
    return prefix + ", ".join(kept) + "."
