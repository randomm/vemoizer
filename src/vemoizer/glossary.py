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
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

#: Whisper's prompt window is small (~224 tokens); the glossary must not
#: overflow it. Terms beyond the budget are dropped with a warning.
_MAX_PROMPT_CHARS = 1000


def _read_lines(path: str | Path | None) -> list[str]:
    if path is None:
        return []
    try:
        text = Path(path).read_text(encoding="utf-8")
    except OSError:
        logger.warning("glossary not readable: %s (continuing without)", path)
        return []
    return [
        line.strip()
        for line in text.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]


def load_glossary(path: str | Path | None) -> list[str]:
    """Prompt terms from the glossary file; ``[]`` when absent (fail-open).

    ``wrong => right`` correction lines contribute their *right* side (the
    canonical spelling is a good recognition seed); the wrong side must
    never appear in a prompt.
    """
    terms: list[str] = []
    for line in _read_lines(path):
        if "=>" in line:
            _wrong, _, right = line.partition("=>")
            right = right.strip()
            if right:
                terms.append(right)
        else:
            terms.append(line)
    if terms:
        logger.info("glossary: %d terms from %s", len(terms), path)
    return terms


def load_corrections(path: str | Path | None) -> dict[str, str]:
    """``wrong => right`` pairs from the glossary file (fail-open).

    Known garble forms ("Blacksit" for Flagship, "Newport" for Nyborg)
    survived the LLM repair pass; a known-bad -> canonical mapping is
    deterministic and must not depend on a model's judgment.
    """
    corrections: dict[str, str] = {}
    for line in _read_lines(path):
        if "=>" not in line:
            continue
        wrong, _, right = line.partition("=>")
        wrong, right = wrong.strip(), right.strip()
        if wrong and right:
            corrections[wrong] = right
    return corrections


def apply_corrections(
    paragraphs: list[dict[str, Any]], corrections: dict[str, str]
) -> list[dict[str, Any]]:
    """Replace known garble forms in paragraph texts (pure, metadata kept).

    Whole-word matches only (a correction must never fire inside another
    word), case-insensitive on the match, canonical spelling as the
    replacement. Compounds like ``Blacksit-hankkeiksi`` are covered because
    the hyphen is a word boundary.
    """
    if not corrections:
        return list(paragraphs)
    patterns = [
        (re.compile(rf"\b{re.escape(wrong)}\b", re.IGNORECASE), right)
        for wrong, right in corrections.items()
    ]
    out: list[dict[str, Any]] = []
    for para in paragraphs:
        text = str(para.get("text", ""))
        for pattern, right in patterns:
            text = pattern.sub(right, text)
        out.append({**para, "text": text})
    return out


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
