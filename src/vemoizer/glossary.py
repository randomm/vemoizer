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
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol

logger = logging.getLogger(__name__)

# mlx-whisper keeps only the LAST 223 prompt tokens per decode window
# (n_text_ctx=448 in decoding.py: prompt_tokens[-(n_ctx // 2 - 1):]).
# The glossary prompt is capped at 150 tokens so the glossary itself can
# always fit in that 223-token keep-window; nothing here reserves tokens
# for the previous-text tail (the decoded text may occupy the rest). The
# glossary actually reaching every window is the per-window re-seeding in
# whisper_transcriber (each 30 s window is its own transcribe() call that
# re-applies the initial_prompt), not a tail reservation.
GLOSSARY_PROMPT_TOKEN_BUDGET = 150

_PROMPT_PREFIX = "Sanasto: "
_SEPARATOR = ", "


class Tokenizer(Protocol):
    """The minimal interface the glossary budgeting needs from a tokenizer.

    mlx-whisper's ``get_tokenizer(True)`` satisfies it (``encode`` returns
    an iterable of token ids); a named protocol keeps mistyped tokenizers
    (e.g. one without ``encode``) visible at the call site, not at the
    first budgeting call deep in :func:`glossary_prompt`.
    """

    def encode(self, text: str) -> Sequence[int]: ...


def _whisper_tokenizer() -> Any | None:
    """The mlx_whisper tokenizer, or ``None`` if unavailable (fail-open).

    ``get_tokenizer`` is ``lru_cache``-wrapped, so the encoding is fetched
    once per process; the per-term budgeting in ``glossary_prompt`` is cheap.
    """
    try:
        from mlx_whisper.tokenizer import get_tokenizer

        return get_tokenizer(True)
    except Exception:  # noqa: BLE001 - fail-open (no prompt, run continues)
        return None


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


def _as_token_ids(tokenizer: Tokenizer, text: str) -> list[int]:
    """Tokenize *text* to plain ids (``encode`` returns an id list)."""
    ids = tokenizer.encode(text)
    if not isinstance(ids, list):
        ids = list(ids)
    return [int(i) for i in ids]


def _token_cost(tokenizer: Tokenizer, text: str) -> int:
    """Tokens *text* occupies in the whisper tokenizer."""
    return len(_as_token_ids(tokenizer, text))


def load_glossary(path: str | Path | None) -> list[str]:
    """Prompt terms from the glossary file; ``[]`` when absent (fail-open).

    Returns every non-``=>`` line, INCLUDING ``@``-prefixed LLM-only terms
    (issue #76/#82), with the ``@`` prefix intact. The ``@`` filter lives
    in ONE place, ``glossary_prompt``: that is where the whisper
    ``initial_prompt`` is built, so @-terms never seed recognition at any
    budget, while the LLM stages (repair / notes, via
    ``llm_tail.apply_llm_tail``) receive the same list with the ``@``
    stripped at the LLM boundary. Both the direct ``--glossary`` path and
    the preset temp-file path flow through this loader, so one seam feeds
    the recognizer and the LLM stages.

    Only explicitly listed terms may seed the prompt; ``wrong => right``
    correction lines are post-recognition fixes and contribute nothing
    (prompt order is load-bearing, so a term kept out on purpose must not
    re-enter through a pair's right side).
    """
    return [line for line in _read_lines(path) if "=>" not in line]


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


def _compile_corrections(
    corrections: dict[str, str],
) -> list[tuple[re.Pattern[str], Any]]:
    """Compile correction pairs to (pattern, callable) tuples.

    Both branches use a callable replacement so that backslashes in the
    right side are treated as literal characters rather than re.sub group
    references or escape sequences. A trailing ``*`` on the wrong side
    matches Finnish inflections and compounds ("epit*" covers epittä and
    epitävaikutuksia): the matched stem becomes the canonical term, and a
    surviving suffix of at least three characters (leading vowel joints
    stripped) is re-attached with a hyphen ("EBITDA-vaikutuksia").
    """
    compiled: list[tuple[re.Pattern[str], Any]] = []
    for wrong, right in corrections.items():
        if wrong.endswith("*"):
            stem = re.escape(wrong[:-1])
            pattern = re.compile(rf"\b{stem}(\w*)", re.IGNORECASE)

            def _repl_prefix(match: re.Match[str], r: str = right) -> str:
                suffix = match.group(1).lstrip("aeiouyäö")
                if len(suffix) >= 3:
                    return f"{r}-{suffix}"
                return r

            compiled.append((pattern, _repl_prefix))
        else:

            def _repl_whole(match: re.Match[str], r: str = right) -> str:
                return r

            compiled.append(
                (re.compile(rf"\b{re.escape(wrong)}\b", re.IGNORECASE), _repl_whole)
            )
    return compiled


def _correct_text(text: str, compiled: list[tuple[re.Pattern[str], Any]]) -> str:
    for pattern, repl in compiled:
        text = pattern.sub(repl, text)
    return text


def apply_corrections(
    paragraphs: list[dict[str, Any]], corrections: dict[str, str]
) -> list[dict[str, Any]]:
    """Replace known garble forms in paragraph texts (pure, metadata kept).

    Whole-word matches (or ``wrong*`` prefix matches for inflections),
    case-insensitive, canonical spelling as the replacement. Compounds
    like ``Blacksit-hankkeiksi`` are covered because the hyphen is a word
    boundary.
    """
    if not corrections:
        return list(paragraphs)
    compiled = _compile_corrections(corrections)
    return [
        {**para, "text": _correct_text(str(para.get("text", "")), compiled)}
        for para in paragraphs
    ]


def apply_corrections_to_notes(
    notes: dict[str, Any], corrections: dict[str, str]
) -> dict[str, Any]:
    """Apply correction pairs over the notes strings (title, summary, lists).

    Garble the LLM copied from the transcript into deliverables dies here
    deterministically, whatever the model did.
    """
    if not corrections or not notes:
        return notes
    compiled = _compile_corrections(corrections)
    out = dict(notes)
    for key in ("title", "summary"):
        if isinstance(out.get(key), str):
            out[key] = _correct_text(str(out[key]), compiled)
    for key in ("key_points", "action_items"):
        if isinstance(out.get(key), list):
            out[key] = [_correct_text(str(item), compiled) for item in out[key]]
    return out


def glossary_prompt(terms: list[str], tokenizer: Tokenizer | None = None) -> str | None:
    """The whisper ``initial_prompt`` seeding recognition with *terms*.

    ``tokenizer`` is the mlx_whisper tokenizer (``get_tokenizer(True)``);
    when omitted it is loaded once via the fail-open ``_whisper_tokenizer``
    helper and the prompt is ``None`` if it cannot be obtained.

    Styled as preceding transcript text (that is what initial_prompt is),
    so the terms read as vocabulary already in use, not as an instruction.

    Budgeted in WHISPER TOKENS, not characters (issue #76): the string must
    fit in ``GLOSSARY_PROMPT_TOKEN_BUDGET`` tokens INCLUDING the
    ``"Sanasto: "`` prefix and the trailing period, so the glossary itself
    can always fit in the 223-token keep-window mlx-whisper retains. The
    previous text is not budgeted here; the per-window re-seeding in
    whisper_transcriber is what guarantees the glossary reaches every
    decoding window.

    Priority is inverted from file order: the LAST-listed term is the
    highest-priority and sits at the TAIL of the prompt string (the part
    mlx-whisper's ``prompt_tokens[-223:]`` slice keeps when truncating).
    Lowest-priority (earliest-listed) terms are dropped first — with a
    ``logger.warning`` naming them, never silently.

    ``@``-prefixed LLM-only terms (M2) are excluded at any budget: they
    must never enter the whisper prompt and cannot occupy the tail via
    the ``@`` path.
    """
    if tokenizer is None:
        tokenizer = _whisper_tokenizer()
        if tokenizer is None:
            logger.warning("whisper tokenizer unavailable; glossary prompt off")
            return None
    asr_terms = [t for t in terms if not t.startswith("@")]
    if not asr_terms:
        return None

    prefix_cost = _token_cost(tokenizer, _PROMPT_PREFIX)
    period_cost = _token_cost(tokenizer, ".")
    sep_cost = _token_cost(tokenizer, _SEPARATOR)
    overhead = prefix_cost + period_cost
    if overhead > GLOSSARY_PROMPT_TOKEN_BUDGET:
        # Degenerate: even an empty glossary overflows. Ship nothing.
        # Unreachable with the default whisper tokenizer (the prefix +
        # period fit with room to spare); a custom tokenizer that
        # assigns more than ~150 tokens to "Sanasto: " would hit
        # this, and the tokenizer type is logged so an operator can
        # quickly identify the cause of the silently-dropped glossary.
        logger.warning(
            "glossary: prefix overhead %d tokens exceeds budget %d; "
            "whisper prompt left empty (tokenizer: %s)",
            overhead,
            GLOSSARY_PROMPT_TOKEN_BUDGET,
            type(tokenizer).__name__,
        )
        return None

    # Build from the LAST term (highest priority) backwards, filling the
    # tail of the budget; drop the earliest-listed (lowest-priority) first.
    budget = GLOSSARY_PROMPT_TOKEN_BUDGET - overhead
    kept: list[str] = []
    dropped: list[str] = []
    used = 0
    for term in reversed(asr_terms):
        term_cost = _token_cost(tokenizer, term)
        # First kept term: cost = term_cost. Each additional term adds
        # term_cost + one separator.
        additional = term_cost if not kept else term_cost + sep_cost
        if used + additional > budget:
            dropped.append(term)
            continue
        kept.append(term)
        used += additional
    kept.reverse()  # prompt order = file order, highest-priority last

    if dropped:
        # The dropped list is in reverse file order: it leads with the
        # LOWEST-priority (earliest-listed) casualties, not the most
        # important ones, so say so explicitly.
        logger.warning(
            "glossary: %d terms dropped (prompt budget %d tokens exceeded); "
            "dropped, lowest priority first: %s",
            len(dropped),
            GLOSSARY_PROMPT_TOKEN_BUDGET,
            ", ".join(dropped[:10]) + ("…" if len(dropped) > 10 else ""),
        )
    return _PROMPT_PREFIX + _SEPARATOR.join(kept) + "."
