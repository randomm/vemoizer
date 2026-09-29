"""glossary_layers: load_layers I/O + pure merge (issue #82).

Covers:
- load_layers reading both ~/.vemozer/glossary.txt and ./.vemozer/glossary.txt
- merge() case-insensitive dedupe keeping project spelling
- correction union with project right-side winning on same wrong-side key
- token-budget drop order and notices
- @-line exclusion from prompt terms (LLM-only terms preserved, budget-skipped)
- --glossary single-file override bypassing merge (load_layers not called)
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pytest

from vemoizer.glossary import GLOSSARY_PROMPT_TOKEN_BUDGET
from vemoizer.glossary_layers import load_layers, merge

# ---------------------------------------------------------------------------
# Fake tokenizer (same shape as tests/test_glossary.py::FakeTokenizer)
# ---------------------------------------------------------------------------


class FakeTokenizer:
    """Deterministic tokenizer: each space-separated token costs 1 (separator)
    plus 1 per non-space character.  Mirrors tests/test_glossary.py."""

    def encode(self, text: str) -> Sequence[int]:
        ids: list[int] = []
        for i, tok in enumerate(text.split()):
            if i > 0:
                ids.append(0)  # separator
            ids.append(len(tok))  # 1 per char
        return ids


# ---------------------------------------------------------------------------
# load_layers — I/O
# ---------------------------------------------------------------------------


class TestLoadLayers:
    """load_layers reads both layer files, fail-open on missing."""

    def test_reads_project_and_home(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        home_g = tmp_path / "home" / ".vemozer" / "glossary.txt"
        home_g.parent.mkdir(parents=True)
        home_g.write_text(
            "Nordea\nFlagship-hanke\nBlacksit => Flagship\n", encoding="utf-8"
        )

        proj_g = tmp_path / "proj" / ".vemozer" / "glossary.txt"
        proj_g.parent.mkdir(parents=True)
        proj_g.write_text("Nordea\nRiihimäki\nNewport => Nyborg\n", encoding="utf-8")

        monkeypatch.chdir(tmp_path / "proj")
        home_terms, home_corr, proj_terms, proj_corr = load_layers(
            home_path=home_g,
            project_path=proj_g,
        )
        assert home_terms == ["Nordea", "Flagship-hanke"]
        assert home_corr == {"Blacksit": "Flagship"}
        assert proj_terms == ["Nordea", "Riihimäki"]
        assert proj_corr == {"Newport": "Nyborg"}

    def test_missing_home_file_is_empty(
        self,
        tmp_path: Path,
    ) -> None:
        proj_g = tmp_path / "glossary.txt"
        proj_g.write_text("Nordea\n", encoding="utf-8")
        ht, hc, pt, pc = load_layers(
            home_path=tmp_path / "nope.txt",
            project_path=proj_g,
        )
        assert ht == [] and hc == {}
        assert pt == ["Nordea"] and pc == {}

    def test_missing_project_file_is_empty(
        self,
        tmp_path: Path,
    ) -> None:
        home_g = tmp_path / "glossary.txt"
        home_g.write_text("Nordea\n", encoding="utf-8")
        ht, hc, pt, pc = load_layers(
            home_path=home_g,
            project_path=tmp_path / "nope.txt",
        )
        assert ht == ["Nordea"] and hc == {}
        assert pt == [] and pc == {}

    def test_at_lines_preserved_in_terms(
        self,
        tmp_path: Path,
    ) -> None:
        """@-prefixed lines stay in the terms list (LLM-only, no stripping)."""
        g = tmp_path / "glossary.txt"
        g.write_text("@Jukka Loikkanen\n@Maija\nNordea\n", encoding="utf-8")
        ht, _, pt, pc = load_layers(home_path=tmp_path / "x", project_path=g)
        assert pt == ["@Jukka Loikkanen", "@Maija", "Nordea"]
        assert pc == {}


# ---------------------------------------------------------------------------
# merge — pure, no I/O, no printing
# ---------------------------------------------------------------------------


class TestMerge:
    """merge() combines project + home layers, returns (terms, corrections, notices)."""

    def test_dedupe_case_insensitive_keeps_project_spelling(self) -> None:
        terms, corr, notices = merge(
            project_terms=["Nordea", "flagship-hanke"],
            project_corrections={},
            home_terms=["Nordea", "Flagship-hanke", "Riihimäki"],
            home_corrections={},
        )
        # "Nordea" deduped, project spelling kept.
        # "flagship-hanke" vs "Flagship-hanke" → project spelling kept.
        assert terms == ["Nordea", "flagship-hanke", "Riihimäki"]
        assert corr == {}
        assert notices == []

    def test_project_terms_first_in_order(self) -> None:
        terms, _, _ = merge(
            project_terms=["Alpha", "Beta"],
            project_corrections={},
            home_terms=["Gamma", "Delta"],
            home_corrections={},
        )
        assert terms == ["Alpha", "Beta", "Gamma", "Delta"]

    def test_correction_union_project_right_wins(self) -> None:
        _, corr, _ = merge(
            project_terms=[],
            project_corrections={"Blacksit": "Flagship", "Newport": "Nyborg"},
            home_terms=[],
            home_corrections={"Blacksit": "OldName", "Riihimääri": "Riihimäki"},
        )
        assert corr == {
            "Blacksit": "Flagship",  # project right-side wins
            "Newport": "Nyborg",
            "Riihimääri": "Riihimäki",
        }

    def test_correction_lines_never_contribute_prompt_terms(self) -> None:
        """A glossary with only => lines has no prompt terms."""
        terms, corr, _ = merge(
            project_terms=[],
            project_corrections={"A": "B"},
            home_terms=[],
            home_corrections={"C": "D"},
        )
        assert terms == []
        assert corr == {"A": "B", "C": "D"}

    def test_at_terms_preserved_and_deduped(self) -> None:
        """@-terms are preserved (LLM-only) and deduped like other terms."""
        terms, _, _ = merge(
            project_terms=["@Jukka", "Nordea"],
            project_corrections={},
            home_terms=["@jukka", "Riihimäki"],
            home_corrections={},
        )
        # "@Jukka" and "@jukka" dedupe case-insensitively; project spelling kept.
        assert terms == ["@Jukka", "Nordea", "Riihimäki"]

    def test_budget_drops_lowest_priority_first(self) -> None:
        """Lowest-priority (earliest-listed) terms are dropped first; notices
        name each dropped term.

        FakeTokenizer counts: overhead = 2 tokens (prefix + period),
        each term = 1 token, each separator = 1 token.
        Budget=5 → remaining=3 → Termi0 (lowest priority) is dropped.
        """
        tok = FakeTokenizer()
        terms, _, notices = merge(
            project_terms=["Termi0", "Termi1", "Termi2"],
            project_corrections={},
            home_terms=[],
            home_corrections={},
            tokenizer=tok,
            budget=5,  # small enough to force drops
        )
        # Termi2 (last = highest priority) and Termi1 kept; Termi0 dropped first.
        assert "Termi2" in terms
        assert "Termi1" in terms
        assert "Termi0" not in terms
        # At least one notice naming the dropped term.
        assert any("Termi0" in n for n in notices)

    def test_budget_none_fails_open_keeps_all(self) -> None:
        """tokenizer=None disables budgeting; all terms kept, no notices."""
        terms, _, notices = merge(
            project_terms=["Termi0", "Termi1", "Termi2"],
            project_corrections={},
            home_terms=[],
            home_corrections={},
            tokenizer=None,
        )
        assert terms == ["Termi0", "Termi1", "Termi2"]
        assert notices == []

    def test_at_terms_skip_budget(self) -> None:
        """@-terms never consume budget tokens; they pass through unchanged."""
        tok = FakeTokenizer()
        terms, _, notices = merge(
            project_terms=["@VeryLongLLMOnlyTerm", "Nordea"],
            project_corrections={},
            home_terms=[],
            home_corrections={},
            tokenizer=tok,
            budget=20,  # small budget
        )
        # @-term survives regardless of budget (LLM-only, no prompt cost).
        assert "@VeryLongLLMOnlyTerm" in terms
        assert "Nordea" in terms

    def test_empty_layers(self) -> None:
        terms, corr, notices = merge([], {}, [], {})
        assert terms == []
        assert corr == {}
        assert notices == []

    def test_full_budget_no_drops(self) -> None:
        """Small glossary within budget: no drops, no notices."""
        tok = FakeTokenizer()
        terms, _, notices = merge(
            project_terms=["Nordea", "Riihimäki"],
            project_corrections={},
            home_terms=[],
            home_corrections={},
            tokenizer=tok,
            budget=GLOSSARY_PROMPT_TOKEN_BUDGET,
        )
        assert terms == ["Nordea", "Riihimäki"]
        assert notices == []

    def test_merge_is_pure_no_print(self, capsys: pytest.CaptureFixture) -> None:
        """merge itself never prints or logs (notices go to the return value)."""
        merge(
            project_terms=["A"],
            project_corrections={},
            home_terms=["B"],
            home_corrections={},
            tokenizer=FakeTokenizer(),
            budget=10,
        )
        captured = capsys.readouterr()
        assert captured.out == ""
        assert captured.err == ""


# ---------------------------------------------------------------------------
# --glossary override bypasses merge (load_layers not called)
# ---------------------------------------------------------------------------


class TestGlossaryOverrideBypass:
    """When --glossary is given explicitly, load_layers is not called;
    the single file replaces both layers entirely.  This is tested via
    the fact that load_layers with only a project_path (no home) returns
    the single-file contents, which batch.py passes directly."""

    def test_single_file_replaces_both_layers(
        self,
        tmp_path: Path,
    ) -> None:
        g = tmp_path / "override.txt"
        g.write_text("Nordea\n@Jukka\nBlacksit => Flagship\n", encoding="utf-8")
        ht, hc, pt, pc = load_layers(
            home_path=tmp_path / "missing.txt",  # missing → empty
            project_path=g,
        )
        assert ht == []
        assert hc == {}
        assert pt == ["Nordea", "@Jukka"]
        assert pc == {"Blacksit": "Flagship"}
        # No home layer read; project file used directly.
        terms, corr, _ = merge(pt, pc, ht, hc)
        assert terms == ["Nordea", "@Jukka"]
        assert corr == {"Blacksit": "Flagship"}
