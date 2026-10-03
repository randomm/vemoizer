"""Tests for ``vemoizer glossary check`` (issue #79, M7).

Unit tests only: no network, no model downloads.  The whisper tokenizer is
mocked (``glossary._whisper_tokenizer``); the layered-glossary seams are
monkeypatched to ``tmp_path`` files.
"""

from __future__ import annotations

from typer.testing import CliRunner

from vemoizer import glossary_check as gc
from vemoizer.cli import app

runner = CliRunner()


class _FakeTokenizer:
    """One token per character (deterministic, no mlx_whisper import)."""

    def encode(self, text: str):
        return list(range(len(text)))


def _patch_tokenizer(monkeypatch, tok=None) -> None:
    """Patch the tokenizer seam in glossary.glossary_prompt (its import path)."""
    import vemoizer.glossary as glossary

    monkeypatch.setattr(glossary, "_whisper_tokenizer", lambda: tok)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def test_glossary_subapp_is_registered() -> None:
    result = runner.invoke(app, ["glossary", "--help"])
    assert result.exit_code == 0
    assert "check" in result.stdout


def test_main_help_lists_glossary() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "glossary" in result.stdout


# ---------------------------------------------------------------------------
# check_file: basic parsing
# ---------------------------------------------------------------------------


def test_missing_file_reports_empty_fail_open(tmp_path) -> None:
    report = gc.check_file(tmp_path / "no-such-file.txt")
    assert report.is_empty
    assert report.terms == []
    assert report.corrections == {}
    assert report.prompt is None


def test_terms_and_corrections_parsed(tmp_path) -> None:
    f = tmp_path / "g.txt"
    f.write_text(
        "Flagship\nNurdea => Nordea\n# comment\nBlacksit* => Flagship\n",
        encoding="utf-8",
    )
    report = gc.check_file(f)
    assert report.terms == ["Flagship"]
    assert report.corrections == {"Nurdea": "Nordea", "Blacksit*": "Flagship"}


def test_malformed_lines_reported_ignored(tmp_path) -> None:
    f = tmp_path / "g.txt"
    f.write_text("foo =>\n => bar\nfoo => bar\n", encoding="utf-8")
    report = gc.check_file(f)
    assert sorted(report.ignored) == sorted(["foo =>", "=> bar"])
    assert report.corrections == {"foo": "bar"}


# ---------------------------------------------------------------------------
# Prompt + token count
# ---------------------------------------------------------------------------


def test_prompt_printed_with_token_count(tmp_path, monkeypatch) -> None:
    _patch_tokenizer(monkeypatch, _FakeTokenizer())
    f = tmp_path / "g.txt"
    f.write_text("hei\nmoi\n", encoding="utf-8")
    report = gc.check_file(f)
    assert report.prompt is not None
    # Neutral form: no label prefix, just terms + trailing period.
    assert not report.prompt.startswith("Sanasto: ")
    # "hei, moi." — 8 chars = 8 tokens for the fake tokenizer
    assert report.prompt == "hei, moi."
    assert report.prompt_tokens == len(report.prompt)
    text = gc.render_report(report)
    assert f"Prompt: {report.prompt}" in text
    assert "Prompt token count:" in text


def test_prompt_none_reports_no_prompt(tmp_path, monkeypatch) -> None:
    _patch_tokenizer(monkeypatch, _FakeTokenizer())
    f = tmp_path / "g.txt"
    f.write_text("@llm-only-term\n", encoding="utf-8")
    report = gc.check_file(f)
    # @-only terms never enter the whisper prompt → glossary_prompt None
    assert report.prompt is None
    text = gc.render_report(report)
    assert "Prompt: (none" in text


def test_dropped_terms_reported(tmp_path, monkeypatch) -> None:
    _patch_tokenizer(monkeypatch, _FakeTokenizer())
    f = tmp_path / "g.txt"
    # 200-char term alone exceeds the 150-token budget (fake tok: 1 tok/char
    # + prefix overhead) → dropped; a short term fits.
    f.write_text("a" * 200 + "\nshort\n", encoding="utf-8")
    report = gc.check_file(f)
    assert "a" * 200 in report.dropped_terms
    assert "short" in report.terms
    text = gc.render_report(report)
    assert "Dropped terms" in text


def test_comma_containing_term_not_false_positive(tmp_path, monkeypatch) -> None:
    """A glossary term containing ", " must not be reported as dropped.

    The old naive split(", ") would break "a, b" into "a" and "b",
    neither of which equals the full term, producing a false positive.
    """
    _patch_tokenizer(monkeypatch, _FakeTokenizer())
    f = tmp_path / "g.txt"
    f.write_text("a, b\nc\nd\n", encoding="utf-8")
    report = gc.check_file(f)
    assert report.prompt == "a, b, c, d."
    assert "a, b" not in report.dropped_terms
    assert "c" not in report.dropped_terms
    assert "d" not in report.dropped_terms


# ---------------------------------------------------------------------------
# Conflict / chained / self-ref / unmatchable
# ---------------------------------------------------------------------------


def test_conflicting_spellings_detected_from_raw_lines(tmp_path) -> None:
    f = tmp_path / "g.txt"
    f.write_text("foo => bar\nfoo => baz\n", encoding="utf-8")
    report = gc.check_file(f)
    assert len(report.conflicts) == 1
    assert "foo" in report.conflicts[0]
    text = gc.render_report(report)
    assert "Conflicting spellings" in text


def test_chained_pairs_detected(tmp_path) -> None:
    f = tmp_path / "g.txt"
    f.write_text("a => b\nb => c\n", encoding="utf-8")
    report = gc.check_file(f)
    assert any("a => b" in c for c in report.chained)
    text = gc.render_report(report)
    assert "Chained pairs" in text


def test_self_referential_pairs_detected(tmp_path) -> None:
    f = tmp_path / "g.txt"
    f.write_text("x => x\ny => z\n", encoding="utf-8")
    report = gc.check_file(f)
    assert report.self_ref == ["x"]
    text = gc.render_report(report)
    assert "Self-referential" in text


def test_unmatchable_wrong_side_detected(tmp_path) -> None:
    f = tmp_path / "g.txt"
    f.write_text("-foo => bar\n", encoding="utf-8")
    report = gc.check_file(f)
    assert len(report.unmatchable) == 1
    assert "-foo => bar" in report.unmatchable[0]
    text = gc.render_report(report)
    assert "unmatchable" in text


def test_numeric_wrong_side_not_unmatchable(tmp_path) -> None:
    """``3.14`` is matchable (digits are word chars in Python's ``re``),
    so it must NOT be reported as unmatchable."""
    f = tmp_path / "g.txt"
    f.write_text("3.14 => 3,14\n", encoding="utf-8")
    report = gc.check_file(f)
    assert report.unmatchable == []


def test_normal_pairs_not_flagged(tmp_path) -> None:
    f = tmp_path / "g.txt"
    f.write_text("Blacksit => Flagship\nfoo\n", encoding="utf-8")
    report = gc.check_file(f)
    assert report.conflicts == []
    assert report.chained == []
    assert report.self_ref == []
    assert report.unmatchable == []


def test_prefix_pattern_unmatchable_check(tmp_path) -> None:
    """A `wrong*` prefix whose stem starts with a non-word char is unmatchable."""
    f = tmp_path / "g.txt"
    f.write_text("*bad* => good\n", encoding="utf-8")
    report = gc.check_file(f)
    assert any("*bad*" in u for u in report.unmatchable)


def test_at_term_under_budget_not_reported_dropped(tmp_path, monkeypatch) -> None:
    """``@``-prefixed terms are LLM-only (M2): never budgeted into the
    whisper prompt, so an under-budget @-term must NOT appear in the
    'Dropped terms (over budget)' line."""
    _patch_tokenizer(monkeypatch, _FakeTokenizer())
    f = tmp_path / "g.txt"
    f.write_text("foo\n@bar\n", encoding="utf-8")
    report = gc.check_file(f)
    assert "@bar" not in report.dropped_terms
    assert "foo" not in report.dropped_terms
    text = gc.render_report(report)
    assert "Dropped terms" not in text


def test_genuinely_over_budget_plain_term_still_dropped(tmp_path, monkeypatch) -> None:
    """A plain (non-@) term over budget is still reported, alongside @-terms."""
    _patch_tokenizer(monkeypatch, _FakeTokenizer())
    f = tmp_path / "g.txt"
    f.write_text("a" * 200 + "\n@bar\n", encoding="utf-8")
    report = gc.check_file(f)
    assert "a" * 200 in report.dropped_terms
    assert "@bar" not in report.dropped_terms


def test_cli_check_non_utf8_file_clean_error(tmp_path, monkeypatch) -> None:
    """A non-UTF-8 glossary file: one clean stderr line, exit 1, no traceback."""
    _patch_tokenizer(monkeypatch, _FakeTokenizer())
    bad = tmp_path / "bad.txt"
    bad.write_bytes(b"\xff\xfe\x00bad")
    result = runner.invoke(app, ["glossary", "check", str(bad)], catch_exceptions=False)
    assert result.exit_code == 1
    assert "error: could not read" in result.stderr
    assert "UnicodeDecodeError" in result.stderr
    assert "Traceback" not in result.stderr
    assert "Traceback" not in result.stdout


# ---------------------------------------------------------------------------
# Merged layers (no file argument)
# ---------------------------------------------------------------------------


def test_no_file_reads_merged_layers(tmp_path, monkeypatch) -> None:
    project = tmp_path / "project" / ".vemoizer" / "glossary.txt"
    project.parent.mkdir(parents=True)
    project.write_text("projterm\npf => pr\n", encoding="utf-8")
    home = tmp_path / "home" / ".vemoizer" / "glossary.txt"
    home.parent.mkdir(parents=True)
    home.write_text("hometerm\nhf => hr\n", encoding="utf-8")

    from vemoizer import glossary_layers as gl

    monkeypatch.setattr(gl, "_nearest_project_glossary", lambda: project)
    monkeypatch.setattr(gl, "_home_glossary_path", lambda: home)
    monkeypatch.setenv("HOME", str(tmp_path / "home"))

    report = gc.check_file(None)
    assert "projterm" in report.terms
    assert "hometerm" in report.terms
    assert report.corrections == {"pf": "pr", "hf": "hr"}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def test_cli_check_missing_file_exits_zero(tmp_path, monkeypatch) -> None:
    _patch_tokenizer(monkeypatch, _FakeTokenizer())
    missing = tmp_path / "missing.txt"
    result = runner.invoke(app, ["glossary", "check", str(missing)])
    assert result.exit_code == 0
    assert "empty" in result.stdout


def test_cli_check_prints_prompt(tmp_path, monkeypatch) -> None:
    _patch_tokenizer(monkeypatch, _FakeTokenizer())
    f = tmp_path / "g.txt"
    f.write_text("hello\n", encoding="utf-8")
    result = runner.invoke(app, ["glossary", "check", str(f)])
    assert result.exit_code == 0
    assert "Prompt: hello." in result.stdout
    assert "Prompt token count:" in result.stdout
