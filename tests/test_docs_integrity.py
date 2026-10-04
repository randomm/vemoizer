"""Docs-integrity guard for ``docs/pipeline-spec.md`` (issue #131).

Three drift checks, all read-only against the spec file:

- no numbered heading carries an ``a``/``b`` suffix (``### 10a.`` etc.) —
  the spec's sections must be sequentially numbered;
- the retired "Planned subcommands" section is gone;
- every ``--flag`` named in the spec's ``transcribe`` flag table exists on
  the real Typer command (parsed via Typer/click params — no subprocess).

Unit test: no pipeline, no network, no CLI invocation.
"""

from __future__ import annotations

import re
from pathlib import Path

import typer
from typer.models import CommandInfo

from vemoizer.cli import transcribe

SPEC_PATH = Path(__file__).resolve().parents[1] / "docs" / "pipeline-spec.md"


def _spec_text() -> str:
    return SPEC_PATH.read_text(encoding="utf-8")


def _transcribe_flag_options() -> set[str]:
    """Option strings (e.g. ``--no-group``) of the real ``transcribe`` command.

    Parsed from the Typer command's click params — no subprocess, no CLI run.
    """
    command_info = CommandInfo(callback=transcribe)
    typer_command = typer.main.get_command_from_info(
        command_info, pretty_exceptions_short=False, rich_markup_mode=None
    )
    opts: set[str] = set()
    for param in typer_command.params:
        if hasattr(param, "opts"):
            opts.update(opt for opt in param.opts if opt.startswith("--"))
    return opts


def _transcribe_table_flags(spec_text: str) -> set[str]:
    """Flag option strings (``--flag``) named in the transcribe flag table."""
    start = spec_text.index("### `vemoizer transcribe")
    section = spec_text[start:]
    next_heading = re.search(r"^### ", section[1:], re.MULTILINE)
    section = section[: 1 + next_heading.start()] if next_heading else section
    flags: set[str] = set()
    for line in section.splitlines():
        if not line.startswith("|"):
            continue
        # flag-table rows start with the flag itself, backticked
        if re.match(r"^\|\s*`--", line):
            flags.update(re.findall(r"--[\w-]+", line.split("|")[1]))
    return flags


def test_spec_has_no_letter_suffix_section_headings() -> None:
    bad = re.findall(r"^#{2,4}\s+\d+[a-z]\.\s", _spec_text(), re.MULTILINE)
    assert not bad, f"spec headings with a/b suffixes must be renumbered: {bad}"


def test_spec_has_no_planned_subcommands_section() -> None:
    assert "Planned subcommands" not in _spec_text()


def test_spec_transcribe_flags_exist_on_real_command() -> None:
    opts = _transcribe_flag_options()
    missing = sorted(_transcribe_table_flags(_spec_text()) - opts)
    assert not missing, (
        f"spec transcribe table lists flags not on the command: {missing}"
    )
