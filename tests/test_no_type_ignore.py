"""Guard: no type-ignore comment suppressions may reappear (issue #115, #128, #141).

AGENTS.md forbids type-ignore comment suppressions; #115 brought the repo to
zero and #128 removed the last five (tests/test_whisper_transcriber.py).
#141 extends the guard to ty's ``ty``-prefixed ignore spelling across src,
tests and scripts. This test fails the suite if either spelling creeps back
in.

The forbidden tokens are assembled by concatenation so this file does not
contain the literal comments in a form a plain source scan would match
(``grep -rn`` over src, tests and scripts must keep returning zero matches).
"""

from pathlib import Path

# Each forbidden spelling is built by concatenation so the literal comment
# string never appears in this source file (which the guard scans too).
_FORBIDDEN_MYPY = "#" + " type: " + "ignore"
_FORBIDDEN_TY = "#" + " ty: " + "ignore"
_FORBIDDEN = (_FORBIDDEN_MYPY, _FORBIDDEN_TY)
_PY_ROOTS = ("src", "tests", "scripts")


def _scan(root: Path, tokens: tuple[str, ...], repo: Path) -> list[str]:
    """Offending ``rel:line: text`` entries for any ``*.py`` under *root*."""
    offenders: list[str] = []
    for path in sorted(root.rglob("*.py")):
        for line_no, line in enumerate(path.read_text().splitlines(), 1):
            if any(token in line for token in tokens):
                rel = path.relative_to(repo)
                offenders.append(f"{rel}:{line_no}: {line.strip()}")
    return offenders


def _find_offenders(repo: Path) -> list[str]:
    """All suppression-comment offenders across every scanned root.

    Missing roots are skipped so that partial repos (e.g. a temp dir that
    only contains ``scripts/``) are still scanned correctly.
    """
    offenders: list[str] = []
    for py_root in _PY_ROOTS:
        root_dir = repo / py_root
        if not root_dir.is_dir():
            continue
        offenders.extend(_scan(root_dir, _FORBIDDEN, repo))
    return offenders


def test_no_type_ignore_suppressions() -> None:
    repo = Path(__file__).resolve().parents[1]
    offenders = _find_offenders(repo)
    assert not offenders, "\n".join(
        ["type-ignore comment suppressions are forbidden (AGENTS.md):", *offenders]
    )


def test_planted_ty_ignore_in_temp_file_is_flagged(tmp_path: Path) -> None:
    """A planted ``ty``-style suppression in a scanned tree must be detected.

    Plants in ``scripts/`` (the last entry of ``_PY_ROOTS``) to prove roots
    past the first are scanned too, and uses a temp file so the planted
    comment never appears in this repo's source scan.
    """
    (tmp_path / "scripts").mkdir()
    planted = tmp_path / "scripts" / "planted.py"
    planted.write_text(f"x: int = 1  # {_FORBIDDEN_TY}[unresolved-attribute]\n")
    offenders = _find_offenders(tmp_path)
    assert offenders, "planted suppression was not detected"
    assert any("scripts/planted.py:1" in line for line in offenders)


def test_planted_type_ignore_in_temp_file_is_flagged(tmp_path: Path) -> None:
    """A planted mypy-style suppression in a scanned tree must be detected."""
    (tmp_path / "src").mkdir()
    planted = tmp_path / "src" / "planted.py"
    planted.write_text(f"x: int = 1  # {_FORBIDDEN_MYPY}\n")
    offenders = _find_offenders(tmp_path)
    assert offenders, "planted suppression was not detected"
    assert any("src/planted.py:1" in line for line in offenders)
