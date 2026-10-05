"""Guard: no type-ignore comment suppressions may reappear (issue #115, #128, #141).

AGENTS.md forbids type-ignore comment suppressions; #115 brought the repo to
zero and #128 removed the last five (tests/test_whisper_transcriber.py). This
test fails the suite if one creeps back in.

Both the mypy-style "type ignore" and the "ty" checker's "ty ignore"
spellings are rejected: #141 found 16 ``ty``-style suppressions that the
single-spelling guard had never seen.

The forbidden tokens are assembled by concatenation (see below) so this file
does not contain the literal comments in a form a plain source scan would
match (``grep -rn`` over src, tests, and scripts must keep returning zero
matches).
"""

from pathlib import Path

# Each forbidden spelling is built by concatenation so the literal comment
# string never appears in this source file (which the guard scans too).
_FORBIDDEN = [
    "#" + " type: " + "ignore",
    "#" + " ty: " + "ignore",
]
_PY_ROOTS = ("src", "tests", "scripts")


def _scan(root: Path, tokens: list[str]) -> list[str]:
    """Offending ``rel:line: text`` entries for any ``*.py`` under *root*."""
    offenders: list[str] = []
    for path in sorted(root.rglob("*.py")):
        for line_no, line in enumerate(path.read_text().splitlines(), 1):
            if any(token in line for token in tokens):
                rel = path.relative_to(root)
                offenders.append(f"{rel}:{line_no}: {line.strip()}")
    return offenders


def _offenders(tokens: list[str], repo: Path) -> list[str]:
    offenders: list[str] = []
    for py_root in _PY_ROOTS:
        offenders.extend(_scan(repo / py_root, tokens))
    return offenders


def test_no_type_ignore_suppressions() -> None:
    repo = Path(__file__).resolve().parents[1]
    offenders = _offenders(_FORBIDDEN, repo)
    assert not offenders, "\n".join(
        ["type-ignore comment suppressions are forbidden (AGENTS.md):", *offenders]
    )


def test_planted_ty_ignore_is_flagged(tmp_path: Path) -> None:
    """A planted ``ty``-style suppression in a temp file is flagged."""
    planted = tmp_path / ("planted" + ".py")
    planted.write_text("x = 1  " + _FORBIDDEN[1] + "[unresolved-attribute]\n")
    offenders = _scan(tmp_path, _FORBIDDEN)
    assert len(offenders) == 1
    assert "planted" + ".py:1" in offenders[0]
    assert _FORBIDDEN[1] in offenders[0]


def test_planted_type_ignore_is_flagged(tmp_path: Path) -> None:
    """A planted mypy-style suppression in a temp file is flagged too."""
    planted = tmp_path / ("planted" + ".py")
    planted.write_text("x = 1  " + _FORBIDDEN[0] + "\n")
    offenders = _scan(tmp_path, _FORBIDDEN)
    assert len(offenders) == 1
    assert "planted" + ".py:1" in offenders[0]
    assert _FORBIDDEN[0] in offenders[0]
