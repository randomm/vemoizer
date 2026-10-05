"""Guard: no type-ignore comment suppressions may reappear (issue #115, #128, #141).

AGENTS.md forbids type-ignore comment suppressions; #115 brought the repo to
zero and #128 removed the last five (tests/test_whisper_transcriber.py).
#141 extends the guard to ty's ``ty``-prefixed ignore spelling across src,
tests and scripts, and now matches *any* spacing between ``#``, the tool
name, the colon, and the word ``ignore``.

The forbidden patterns are assembled by concatenation + ``re.compile`` so
this file does not contain a literal suppression comment in a form that a
plain source scan would match (``grep -rn`` over src, tests and scripts must
keep returning zero matches).
"""

import re
from pathlib import Path

import pytest

# Each forbidden spelling is built by concatenation so the literal comment
# string never appears in this source file (which the guard scans too).
# The regex allows any whitespace (including none) between ``#``, the tool
# name, the colon, and ``ignore`` — so any spacing variant is caught.
_FORBIDDEN_TY = re.compile(r"#\s*" + "ty" + r"\s*:\s*" + "ignore")
_FORBIDDEN_TYPE = re.compile(r"#\s*" + "type" + r"\s*:\s*" + "ignore")
_FORBIDDEN_MYPY = re.compile(r"#\s*" + "mypy" + r"\s*:\s*" + "ignore")
_FORBIDDEN_PYRIGHT = re.compile(r"#\s*" + "pyright" + r"\s*:\s*" + "ignore")
_FORBIDDEN = (_FORBIDDEN_TY, _FORBIDDEN_TYPE, _FORBIDDEN_MYPY, _FORBIDDEN_PYRIGHT)
_PY_ROOTS = ("src", "tests", "scripts")


def _scan(root: Path, tokens: tuple[re.Pattern[str], ...], repo: Path) -> list[str]:
    """Offending ``rel:line: text`` entries for any ``*.py`` under *root*."""
    offenders: list[str] = []
    for path in sorted(root.rglob("*.py")):
        for line_no, line in enumerate(path.read_text().splitlines(), 1):
            if any(pat.search(line) for pat in tokens):
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


# --- Planted-suppression test matrix ----------------------------------------

# Build the literal suppression strings by concatenation so they never appear
# in this source file (which the guard scans).
_TY_STD = "# " + "ty" + ": ignore[unresolved-attribute]"
_TY_NOSPACE = "#" + "ty" + ":" + "ignore[unresolved-attribute]"
_TY_WIDE = "#  " + "ty" + ":  " + "ignore"
_TYPE_STD = "# " + "type" + ": " + "ignore"
_TYPE_NOSPACE = "#" + "type" + ":" + "ignore"
_TYPE_WIDE = "#  " + "type" + ":  " + "ignore[misc]"

# (label, line_text, expected_flagged)
# Each line is what would appear in a planted .py file.
# ``expected_flagged`` is True if the guard should detect it.
_PLANTED_CASES: list[tuple[str, str, bool]] = [
    # Spacings that MUST be flagged
    ("ty-std", "x: int = 1  " + _TY_STD, True),
    ("ty-nospace", "x: int = 1  " + _TY_NOSPACE, True),
    ("ty-wide", "x: int = 1  " + _TY_WIDE, True),
    ("type-std", "x: int = 1  " + _TYPE_STD, True),
    ("type-nospace", "x: int = 1  " + _TYPE_NOSPACE, True),
    ("type-wide", "x: int = 1  " + _TYPE_WIDE, True),
    # Negatives that must NOT be flagged
    ("plain-text-1", "# typing is fine", False),
    ("plain-text-2", "# ignore this", False),
    ("plain-text-3", "# the ty checker", False),
    (
        "plain-text-4",
        "# some code that has the word " + "ignore" + " after a hash",
        False,
    ),
]


@pytest.mark.parametrize(
    "label, line_text, expected_flagged",
    [pytest.param(c[0], c[1], c[2], id=c[0]) for c in _PLANTED_CASES],
)
def test_planted_suppression_matrix(
    tmp_path: Path, label: str, line_text: str, expected_flagged: bool
) -> None:
    """Parametrized matrix: each spelling must be flagged or not as specified."""
    (tmp_path / "scripts").mkdir()
    planted = tmp_path / "scripts" / "planted.py"
    planted.write_text(line_text + "\n")
    offenders = _find_offenders(tmp_path)
    flagged = bool(offenders)
    assert flagged == expected_flagged, (
        f"{label}: expected flagged={expected_flagged}, got {flagged}; "
        f"offenders={offenders}"
    )


def test_planted_ty_ignore_in_temp_file_is_flagged(tmp_path: Path) -> None:
    """A planted ``ty``-style suppression in a scanned tree must be detected.

    Plants in ``scripts/`` (the last entry of ``_PY_ROOTS``) to prove roots
    past the first are scanned too, and uses a temp file so the planted
    comment never appears in this repo's source scan.
    """
    (tmp_path / "scripts").mkdir()
    planted = tmp_path / "scripts" / "planted.py"
    planted.write_text("x: int = 1  " + _TY_STD + "\n")
    offenders = _find_offenders(tmp_path)
    assert offenders, "planted suppression was not detected"
    assert any("scripts/planted.py:1" in line for line in offenders)


def test_planted_type_ignore_in_temp_file_is_flagged(tmp_path: Path) -> None:
    """A planted mypy-style suppression in a scanned tree must be detected."""
    (tmp_path / "src").mkdir()
    planted = tmp_path / "src" / "planted.py"
    planted.write_text("x: int = 1  " + _TYPE_STD + "\n")
    offenders = _find_offenders(tmp_path)
    assert offenders, "planted suppression was not detected"
    assert any("src/planted.py:1" in line for line in offenders)
