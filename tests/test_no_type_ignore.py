"""Guard: no type-ignore comment suppressions may reappear (issue #115, #128).

AGENTS.md forbids type-ignore comment suppressions; #115 brought the repo to zero and
#128 removed the last five (tests/test_whisper_transcriber.py). This test
fails the suite if one creeps back in.

The forbidden token is assembled by concatenation so this file does not
contain the literal comment in a form a plain source scan would match
(`grep -rn` over src and tests must keep returning zero matches).
"""

from pathlib import Path

_FORBIDDEN = "#" + " type: " + "ignore"
_PY_ROOTS = ("src", "tests", "scripts")


def test_no_type_ignore_suppressions() -> None:
    repo = Path(__file__).resolve().parents[1]
    offenders = []
    for root in _PY_ROOTS:
        for path in sorted((repo / root).rglob("*.py")):
            for line_no, line in enumerate(path.read_text().splitlines(), 1):
                if _FORBIDDEN in line:
                    rel = path.relative_to(repo)
                    offenders.append(f"{rel}:{line_no}: {line.strip()}")
    assert not offenders, "\n".join(
        ["type-ignore comment suppressions are forbidden (AGENTS.md):", *offenders]
    )
