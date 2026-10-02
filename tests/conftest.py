"""Test-suite configuration for vemoizer.

No hardware/model-backed scripts exist yet. When the first entry lands in
``collect_ignore``, re-add the ``pytest_configure`` stale-entry guard in
the same PR so the guard and its data ship together.

Session guard: after the full test run, the repo root must not contain
any new untracked output files (txt/json/srt/vtt/md). This catches any
test that forgets to ``monkeypatch.chdir(tmp_path)`` and writes its
transcript outputs into the actual working directory (the repo root).
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

# Hermetic CLI help output: typer.rich_utils reads GITHUB_ACTIONS / FORCE_COLOR /
# PY_COLORS at IMPORT time (rich_utils.py, lines 77-84) to force a coloured
# terminal, and NO_COLOR/TERM/COLUMNS affect rich's Console sizing. GitHub
# Actions sets GITHUB_ACTIONS=true, which makes --help emit ANSI + box-drawing
# and breaks substring assertions (issue #102, run 36991883722). typer is not
# imported yet at this point (conftest runs before test-module imports), so
# pinning the environment here keeps the suite environment-independent.
for _var in ("GITHUB_ACTIONS", "FORCE_COLOR", "PY_COLORS", "CI"):
    os.environ.pop(_var, None)
os.environ["NO_COLOR"] = "1"
os.environ["TERM"] = "dumb"
os.environ["COLUMNS"] = "120"

# Extensions the test suite can write as transcript outputs.
_OUTPUT_EXTS = {".txt", ".json", ".srt", ".vtt", ".md"}


def _repo_root() -> Path:
    # tests/ lives one level below the repo root.
    return Path(__file__).parent.parent


def _snapshot_root() -> set[str]:
    """Names of existing files in the repo root that have a transcript-output ext."""
    root = _repo_root()
    return {p.name for p in root.iterdir() if p.is_file() and p.suffix in _OUTPUT_EXTS}


@pytest.fixture(autouse=True, scope="session")
def _guard_no_repo_root_output_files():
    """Assert no new transcript-output files appeared in the repo root.

    The snapshot is taken inside the fixture setup (before ``yield``), so
    it reflects the repo state at the start of the session — not at
    import time, when collection-time module imports could still run.
    """
    before = _snapshot_root()
    yield
    after = _snapshot_root()
    new_files = after - before
    if new_files:
        pytest.fail(
            f"Session guard: new file(s) appeared in repo root during the "
            f"test session: {sorted(new_files)}. A test may have written "
            f"transcript outputs to the CWD instead of "
            f"monkeypatch.chdir(tmp_path) (or the file pre-existed and "
            f"the snapshot missed it)."
        )
