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

# Extensions the test suite can write as transcript outputs.
_OUTPUT_EXTS = {".txt", ".json", ".srt", ".vtt", ".md"}


def _repo_root() -> Path:
    # tests/ lives one level below the repo root.
    return Path(__file__).parent.parent


def _snapshot_root() -> set[str]:
    """Names of existing files in the repo root that have a transcript-output ext."""
    root = _repo_root()
    result = set()
    for entry in os.listdir(root):
        p = root / entry
        if p.is_file() and p.suffix in _OUTPUT_EXTS:
            result.add(entry)
    return result


# Snapshot taken at import time (before any test runs).
_ROOT_FILES_BEFORE = _snapshot_root()


@pytest.fixture(autouse=True, scope="session")
def _guard_no_repo_root_output_files():
    """Assert no new untracked transcript-output files appeared in the repo root.

    Run after all tests (yield is session-scoped).
    """
    yield
    after = _snapshot_root()
    new_files = after - _ROOT_FILES_BEFORE
    if new_files:
        pytest.fail(
            f"Session guard: new untracked output file(s) in repo root: "
            f"{sorted(new_files)}. A test wrote transcript outputs to the "
            f"CWD instead of monkeypatch.chdir(tmp_path)."
        )
