"""Test-suite configuration for vemoizer.

Pattern: collect_ignore + stale-entry guard (port of
kuiskaus/tests/conftest.py). No hardware scripts exist yet; the list
starts empty and each entry must land in the same PR as the file it
names, so a stale entry can never silently let a model-backed script
into unit collection.
"""

from pathlib import Path

import pytest

collect_ignore: list[str] = []

_TESTS_DIR = Path(__file__).parent


def pytest_configure(config: pytest.Config) -> None:
    """Fail fast if a collect_ignore entry no longer matches a file on disk."""
    stale = [name for name in collect_ignore if not (_TESTS_DIR / name).is_file()]
    if stale:
        raise RuntimeError(
            f"conftest.collect_ignore entries no longer match a file on disk: {stale}"
        )
