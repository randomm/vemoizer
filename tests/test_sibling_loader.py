"""Tests for ``scripts/_sibling_loader.py``.

The loader helper is imported via ``importlib`` the same way
``tests/test_gen_real_speech_corpus.py`` loads its scripts (direct
``spec_from_file_location``), so these tests exercise the same import
path the tests use in production (no ``scripts/`` on ``sys.path``).
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest

_SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
_HELPER = _SCRIPTS_DIR / "_sibling_loader.py"


def _load_helper() -> types.ModuleType:
    spec = importlib.util.spec_from_file_location("_sibling_loader_under_test", _HELPER)
    if spec is None:
        raise RuntimeError(f"could not load {_HELPER}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_sibling_loader_under_test"] = mod
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


helper = _load_helper()


class TestLoadSiblingExecFailure:
    """``_load_from_disk``: exec failure must not poison ``sys.modules``.

    A sibling that raises during ``exec_module`` leaves a half-built module
    behind. Without the ``pop`` fix, the next ``load_sibling`` call returns
    that partial module instead of re-raising the original error (the
    ``sys.modules`` cache hit short-circuits before the load is retried).
    """

    def test_exec_failure_leaves_no_modules_entry(self, tmp_path: Path) -> None:
        """A sibling that raises on import leaves no sys.modules entry."""
        bad_sibling = tmp_path / "bad_sibling.py"
        bad_sibling.write_text("raise RuntimeError('boom on import')\n")

        modname = "scripts.bad_sibling_under_test"
        sys.modules.pop(modname, None)

        with pytest.raises(RuntimeError, match="boom on import"):
            helper._load_from_disk(modname, bad_sibling)

        assert modname not in sys.modules, (
            "exec failure left a half-built module in sys.modules; "
            "a subsequent load_sibling call would return the partial module "
            "instead of re-raising the original error"
        )

    def test_second_call_raises_same_error_again(self, tmp_path: Path) -> None:
        """After a failed exec, a second call re-raises (not a cached partial)."""
        bad_sibling = tmp_path / "bad_sibling.py"
        bad_sibling.write_text("raise RuntimeError('boom on import')\n")

        modname = "scripts.bad_sibling_under_test2"
        sys.modules.pop(modname, None)

        with pytest.raises(RuntimeError, match="boom on import"):
            helper._load_from_disk(modname, bad_sibling)

        with pytest.raises(RuntimeError, match="boom on import"):
            helper._load_from_disk(modname, bad_sibling)


class TestLoadSiblingCacheFileCheck:
    """``load_sibling``: a cached module with a different ``__file__`` is not trusted.

    If an unrelated top-level module happens to occupy ``sys.modules[name]``,
    the old code returned it unconditionally. The ``__file__`` check catches
    this: a hit is only trusted when its ``__file__`` points to the expected
    sibling under the scripts dir.
    """

    def test_unrelated_cached_module_with_different_file_is_not_returned(
        self, tmp_path: Path
    ) -> None:
        """A sys.modules entry under the sibling name with a different __file__
        is NOT returned; the loader falls through to disk."""
        fake = types.ModuleType("wav_header")
        fake.__file__ = str(tmp_path / "some_other_file.py")
        sys.modules["wav_header"] = fake
        try:
            result = helper.load_sibling("wav_header")
        finally:
            sys.modules.pop("wav_header", None)

        assert result is not fake, (
            "load_sibling returned an unrelated cached module whose __file__ "
            "does not point to the expected sibling under the scripts dir"
        )
        assert "scripts" in result.__file__

    def test_real_sibling_cached_module_is_returned(self, tmp_path: Path) -> None:
        """When sys.modules already has the correct sibling (matching __file__),
        the cached module IS returned (no redundant disk load)."""
        sys.modules.pop("wav_header", None)
        mod = helper.load_sibling("wav_header")
        cached_file = mod.__file__

        result = helper.load_sibling("wav_header")
        assert result is mod
        assert result.__file__ == cached_file
