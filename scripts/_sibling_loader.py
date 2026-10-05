"""Shared sibling-module loader for the ``scripts/`` helper modules.

The dev-time scripts (``gen_real_speech_corpus.py`` -> ``fleurs_source.py``
-> ``wav_header.py``) are plain files, not a packaged module, but they must
stay importable both as ``python scripts/<script>.py`` (the script's
directory on ``sys.path``) and via
``importlib.util.spec_from_file_location`` (``tests/test_gen_real_speech_corpus.py``).
This helper is the single place that resolves a sibling ``scripts/*.py``
file into a module; each consumer just imports its sibling at the top of
its file.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

_SCRIPTS_DIR = Path(__file__).resolve().parent


def load_sibling(name: str) -> ModuleType:
    """Return module *name* (a sibling ``scripts/<name>.py``), cached.

    Prefers an ordinary package-relative import (``scripts.<name>``) so the
    module ends up under the package in ``sys.modules`` with full first-party
    resolution; falls back to a direct ``<name>`` import (script directory
    on ``sys.path``); and last to loading the file next to this one via
    ``importlib`` (no sibling on ``sys.path`` at all, as in the tests).
    Repeated calls return the same module object from ``sys.modules``.

    A cached hit is trusted only when its ``__file__`` points to the expected
    sibling under the scripts dir; an unrelated top-level module of the same
    name (e.g. a test fixture) falls through to a fresh load.
    """
    cached = sys.modules.get(name)
    if cached is not None:
        file_attr = getattr(cached, "__file__", None)
        if file_attr is not None and _is_expected_sibling(name, file_attr):
            return cached
    try:
        mod = importlib.import_module(f"scripts.{name}")
    except ImportError:
        try:
            mod = importlib.import_module(name)
        except ImportError:
            mod = None
    # ``importlib.import_module(name)`` can return an unrelated module that
    # already occupies ``sys.modules[name]`` (Python's import system checks
    # ``sys.modules`` first). Validate the result; if its ``__file__`` does
    # not point to the expected sibling, discard the stale entry and load
    # from disk directly.
    if mod is not None:
        file_attr = getattr(mod, "__file__", None)
        if file_attr is not None and _is_expected_sibling(name, file_attr):
            return mod
        sys.modules.pop(name, None)
    # Last resort: the helper itself is not importable, or a stale module
    # was cached under the sibling's name. Load the sibling file directly
    # from disk via ``importlib``.
    return _load_from_disk(f"scripts.{name}", _SCRIPTS_DIR / f"{name}.py")


def _is_expected_sibling(name: str, file_attr: object) -> bool:
    """Return ``True`` when *file_attr* is the expected sibling path."""
    if not isinstance(file_attr, str):
        return False
    return Path(file_attr).resolve() == _SCRIPTS_DIR / f"{name}.py"


def _load_from_disk(modname: str, path: Path) -> ModuleType:
    """Load *path* under *modname* via importlib (no sys.path entry)."""
    spec = importlib.util.spec_from_file_location(modname, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot build an import spec for {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[modname] = mod
    try:
        spec.loader.exec_module(mod)
    except BaseException:
        # Remove the half-built module so a later load_sibling call does
        # not return a partial object from the sys.modules cache; the
        # register-before-exec ordering is kept (dataclass modules need it).
        sys.modules.pop(modname, None)
        raise
    return mod
