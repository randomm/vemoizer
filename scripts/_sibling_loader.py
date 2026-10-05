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
    """
    if name in sys.modules:
        return sys.modules[name]
    try:
        return importlib.import_module(f"scripts.{name}")
    except ImportError:
        pass
    try:
        return importlib.import_module(name)
    except ImportError:
        pass
    # Last resort: the helper itself is not importable (the tests' importlib
    # path, where no sibling is on ``sys.path`` at all). Load the sibling
    # file directly from disk via ``importlib``.
    return _load_from_disk(f"scripts.{name}", _SCRIPTS_DIR / f"{name}.py")


def _load_from_disk(modname: str, path: Path) -> ModuleType:
    """Load *path* under *modname* via importlib (no sys.path entry)."""
    spec = importlib.util.spec_from_file_location(modname, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot build an import spec for {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[modname] = mod
    spec.loader.exec_module(mod)
    return mod
