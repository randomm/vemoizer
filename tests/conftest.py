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

import logging
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

import pytest

from vemoizer import run_log as _run_log_module

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


@pytest.fixture(autouse=True)
def _no_real_system_effects(
    monkeypatch: pytest.MonkeyPatch, request: pytest.FixtureRequest
) -> Iterator[dict[str, list[list[str]]] | None]:
    """Stub the three real-system-call seams so the suite never posts a
    macOS notification, spawns ``caffeinate``, or plays audio.

    The production code keeps NO test awareness (no PYTEST_CURRENT_TEST
    checks); each side-effect module exposes one tiny private seam that
    performs the real system call, and this autouse fixture replaces those
    seams with recorders/no-ops. Patching ``subprocess`` or ``sys`` globally
    is deliberately avoided: it would break the unrelated ffmpeg/ingest
    tests that run real ``subprocess.run``.

    Opt-out: tests (or their module) marked ``real_system_calls`` exercise
    the *real* seam and patch ``subprocess`` themselves; for those the
    fixture is a no-op so the real seam (and the test's own subprocess
    patches) run. The per-test patches in test_notify.py / test_caffeinate.py
    / test_speaker_clips.py keep working because the real seam is restored.
    """
    if request.node.get_closest_marker("real_system_calls") is not None:
        # This test (or its module/class) opts out: the real seam runs and
        # the test patches subprocess itself. Do not install the stubs.
        yield None
        return

    from vemoizer import caffeinate, notify, speaker_clips

    # The per-test recorder, published to dependents (the ``system_effects``
    # fixture) through the fixture's own yielded value — no closure
    # introspection required.
    recorded: dict[str, list[list[str]]] = {
        "notify": [],
        "caffeinate": [],
        "afplay": [],
    }

    def _post(argv: list[str]) -> None:
        recorded["notify"].append(argv)

    class _FakeProcess:
        """A stand-in for the ``caffeinate`` Popen.

        ``terminate``/``wait``/``kill``/``poll`` satisfy
        ``caffeinate_context``'s exit path without touching a real process.
        """

        def terminate(self) -> None:  # pragma: no cover - trivial no-op
            pass

        def kill(self) -> None:  # pragma: no cover - trivial no-op
            pass

        def wait(self, timeout: float | None = None) -> int:
            return 0

        def poll(self) -> int | None:
            return 0

    def _spawn(argv: list[str]) -> Any:
        recorded["caffeinate"].append(argv)
        return _FakeProcess()

    def _run_player(argv: list[str]) -> int:
        recorded["afplay"].append(argv)
        return 0

    monkeypatch.setattr(notify, "_post", _post)
    monkeypatch.setattr(caffeinate, "_spawn", _spawn)
    monkeypatch.setattr(speaker_clips, "_run_player", _run_player)
    yield recorded


@pytest.fixture
def system_effects(
    _no_real_system_effects: dict[str, list[list[str]]] | None,
) -> dict[str, list[list[str]]]:
    """The recorded system effects for the current test.

    Depends on the autouse ``_no_real_system_effects`` fixture, which yields
    the per-test ``recorded`` dict (or ``None`` when the test opted out via
    the ``real_system_calls`` marker) so this fixture can hand it to the test
    for asserting on the recorded argv lists.
    """
    if _no_real_system_effects is None:
        raise RuntimeError(
            "system_effects: this test is marked ``real_system_calls`` and "
            "opted out of the autouse system-effects stub, so no effects "
            "are being recorded. Remove the marker (or drop the "
            "``system_effects`` fixture) to use this fixture."
        )
    return _no_real_system_effects


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


# --- logging-state snapshot (issue #111 M4c) ---------------------------------
# The ``file_log`` context manager attaches a ``QuietFileHandler`` to the
# root logger (and optionally the ``huggingface_hub`` logger) and raises
# the root level to INFO. If a test forgets to exit the ``with`` block (or
# the context manager raises before ``finally``), the handler and level
# change leak into subsequent tests, breaking ``caplog``-based tests and
# polluting the test environment. This autouse fixture snapshots the
# handler list, level, ``propagate``, and per-handler filters of the three
# loggers the design names (root, ``vemoizer``, ``huggingface_hub``) before
# each test and ASSERTS+restores them after, so any leak is caught (the
# test fails with a clear ``AssertionError``) and rolled back.
#
# Handlers owned by pytest itself (``LogCaptureHandler`` from ``caplog``,
# ``_LiveLoggingNullHandler`` from the ``live-logging`` plugin, and any
# other capture handler the logging plugin adds during the test) are
# EXCLUDED from the comparison so legitimate pytest behaviour does not
# cause spurious failures.


def _is_pytest_handler(h: logging.Handler) -> bool:
    """True if *h* is a handler owned by pytest itself (excluded from the
    logging-state assertion)."""
    if isinstance(h, logging.NullHandler):
        return True
    cls_name = type(h).__name__
    if cls_name in ("LogCaptureHandler", "_LiveLoggingNullHandler"):
        return True
    mod = type(h).__module__
    return mod.startswith("_pytest") or mod.startswith("pytest")


def _is_library_handler(h: logging.Handler) -> bool:
    """True if *h* is a handler added by a third-party library (not the test).

    ``huggingface_hub`` adds a ``StreamHandler`` to its own logger when
    imported; this is expected library behaviour, not a test leak.
    """
    if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler):
        # HF's default handler writes to stderr via a _FileIO stream.
        stream = getattr(h, "stream", None)
        if stream is not None:
            stream_name = getattr(stream, "name", "")
            if stream_name in ("/dev/stderr", "stderr"):
                return True
            # Check if the stream is sys.stderr by fileno.
            try:
                import sys

                if stream.fileno() == sys.stderr.fileno():
                    return True
            except (OSError, ValueError, AttributeError):
                pass
    return False


def _logger_snapshot(name: str) -> tuple[list[logging.Handler], int, bool]:
    """Snapshot a logger's handler list, level, and propagate flag."""
    lg = logging.getLogger(name)
    return list(lg.handlers), lg.level, lg.propagate


def _handler_filter_snapshot(handler: logging.Handler) -> list[logging.Filter]:
    """Snapshot a handler's filter list."""
    return [cast(logging.Filter, f) for f in handler.filters]


def _snapshot_all_loggers() -> dict[str, tuple[list[logging.Handler], int, bool]]:
    """Snapshot the three loggers the design names."""
    return {
        "": _logger_snapshot(""),
        "vemoizer": _logger_snapshot("vemoizer"),
        "huggingface_hub": _logger_snapshot("huggingface_hub"),
    }


def _restore_all_loggers(
    snap: dict[str, tuple[list[logging.Handler], int, bool]],
    handler_filters: dict[int, list[logging.Filter]] | None = None,
) -> None:
    """Restore the three loggers to their snapshot."""
    for name, (handlers, level, propagate) in snap.items():
        lg = logging.getLogger(name)
        for h in list(lg.handlers):
            lg.removeHandler(h)
        for h in handlers:
            lg.addHandler(h)
        lg.setLevel(level)
        lg.propagate = propagate
        if handler_filters is not None:
            for h in lg.handlers:
                snap_filters = handler_filters.get(id(h))
                if snap_filters is not None:
                    for f in list(h.filters):
                        h.removeFilter(f)
                    for f in snap_filters:
                        h.addFilter(f)


@pytest.fixture(autouse=True)
def _guard_logging_state():
    """Assert+restore logging state around every test (issue #111 M4c).

    After each test, the post-test state (handlers, levels, per-handler
    filters, and ``propagate`` of root, ``vemoizer``, and
    ``huggingface_hub``) is compared with the pre-test snapshot. On
    mismatch the test fails with a clear ``AssertionError``; the snapshot
    is restored either way so a leaking test cannot poison subsequent
    tests. Pytest-owned handlers are excluded from the comparison.
    """
    before = _snapshot_all_loggers()
    handler_filters: dict[int, list[logging.Filter]] = {}
    for _name, (handlers, _level, _prop) in before.items():
        for h in handlers:
            handler_filters[id(h)] = _handler_filter_snapshot(h)
    _run_log_module.reset_run_log()
    try:
        yield
    finally:
        _assert_logging_state(before)
        _restore_all_loggers(before, handler_filters)
        _run_log_module.reset_run_log()


def _assert_logging_state(
    snap: dict[str, tuple[list[logging.Handler], int, bool]],
) -> None:
    """Assert the post-test logging state matches the pre-test snapshot.

    Excludes pytest-owned handlers (``LogCaptureHandler``,
    ``_LiveLoggingNullHandler``, ``NullHandler``, and any handler from the
    ``_pytest``/``pytest`` package) so legitimate pytest behaviour does
    not cause spurious failures.
    """
    for name, (handlers, level, propagate) in snap.items():
        lg = logging.getLogger(name)
        expected = [
            h
            for h in handlers
            if not _is_pytest_handler(h) and not _is_library_handler(h)
        ]
        actual = [
            h
            for h in lg.handlers
            if not _is_pytest_handler(h) and not _is_library_handler(h)
        ]
        if actual != expected:
            raise AssertionError(
                f"Logging state leak on logger {name!r}: "
                f"handlers changed from {expected!r} to {actual!r}. "
                f"A test leaked a handler or failed to clean up."
            )
        # Level check: the HF library manages its own level internally
        # (it may set WARNING during model loading), so only assert the
        # level for root and vemoizer, not huggingface_hub.
        if name != "huggingface_hub" and lg.level != level:
            raise AssertionError(
                f"Logging state leak on logger {name!r}: "
                f"level changed from {level} to {lg.level}. "
                f"A test changed the logger level and did not restore it."
            )
        if lg.propagate != propagate:
            raise AssertionError(
                f"Logging state leak on logger {name!r}: "
                f"propagate changed from {propagate} to {lg.propagate}. "
                f"A test changed the propagate flag and did not restore it."
            )
