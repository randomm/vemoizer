"""Hold a macOS wake assertion during transcription via caffeinate (issue #14).

The context manager spawns ``caffeinate -ims`` (idle, disk, network; the
display may sleep) as a daemon process that holds the assertion until
terminated. The caller runs the transcription work inside the ``with``
block; on exit
the process is terminated and reaped.

Fail-open: on non-darwin platforms or any spawn error, the context is a
no-op — the work runs without the wake assertion and no exception is
raised. This matches the local-first, fail-open invariant: a missing
caffeinate must never block a transcription.
"""

from __future__ import annotations

import subprocess
import sys
from collections.abc import Generator
from contextlib import contextmanager, suppress

__all__ = ["caffeinate_context"]

# Module-level flag to prevent nested caffeinate contexts from double-spawning.
_active: bool = False


def _spawn(argv: list[str]) -> subprocess.Popen[bytes] | None:
    """Spawn *argv* (``caffeinate -ims``) with its fail-open handling.

    The single seam that performs the real system call; it returns the
    spawned process, or ``None`` when spawn fails (fail-open). Tests stub
    it via the ``system_effects`` autouse fixture without globally patching
    ``subprocess`` or ``sys``.
    """
    try:
        return subprocess.Popen(
            argv,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
        )
    except (FileNotFoundError, OSError, subprocess.SubprocessError):
        # Fail-open: no assertion held, but the work still runs.
        return None


@contextmanager
def caffeinate_context() -> Generator[None, None, None]:
    """Context manager that holds a macOS wake assertion during a block.

    Spawns ``caffeinate -ims`` (idle, disk, network; the display may
    sleep) with no command, so it holds the assertion until terminated.
    On exit the process is terminated and waited on.

    On non-darwin platforms or spawn errors the context is a no-op.
    Nested contexts are also no-ops (the outer context owns the process).

    Usage::

        with caffeinate_context():
            run_transcription()  # wake assertion held during this block
    """
    global _active

    if sys.platform != "darwin" or _active:
        # Non-darwin or nested → no-op
        yield
        return

    _active = True

    try:
        proc = _spawn(["caffeinate", "-ims"])

        try:
            yield
        finally:
            if proc is not None:
                with suppress(OSError):
                    proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    with suppress(OSError):
                        proc.kill()
    finally:
        _active = False
