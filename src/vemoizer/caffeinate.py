"""Hold a macOS wake assertion during transcription via caffeinate (issue #14).

The context manager spawns ``caffeinate -dims`` (display, idle, disk,
network) as a daemon process that holds the assertion until terminated.
The caller runs the transcription work inside the ``with`` block; on exit
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


@contextmanager
def caffeinate_context() -> Generator[None, None, None]:
    """Context manager that holds a macOS wake assertion during a block.

    Spawns ``caffeinate -dims`` (no command → holds assertion until
    terminated). On exit the process is terminated and waited on.

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
    proc: subprocess.Popen[bytes] | None = None

    try:
        try:
            proc = subprocess.Popen(
                ["caffeinate", "-dims"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                stdin=subprocess.DEVNULL,
            )
        except (FileNotFoundError, OSError, subprocess.SubprocessError):
            # Fail-open: no assertion held, but the work still runs
            proc = None

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
