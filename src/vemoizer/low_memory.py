"""Low-memory mode detection and configuration (issue #14, --low-memory).

On macOS, ``sysctl -n hw.memsize`` returns the total RAM in bytes.
Macs with 16 GiB or less of unified memory benefit from a low-memory
model-loading strategy (sequential load/unload instead of simultaneous
in-memory models). This module exposes the detection function and a
no-op apply function (the actual model lifecycle wiring is deferred to
the consensus pipeline task).

Fail-open: on non-darwin platforms or sysctl errors, ``total_ram_bytes()``
returns ``None`` and ``default_low_memory()`` returns False (off), so the
flag is opt-in, not opt-out, when detection fails.
"""

from __future__ import annotations

import subprocess
import sys

__all__ = ["apply_low_memory_mode", "default_low_memory", "total_ram_bytes"]

#: Threshold (bytes) at and below which low-memory mode is enabled by default.
_16_GIB = 16 * 1024**3


def total_ram_bytes() -> int | None:
    """Return the total system RAM in bytes, or None if detection fails.

    Uses ``sysctl -n hw.memsize`` on macOS. Returns None on non-darwin
    platforms, non-zero exit codes, or any subprocess error.
    """
    if sys.platform != "darwin":
        return None

    try:
        proc = subprocess.run(
            ["sysctl", "-n", "hw.memsize"],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return None

    if proc.returncode != 0:
        return None

    try:
        return int(proc.stdout.strip())
    except ValueError:
        return None


def default_low_memory() -> bool:
    """Return True if low-memory mode should be enabled by default.

    Rule: enabled when total RAM is 16 GiB or less (the boundary is
    inclusive). Returns False when RAM detection fails (None), so the
    flag defaults off in the absence of a reliable signal.
    """
    ram = total_ram_bytes()
    if ram is None:
        return False
    return ram <= _16_GIB


def apply_low_memory_mode(enabled: bool) -> None:
    """Apply (or clear) the low-memory mode flag.

    Currently a no-op: the actual model-loading strategy (sequential
    load/unload via ``del model`` + ``mx.clear_cache()``) is
    deferred to the consensus pipeline task. This function exists so
    the CLI flag can be parsed and validated now, and so the call site
    is in place for when the pipeline is wired.
    """
    # No-op for now. The pipeline will read this flag when loading
    # models and switch between simultaneous and sequential loading.
    # Intentionally empty — see docstring.
    del enabled
