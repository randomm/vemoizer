"""Check whether the Mac is running on battery power (issue #14).

Uses ``pmset -g batt`` to detect the power source. The function is
fail-open: on non-darwin platforms or any subprocess error, it returns
False (no warning), so a pmset failure never blocks a transcription run.
"""

from __future__ import annotations

import subprocess
import sys

__all__ = ["on_battery"]


def on_battery() -> bool:
    """Return True if the Mac is currently running on battery power.

    Uses ``pmset -g batt`` and checks for ``"Battery Power"`` in the
    output. Returns False if:
    - the platform is not darwin
    - pmset is not found or fails
    - the output does not mention "Battery Power"

    Never raises.
    """
    if sys.platform != "darwin":
        return False

    try:
        proc = subprocess.run(
            ["pmset", "-g", "batt"],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return False

    if proc.returncode != 0:
        return False

    return "Battery Power" in proc.stdout
