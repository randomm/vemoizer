"""Copy transcript text to the macOS clipboard via pbcopy (issue #14).

Fail-open: on non-darwin platforms or any subprocess error, the function
returns False (no exception) so a failed copy never crashes the run.

The transcript file is always written; ``--copy`` is an *additional*
output to the clipboard, not a replacement.
"""

from __future__ import annotations

import subprocess
import sys

__all__ = ["copy_to_clipboard"]


def copy_to_clipboard(text: str) -> bool:
    """Copy *text* to the macOS system clipboard via pbcopy.

    Returns True on success, False on failure (non-darwin, pbcopy missing,
    non-zero exit code, timeout, or any OS-level error).

    Never raises.
    """
    if sys.platform != "darwin":
        return False

    try:
        proc = subprocess.run(
            ["pbcopy"],
            input=text.encode("utf-8"),
            check=False,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        return False

    return proc.returncode == 0
