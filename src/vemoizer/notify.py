"""Post a macOS completion notification via osascript (issue #100, M4a).

Fires once per input file (or per written/failed group) on success and
failure from every run path (expert ``transcribe``, ``meeting``, ``memo``).
Fail-open: on non-darwin platforms or any subprocess error it does nothing,
so a missing ``osascript`` or a hung post never changes an exit code or
blocks a transcription run.

The message names the NFC-normalised file stem and the outcome and, on
failure, the one-line reason already printed on stderr. It never carries
transcript text, audio-derived content, or secrets (the local-first
invariant: audio and transcripts never leave the machine).
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

__all__ = [
    "NOTIFY_TITLE",
    "escape_apple_script",
    "notify",
    "notify_result",
    "notify_write",
]

#: Subprocess timeout for the ``osascript`` call (seconds). A notification
#: is best-effort; a few seconds is plenty and bounds how long a hung
#: osascript can stall the run.
_TIMEOUT = 5

#: Cap for the failure reason carried in a notification message (characters)
#: — the reason is one line, but a long path in a write error must not
#: produce a notification nobody can read.
_REASON_MAX = 200

#: The single AppleScript source line. ``{message}`` / ``{title}`` are
#: escaped by :func:`escape_apple_script` before interpolation.
_SCRIPT_TEMPLATE = 'display notification "{message}" with title "{title}"'


def escape_apple_script(value: str) -> str:
    """Escape *value* for use inside an AppleScript double-quoted literal.

    AppleScript string literals use ``\\b`` for a double quote and ``\\n``
    for a newline; a literal backslash is doubled. Backslashes are escaped
    *first* so the later inserted backslashes are not re-escaped.
    """
    return value.replace("\\", "\\\\").replace('"', "\\b").replace("\n", "\\n")


def notify(title: str, message: str) -> None:
    """Post *message* as a macOS notification titled *title*.

    No-op (never raises) when:
    - the platform is not darwin
    - ``osascript`` is missing, fails, or times out
    - any other OS-level error occurs

    The argv is always built as a *list* (never passed to a shell), and
    *title* / *message* are escaped for AppleScript string literals, so a
    value containing quotes, backslashes, or newlines can neither inject
    extra AppleScript nor break the ``osascript`` invocation.

    Never raises and never changes the caller's exit code.
    """
    if sys.platform != "darwin":
        return

    script = _SCRIPT_TEMPLATE.format(
        title=escape_apple_script(title),
        message=escape_apple_script(message),
    )
    argv = ["osascript", "-e", script]
    try:
        subprocess.run(
            argv,
            capture_output=True,
            text=True,
            check=False,
            timeout=_TIMEOUT,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
        # Fail-open: a notification is best-effort; never raise, never print
        # a traceback, never change the run's exit code.
        return


#: The fixed notification title for every vemoizer notification.
NOTIFY_TITLE = "vemoizer"


def _nfc_group_stem(label: Path | str) -> str:
    """The NFC-normalised stem of a group's *first part*.

    A multi-part meeting group is labelled ``"a.m4a+b.m4a"``; the
    notification names the first part's stem (the same stem ``write_group``
    uses for the pair), never the joined label. A plain file name passes
    through unchanged.
    """
    from vemoizer.output.naming import nfc_stem_and_suffix

    first = str(label).split("+", 1)[0]
    stem, _ = nfc_stem_and_suffix(first)
    return stem


def _reason(message: str) -> str:
    """Strip the leading ``error: `` from a stderr line, keep one line,
    and cap it to a readable length."""
    line = message.strip()
    if line.startswith("error:"):
        line = line.removeprefix("error:").strip()
    # Keep it to one line: the first line of a multi-line reason.
    line = line.splitlines()[0] if line else ""
    if len(line) > _REASON_MAX:
        line = line[: _REASON_MAX - 1] + "…"
    return line


def notify_result(label: Path | str, outcome: str, reason: str = "") -> None:
    """Post the one notification for a file/group *outcome* (M4a seam helper).

    *label* is the file (``Path``) or a group label (``"a.m4a+b.m4a"``);
    the message carries the NFC-normalised stem (a group's first part).
    *outcome* is ``"done"`` or ``"failed"``; on failure *reason* is the
    one-line stderr line for that seam (the leading ``error: `` is
    stripped). Never raises — a notification must never change a run's
    exit code.
    """
    try:
        stem = _nfc_group_stem(label)
        if outcome == "done":
            message = f"{stem}: done"
        else:
            reason = _reason(reason)
            message = f"{stem}: failed" if not reason else f"{stem}: failed - {reason}"
        notify(NOTIFY_TITLE, message)
    except Exception:  # noqa: BLE001 - best-effort: a notification must never change a run's exit code
        pass


def notify_write(
    label: Path | str, written: int, expected: int, reason: str = ""
) -> None:
    """Post the one notification for an output write (M4a seam helper).

    *written* is how many files of the expected pair landed; fewer than
    *expected* (a partial pair — the ``.json`` write failed after the
    ``.md``) is a FAILURE even though a file landed. Never raises.
    """
    try:
        if written < expected:
            notify_result(label, "failed", reason)
        else:
            notify_result(label, "done")
    except Exception:  # noqa: BLE001 - best-effort: a notification must never change a run's exit code
        pass
