"""The console voice of ``vemoizer listen`` (issue #236).

Recording is waiting, not working: a 38-minute call used to print ~110 skew
lines, a clipping warning that glued itself onto the half-drawn ``● REC``
line, and then nothing at all after Ctrl-C while the stems were flushed, the
mix was rendered and the models loaded — long enough that the owner did not
know whether to press Ctrl-C again (which cancels the work, though the
recording is kept).

Everything here is policy about *what the terminal sees*; the lines go
through :func:`vemoizer.status_line.say`, which clears the REC status first
so no message is ever glued onto it. The logger keeps the full detail either
way (``capture.SkewLogger``), so a quiet console never loses information.
"""

from __future__ import annotations

import signal
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path

from vemoizer.capture import skew_message
from vemoizer.status_line import hms, say

#: Skew past this reaches the console even without ``--verbose``.
#:
#: Both stems are written by one process on one session clock, so a healthy
#: call sits within tens of milliseconds (a real 38-minute call stayed inside
#: ±125 ms). Half a second is far outside anything clock drift produces; by
#: then the turn order in the mixed transcript is wrong enough to be worth
#: interrupting the user for.
ABNORMAL_SKEW_S = 0.5

#: What to tell the user about a hot input, per track. Clipping is
#: irreversible, so it must be said while the call is still running; "too
#: loud" plus the fix, because "input exceeds +/-1.0" is not actionable.
_CLIP_NOTICES = {
    "mic": "warning: the microphone is too loud and the audio is clipping; "
    "lower the input gain",
    "tap": "warning: the call audio is too loud and it is clipping; lower its volume",
}


def clip_notice(track: str) -> str:
    """The one-line wording for a clipping *track* (``mic`` or ``tap``)."""
    return _CLIP_NOTICES[track]


def kept_message(call_dir: Path) -> str:
    """The line telling the user the call folder survived."""
    return f"audio kept: {call_dir.as_posix()}/"


def clip_notifier(track: str) -> Callable[[], None]:
    """The callback :class:`~vemoizer.capture.TrackWriter` calls once per recording.

    It goes through the same echo as every other ``listen`` line, which ends
    the ``● REC`` status line first; the writer runs it on its pump thread,
    where ``say`` is lock-protected.
    """
    line = clip_notice(track)
    return lambda: say(line)


class SkewConsole:
    """When a measured skew is worth the terminal (issue #236).

    The once-a-minute line is debugging output: ``--verbose`` or the log file,
    never an unasked-for stream of lines. Without ``--verbose`` the console
    hears only about an abnormal skew, once per excursion: the line is printed
    when the magnitude first crosses :data:`ABNORMAL_SKEW_S` and not again
    until it has fallen back below the threshold and crossed it once more.
    """

    def __init__(
        self,
        *,
        verbose: bool = False,
        threshold_s: float = ABNORMAL_SKEW_S,
        echo: Callable[[str], None] = say,
    ) -> None:
        self._verbose = verbose
        self._threshold = threshold_s
        self._echo = echo
        self._above = False

    def interval(self, skew: float) -> None:
        """The logging-interval line: the console only asked for it."""
        if self._verbose:
            self._echo(skew_message(skew))

    def watch(self, skew: float) -> None:
        """Track the magnitude every tick; announce a fresh excursion once."""
        if self._verbose:
            return  # the interval line already prints the skew as it drifts
        above = abs(skew) >= self._threshold
        if above and not self._above:
            self._echo(
                f"warning: {skew_message(skew)} "
                f"(past {self._threshold * 1000:.0f} ms; the tracks are drifting)"
            )
        self._above = above

    def final(self, skew: float) -> None:
        """The stop line: console only with ``--verbose``."""
        if self._verbose:
            self._echo(skew_message(skew))


class CallStatus:
    """The lines between Ctrl-C and the preset's own stage display.

    After the interrupt ``listen`` joins the capture threads, flushes the FLAC
    stems, renders ``call.flac`` with ffmpeg and loads the models — minutes
    for a long call, with nothing printed. One line per phase, the first one
    before any of it, so the owner knows the audio is already on disk and
    what a second Ctrl-C would cost. ``--quiet`` silences the phase lines just
    as it silences the stage display; what it never silences is the line that
    says the recording was kept.
    """

    def __init__(
        self, *, quiet: bool = False, echo: Callable[[str], None] = say
    ) -> None:
        self._quiet = quiet
        self._echo = echo

    def stopped(self, elapsed_s: float, call_dir: Path) -> None:
        """Printed the moment Ctrl-C ends the recording, before any shutdown."""
        self._echo(
            f"■ Recording stopped at {hms(elapsed_s)}. "
            f"The recording is saved in {call_dir}."
        )
        self._echo(
            "Pressing Ctrl-C again cancels the transcription; the recording is kept."
        )

    @contextmanager
    def saving(self, *, announce: bool = True) -> Iterator[None]:
        """Announce the save and hold off Ctrl-C until the stems are closed."""
        if announce:
            self._phase("Saving the tracks…")
        with ignore_interrupts():
            yield

    def mixing(self) -> None:
        self._phase("Mixing the tracks…")

    def transcribing(self) -> None:
        self._phase("Transcribing — this can take several minutes for a long call.")

    def _phase(self, line: str) -> None:
        if not self._quiet:
            self._echo(line)


@contextmanager
def ignore_interrupts() -> Iterator[None]:
    """Hold off Ctrl-C while the stems are flushed and closed.

    Saving takes seconds, and an interrupt inside it would skip the FLAC
    writers' close and leave the stems unfinalised. Ctrl-C is ignored until
    the tracks are on disk, so "Ctrl-C again cancels the transcription" stays
    true. Off the main thread signal handlers cannot be changed; then this
    is a no-op.
    """
    if threading.current_thread() is not threading.main_thread():
        yield
        return
    previous = signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, previous)
