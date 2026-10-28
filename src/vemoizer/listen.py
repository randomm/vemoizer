"""``vemoizer listen``: record a call, mix it, run the meeting preset (issue #173).

Orchestration only. Hardware is reached through ``capture_sources`` (the one
module that touches catap/ffmpeg), the per-source threads and the preflight
live in ``listen_capture``, the stems are ``capture.TrackWriter`` files, and
the ``--live`` draft around the recording is owned by ``listen_live``.
The recording is kept under ``calls/<timestamp>/`` and is never reported as
an output file.
"""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

from vemoizer import capture_sources, speaker_clips
from vemoizer.caffeinate import caffeinate_context
from vemoizer.capture import (
    Chunk,
    SilenceWatchdog,
    SkewLogger,
    TrackWriter,
    skew_message,
)
from vemoizer.cli_support import parse_speakers
from vemoizer.listen_capture import (
    START_WAIT_S,
    TAP_START_WAIT_S,
    Problem,
    TrackState,
    preflight,
    pump,
)
from vemoizer.listen_live import run_session
from vemoizer.owner_config import RunOwner, resolve_run_owner
from vemoizer.preset_cli import preset_interrupt_exit
from vemoizer.preset_interrupt import InterruptTracker, set_interrupt_stage
from vemoizer.presets import LANGUAGE_VALUES
from vemoizer.status_line import rec_tick, say
from vemoizer.tracks import TrackPaths

if TYPE_CHECKING:
    from vemoizer.live_session import LiveModels, LiveSession

CALLS_DIR = "calls"
MIX_NAME = "call.flac"
ME_NAME = "me.flac"
THEM_NAME = "them.flac"

#: Seconds of silence on the system side before the watchdog warns once.
WATCHDOG_S = 180.0

#: Seconds between record-loop ticks (REC line, watchdog, duration).
TICK_S = 0.5

#: Seconds each pump thread may take to drain after its source stops.
_JOIN_S = 10.0

CONSENT = (
    "Recording a call: tell everyone on it that you are recording and get "
    "their consent first. Ctrl-C stops the recording."
)
_CATAP_HINT = (
    "error: live capture needs the listen extra: run `uv sync --extra listen` "
    "or `pip install 'vemoizer[listen]'`"
)
_LABELS = {"mic": "microphone", "tap": "system audio"}

#: The system sound the doctor probe plays while the tap listens (issue #174).
TONE_PATH = "/System/Library/Sounds/Tink.aiff"
_MIC_HINT = (
    "check Privacy & Security > Microphone for Ghostty (or the terminal app "
    "that runs vemoizer), restart Ghostty, keep the lid open, or pick the "
    "device with `listen --mic NAME`"
)
_TAP_HINT = (
    "check Privacy & Security > Screen & System Audio Recording > System Audio "
    "Recording Only for Ghostty (or the terminal app that runs vemoizer), "
    "restart Ghostty, and check the default output device"
)
_NOT_STARTED_DEADLINES_S = {"mic": START_WAIT_S, "tap": TAP_START_WAIT_S}


def _not_started_phrase(track: str) -> str:
    """The start-failure detail: how long the source was given (issue #205)."""
    return f"no audio within {int(_NOT_STARTED_DEADLINES_S[track])} s"


_PROBE_CATAP_HINT = (
    "install the listen extra: run `uv sync --extra listen` or "
    "`pip install 'vemoizer[listen]'`"
)

Sleep = Callable[[float], None]
Hardware = Callable[[str | None], tuple[Any, Any]]


@dataclass(frozen=True)
class ListenRun:
    """The ``listen`` options after the CLI has parsed them."""

    me: str | None = None
    mic: str | None = None
    duration: float | None = None
    yes: bool = False
    quiet: bool = False
    config_path: str | None = None
    glossary_path: str | None = None
    repair: bool | None = True
    no_diarize: bool = False
    speakers: str | None = None
    language: str = "auto"
    live: bool = False
    keep_live: bool = False


def catap_installed() -> bool:
    """Whether the ``listen`` extra is present (checked without importing it)."""
    return importlib.util.find_spec("catap") is not None


@dataclass(frozen=True)
class ProbeResult:
    """One ``doctor --listen`` line: ``name``, pass/fail, and the fix hint."""

    name: str
    ok: bool
    hint: str = ""


class _Discard:
    """A track writer for the probe: the audio is judged, never kept."""

    def write(self, chunk: Chunk) -> bool:
        return True


def _probe_hint(track: str, kind: str | None) -> str:
    if kind is None:
        return ""
    if kind == "not_started":
        return f"did not start ({_not_started_phrase(track)})"
    return _MIC_HINT if track == "mic" else _TAP_HINT


def probe_capture(
    *,
    hardware: Hardware | None = None,
    clock: Callable[[], float] = time.monotonic,
    sleep: Sleep = time.sleep,
) -> list[ProbeResult]:
    """Check the catap install and both capture sources; nothing is recorded.

    The mic must not be exact zeros for its first 2 s. The system tap must not
    be exact zeros while the Tink tone plays through the speaker clip seam,
    so a revoked Screen & System Audio grant is caught before a call.
    """
    if hardware is None and not catap_installed():
        return [ProbeResult("catap importable", False, _PROBE_CATAP_HINT)]
    open_sources = hardware or capture_sources._open_real_sources
    try:
        mic, tap = open_sources(None)
    except (OSError, RuntimeError) as exc:
        return [
            ProbeResult("catap importable", True),
            ProbeResult("capture devices", False, str(exc)),
        ]
    mic_state, tap_state = TrackState("mic"), TrackState("tap")
    threads = [pump(mic, _Discard(), mic_state), pump(tap, _Discard(), tap_state)]
    try:
        speaker_clips._run_player(["afplay", TONE_PATH])
        problems = preflight([mic_state, tap_state], clock, sleep, clock())
    finally:
        _shutdown([mic, tap], threads)
    kinds = {p.track: p.kind for p in problems}
    return [
        ProbeResult("catap importable", True),
        ProbeResult(
            "microphone (2 s exact-zero check)",
            "mic" not in kinds,
            _probe_hint("mic", kinds.get("mic")),
        ),
        ProbeResult(
            "system audio (tone + tap, exact-zero check)",
            "tap" not in kinds,
            _probe_hint("tap", kinds.get("tap")),
        ),
    ]


def _run_mix(argv: list[str]) -> None:
    subprocess.run(argv, check=True, capture_output=True)


def build_mix_argv(me: Path | None, them: Path | None, out: Path) -> list[str]:
    """ffmpeg argv that mixes the two stems into one mono 16 kHz FLAC.

    ``amix`` with ``duration=longest`` keeps the whole call, and the stems
    stay on their own clocks (no ``adelay``). A missing stem gives a
    single-track mix of the other.
    """
    inputs = [p for p in (me, them) if p is not None]
    argv = ["ffmpeg", "-hide_banner", "-nostdin", "-y"]
    for path in inputs:
        argv += ["-i", str(path)]
    if len(inputs) == 2:
        argv += ["-filter_complex", "amix=inputs=2:normalize=0:duration=longest"]
    argv += ["-ac", "1", "-ar", "16000", "-c:a", "flac", str(out)]
    return argv


def _echo(line: str) -> None:
    say(line)


def kept_message(call_dir: Path) -> str:
    """The line telling the user the call folder survived."""
    return f"audio kept: {call_dir.as_posix()}/"


def _report_problems(problems: Sequence[Problem]) -> bool:
    """Print each preflight finding; True when one of them stops the run."""
    fatal = False
    for problem in problems:
        label = _LABELS[problem.track]
        if problem.kind == "not_started":
            _echo(
                f"error: the {label} source did not start "
                f"({_not_started_phrase(problem.track)})"
            )
            fatal = True
        elif problem.track == "mic":
            _echo(
                "error: the microphone delivered 2 s of exact zeros (a dead input). "
                "Check Privacy & Security > Microphone for Ghostty, restart Ghostty, "
                "keep the lid open, or pick the device with --mic NAME."
            )
            fatal = True
        else:
            _echo(
                "note: system audio is silent so far. That is fine if nothing "
                "is playing yet; if the call stays silent in the transcript, "
                "check Privacy & Security > Screen & System Audio Recording > "
                "System Audio Recording Only for Ghostty, restart Ghostty, and "
                "check the default output device."
            )
    return fatal


def _skew_tick(skew: SkewLogger, mic: TrackState, tap: TrackState, t0: float) -> None:
    """Echo the measured me/them skew when a logging interval is crossed."""
    if (me_t := mic.end_t) is None or (them_t := tap.end_t) is None:
        return
    if (line := skew.observe(me_t - t0, them_t - t0)) is not None:
        _echo(skew_message(line))


def _record(
    clock: Callable[[], float],
    sleep: Sleep,
    t0: float,
    duration: float | None,
    mic: TrackState,
    tap: TrackState,
    output_name: Callable[[], str],
    skew: SkewLogger,
) -> None:
    """Tick until the duration ends; a Ctrl-C is caught by the caller."""
    rewrite = sys.stderr.isatty()
    while True:
        elapsed = clock() - t0
        rec_tick(elapsed, rewrite=rewrite)
        _skew_tick(skew, mic, tap, t0)
        if tap.check_watchdog():
            _echo(
                "warning: no sound from the call for 3 min "
                f"(output device: {output_name()}); check the default output device"
            )
        if duration is not None and elapsed >= duration:
            break
        sleep(TICK_S)
    if rewrite:
        sys.stderr.write("\n")


def capture_call(
    run: ListenRun,
    call_dir: Path,
    open_sources: Hardware,
    clock: Callable[[], float],
    sleep: Sleep,
    output_name: Callable[[], str],
    tracker: InterruptTracker,
    t0: float,
    live: LiveSession | None,
) -> tuple[TrackWriter, TrackWriter] | None:
    """Preflight and record; the closed writers, or None when preflight failed.

    The first Ctrl-C is caught here: it ends the recording and the tracks are
    finalized normally. A Ctrl-C during finalize propagates to the caller.
    """
    me_w = TrackWriter(call_dir / ME_NAME, t0)
    them_w = TrackWriter(call_dir / THEM_NAME, t0)
    if live is not None:
        me_w.subscribe(live.subscriber("me"))
        them_w.subscribe(live.subscriber("them"))
    try:
        mic, tap = open_sources(run.mic)
    except (OSError, RuntimeError) as exc:
        _echo(f"error: could not open the capture devices: {exc}")
        me_w.close()
        them_w.close()
        return None
    mic_state = TrackState("mic")
    tap_state = TrackState("tap", SilenceWatchdog(WATCHDOG_S, clock))
    threads = [pump(mic, me_w, mic_state), pump(tap, them_w, tap_state)]
    skew = SkewLogger()
    failed = False
    try:
        failed = _report_problems(preflight([mic_state, tap_state], clock, sleep, t0))
        if not failed:
            set_interrupt_stage(tracker, "recording")
            _record(
                clock, sleep, t0, run.duration, mic_state, tap_state, output_name, skew
            )
    except KeyboardInterrupt:
        pass  # the first Ctrl-C ends the recording
    _shutdown([mic, tap], threads)
    _skew_final(skew, mic_state, tap_state, t0)
    me_w.close()
    them_w.close()
    return None if failed else (me_w, them_w)


def _skew_final(skew: SkewLogger, mic: TrackState, tap: TrackState, t0: float) -> None:
    """Log the skew once more at stop, from the drained tracks."""
    if (me_t := mic.end_t) is None or (them_t := tap.end_t) is None:
        return
    _echo(skew_message(skew.final(me_t - t0, them_t - t0)))


def _shutdown(sources: Sequence[Any], threads: Sequence[Any]) -> None:
    """Stop each source, then let its pump drain to the end of its queue."""
    for source in sources:
        source.stop()
    for thread in threads:
        thread.join(timeout=_JOIN_S)


def mix_call(
    call_dir: Path,
    me_w: TrackWriter,
    them_w: TrackWriter,
    start_epoch: float,
) -> Path | int:
    """Mix the stems into ``call.flac``: its path, or the exit code to stop with."""
    me = call_dir / ME_NAME if me_w.position_s > 0 else None
    them = call_dir / THEM_NAME if them_w.position_s > 0 else None
    mix = call_dir / MIX_NAME
    try:
        _run_mix(build_mix_argv(me, them, mix))
    except KeyboardInterrupt:
        mix.unlink(missing_ok=True)
        _echo(kept_message(call_dir))
        return 130
    except (OSError, subprocess.CalledProcessError) as exc:
        mix.unlink(missing_ok=True)
        _echo(f"error: mixing the call failed: {exc}; {kept_message(call_dir)}")
        return 1
    # The call is dated by when it started, not when it was stopped.
    os.utime(mix, (start_epoch, start_epoch))
    return mix


def decode_call(
    run: ListenRun,
    call_dir: Path,
    owner: RunOwner,
    mix: Path,
    speakers: Any,
    tracker: InterruptTracker,
) -> int:
    """The meeting preset on the mix; the display exists only from here on."""
    from vemoizer.batch import run_preset
    from vemoizer.progress_wiring import make_batch_display

    display = make_batch_display(quiet=run.quiet)
    tracks = TrackPaths(me=call_dir / ME_NAME, them=call_dir / THEM_NAME)
    try:
        return run_preset(
            [mix],
            command="meeting",
            config_path=run.config_path,
            glossary_path=run.glossary_path,
            repair=run.repair,
            diarize=False if run.no_diarize else None,
            speakers=speakers,
            language=run.language.strip().lower(),
            quiet=run.quiet,
            yes=run.yes,
            display=display,
            tracker=tracker,
            tracks=tracks,
            owner=owner,
        )
    except KeyboardInterrupt:
        preset_interrupt_exit(tracker, display, kept=call_dir)
    except ValueError as exc:
        _echo(f"error: {exc}")
        return 2
    finally:
        if display is not None:
            display.close()


def run_listen(
    run: ListenRun,
    *,
    tracker: InterruptTracker,
    hardware: Hardware | None = None,
    clock: Callable[[], float] = time.monotonic,
    sleep: Sleep = time.sleep,
    input_fn: Callable[[str], str] = input,
    isatty: Callable[[], bool] | None = None,
    output_name: Callable[[], str] | None = None,
    live_models: LiveModels | None = None,
) -> int:
    """Record one call, then run the meeting preset on its mix; the exit code.

    The hardware, clock, sleep, prompt, TTY and live-model seams default to the
    real ones; unit tests inject fakes. A Ctrl-C during decoding exits 130 via
    the preset handler, which also names the kept call folder.
    """
    language = run.language.strip().lower()
    if language not in LANGUAGE_VALUES:
        known = ", ".join(LANGUAGE_VALUES)
        _echo(f"error: unknown language {language!r} (known: {known})")
        return 2
    # Validated before any folder or prompt: a typo must cost nothing.
    speakers = parse_speakers(run.speakers)
    if hardware is None and not catap_installed():
        _echo(_CATAP_HINT)
        return 2
    _echo(CONSENT)
    interactive = not run.yes and (isatty or sys.stdin.isatty)()
    owner = resolve_run_owner(run.me, prompt=interactive, input_fn=input_fn, echo=_echo)
    start_epoch = time.time()
    call_dir = Path(CALLS_DIR) / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    try:
        call_dir.mkdir(parents=True, exist_ok=False)
    except OSError as exc:
        _echo(f"error: cannot create {call_dir}: {exc}")
        return 1
    with caffeinate_context():
        return run_session(
            run,
            call_dir,
            owner,
            tracker,
            hardware or capture_sources._open_real_sources,
            clock,
            sleep,
            output_name or capture_sources.current_output_name,
            start_epoch,
            speakers,
            live_models,
        )
