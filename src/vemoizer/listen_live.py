"""The ``listen --live`` draft around a recording (issue #180).

``listen`` owns the recording; this module owns the optional draft beside it
and the session lifecycle around it: the models load before the recording
clock starts, the draft is opened before recording and closed on every path.
"""

from __future__ import annotations

import shutil
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

import typer

if TYPE_CHECKING:
    from vemoizer.listen import Hardware, ListenRun, Sleep
    from vemoizer.live_session import LiveModels, LiveSession, LoadedModels
    from vemoizer.owner_config import RunOwner
    from vemoizer.preset_interrupt import InterruptTracker


def load_live(run: ListenRun, models: LiveModels | None) -> LoadedModels | None:
    """The draft's models, loaded; None when the draft is off or they will not load."""
    if not run.live:
        return None
    from vemoizer.live_session import default_models
    from vemoizer.vad import VadModelError

    models = models or default_models(run.glossary_path, run.language.strip().lower())
    try:
        return models.load()
    except VadModelError as exc:
        typer.echo(
            f"warning: live draft unavailable: {exc}; recording without it", err=True
        )
        return None


def start_live(
    call_dir: Path,
    owner: str,
    tracker: InterruptTracker,
    loaded: LoadedModels | None,
    t0: float,
    clock: Callable[[], float],
) -> LiveSession | None:
    """The draft over already-loaded models, started at ``t0``; None when off."""
    if loaded is None:
        return None
    # ``listen`` owns the stderr status line, so the draft lines print
    # through its locked writer (lazy import: ``listen`` imports this module).
    from vemoizer.live_session import DRAFT_NAME, LiveSession
    from vemoizer.status_line import say

    return LiveSession.open(
        call_dir / DRAFT_NAME,
        owner=owner,
        models=loaded,
        t0=t0,
        tracker=tracker,
        clock=clock,
        print_line=say,
    )


def stop_live(live: LiveSession | None) -> None:
    """Drain the draft and print its lag summary before the final pass."""
    if live is None:
        return
    from vemoizer.live_session import lag_line

    typer.echo(lag_line(live.stop()), err=True)


def run_session(
    run: ListenRun,
    call_dir: Path,
    owner: RunOwner,
    tracker: InterruptTracker,
    open_sources: Hardware,
    clock: Callable[[], float],
    sleep: Sleep,
    output_name: Callable[[], str],
    start_epoch: float,
    speakers: Any,
    live_models: LiveModels | None,
) -> int:
    """Capture, mix, decode; each stage's failure is an exit code.

    The live draft (``--live``) is opened before recording and closed on every
    path. It is removed only when the final pass succeeded and ``--keep-live``
    is off, so a failed or interrupted call keeps whatever the draft captured.
    The models load before ``t0``, so the recording clock starts only once
    they are ready. The draft labels the owner's speech with ``owner.label``;
    the final pass gets the ``RunOwner`` itself.
    """
    loaded = load_live(run, live_models)
    t0 = clock()
    live = start_live(call_dir, owner.label, tracker, loaded, t0, clock)
    code: int | None = None
    try:
        code = _record_mix_decode(
            run,
            call_dir,
            owner,
            tracker,
            open_sources,
            clock,
            sleep,
            output_name,
            start_epoch,
            speakers,
            t0,
            live,
        )
        return code
    finally:
        if live is not None:
            live.close(remove_draft=code == 0 and not run.keep_live)


def _record_mix_decode(
    run: ListenRun,
    call_dir: Path,
    owner: RunOwner,
    tracker: InterruptTracker,
    open_sources: Hardware,
    clock: Callable[[], float],
    sleep: Sleep,
    output_name: Callable[[], str],
    start_epoch: float,
    speakers: Any,
    t0: float,
    live: LiveSession | None,
) -> int:
    # The recording stages live in ``listen``, which imports this module, so
    # they are imported here at call time to keep the import cycle one-way.
    from vemoizer.listen import capture_call, decode_call, kept_message, mix_call

    try:
        recorded = capture_call(
            run, call_dir, open_sources, clock, sleep, output_name, tracker, t0, live
        )
    except KeyboardInterrupt:
        typer.echo(kept_message(call_dir), err=True)
        return 130
    stop_live(live)
    if recorded is None:
        shutil.rmtree(call_dir, ignore_errors=True)
        return 1
    me_w, them_w = recorded
    if me_w.position_s == 0 and them_w.position_s == 0:
        typer.echo("nothing was recorded; no transcript written", err=True)
        shutil.rmtree(call_dir, ignore_errors=True)
        return 1
    mixed = mix_call(call_dir, me_w, them_w, start_epoch)
    if isinstance(mixed, int):
        return mixed
    return decode_call(run, call_dir, owner, mixed, speakers, tracker)
