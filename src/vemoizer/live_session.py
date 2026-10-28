"""Live draft session for ``vemoizer listen --live`` (issue #180).

Per track, the capture subscriber queues each block the stem wrote; a track
thread feeds a :class:`StreamingVad` and hands closed utterances to one shared
:class:`DecodeWorker`, which appends lines to the draft. The draft is a
single-decode preview, never the deliverable: the post-call pass always runs
the full pipeline.

Nothing here can stop a recording. Every failure is logged and the affected
track (or the whole draft) goes quiet while the stems keep being written.
"""

from __future__ import annotations

import contextlib
import logging
import queue
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TextIO

import numpy as np

from vemoizer.audio_contract import SAMPLE_RATE
from vemoizer.live_decode import Decoded, DecodeWorker, LagReport, Utterance
from vemoizer.live_output import (
    BLEED_WINDOW_S,
    BleedWarner,
    Line,
    LiveWriter,
    Track,
    live_draft,
    render_terminal_line,
)
from vemoizer.preset_interrupt import InterruptTracker
from vemoizer.transcriber import Transcriber
from vemoizer.vad import SpeechSegment, VadModel
from vemoizer.vad_stream import MAX_UTTERANCE_S, StreamingVad

log = logging.getLogger(__name__)

TRACKS: tuple[Track, ...] = ("me", "them")

#: Audio kept per track to slice utterances from: twice the longest utterance.
_TAIL_S = 2 * MAX_UTTERANCE_S

#: The draft's file name inside the call folder.
DRAFT_NAME = "live.md"


@dataclass(frozen=True)
class LoadedModels:
    """The two models a session runs on, loaded and ready."""

    vads: dict[Track, VadModel]
    transcriber: Transcriber


@dataclass(frozen=True)
class LiveModels:
    """Factories for the two models; the real ones come from :func:`default_models`.

    Call :meth:`load` before the recording clock starts: a load takes seconds
    and must not count as recording time.
    """

    vad: Callable[[], VadModel]
    transcriber: Callable[[], Transcriber]

    def load(self) -> LoadedModels:
        vads = {track: self.vad() for track in TRACKS}
        return LoadedModels(vads=vads, transcriber=self.transcriber())


def default_models(glossary_path: str | None, language: str) -> LiveModels:
    """silero-vad and whisper-turbo, the glossary prompt seeding recognition.

    ``language`` ``auto`` lets whisper detect per window (invariant 3); ``fi``
    or ``en`` is the explicit user override, as in ``meeting``.
    """
    from vemoizer.glossary import glossary_prompt, load_glossary
    from vemoizer.vad import load_model
    from vemoizer.whisper_transcriber import WhisperTranscriber

    def transcriber() -> Transcriber:
        prompt = glossary_prompt(load_glossary(glossary_path))
        pinned = None if language == "auto" else language
        return WhisperTranscriber(language=pinned, initial_prompt=prompt)

    return LiveModels(vad=load_model, transcriber=transcriber)


def lag_line(report: LagReport) -> str:
    """The one-line lag summary printed when the live draft stops."""
    u = report.utterance
    if u.count == 0:
        return "live draft lag: no utterances"
    return (
        f"live draft lag: utterance end p50 {u.p50:.1f} s, p90 {u.p90:.1f} s, "
        f"max {u.max:.1f} s (n={u.count})"
    )


class _Tail:
    """The newest ``_TAIL_S`` seconds of one track, addressed by stream sample."""

    def __init__(self) -> None:
        self._buf = np.zeros(0, dtype=np.float32)
        self._end = 0  # stream sample just after the newest sample

    def append(self, block: np.ndarray) -> None:
        keep = round(_TAIL_S * SAMPLE_RATE)
        # Each append builds a new array; older views are never mutated.
        self._buf = np.concatenate([self._buf, block])[-keep:]
        self._end += len(block)

    def slice(self, start: int, end: int) -> np.ndarray:
        origin = self._end - len(self._buf)
        lo = min(max(start, origin), self._end) - origin
        hi = min(max(end, origin), self._end) - origin
        return self._buf[lo:hi]


class _TrackLive:
    """One track's VAD thread: blocks in, closed utterances out to the worker."""

    def __init__(self, track: Track, model: VadModel, worker: DecodeWorker) -> None:
        self._track = track
        self._vad = StreamingVad(model)
        self._worker = worker
        self._tail = _Tail()
        self._blocks: queue.SimpleQueue[np.ndarray | None] = queue.SimpleQueue()
        self._detached = False
        self._thread = threading.Thread(
            target=self._run, name=f"live-vad-{track}", daemon=True
        )

    def start(self) -> None:
        self._thread.start()

    def feed(self, block: np.ndarray) -> None:
        """Runs on the capture pump thread: queue the block and return."""
        if not self._detached:
            self._blocks.put(block)

    def finish(self) -> None:
        """Stop accepting blocks, flush the VAD, and wait for the thread."""
        self._detached = True
        self._blocks.put(None)
        self._thread.join()

    def _run(self) -> None:
        try:
            while (block := self._blocks.get()) is not None:
                self._tail.append(block)
                for seg in self._vad.feed(block):
                    self._submit(seg)
            for seg in self._vad.flush():
                self._submit(seg)
        except Exception as exc:  # the recording must go on without the draft
            self._detached = True
            log.warning(
                "live draft stopped for the %s track: %s; the recording continues",
                self._track,
                exc,
            )

    def _submit(self, seg: SpeechSegment) -> None:
        audio = self._tail.slice(seg.start, seg.end)
        if len(audio) == 0:
            return
        self._worker.submit(
            self._track,
            Utterance(start_s=seg.start / SAMPLE_RATE, audio=audio),
        )


def _line_writer(out: TextIO) -> Callable[[str], None]:
    """One flushed line per call on *out*."""

    def emit(line: str) -> None:
        out.write(line + "\n")
        out.flush()

    return emit


class LiveSession:
    """The live draft for one call: opened before recording, stopped before decode."""

    def __init__(
        self,
        *,
        writer: LiveWriter,
        stack: contextlib.ExitStack,
        path: Path,
        vads: dict[Track, VadModel],
        transcriber: Transcriber,
        t0: float,
        warn: TextIO,
        clock: Callable[[], float],
        print_line: Callable[[str], None] | None = None,
    ) -> None:
        self._writer = writer
        self._stack = stack
        self._path = path
        # The terminal sink (``status_line.say`` from ``listen``) holds the
        # stderr lock and clears the REC line; a bare stream is the fallback.
        self._bleed = BleedWarner(
            print_line if print_line is not None else _line_writer(warn)
        )
        self._print_line = print_line
        self._lines: dict[Track, list[Line]] = {"me": [], "them": []}
        self._report: LagReport | None = None
        self._worker = DecodeWorker(transcriber, self._emit, capture_t0=t0, clock=clock)
        self._tracks = {
            track: _TrackLive(track, vads[track], self._worker) for track in TRACKS
        }
        self._worker.start()
        for track_live in self._tracks.values():
            track_live.start()

    @classmethod
    def open(
        cls,
        path: Path,
        *,
        owner: str,
        models: LoadedModels,
        t0: float,
        warn: TextIO | None = None,
        tracker: InterruptTracker | None = None,
        clock: Callable[[], float] = time.monotonic,
        print_line: Callable[[str], None] | None = None,
    ) -> LiveSession:
        """Create the draft and start the threads; ``t0`` is the recording start."""
        stack = contextlib.ExitStack()
        writer = stack.enter_context(
            live_draft(path, owner=owner, keep=True, tracker=tracker)
        )
        return cls(
            writer=writer,
            stack=stack,
            path=path,
            vads=models.vads,
            transcriber=models.transcriber,
            t0=t0,
            warn=warn if warn is not None else sys.stderr,
            clock=clock,
            print_line=print_line,
        )

    def subscriber(self, track: Track) -> Callable[[np.ndarray], None]:
        """The capture callback for one track's stem."""
        return self._tracks[track].feed

    def stop(self) -> LagReport:
        """Flush both VADs, drain and tear down the decoder, report lag. Idempotent."""
        if self._report is None:
            for track_live in self._tracks.values():
                track_live.finish()
            self._report = self._worker.stop()
        return self._report

    def close(self, *, remove_draft: bool) -> None:
        """Stop if still running, close the draft, and delete it when asked."""
        self.stop()
        self._stack.close()
        if remove_draft:
            self._path.unlink(missing_ok=True)

    def _emit(self, turn: Decoded) -> None:
        """Runs on the decode thread: one draft line, then the headphone check."""
        if not turn.text:
            return
        self._writer.write_turn(turn.start_s, turn.track, turn.text)
        if self._print_line is not None:
            # Streamed preview in the terminal; the callable owns the
            # stderr lock it shares with the REC status line.
            self._print_line(
                render_terminal_line(
                    turn.start_s, self._writer.label_for(turn.track), turn.text
                )
            )
        line: Line = (turn.start_s, turn.text)
        self._lines[turn.track].append(line)
        other: Track = "them" if turn.track == "me" else "me"
        near = [
            other_line
            for other_line in self._lines[other]
            if abs(other_line[0] - turn.start_s) <= BLEED_WINDOW_S
        ]
        if turn.track == "me":
            self._bleed.check([line], near)
        else:
            self._bleed.check(near, [line])
