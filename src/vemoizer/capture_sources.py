"""Live-capture hardware seam (issue #172).

The only module that imports ``catap``/``soxr`` or runs ffmpeg. Both are
imported lazily inside functions, so importing vemoizer never loads Core
Audio. The catap 0.6.0 public API (py.typed) is the authority; catap has no
Context7 docs.

Callbacks from catap's worker thread only enqueue. All resampling happens on
our own worker thread (AGENTS.md: no soxr or stop() on the callback thread).
"""

from __future__ import annotations

import atexit
import logging
import queue
import signal
import subprocess
import threading
import time
from collections.abc import Callable, Iterator
from typing import Any

import numpy as np

from vemoizer.audio_contract import SAMPLE_RATE
from vemoizer.capture import Chunk, drain, is_digital_silence
from vemoizer.capture_tap import TapConverter, UnsupportedTapFormat

log = logging.getLogger(__name__)

STABLE_WAIT_S = 1.5
STABLE_POLL_S = 0.25
MAX_RESTARTS = 3
STOP_TIMEOUT_S = 3.0
READ_BYTES = 4096
ZERO_WATCH_S = 2.0


def default_input_name() -> str:
    """Name of the Mac's default input device, as avfoundation expects it."""
    import catap

    for dev in catap.list_audio_devices():
        if dev.is_default_input:
            return str(dev.name)
    raise RuntimeError("no default input device found")


class FfmpegMicSource:
    """Microphone via ffmpeg avfoundation, converted to 16 kHz mono f32le."""

    def __init__(
        self,
        name: str | None = None,
        *,
        clock: Callable[[], float] = time.monotonic,
        popen: Callable[..., Any] = subprocess.Popen,
        resolve: Callable[[], str] = default_input_name,
    ) -> None:
        self._name = name
        self._clock = clock
        self._popen = popen
        self._resolve = resolve
        self._out: queue.SimpleQueue[Chunk | None] = queue.SimpleQueue()
        self._proc: Any = None
        self._thread: threading.Thread | None = None
        self._stopping = False
        self._failed = False
        self._restarts = 0
        self._leftover = b""
        self._anchor: float | None = None
        self._frames = 0
        self._prev_end: float | None = None
        self._restart_t: float | None = None
        self._zero_warned = False

    @property
    def failed(self) -> bool:
        return self._failed

    def start(self) -> None:
        self._proc = self._launch()
        self._thread = threading.Thread(
            target=self._pump, name="ffmpeg-mic", daemon=True
        )
        self._thread.start()
        atexit.register(self.stop)

    def chunks(self) -> Iterator[Chunk]:
        return drain(self._out)

    def stop(self) -> None:
        self._stopping = True
        proc = self._proc
        if proc is not None and proc.poll() is None:
            proc.send_signal(signal.SIGINT)
            try:
                proc.wait(timeout=STOP_TIMEOUT_S)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()

    def _launch(self) -> Any:
        # Re-resolved on every launch when no --mic is given: a restart after a
        # device switch must follow the new default (AirPods, clamshell).
        name = self._name or self._resolve()
        argv = [
            "ffmpeg", "-hide_banner", "-nostdin", "-fflags", "nobuffer",
            "-flush_packets", "1", "-f", "avfoundation", "-i", f":{name}",
            "-ac", "1", "-ar", "16000", "-af", "aresample=async=1",
            "-f", "f32le", "pipe:1",
        ]  # fmt: skip
        # aresample=async=1 follows the device timestamps: avfoundation delivers
        # ~89 % of nominal samples at 48 kHz, which otherwise drifts the stem (#199).
        proc = self._popen(
            argv,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True,
        )
        threading.Thread(
            target=self._drain_stderr,
            args=(proc,),
            name="ffmpeg-mic-stderr",
            daemon=True,
        ).start()
        return proc

    def _drain_stderr(self, proc: Any) -> None:
        # Own thread: a chatty ffmpeg must never fill this pipe and stall stdout.
        while data := proc.stderr.read(READ_BYTES):
            for line in (
                data.decode("utf-8", "replace").replace("\r", "\n").splitlines()
            ):
                if line.strip():
                    log.debug("ffmpeg mic: %s", line.strip())

    def _pump(self) -> None:
        while True:
            samples = self._read()
            if samples is None:
                if self._stopping or not self._restart():
                    break
                continue
            self._emit(samples)
        self._out.put(None)

    def _read(self) -> np.ndarray | None:
        while True:
            data = self._proc.stdout.read(READ_BYTES)
            if not data:
                return None
            buf = self._leftover + data
            n = len(buf) // 4
            self._leftover = buf[n * 4 :]
            if n:
                return np.frombuffer(buf[: n * 4], dtype="<f4").copy()

    def _emit(self, samples: np.ndarray) -> None:
        now = self._clock()
        disc = 0
        if self._anchor is None:
            # Chunk time is the sample count since (re)start, anchored at the
            # first chunk's arrival minus its duration: jitter never shifts it.
            self._anchor = now - len(samples) / SAMPLE_RATE
            self._frames = 0
            if self._prev_end is not None:
                disc = max(0, round((self._anchor - self._prev_end) * SAMPLE_RATE))
        first_t = self._anchor + self._frames / SAMPLE_RATE
        self._frames += len(samples)
        self._prev_end = first_t + len(samples) / SAMPLE_RATE
        self._restarts = 0
        if (
            self._restart_t is not None
            and now - self._restart_t <= ZERO_WATCH_S
            and not self._zero_warned
            and is_digital_silence(samples)
        ):
            self._zero_warned = True
            log.warning(
                "mic delivered exact zeros for %.0f s after restart "
                "(clamshell built-in mic?)",
                ZERO_WATCH_S,
            )
        self._out.put(Chunk(samples, first_t, disc))

    def _restart(self) -> bool:
        code = self._proc.wait()
        while True:
            if self._stopping:
                return False
            if self._restarts >= MAX_RESTARTS:
                log.error("ffmpeg mic keeps exiting (last code %s); giving up", code)
                self._failed = True
                return False
            self._restarts += 1
            try:
                self._proc = self._launch()
            except (OSError, RuntimeError) as exc:
                log.warning("ffmpeg mic restart failed: %s", exc)
                continue
            log.warning("ffmpeg mic exited (code %s); restarted", code)
            self._leftover = b""
            self._anchor = None
            self._restart_t = self._clock()
            self._zero_warned = False
            return True


def current_output_name() -> str:
    """Name of the Mac's current playback device, for the silence warning (#173)."""
    import catap

    return _default_output_name(catap)


def _default_output(catap: Any) -> Any:
    # Playback device, not is_default_system_output (the alert/sound-effects device).
    return next((d for d in catap.list_audio_devices() if d.is_default_output), None)


def _default_output_name(catap: Any) -> str:
    dev = _default_output(catap)
    return str(dev.name) if dev is not None else "unknown"


def _output_format(catap: Any) -> tuple[tuple[float, int, int], ...]:
    dev = _default_output(catap)
    if dev is None:
        return ()
    return tuple(
        (s.sample_rate, s.num_channels, s.bits_per_channel)
        for s in dev.streams
        if s.direction == "output"
    )


class CatapSystemSource:
    """Global system-output mix via catap ``record_system_audio`` (mono)."""

    def __init__(
        self,
        *,
        clock: Callable[[], float] = time.monotonic,
        poll_s: float = 0.25,
    ) -> None:
        self._clock = clock
        self._poll_s = poll_s
        self._buffers: queue.SimpleQueue[tuple[Any, float] | None] = queue.SimpleQueue()
        self._out: queue.SimpleQueue[Chunk | None] = queue.SimpleQueue()
        self._catap: Any = None
        self._session: Any = None
        self._watch: Any = None
        self._failed = False

    @property
    def failed(self) -> bool:
        return self._failed

    def start(self) -> None:
        import catap
        import soxr
        from catap import events

        self._catap = catap
        self._session = self._record()
        self._watch = events.watch_default_output_device(self._on_output_change)
        threading.Thread(
            target=self._run, args=(soxr,), name="tap-worker", daemon=True
        ).start()

    def chunks(self) -> Iterator[Chunk]:
        return drain(self._out)

    def stop(self) -> None:
        if self._session is not None:
            try:
                self._session.stop()
            except (OSError, RuntimeError) as exc:
                log.warning("stopping the system tap: %s", exc)
        if self._watch is not None:
            self._watch.close()
        self._buffers.put(None)

    def _record(self) -> Any:
        session = self._catap.record_system_audio(on_buffer=self._on_buffer, mono=True)
        session.start()
        return session

    def _on_buffer(self, buf: Any) -> None:
        # catap's worker thread: enqueue only.
        self._buffers.put((buf, self._clock()))

    def _on_output_change(self, _event: Any) -> None:
        log.warning("default output changed to %s", _default_output_name(self._catap))

    def _run(self, soxr: Any) -> None:
        conv = TapConverter(soxr)
        try:
            while True:
                try:
                    item = self._buffers.get(timeout=self._poll_s)
                except queue.Empty:
                    if self._session.capture_failed:
                        self._reopen(conv)
                        if self._failed:
                            break
                    continue
                if item is None:
                    break
                self._send(conv.feed(*item))
        except UnsupportedTapFormat as exc:
            log.error("system tap unusable: %s", exc)
            self._failed = True
        if not self._failed:
            self._send(conv.flush())
        self._out.put(None)

    def _reopen(self, conv: TapConverter) -> None:
        log.warning("system tap failed; reopening")
        try:
            self._session.stop()
        except (OSError, RuntimeError) as exc:
            log.warning("stopping the failed tap: %s", exc)
        while True:
            try:
                self._buffers.get_nowait()
            except queue.Empty:
                break
        self._send(conv.reset())
        try:
            self._session = self._record()
        except (OSError, RuntimeError) as exc:
            log.error("could not reopen the system tap: %s", exc)
            self._failed = True

    def _send(self, chunks: list[Chunk]) -> None:
        for chunk in chunks:
            self._out.put(chunk)


def _wait_output_stable(catap: Any) -> None:
    """Wait (<= STABLE_WAIT_S) for the default output format to stop changing.

    AirPods entering HFP switch the output format when the mic opens (F5), so
    the tap must not open until two polls agree.
    """
    deadline = time.monotonic() + STABLE_WAIT_S
    last = None
    while True:
        current = _output_format(catap)
        if current == last:
            return
        last = current
        if time.monotonic() >= deadline:
            log.warning("default output format still changing; opening the tap anyway")
            return
        time.sleep(STABLE_POLL_S)


def _open_real_sources(
    mic_name: str | None,
) -> tuple[FfmpegMicSource, CatapSystemSource]:
    """The only hardware factory. Mic starts first; the tap waits for the output."""
    import catap

    mic = FfmpegMicSource(mic_name)
    mic.start()
    _wait_output_stable(catap)
    tap = CatapSystemSource()
    tap.start()
    return mic, tap
