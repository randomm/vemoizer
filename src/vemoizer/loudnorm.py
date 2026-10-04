"""Opt-in two-pass ``loudnorm`` preprocessing for far-field recordings
(issue #135).

``preprocess_audio`` runs ffmpeg's ``loudnorm`` filter in the two-pass
form:

* **Pass 1** measures the source: the same mono-16 kHz decode argv as the
  plain decode with ``loudnorm=...:print_format=json`` appended and the
  output sent to the null muxer (``-f null -``), so no audio is produced
  and there is no ambiguity about a half-processed result. The measured
  JSON (``input_i`` / ``input_tp`` / ``input_lra`` / ``input_thresh`` /
  ``target_offset``) arrives on stderr.
* **Pass 2** is the normal decode with
  ``loudnorm=...:measured_I=..:measured_TP=..:measured_LRA=..:measured_thresh=..:offset=..:linear=true``
  so the measured values drive a deterministic linear gain
  (``linear=true`` never applies dynamic gain beyond the measurement —
  silent input can never be amplified unboundedly).

The filter is appended AFTER ``-ar 16000 -ac 1`` (the resample/mono
downmix happens first, chosen by measurement on a synthetic pink-noise
signal: the true peak stays below 0 dBFS, and the filter operates on the
exact signal the decodes see). The audio contract is unchanged: pass 2
still emits 16 kHz mono float32, and a linear gain cannot change the
sample count, so decoded sample counts across the pipeline stay equal.

Fail open: if pass 1 fails, its JSON is missing/malformed, or a measured
value is ``-inf``/NaN/absurd (silent input), log ONE warning (file name
only, no transcript or path content) and run the plain unprocessed
decode — never abort the run, never emit an unbounded gain.

The measurement itself is memoized per file (``_measurement_for``):
the pass-1 decode is a full-file pass, and a group run measures it
several times (``part_offsets`` per part, the group's own decode, the
sidecar durations). The key is ``(resolved path, size, mtime)`` — an
unchanged file is measured once, a changed or deleted file is
re-measured, and a FAILED measurement is cached too (``None``, one
warning, no re-measurement of a file that just failed).
"""

from __future__ import annotations

import contextlib
import json
import logging
import math
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .ingest import _FFMPEG_AUDIO_ARGS, ingest_audio

logger = logging.getLogger(__name__)

# Target levels (issue #135): I=-16 LUFS (speech-centric target; the EBU
# R128 broadcast standard -23 LUFS is far too quiet for near-field
# dictation), TP=-1.5 dBFS (headroom so the gain cannot clip float32),
# LRA=11 (the ffmpeg loudnorm default dynamic-range ceiling, which
# linear=true then honours).
LOUDNORM_I = -16
LOUDNORM_TP = -1.5
LOUDNORM_LRA = 11

# Pass 1 budget: the measurement decodes the whole file at ~100x
# realtime (measured: 10 min of pink noise in ~7 s). A typical memo
# costs well under a second, so this covers absurdly long inputs.
_MEASURE_TIMEOUT_S = 600.0

# Bounded measurement-cache size: a cache hit saves a full-file ffmpeg
# pass, so a handful of files is plenty — a run's hot set is one file
# (the merged group) plus its parts; an evicted entry is re-measured on
# its next access. Failures (``None``) are cached too (a failing file
# must not be re-measured — and re-warned — on every call).
_MEASURE_CACHE_MAX = 32

#: The measurement memo, keyed by (resolved path, size, mtime).
_MEASURE_CACHE: dict[tuple[str, int, int], LoudnormMeasurement | None] = {}

# Sanity bounds on measured values (ffmpeg emits "-inf" for silent
# input): anything outside these is treated as "no measurement" and the
# run falls back to the plain decode (never an unbounded gain).
_MIN_I = -100.0
_MAX_I = 0.0
_MIN_TP = -100.0
_MAX_TP = 0.0
_MIN_LRA = 0.0
_MAX_LRA = 100.0
_MAX_ABS_THRESH = 100.0
_MAX_ABS_OFFSET = 100.0


@dataclass(frozen=True)
class LoudnormMeasurement:
    """The measured input values from a loudnorm pass 1 (the JSON block)."""

    input_i: float
    input_tp: float
    input_lra: float
    input_thresh: float
    offset: float


def loudnorm_pass1_filter() -> str:
    """The loudnorm filter string for pass 1 (measurement, JSON on stderr).

    ``print_format=json`` (equivalent to ``print_format=1``; ``json`` is
    used for clarity — the issue text mentions both forms).
    """
    return (
        f"loudnorm=I={LOUDNORM_I}:TP={LOUDNORM_TP}:LRA={LOUDNORM_LRA}:print_format=json"
    )


def loudnorm_pass2_filter(m: LoudnormMeasurement) -> str:
    """The loudnorm filter string for pass 2 (the actual normalization).

    ``measured_*`` values drive a linear gain (``linear=true``): the gain
    is computed from the measurement alone, so it is deterministic and
    bounded — measurements that would imply an unbounded gain (silent
    input) are rejected by :func:`_sanitize_measurement` before this
    string is ever built.
    """
    return (
        f"loudnorm=I={LOUDNORM_I}:TP={LOUDNORM_TP}:LRA={LOUDNORM_LRA}"
        f":measured_I={m.input_i:g}:measured_TP={m.input_tp:g}"
        f":measured_LRA={m.input_lra:g}:measured_thresh={m.input_thresh:g}"
        f":offset={m.offset:g}:linear=true"
    )


def _sanitize_measurement(raw: dict[str, Any]) -> LoudnormMeasurement | None:
    """Validate the parsed loudnorm JSON; ``None`` when unusable.

    Rejects missing keys, non-numeric values, NaN, and ``-inf``/absurd
    magnitudes (ffmpeg reports silent input as ``"input_i": "-inf"``).
    """
    try:
        vals = {
            "input_i": float(raw["input_i"]),
            "input_tp": float(raw["input_tp"]),
            "input_lra": float(raw["input_lra"]),
            "input_thresh": float(raw["input_thresh"]),
            "offset": float(raw.get("target_offset", 0.0)),
        }
    except (KeyError, TypeError, ValueError):
        return None

    bounds = {
        "input_i": (_MIN_I, _MAX_I),
        "input_tp": (_MIN_TP, _MAX_TP),
        "input_lra": (_MIN_LRA, _MAX_LRA),
        "input_thresh": (-_MAX_ABS_THRESH, _MAX_ABS_THRESH),
        "offset": (-_MAX_ABS_OFFSET, _MAX_ABS_OFFSET),
    }
    for key, (lo, hi) in bounds.items():
        v = vals[key]
        if not math.isfinite(v) or not (lo <= v <= hi):
            return None
    return LoudnormMeasurement(**vals)


def _json_blocks(text: str) -> list[str]:
    """Brace-balanced ``{...}`` blocks in *text* (nesting-safe).

    A depth walk (string-aware) instead of a regex: ffmpeg's warning
    lines contain braces (``[in#0 @ 0x…]``) and the loudnorm JSON itself
    nests only at top level, so the balanced walk is the robust parse.
    """
    blocks: list[str] = []
    depth = 0
    start = -1
    in_string = False
    escape = False
    for i, ch in enumerate(text):
        if depth == 0:
            if ch == "{":
                depth = 1
                start = i
            continue
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                blocks.append(text[start : i + 1])
    return blocks


def parse_loudnorm_json(stderr_text: str) -> LoudnormMeasurement | None:
    """Extract the loudnorm measurement from ffmpeg stderr.

    The JSON block is the *last* balanced ``{...}`` block that parses as
    a valid measurement — ffmpeg's own log lines appear before it and
    can contain braces, so we walk from the end. A missing/malformed
    block yields ``None`` (the caller fails open to the plain decode).
    """
    for block in reversed(_json_blocks(stderr_text)):
        try:
            raw = json.loads(block)
        except (json.JSONDecodeError, ValueError):
            continue
        if not isinstance(raw, dict):
            continue
        m = _sanitize_measurement(raw)
        if m is not None:
            return m
    return None


def decode_argv(p: Path, preprocess: str | None) -> list[str]:
    """The decode argv for a *streaming* (byte-count) decode.

    Same contract as :func:`vemoizer.ingest.ingest_audio` (the plain
    decode argv, raw f32le on stdout — byte count, never ffprobe), with
    the loudnorm pass-2 filter added when *preprocess* is ``"loudnorm"``
    (the issue #135 consistency rule: part offsets and source durations
    must match the processed decode). The filter is built by the same
    single ``_decode_args`` builder the transcript's pass 2 uses, so the
    two decodes share one argv shape; the measurement comes from the
    in-process cache (one pass-1 per unchanged file, shared with
    ``ingest_audio``'s pass 2), and a measurement failure falls back to
    the plain decode (fail-open, matching :func:`ingest_audio`'s
    behaviour). ``-f null -`` is NOT used here: the output goes to
    stdout for the byte count.
    """
    plain = ["ffmpeg", *_FFMPEG_AUDIO_ARGS, "-i", str(p)]
    if preprocess != "loudnorm":
        return plain
    measurement = _measurement_for(p)
    if measurement is None:
        return plain
    # The same single builder the transcript's pass 2 uses (exactly ONE
    # pass-2 argv builder, issue #135): same filter, same ordering, so
    # the duration decode and the transcript decode stay in lock-step.
    return _decode_args(("-af", loudnorm_pass2_filter(measurement))) + [str(p)]


def _measurement_for(path: Path) -> LoudnormMeasurement | None:
    """The loudnorm measurement for *path*, memoized (fail open).

    Memoized in an in-process cache keyed by ``(resolved path, size,
    mtime)``: a group run measures the same file several times (per-part
    offsets, the group's own decode, the sidecar durations), and pass 1
    is a full-file pass — an unchanged file is measured once, a changed
    or deleted file is re-measured. A FAILED measurement is cached too
    (``None``): a file that just failed is not re-measured (or
    re-warned) on every call, and the warning is emitted by the caller
    that finds ``None`` — once per call site, as before.
    """
    try:
        st = path.stat()
    except OSError:
        # A vanished file has no stable identity — do not cache it.
        return _measure_loudnorm(path)
    key = (str(path.resolve()), st.st_size, st.st_mtime_ns)
    cached = _MEASURE_CACHE.get(key)
    if cached is not None:
        return cached
    if key in _MEASURE_CACHE:
        return None  # a cached failure (``None``)
    m = _measure_loudnorm(path)
    if len(_MEASURE_CACHE) >= _MEASURE_CACHE_MAX:
        _MEASURE_CACHE.pop(next(iter(_MEASURE_CACHE)))
    _MEASURE_CACHE[key] = m
    return m


def _measure_loudnorm(path: Path) -> LoudnormMeasurement | None:
    """Run pass 1 (measurement) and return the sanitized values.

    ``None`` on ANY failure (fail open — the caller runs the plain
    decode): ffmpeg missing (``FileNotFoundError``), an ``OSError`` from
    the stderr temp file, a non-zero exit, a timeout, or a JSON block
    that is missing/malformed/absurd.

    Note: ``-v error`` is NOT used here (unlike the plain decode) because
    ffmpeg 9.x suppresses the loudnorm filter's JSON output (``input_i``
    etc.) at ``-v info`` level — the JSON is only emitted at the default
    ``-v info`` level. The null muxer (``-f null -``) means no audio
    output is produced, so the extra stderr noise from info-level logging
    is harmless.
    """
    argv = [
        "ffmpeg",
        "-nostdin",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_f32le",
        "-af",
        loudnorm_pass1_filter(),
        "-f",
        "null",
        "-",
        "-i",
        str(path),
    ]
    proc: subprocess.Popen | None = None
    stderr_file = tempfile.TemporaryFile()  # noqa: SIM115
    try:
        try:
            proc = subprocess.Popen(
                argv,
                stdout=subprocess.DEVNULL,
                stderr=stderr_file,
                stdin=subprocess.DEVNULL,
            )
        except (FileNotFoundError, OSError):
            # No ffmpeg (or the launch failed): fail open, the plain
            # decode raises a clean IngestError for a missing ffmpeg.
            return None
        proc.wait(timeout=_MEASURE_TIMEOUT_S)
        stderr: bytes
        stderr_file.seek(0)
        stderr = stderr_file.read()
    except subprocess.TimeoutExpired:
        if proc is not None:
            _kill_and_reap(proc)
        return None
    except OSError:
        # The stderr temp file's seek/read failed: the measurement is
        # unusable — fail open like any other measurement failure.
        if proc is not None:
            _kill_and_reap(proc)
        return None
    finally:
        stderr_file.close()

    if proc.returncode != 0:
        return None
    return parse_loudnorm_json(stderr.decode("utf-8", errors="replace"))


def _kill_and_reap(proc: subprocess.Popen) -> None:
    if proc.poll() is None:
        with contextlib.suppress(OSError):
            proc.kill()
    with contextlib.suppress(OSError):
        proc.wait(timeout=5.0)


def preprocess_audio(
    path: Path | str,
    preprocess: str | None = None,
) -> np.ndarray:
    """Decode *path* to 16 kHz mono float32, applying ``loudnorm`` when
    *preprocess* is ``"loudnorm"`` (two-pass form, fail open).

    ``preprocess=None`` (the default): exactly the plain decode —
    ``ingest_audio``'s argv unchanged. ``"loudnorm"``: two-pass
    normalization; any measurement failure falls back to the plain
    decode with ONE warning (file name only).
    """
    p = Path(path)
    if not p.is_file():
        from .ingest import IngestError

        raise IngestError(f"audio file not found: {p}")

    if preprocess == "loudnorm":
        measurement = _measurement_for(p)
        if measurement is None:
            # Fail open: ONE warning, file name only (no transcript or
            # path content), then the plain unprocessed decode.
            logger.warning(
                "loudnorm: measurement failed for %s; running the plain "
                "unprocessed decode (fail-open)",
                p.name,
            )
            return ingest_audio(p)
        return _decode_pass2(p, measurement)

    return ingest_audio(p)


def _decode_args(extra: tuple[str, ...] = ()) -> list[str]:
    """The plain decode argv with *extra* filter args inserted just
    before ``-i`` (the filter chain runs after resample/downmix, before
    the f32le output — the order chosen by the synthetic-signal
    measurement, issue #135). No args → the argv is literally unchanged
    (the flag-less contract a test pins).

    The ``-f f32le -`` output spec comes BEFORE ``-i`` (the raw stream
    output goes to stdout before the input is opened — the ffmpeg CLI
    ordering that keeps ``-f`` from being misinterpreted as the output
    file name). ``-i`` is the last element, so the caller appends the
    path: ``_decode_args(...) + [str(path)]``.
    """
    base = list(_FFMPEG_AUDIO_ARGS)
    # base[:9] = -nostdin -v error -ac 1 -ar 16000 -c:a pcm_f32le
    # base[9:]  = -f f32le -
    return ["ffmpeg", *base[:9], *extra, *base[9:], "-i"]


def _decode_pass2(p: Path, measurement: LoudnormMeasurement) -> np.ndarray:
    """Pass 2: the normal decode with the measured loudnorm filter."""
    from .ingest import IngestError

    argv = _decode_args(("-af", loudnorm_pass2_filter(measurement))) + [str(p)]
    try:
        proc = subprocess.run(argv, capture_output=True, check=False)
    except FileNotFoundError:
        raise IngestError(
            "ffmpeg not found on PATH; install ffmpeg (e.g. `brew install ffmpeg`)"
        ) from None
    if proc.returncode != 0:
        stderr = proc.stderr.decode("utf-8", errors="replace").strip()
        raise IngestError(
            f"ffmpeg failed to decode {p} (exit {proc.returncode}): {stderr}",
            returncode=proc.returncode,
        )
    raw = proc.stdout
    n = len(raw) // 4
    if n == 0:
        return np.zeros(0, dtype=np.float32)
    return np.frombuffer(raw, dtype=np.float32, count=n).copy()
