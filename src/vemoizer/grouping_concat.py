"""FFmpeg concat demuxer and part offsets for M3 split-recording grouping.

``concat_groups`` joins a group's parts with the ffmpeg concat demuxer
(``-c copy``, no re-encode). ``part_offsets`` measures each part's
cumulative decoded-PCM start offset (``pcm_duration_seconds``: decoded
PCM byte count, never ffprobe or container metadata).

Split from ``grouping.py`` so the concat/offset machinery (impure ffmpeg
I/O) has its own module separate from the pure heuristic and the edit
partition parser.
"""

from __future__ import annotations

import subprocess
import tempfile
import time
from collections.abc import Sequence
from contextlib import suppress
from pathlib import Path

from .grouping_common import PartOffset, _escape_concat_path

# Default cumulative wall-clock budget (seconds) for ``part_offsets``.
# The budget shrinks only by the wall time each decode actually spends
# (bounded above by its granted per-call timeout); a stalled decode burns
# at most its share of the budget, a fast one barely any.
PART_OFFSETS_TOTAL_TIMEOUT = 900.0


def concat_groups(group: Sequence[Path | str]) -> Path:
    """Join *group* into one temp .m4a with the ffmpeg concat demuxer.

    ``-c copy`` (no re-encode). The concat demuxer needs every part to
    share the same audio stream, so a per-part ffprobe check runs first
    and a mismatch raises :class:`GroupingError` naming the offending
    files — ``-c copy`` alone would silently corrupt the output. A
    single-file group is a passthrough: no ffmpeg call, no temp file,
    the input itself is returned.
    """
    from .grouping import GroupingError

    parts = [Path(p) for p in group]
    if len(parts) == 1:
        return parts[0]

    for part in parts:
        if not part.is_file():
            raise GroupingError(f"concat: part file not found: {part}")

    # Deferred import so that monkeypatch.setattr(grouping, "_probe_stream",
    # ...) in tests patches the name that concat_groups actually looks up.
    from .grouping import _probe_stream as _ps

    probes = [(p, _ps(p)) for p in parts]
    (first_part, first_sig) = probes[0]
    mismatched = [(p, sig) for p, sig in probes[1:] if sig != first_sig]
    if mismatched:
        names = ", ".join(p.name for p, _ in probes)
        raise GroupingError(
            f"concat: audio stream mismatch among {names}: "
            f"{first_part.name} has (codec, sample_rate, channels) "
            f"{first_sig!r} but {mismatched[0][0].name} has "
            f"{mismatched[0][1]!r}"
        )

    # The merged group holds every second of the split recording — a
    # world-readable temp file would expose hours of private audio in a
    # shared temp dir, so the concat output lives in a 0o700 directory.
    tmp_dir = Path(tempfile.mkdtemp(prefix="vemoizer-concat-"))
    suffix = parts[0].suffix or ".m4a"
    out_path = tmp_dir / ("group" + suffix)
    list_path = tmp_dir / "concat.txt"
    # Newline/carriage return in a part's name would split the concat list
    # line — the concat demuxer format has no escape for them — so reject
    # such names BEFORE the list file is written.
    for part in parts:
        name = part.name
        if "\n" in name or "\r" in name:
            raise GroupingError(
                f"concat: part file name contains a newline or carriage "
                f"return, which the ffmpeg concat list format cannot "
                f"escape: {name!r}"
            )
    success = False
    try:
        list_path.write_text(
            "".join(_escape_concat_path(p) + "\n" for p in parts),
            encoding="utf-8",
        )
        try:
            proc = subprocess.run(
                [
                    "ffmpeg",
                    "-nostdin",
                    "-v",
                    "error",
                    "-f",
                    "concat",
                    "-safe",
                    "0",
                    "-i",
                    str(list_path),
                    "-c",
                    "copy",
                    "-y",
                    str(out_path),
                ],
                capture_output=True,
                check=False,
                timeout=120.0,
            )
        except FileNotFoundError as e:
            raise GroupingError(
                "ffmpeg not found on PATH; install ffmpeg (e.g. `brew install ffmpeg`)"
            ) from e
        except subprocess.TimeoutExpired:
            raise GroupingError("concat: ffmpeg timed out") from None
        if proc.returncode != 0:
            # Truncate and collapse stderr to a single bounded paragraph
            # that still names the offending files (round 3 finding 8).
            stderr = proc.stderr.decode("utf-8", errors="replace").strip()
            stderr = " ".join(stderr.split())
            if len(stderr) > 200:
                stderr = stderr[:200] + "…"
            names = ", ".join(p.name for p in parts)
            raise GroupingError(
                f"concat: ffmpeg failed for {names} (exit {proc.returncode}): {stderr}"
            )
        success = True
        return out_path
    finally:
        # The temp dir must outlive the call on success: the caller owns
        # the merged file and deletes it after transcription (the list
        # file always goes). On ANY non-success path (missing list file,
        # ffmpeg error, timeout, unexpected exception, KeyboardInterrupt)
        # ffmpeg's ``-y`` may have left a PARTIAL merged file behind, so
        # remove it together with the temp dir — no partial private audio
        # or empty dir leaks, including on KeyboardInterrupt (round 3
        # finding 12).
        list_path.unlink(missing_ok=True)
        if not success:
            out_path.unlink(missing_ok=True)
            with suppress(OSError):
                tmp_dir.rmdir()


def remove_concat_output(merged: Path) -> None:
    """Delete a :func:`concat_groups` temp file AND its 0o700 temp dir.

    The caller (``run_batch``) owns the merged file — delete it on every
    path after transcription, including the error/exit paths, so no
    merged private audio is left behind in the temp dir.
    """
    merged.unlink(missing_ok=True)
    # A non-empty or vanished dir cannot be unlinked either way — the
    # file itself is already gone, so nothing more to clean.
    with suppress(OSError):
        merged.parent.rmdir()


def part_offsets(
    group: Sequence[Path | str], total_timeout: float = PART_OFFSETS_TOTAL_TIMEOUT
) -> list[PartOffset]:
    """Cumulative decoded-PCM start offset per part of *group*.

    Part 1 starts at 0.0; part N starts at the sum of the decoded
    durations of parts 1..N-1. Durations come from
    ``pcm_duration_seconds`` (streamed decoded-PCM byte count) — never
    ffprobe or container metadata (iOS Voice Memos edit lists make
    container duration lie). The decode streams in bounded chunks and
    keeps only the byte count, so measuring a group of hour-long parts
    does not materialise their full float32 PCM.

    The measurement loop is bounded by a CUMULATIVE wall-clock budget
    (*total_timeout*, default :data:`PART_OFFSETS_TOTAL_TIMEOUT` seconds):
    the budget shrinks only by real elapsed wall-clock time, measured with
    ``time.monotonic()`` around each decode — never by the parts' decoded
    audio durations, which are unrelated to how long the decode itself took
    (two 30-minute parts decoded in milliseconds each do not exhaust the
    budget). Each part's decode gets the REMAINING budget as its per-call
    ``timeout``; a decode that stalls until that timeout expires has, by
    definition, spent that wall time and burns it. When the remaining
    budget is exhausted, an :class:`IngestError` is raised naming the part
    that could not be measured and the total budget — the loop never starts
    a decode it cannot possibly afford.
    """
    # Deferred import so that monkeypatch.setattr(grouping,
    # "pcm_duration_seconds", ...) in tests patches the name that
    # part_offsets actually looks up.
    from .grouping import pcm_duration_seconds as _pcm_duration_seconds
    from .ingest import IngestError

    parts = [Path(p) for p in group]
    offsets: list[PartOffset] = []
    total = 0.0
    # The budget shrinks by the wall time each decode actually spent
    # (bounded above by its granted timeout), never by the audio duration.
    remaining = total_timeout
    for i, part in enumerate(parts, start=1):
        offsets.append(
            PartOffset(
                part_number=i,
                source_filename=part.name,
                start_offset=total,
            )
        )
        if remaining <= 0:
            raise IngestError(
                f"part offsets: wall-clock budget of {total_timeout:.0f}s "
                f"exhausted before measuring {part.name}"
            )
        granted = remaining
        start = time.monotonic()
        d = _pcm_duration_seconds(part, timeout=granted)
        remaining -= time.monotonic() - start
        total += d
    return offsets
