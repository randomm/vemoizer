"""Loudnorm pass-1 caching in the group boundary decode (issue #135, fix pass 2).

The lens review found that ``grouping_decode._decode_edge_window`` calls
the raw ``_measure_loudnorm`` directly, bypassing the memoized
``_measurement_for`` cache. For an N-part group, every boundary tail/head
edge-window decode re-runs the FULL-file pass-1 measurement — about
2(N-1) uncached full-file measurements for a two-part group — defeating
the cache and contradicting the module docstring's "one pass-1 per
unchanged file" contract.

This test uses a counting spy around ``_measure_loudnorm`` and a real
tiny two-part synthetic group (ffmpeg lavfi, .m4a output) to assert that
pass 1 ran ONCE PER PART FILE in total across ``decode_boundaries`` +
``part_offsets`` + ``pcm_duration_seconds``.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from vemoizer import loudnorm as ln
from vemoizer.grouping_concat import part_offsets
from vemoizer.ingest import pcm_duration_seconds

HAS_FFMPEG = shutil.which("ffmpeg") is not None


def _make_m4a_lavfi(path: Path, seconds: float = 2.0) -> Path:
    """Generate a real .m4a from a lavfi source (needs real ffmpeg)."""
    subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"sine=frequency=440:duration={seconds}",
            "-ac",
            "1",
            "-ar",
            "16000",
            str(path),
        ],
        check=True,
        capture_output=True,
    )
    return path


@pytest.fixture(autouse=True)
def _clean_cache():
    ln._MEASURE_CACHE.clear()
    yield
    ln._MEASURE_CACHE.clear()


class _FakePopen:
    """A minimal fake Popen for the plain decode (no measurement)."""

    def __init__(self) -> None:
        self.returncode = 0
        self.stdout = _FakePipe(b"")
        self.stderr = b""

    def poll(self) -> int | None:
        return self.returncode

    def wait(self, timeout: float | None = None) -> int:
        return self.returncode

    def kill(self) -> None:
        pass


class _FakePipe:
    def __init__(self, data: bytes = b"") -> None:
        self._data = data
        self._pos = 0

    def read(self, size: int | None = None) -> bytes:
        chunk = (
            self._data[self._pos :]
            if size is None
            else self._data[self._pos : self._pos + size]
        )
        self._pos += len(chunk)
        return chunk


def test_group_boundary_pass1_runs_once_per_file(
    tmp_path: Path,
) -> None:
    """With a two-part group, pass 1 must run ONCE PER PART FILE in total
    across decode_boundaries (edge windows) + part_offsets +
    pcm_duration_seconds.

    Before the fix: ``_decode_edge_window`` called raw ``_measure_loudnorm``
    (bypassing the cache), so for a two-part group with one boundary (tail
    of part 0, head of part 1), pass 1 ran at least 2 extra times from the
    edge windows on top of the cached part_offsets + pcm_duration_seconds
    measurements — total 4 calls instead of 2.

    After the fix: all call sites use the memoized ``measurement_for``,
    so pass 1 runs exactly ONCE PER PART FILE = 2 total.
    """
    if not HAS_FFMPEG:
        pytest.skip("ffmpeg not available")

    part_a = _make_m4a_lavfi(tmp_path / "part_a.m4a", 25.0)
    part_b = _make_m4a_lavfi(tmp_path / "part_b.m4a", 25.0)

    calls: list[str] = []

    def fake_measure(p: Path):
        calls.append(p.name)
        return ln.LoudnormMeasurement(-16.0, -3.0, 0.5, -26.0, 0.0)

    def fake_popen(argv, **kw):
        return _FakePopen()

    def fake_run(argv, **kw):
        return subprocess.CompletedProcess(
            argv, 0, stdout=np.zeros(32000, dtype=np.float32).tobytes(), stderr=b""
        )

    # Patch ffprobe to return a valid duration (2.0s) so the edge windows
    # actually decode (the real ffprobe would work, but we want to be sure
    # the edge windows run through the measurement path).
    def fake_probe(path):
        return 2.0

    with (
        patch.object(ln, "_measure_loudnorm", side_effect=fake_measure),
        patch("vemoizer.ingest.subprocess.Popen", side_effect=fake_popen),
        patch("vemoizer.ingest.subprocess.run", side_effect=fake_run),
        patch("vemoizer.grouping_decode.subprocess.run", side_effect=fake_run),
        patch("vemoizer.grouping.probe_duration_seconds", side_effect=fake_probe),
    ):
        # 1. Boundary decode (edge windows): tail of part_a, head of part_b
        from vemoizer.grouping_decode import decode_boundaries

        # Use a no-op transcribe_fn to avoid loading Whisper
        decode_boundaries(
            [part_a, part_b], lambda audio: {"text": ""}, preprocess="loudnorm"
        )

        # 2. Part offsets (decodes each part's full duration)
        part_offsets([part_a, part_b], preprocess="loudnorm")

        # 3. pcm_duration_seconds on part_a (as the pipeline would)
        pcm_duration_seconds(part_a, preprocess="loudnorm")

    # Pass 1 must have run exactly ONCE PER PART FILE = 2 total
    assert len(calls) == 2, (
        f"expected 2 pass-1 calls (one per part file), got {len(calls)}: {calls}"
    )
    assert calls[0] == "part_a.m4a"
    assert calls[1] == "part_b.m4a"
