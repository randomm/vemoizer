"""Mix argv and dating for ``vemoizer listen`` (issue #173).

``build_mix_argv`` is pure and pinned. The duration test runs real ffmpeg on
two tiny generated tones (skipped without ffmpeg); no capture hardware.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from vemoizer.batch_preset import _mtime_date_str
from vemoizer.listen import build_mix_argv

SR = 16_000
needs_ffmpeg = pytest.mark.skipif(
    shutil.which("ffmpeg") is None, reason="ffmpeg is not on PATH"
)


def test_two_track_argv_pins_amix_filter_and_output_format() -> None:
    argv = build_mix_argv(Path("me.flac"), Path("them.flac"), Path("call.flac"))
    assert argv == [
        "ffmpeg",
        "-hide_banner",
        "-nostdin",
        "-y",
        "-i",
        "me.flac",
        "-i",
        "them.flac",
        "-filter_complex",
        "amix=inputs=2:normalize=0:duration=longest",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "flac",
        "call.flac",
    ]


def test_two_track_argv_has_no_delay_filter() -> None:
    argv = build_mix_argv(Path("me.flac"), Path("them.flac"), Path("call.flac"))
    assert "adelay" not in " ".join(argv)


def test_single_track_argv_when_mic_stem_is_missing() -> None:
    argv = build_mix_argv(None, Path("them.flac"), Path("call.flac"))
    assert "-filter_complex" not in argv
    assert "amix=inputs=2" not in " ".join(argv)
    assert argv.count("-i") == 1
    assert "them.flac" in argv
    assert argv[-1] == "call.flac"


@needs_ffmpeg
def test_mix_duration_is_the_longer_track(tmp_path: Path) -> None:
    me = tmp_path / "me.flac"
    them = tmp_path / "them.flac"
    out = tmp_path / "call.flac"
    tone = np.full(SR * 2, 0.2, dtype=np.float32)  # 2 s
    longer = np.full(SR * 3, 0.2, dtype=np.float32)  # 3 s
    sf.write(me, tone, SR, format="FLAC", subtype="PCM_24")
    sf.write(them, longer, SR, format="FLAC", subtype="PCM_24")

    subprocess.run(
        build_mix_argv(me, them, out), check=True, capture_output=True, timeout=60
    )

    info = sf.info(out)
    assert info.channels == 1
    assert info.samplerate == SR
    assert info.duration == pytest.approx(3.0, abs=0.05)


def test_mix_is_dated_by_its_start_across_midnight(tmp_path: Path) -> None:
    mix = tmp_path / "call.flac"
    mix.write_bytes(b"")
    started = datetime(2026, 10, 26, 23, 59, 30).timestamp()
    os.utime(mix, (started, started))  # listen stamps the start, not the stop
    assert _mtime_date_str(mix) == "2026-10-26"
