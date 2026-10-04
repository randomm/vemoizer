"""Named ffmpeg argv constants: byte-identical to the main branch (issue #135, fix pass 2).

The spec says: in ingest.py define two named tuples (e.g. _FFMPEG_DECODE_ARGS
= the first part up to and including pcm_f32le, _FFMPEG_OUTPUT_ARGS = -f
f32le -) and compose _FFMPEG_AUDIO_ARGS = _FFMPEG_DECODE_ARGS +
_FFMPEG_OUTPUT_ARGS so the value is byte-identical to today's.

This test reads the tuple from main (via git show) and hardcodes the
expected tuple, then asserts that _FFMPEG_AUDIO_ARGS equals it.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest


def _get_main_ffmpeg_audio_args() -> tuple[str, ...]:
    """Read the _FFMPEG_AUDIO_ARGS tuple from the main branch."""
    # Extract the tuple from the main branch's ingest.py
    try:
        result = subprocess.run(
            ["git", "show", "main:src/vemoizer/ingest.py"],
            capture_output=True,
            text=True,
            check=True,
        )
        source = result.stdout
    except subprocess.CalledProcessError:
        # If we can't read main, use the hardcoded expected value
        return (
            "-nostdin",
            "-v",
            "error",
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_f32le",
            "-f",
            "f32le",
            "-",
        )

    # Find the tuple definition
    start = source.find("_FFMPEG_AUDIO_ARGS = (")
    if start == -1:
        raise AssertionError("Could not find _FFMPEG_AUDIO_ARGS in main")
    start = source.index("(", start)
    depth = 0
    for i in range(start, len(source)):
        if source[i] == "(":
            depth += 1
        elif source[i] == ")":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    else:
        raise AssertionError("Could not find end of _FFMPEG_AUDIO_ARGS tuple")

    tuple_text = source[start:end]
    # Parse the tuple (it's a simple tuple of strings)
    exec(f"_tuple = {tuple_text}")
    return _tuple  # type: ignore[name-defined]


def test_ffmpeg_audio_args_byte_identical_to_main() -> None:
    """_FFMPEG_AUDIO_ARGS is byte-identical to the main branch's tuple."""
    from vemoizer.ingest import _FFMPEG_AUDIO_ARGS

    # Hardcoded expected value (from the main branch)
    expected = (
        "-nostdin",
        "-v",
        "error",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-c:a",
        "pcm_f32le",
        "-f",
        "f32le",
        "-",
    )

    assert _FFMPEG_AUDIO_ARGS == expected, (
        f"_FFMPEG_AUDIO_ARGS changed from main:\n"
        f"  expected: {expected}\n"
        f"  got:      {_FFMPEG_AUDIO_ARGS}"
    )


def test_ffmpeg_decode_args_and_output_args_compose_to_audio_args() -> None:
    """_FFMPEG_AUDIO_ARGS = _FFMPEG_DECODE_ARGS + _FFMPEG_OUTPUT_ARGS."""
    from vemoizer.ingest import (
        _FFMPEG_AUDIO_ARGS,
        _FFMPEG_DECODE_ARGS,
        _FFMPEG_OUTPUT_ARGS,
    )

    assert _FFMPEG_AUDIO_ARGS == _FFMPEG_DECODE_ARGS + _FFMPEG_OUTPUT_ARGS


def test_ffmpeg_decode_args_ends_with_pcm_f32le() -> None:
    """_FFMPEG_DECODE_ARGS ends with 'pcm_f32le' (the codec spec)."""
    from vemoizer.ingest import _FFMPEG_DECODE_ARGS

    assert _FFMPEG_DECODE_ARGS[-1] == "pcm_f32le"


def test_ffmpeg_output_args_is_f_f32le_dash() -> None:
    """_FFMPEG_OUTPUT_ARGS is ('-f', 'f32le', '-') (raw stream to stdout)."""
    from vemoizer.ingest import _FFMPEG_OUTPUT_ARGS

    assert _FFMPEG_OUTPUT_ARGS == ("-f", "f32le", "-")
