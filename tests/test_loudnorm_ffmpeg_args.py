"""Named ffmpeg argv constants: byte-identical to the main branch
(issue #135, fix pass 2).

The spec says: in ingest.py define two named tuples (e.g. _FFMPEG_DECODE_ARGS
= the first part up to and including pcm_f32le, _FFMPEG_OUTPUT_ARGS = -f
f32le -) and compose _FFMPEG_AUDIO_ARGS = _FFMPEG_DECODE_ARGS +
_FFMPEG_OUTPUT_ARGS so the value is byte-identical to today's.

The expected tuple is hardcoded in this test — it is the literal value
of ``_FFMPEG_AUDIO_ARGS`` from the main branch at the time of the change.
It intentionally pins today's argv: if main's argv is changed on purpose,
this test is updated with it.
"""

from __future__ import annotations


def _get_main_ffmpeg_audio_args() -> tuple[str, ...]:
    """The _FFMPEG_AUDIO_ARGS tuple from the main branch (hardcoded).

    The spec says: use `git show main:src/vemoizer/ingest.py` to read it
    and hardcode the expected tuple in the test. The tuple has been
    stable since the original implementation, so we hardcode it here.
    """
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

    assert expected == _FFMPEG_AUDIO_ARGS, (
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
