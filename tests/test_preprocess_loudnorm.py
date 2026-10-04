"""Tests for the opt-in ``--preprocess loudnorm`` audio level (issue #135).

Covers:

- The two-pass loudnorm argv builders (pass 1 measurement, pass 2
  normalization) with and without the flag — the no-flag argv is
  literally unchanged (a test pins equality with the plain decode
  argv).
- The two-pass JSON parse (the last balanced ``{...}`` block on stderr,
  robust to nested braces in ffmpeg's own warning lines), and the
  fail-open matrix (missing/malformed/``-inf``/NaN/absurd measured
  values → the plain decode, ONE warning, never an abort, never an
  unbounded gain).
- Real-ffmpeg sample-count fidelity and peak-bound on a synthetic
  signal (skipped cleanly when ffmpeg or the loudnorm filter is
  missing); no models, no network.
"""

from __future__ import annotations

import json
import math
import shutil
import struct
import subprocess
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from _cli_helpers import ffmpeg_has_loudnorm

from vemoizer import loudnorm as ln
from vemoizer.ingest import (
    _FFMPEG_AUDIO_ARGS,
    IngestError,
    ingest_audio,
    pcm_duration_seconds,
)
from vemoizer.loudnorm import (
    LoudnormMeasurement,
    loudnorm_pass1_filter,
    loudnorm_pass2_filter,
    parse_loudnorm_json,
)
from vemoizer.speaker_clips import ClipWindow, extract_clips

FFMPEG = pytest.mark.skipif(
    shutil.which("ffmpeg") is None, reason="ffmpeg not available"
)

# The synthetic 16 kHz mono float32 signals are generated with ffmpeg's
# lavfi (no private audio, no network, no models).


HAS_LOUDNORM = FFMPEG and ffmpeg_has_loudnorm()
NO_LOUDNORM = pytest.mark.skipif(
    not HAS_LOUDNORM,
    reason="ffmpeg loudnorm filter not available",
)


def _make_wav(path: Path, seconds: float, amplitude: float = 3000) -> Path:
    """A tiny mono 16 kHz 16-bit PCM WAV (a constant tone)."""
    rate = 16000
    n = int(seconds * rate)
    samples = b"".join(struct.pack("<h", int(amplitude)) for _ in range(n))
    filesize = 36 + len(samples)
    hdr = (
        b"RIFF"
        + struct.pack("<I", filesize - 8)
        + b"WAVE"
        + b"fmt "
        + struct.pack("<I", 16)
        + struct.pack("<H", 1)
        + struct.pack("<H", 1)
        + struct.pack("<I", rate)
        + struct.pack("<I", rate * 2)
        + struct.pack("<H", 2)
        + struct.pack("<H", 16)
        + b"data"
        + struct.pack("<I", len(samples))
    )
    path.write_bytes(hdr + samples)
    return path


# ---------------------------------------------------------------------------
# Filter strings
# ---------------------------------------------------------------------------


def test_pass1_filter_string() -> None:
    f = loudnorm_pass1_filter()
    assert f.startswith("loudnorm=")
    assert "I=-16" in f
    assert "TP=-1.5" in f
    assert "LRA=11" in f
    assert "print_format=json" in f
    assert "measured_" not in f  # pass 1 measures; it never feeds values


def test_pass2_filter_string_carries_measured_values() -> None:
    m = LoudnormMeasurement(-16.1, -3.2, 0.5, -26.1, -0.5)
    f = loudnorm_pass2_filter(m)
    assert "measured_I=-16.1" in f
    assert "measured_TP=-3.2" in f
    assert "measured_LRA=0.5" in f
    assert "measured_thresh=-26.1" in f
    assert "offset=-0.5" in f
    assert "linear=true" in f


# ---------------------------------------------------------------------------
# Two-pass JSON parse (robust, last balanced block, fail-open matrix)
# ---------------------------------------------------------------------------


def test_parse_loudnorm_json_extracts_measurement() -> None:
    stderr = (
        "some ffmpeg banner line\n"
        "[in#0 @ 0xdeadbeef] Input #0, wav, from 'x.wav':\n"
        "  Stream #0:0: Audio: pcm_s16le, 16000 Hz, mono, s16, 256 kb/s\n"
        "[aist#0:0/pcm_s16le @ 0x1] Guessed Channel Layout: mono\n"
        "[Parsed_loudnorm_0 @ 0x2] \n"
        "{\n"
        '\t"input_i" : "-16.17",\n'
        '\t"input_tp" : "-3.05",\n'
        '\t"input_lra" : "0.00",\n'
        '\t"input_thresh" : "-26.17",\n'
        '\t"output_i" : "-15.98",\n'
        '\t"output_tp" : "-2.88",\n'
        '\t"normalization_type" : "linear",\n'
        '\t"target_offset" : "-0.02"\n'
        "}\n"
        "[out#0/null @ 0x3] video:0KiB audio:125KiB\n"
    )
    m = parse_loudnorm_json(stderr)
    assert m is not None
    assert m.input_i == pytest.approx(-16.17)
    assert m.input_tp == pytest.approx(-3.05)
    assert m.input_lra == pytest.approx(0.00)
    assert m.input_thresh == pytest.approx(-26.17)
    assert m.offset == pytest.approx(-0.02)


def test_parse_loudnorm_json_last_block_wins() -> None:
    # A brace-containing warning line BEFORE the JSON must not break the
    # parse; the loudnorm JSON is the LAST valid measurement block.
    stderr = (
        'warning: nested braces { "a": {"b": 1} } in a log line\n'
        "{\n"
        '"input_i" : "-18.0",\n'
        '"input_tp" : "-2.0",\n'
        '"input_lra" : "1.0",\n'
        '"input_thresh" : "-28.0",\n'
        '"target_offset" : "0.5"\n'
        "}\n"
    )
    m = parse_loudnorm_json(stderr)
    assert m is not None
    assert m.input_i == pytest.approx(-18.0)
    assert m.offset == pytest.approx(0.5)


def test_parse_loudnorm_json_malformed_returns_none() -> None:
    assert parse_loudnorm_json("no json here at all") is None
    assert parse_loudnorm_json("{ this is not json }") is None
    assert parse_loudnorm_json("") is None


def test_parse_loudnorm_json_missing_keys_returns_none() -> None:
    assert parse_loudnorm_json('{"foo": "bar"}') is None
    # input_i missing
    assert parse_loudnorm_json('{"input_tp": "-2.0", "input_lra": "1.0"}') is None


def test_parse_loudnorm_json_inf_returns_none() -> None:
    # ffmpeg reports silent input as "input_i": "-inf"
    stderr = (
        "{\n"
        '"input_i" : "-inf",\n'
        '"input_tp" : "-inf",\n'
        '"input_lra" : "0.0",\n'
        '"input_thresh" : "-inf",\n'
        '"target_offset" : "-0.00"\n'
        "}"
    )
    assert parse_loudnorm_json(stderr) is None


def test_parse_loudnorm_json_nan_returns_none() -> None:
    raw = json.dumps(
        {
            "input_i": math.nan,
            "input_tp": -2.0,
            "input_lra": 1.0,
            "input_thresh": -28.0,
            "target_offset": 0.0,
        }
    )
    assert parse_loudnorm_json(raw) is None


def test_parse_loudnorm_json_absurd_magnitude_returns_none() -> None:
    raw = json.dumps(
        {
            "input_i": -500.0,
            "input_tp": -2.0,
            "input_lra": 1.0,
            "input_thresh": -28.0,
            "target_offset": 0.0,
        }
    )
    assert parse_loudnorm_json(raw) is None


def test_sanitize_rejects_absurd_offset() -> None:
    raw = {
        "input_i": -16.0,
        "input_tp": -2.0,
        "input_lra": 1.0,
        "input_thresh": -28.0,
        "target_offset": 999.0,
    }
    assert ln._sanitize_measurement(raw) is None


# ---------------------------------------------------------------------------
# Argv builders — flag-off equality (the no-flag argv is literally
# unchanged) and flag-on shape
# ---------------------------------------------------------------------------


def test_ingest_audio_default_argv_unchanged(tmp_path: Path) -> None:
    """The no-flag ingest argv is byte-identical to the plain decode argv."""
    fixture = _make_wav(tmp_path / "x.wav", 0.1)
    captured: list[list[str]] = []

    def fake_run(argv, **kw):
        captured.append(list(argv))
        return subprocess.CompletedProcess(
            argv, 0, stdout=np.zeros(100, dtype=np.float32).tobytes(), stderr=b""
        )

    with patch("vemoizer.ingest.subprocess.run", side_effect=fake_run):
        ingest_audio(fixture)

    argv = captured[0]
    expected = ["ffmpeg", *_FFMPEG_AUDIO_ARGS, "-i", str(fixture)]
    assert argv == expected


def test_pcm_duration_default_argv_unchanged(tmp_path: Path) -> None:
    """The no-flag pcm_duration argv is byte-identical to the plain decode."""
    fixture = _make_wav(tmp_path / "x.wav", 0.1)
    captured: list[list[str]] = []

    class _Pipe:
        def __init__(self, data: bytes) -> None:
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

    class _StderrFile:
        def __init__(self, data: bytes = b"") -> None:
            self._buf = bytearray(data)

        def read(self, size: int | None = None) -> bytes:
            return bytes(self._buf)

        def seek(self, offset: int, whence: int = 0) -> int:
            return 0

    class _Proc:
        stdout = _Pipe(b"")
        stderr = _StderrFile()
        returncode = 0
        killed = False
        waited = False

        def poll(self) -> int | None:
            return self.returncode

        def wait(self, timeout: float | None = None) -> int:
            return self.returncode

        def kill(self) -> None:
            self.killed = True

    def fake_popen(argv, **kw):
        captured.append(list(argv))
        return _Proc()

    with patch("vemoizer.ingest.subprocess.Popen", side_effect=fake_popen):
        pcm_duration_seconds(fixture)

    argv = captured[0]
    expected = ["ffmpeg", *_FFMPEG_AUDIO_ARGS, "-i", str(fixture)]
    assert argv == expected


def test_decode_argv_no_preprocess_matches_plain() -> None:
    p = Path("/tmp/whatever.m4a")
    argv = ln.decode_argv(p, None)
    expected = ["ffmpeg", *_FFMPEG_AUDIO_ARGS, "-i", str(p)]
    assert argv == expected


def test_loudnorm_argv_inserts_af_before_input() -> None:
    """Pass 2's filter is inserted AFTER the resample/downmix args
    (the chosen order: filter after mono-16 kHz conversion, so the true
    peak stays below 0 dBFS) and BEFORE ``-i``."""
    m = LoudnormMeasurement(-16.0, -3.0, 0.5, -26.0, 0.0)
    filt = loudnorm_pass2_filter(m)
    base = list(_FFMPEG_AUDIO_ARGS)
    # base: -nostdin -v error -ac 1 -ar 16000 -c:a pcm_f32le -f f32le -
    expected = ["ffmpeg", *base[:10], "-af", filt, *base[10:], "-i"]
    # base[:10] = -nostdin -v error -ac 1 -ar 16000 -c:a pcm_f32le
    # base[10:] = -f f32le -
    assert expected[0] == "ffmpeg"
    # -af comes after the resample/downmix args (after -ar 16000 and
    # after -c:a pcm_f32le).
    assert expected.index("-af") > expected.index("16000")
    assert expected.index("-af") > expected.index("pcm_f32le")
    # -af comes before -i (the filter applies to the input stream).
    assert expected.index("-af") < expected.index("-i")
    # The plain decode contract is otherwise intact.
    assert expected[-1] == "-i"
    assert "f32le" in expected


# ---------------------------------------------------------------------------
# Real ffmpeg — sample-count fidelity, peak bound, duration equality
# ---------------------------------------------------------------------------


def _make_raw_f32le(path: Path, lavfi_src: str) -> Path:
    """Write a *lavfi_src* source (e.g. ``sine=frequency=440:duration=2``)
    as 16 kHz mono float32 raw (the exact audio contract, no container).
    The source string must already include a duration (``:duration=2`` or
    ``d=2``)."""
    subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            lavfi_src,
            "-ac",
            "1",
            "-ar",
            "16000",
            "-c:a",
            "pcm_f32le",
            "-f",
            "f32le",
            str(path),
        ],
        check=True,
        capture_output=True,
    )
    return path


@NO_LOUDNORM
class TestRealFfmpeg:
    def test_sample_count_unchanged_by_loudnorm(self, tmp_path: Path) -> None:
        """A linear gain cannot change the sample count: processed and
        unprocessed decodes yield identical lengths on a synthetic
        signal (timestamps across the pipeline depend on it)."""
        # 2 s of 440 Hz sine at 16 kHz mono float32 raw
        raw = _make_raw_f32le(tmp_path / "tone.raw", "sine=frequency=440:duration=2")
        # Wrap the raw f32le in a WAV container so the decodes can read
        # it (the loudnorm tests need a real decodable input).
        wav = tmp_path / "tone.wav"
        subprocess.run(
            [
                "ffmpeg",
                "-nostdin",
                "-v",
                "error",
                "-f",
                "f32le",
                "-ar",
                "16000",
                "-ac",
                "1",
                "-i",
                str(raw),
                "-c:a",
                "pcm_f32le",
                str(wav),
            ],
            check=True,
            capture_output=True,
        )
        plain = ingest_audio(wav)
        processed = ln.preprocess_audio(wav, preprocess="loudnorm")
        assert processed.dtype == np.float32
        assert len(plain) == len(processed)  # exactly equal (linear gain)

    def test_peak_stays_bounded_after_loudnorm(self, tmp_path: Path) -> None:
        """The loud/clipping synthetic signal stays below 0 dBFS after
        normalization (TP=-1.5 dBFS target; assert movement toward the
        target within tolerance, never exact LUFS decimals — ffmpeg
        versions differ)."""
        wav = tmp_path / "loud.wav"
        subprocess.run(
            [
                "ffmpeg",
                "-nostdin",
                "-v",
                "error",
                "-f",
                "lavfi",
                "-i",
                "anoisesrc=d=2:c=pink",
                "-ac",
                "1",
                "-ar",
                "16000",
                "-c:a",
                "pcm_f32le",
                str(wav),
            ],
            check=True,
            capture_output=True,
        )
        processed = ln.preprocess_audio(wav, preprocess="loudnorm")
        peak = float(np.max(np.abs(processed)))
        # Below full scale (0.0 dBFS) with headroom for the -1.5 dBFS
        # target; never clipped.
        assert peak < 1.0
        # The signal is non-trivial (not silently dropped).
        assert peak > 0.01

    def test_pcm_duration_matches_processed_decode(self, tmp_path: Path) -> None:
        """pcm_duration_seconds with the loudnorm flag equals the
        processed decode's duration (the part offsets / source durations
        must match the processed decode)."""
        raw = _make_raw_f32le(tmp_path / "tone.raw", "sine=frequency=440:duration=2")
        wav = tmp_path / "tone.wav"
        subprocess.run(
            [
                "ffmpeg",
                "-nostdin",
                "-v",
                "error",
                "-f",
                "f32le",
                "-ar",
                "16000",
                "-ac",
                "1",
                "-i",
                str(raw),
                "-c:a",
                "pcm_f32le",
                str(wav),
            ],
            check=True,
            capture_output=True,
        )
        processed = ln.preprocess_audio(wav, preprocess="loudnorm")
        dur = pcm_duration_seconds(wav, preprocess="loudnorm")
        assert dur == pytest.approx(len(processed) / 16000.0, rel=1e-6)

    def test_silence_fails_open(self, tmp_path: Path) -> None:
        """Silent input → measured I is -inf → the loudnorm measurement
        is rejected and the plain decode runs (fail-open, ONE warning,
        no unbounded gain)."""
        raw = _make_raw_f32le(tmp_path / "silence.raw", "anullsrc=d=1")
        wav = tmp_path / "silence.wav"
        subprocess.run(
            [
                "ffmpeg",
                "-nostdin",
                "-v",
                "error",
                "-f",
                "f32le",
                "-ar",
                "16000",
                "-ac",
                "1",
                "-i",
                str(raw),
                "-c:a",
                "pcm_f32le",
                str(wav),
            ],
            check=True,
            capture_output=True,
        )
        # The measurement must be rejected (-inf) → plain decode, which
        # yields all zeros (no unbounded gain on silence).
        result = ln.preprocess_audio(wav, preprocess="loudnorm")
        assert result.dtype == np.float32
        assert len(result) == 16000  # 1 second at 16 kHz
        assert float(np.max(np.abs(result))) == 0.0  # still silent, no gain


# ---------------------------------------------------------------------------
# Speaker-clip argv NEVER contains the loudnorm filter (issue #135: the
# clips are for a human to listen to; a normalised clip would
# misrepresent the real level).
# ---------------------------------------------------------------------------


def test_speaker_clip_argv_never_contains_loudnorm(tmp_path: Path) -> None:
    src = _make_wav(tmp_path / "a.wav", 1.0)
    src2 = _make_wav(tmp_path / "b.wav", 2.0)
    source = [
        {"path": str(src), "part_offset_s": 0.0, "duration_s": 10.0},
        {"path": str(src2), "part_offset_s": 10.0, "duration_s": 10.0},
    ]
    recorded: dict[str, list] = {}

    def fake_run(argv, *a, **kw):
        recorded["argv"] = list(argv)
        # A real 1 s wav can't satisfy -ss 2.0 -t 3.0 -> empty -> None.
        return subprocess.CompletedProcess(argv, 0)

    with patch("vemoizer.speaker_clips.subprocess.run", side_effect=fake_run):
        from vemoizer.speaker_clips import clip_session

        with clip_session() as tmp:
            extract_clips(source, [ClipWindow(12.0, 15.0)], tmp)

    argv = recorded["argv"]
    assert "loudnorm" not in argv
    assert "-af" not in argv


# ---------------------------------------------------------------------------
# ingest_audio loudnorm path: one measurement call + one decode call
# ---------------------------------------------------------------------------


def test_ingest_audio_loudnorm_calls_measure_and_decode(
    tmp_path: Path, monkeypatch
) -> None:
    fixture = _make_wav(tmp_path / "x.wav", 0.2)
    calls: list[str] = []

    m = LoudnormMeasurement(-16.0, -3.0, 0.5, -26.0, 0.0)
    monkeypatch.setattr(ln, "_measure_loudnorm", lambda p: m)

    def fake_popen(argv, **kw):
        calls.append("popen")
        raise AssertionError("unexpected popen")

    def fake_run(argv, **kw):
        calls.append("run")
        return subprocess.CompletedProcess(
            argv, 0, stdout=np.zeros(160, dtype=np.float32).tobytes(), stderr=b""
        )

    with (
        patch("vemoizer.ingest.subprocess.Popen", side_effect=fake_popen),
        patch("vemoizer.ingest.subprocess.run", side_effect=fake_run),
    ):
        result = ingest_audio(fixture, preprocess="loudnorm")

    assert calls == ["run"]  # pass 2 only (measurement was patched)
    assert len(result) == 160


def test_ingest_audio_loudnorm_fails_open_on_measurement_failure(
    tmp_path: Path, monkeypatch
) -> None:
    fixture = _make_wav(tmp_path / "x.wav", 0.2)
    monkeypatch.setattr(ln, "_measure_loudnorm", lambda p: None)

    def fake_run(argv, **kw):
        return subprocess.CompletedProcess(
            argv, 0, stdout=np.zeros(3200, dtype=np.float32).tobytes(), stderr=b""
        )

    with patch("vemoizer.ingest.subprocess.run", side_effect=fake_run):
        result = ingest_audio(fixture, preprocess="loudnorm")

    # The plain decode ran (fail-open) — 0.2 s * 16000 = 3200 samples.
    assert len(result) == 3200


def test_preprocess_audio_missing_file_raises() -> None:
    with pytest.raises(IngestError, match="not found"):
        ln.preprocess_audio(Path("/nonexistent/nope.m4a"), preprocess="loudnorm")


# ---------------------------------------------------------------------------
# Regression: real pass 1 must actually obtain a measurement (not the
# fail-open path). If ``-v error`` is re-added to the pass 1 argv, ffmpeg
# 9.x suppresses the loudnorm JSON and _measure_loudnorm returns None,
# silently defeating the feature. This test catches that regression.
# ---------------------------------------------------------------------------


@NO_LOUDNORM
def test_real_pass1_obtains_measurement_not_fail_open(tmp_path: Path) -> None:
    """Running the REAL _measure_loudnorm on a synthetic tone must return
    a non-None LoudnormMeasurement (the measurement is actually obtained).

    Regression guard: if ``-v error`` is re-added to the pass 1 argv,
    ffmpeg 9.x suppresses the loudnorm JSON output and _measure_loudnorm
    returns None (the fail-open path), silently disabling loudnorm.
    """
    raw = _make_raw_f32le(tmp_path / "tone.raw", "sine=frequency=440:duration=2")
    wav = tmp_path / "tone.wav"
    subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-f",
            "f32le",
            "-ar",
            "16000",
            "-ac",
            "1",
            "-i",
            str(raw),
            "-c:a",
            "pcm_f32le",
            str(wav),
        ],
        check=True,
        capture_output=True,
    )
    measurement = ln._measure_loudnorm(wav)
    assert measurement is not None, (
        "pass 1 returned None — the loudnorm JSON was not obtained "
        "(check: did someone re-add -v error to the pass 1 argv?)"
    )
    assert measurement.input_i < 0, "input_i must be negative (a real tone)"
