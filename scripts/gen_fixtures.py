"""Generate the Finnish fixture corpus under ``tests/fixtures/corpus/``.

The checked-in corpus is the default mode: deterministic 16 kHz mono WAV
files synthesized locally (``wave`` + ``numpy``), paired by stem with a
``.txt`` reference transcript. This keeps the fixtures reproducible without
touching the network and keeps CI free of model downloads.

Piper mode
----------

The Piper Finnish voice (``fi_FY``) can optionally be used to produce
real TTS audio for the same stems. Piper is **not** a runtime dependency
of vemoizer — install it manually if you want real Finnish speech in the
fixtures::

    pip install piper-tts

Then run::

    uv run python scripts/gen_fixtures.py --piper

Piper emits 22050 Hz 16-bit mono WAV; the script resamples to the 16 kHz
mono contract using ffmpeg and writes the paired ``.txt`` next to each
WAV. The corpus directory is recreated on every run.

Stem-paired contract (consumed by ``vemoizer eval`` in issue #11)::

    tests/fixtures/corpus/<stem>.wav  <->  tests/fixtures/corpus/<stem>.txt

Every WAV MUST have a same-stem ``.txt`` with the reference transcript.
"""

from __future__ import annotations

import argparse
import math
import shutil
import subprocess
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# piper-tts is an optional dependency (see the --piper flag below). We keep
# it out of pyproject.toml's runtime deps on purpose — the checked-in
# fixture corpus does NOT need Piper, and the CI path should not download
# TTS weights. The import is guarded so the script's default mode works
# without Piper installed.
try:  # pragma: no cover - depends on install-time state
    from piper import PiperVoice

    _PIPER_AVAILABLE: bool = True
except ImportError:  # pragma: no cover - depends on install-time state
    PiperVoice = None  # type: ignore[assignment, misc]
    _PIPER_AVAILABLE = False

SAMPLE_RATE = 16_000
SAMPLE_WIDTH = 2  # 16-bit PCM


@dataclass(frozen=True)
class Fixture:
    stem: str
    transcript: str
    plan: tuple[tuple[float, float, float, float], ...]


#: Each entry: (stem, reference transcript, synthesis plan).
#: The plan is a list of (start_s, end_s, freq_hz, amplitude) segments used
#: by the deterministic synthesizer to approximate speech-like cadence.
#
#: Transcripts intentionally mix Finnish with English technical terms —
#: the exact code-switching profile this project is built for. See
#: AGENTS.md invariant #3.
FIXTURES: tuple[Fixture, ...] = (
    Fixture(
        stem="fi_code_switch_short",
        transcript=(
            "Hei, tänään testataan uude API endpoint ja se toimii "
            "toivomusten mukaisesti."
        ),
        plan=(
            (0.0, 0.3, 300.0, 0.4),
            (0.35, 0.55, 280.0, 0.35),
            (0.6, 0.9, 320.0, 0.4),
            (0.95, 1.3, 310.0, 0.38),
            (1.35, 1.7, 290.0, 0.35),
            (1.75, 2.1, 305.0, 0.4),
            (2.15, 2.5, 315.0, 0.4),
        ),
    ),
    Fixture(
        stem="fi_english_mixed",
        transcript=(
            "Meidän deployment pipeline käyttää GitHub Actions ja "
            "Kamal kun deployaamme productioniin."
        ),
        plan=(
            (0.0, 0.35, 295.0, 0.4),
            (0.4, 0.7, 310.0, 0.38),
            (0.75, 1.1, 300.0, 0.4),
            (1.15, 1.55, 285.0, 0.35),
            (1.6, 2.0, 320.0, 0.4),
            (2.05, 2.4, 305.0, 0.38),
            (2.45, 2.85, 290.0, 0.35),
            (2.9, 3.3, 315.0, 0.4),
        ),
    ),
    Fixture(
        stem="fi_technical_terms",
        transcript=(
            "Backendissa on useita API rate limit ja se pitää "
            "huomioida load testingissä."
        ),
        plan=(
            (0.0, 0.4, 300.0, 0.4),
            (0.45, 0.8, 285.0, 0.35),
            (0.85, 1.25, 315.0, 0.4),
            (1.3, 1.7, 295.0, 0.38),
            (1.75, 2.15, 310.0, 0.4),
            (2.2, 2.6, 290.0, 0.35),
            (2.65, 3.1, 320.0, 0.4),
            (3.15, 3.5, 305.0, 0.38),
        ),
    ),
)

CORPUS_DIR = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "corpus"


def synthesize_wav(fix: Fixture, out_path: Path) -> None:
    """Write a 16 kHz mono 16-bit WAV approximating the fixture's transcript.

    The synthesis is deterministic: a sum of per-segment sine waves shaped
    by the fixture's plan. This is *not* speech — it is a stable,
    reproducible placeholder that satisfies the 16 kHz mono contract and
    lets the WER harness and alignment stages exercise their code paths
    without downloading Piper or any other model.
    """
    total_frames = int(math.ceil(fix.plan[-1][1] * SAMPLE_RATE))
    frames = np.zeros(total_frames, dtype=np.int16)

    for start_s, end_s, freq_hz, amplitude in fix.plan:
        start = int(start_s * SAMPLE_RATE)
        end = min(int(end_s * SAMPLE_RATE), total_frames)
        if end <= start:
            continue
        t = np.arange(end - start) / SAMPLE_RATE
        tone = np.sin(2.0 * math.pi * freq_hz * t)
        # Short attack/release envelope to avoid clicks at segment edges.
        env = np.ones_like(t)
        attack = min(8, len(t) // 4)
        release = min(8, len(t) // 4)
        if attack > 0:
            env[:attack] *= np.linspace(0.0, 1.0, attack)
        if release > 0:
            env[-release:] *= np.linspace(1.0, 0.0, release)
        peak = int(amplitude * 32767)
        frames[start:end] += (tone * env * peak).astype(np.int16)

    frames = np.clip(frames, -32768, 32767).astype(np.int16)

    with wave.open(str(out_path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(SAMPLE_WIDTH)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(frames.tobytes())


def resample_piper_output(src: Path, dst: Path) -> None:
    """Resample a Piper 22050 Hz WAV to the 16 kHz mono contract via ffmpeg."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError(
            "ffmpeg not found on PATH; required for Piper resampling. "
            "Install ffmpeg or use the default (non-Piper) fixture mode."
        )
    subprocess.run(
        [
            ffmpeg,
            "-nostdin",
            "-v",
            "error",
            "-y",
            "-i",
            str(src),
            "-ac",
            "1",
            "-ar",
            str(SAMPLE_RATE),
            "-c:a",
            "pcm_s16le",
            str(dst),
        ],
        check=True,
    )


def _require_piper() -> None:
    if not _PIPER_AVAILABLE:
        raise RuntimeError(
            "piper-tts is not installed. Install it with `pip install piper-tts` "
            "and re-run this script, or drop --piper to use the deterministic "
            "built-in synthesis (the default checked-in mode)."
        )


def generate_piper(fix: Fixture, corpus_dir: Path, piper_voice: str) -> None:
    """Synthesize a fixture with Piper TTS and resample to 16 kHz mono."""
    _require_piper()
    assert PiperVoice is not None  # guard for ty: checked via _require_piper()
    voice = PiperVoice.load(piper_voice)
    out_path = corpus_dir / f"{fix.stem}.wav"
    voice.synthesize(fix.transcript, wav_file=str(out_path))
    resample_piper_output(out_path, out_path)


def generate_corpus(
    corpus_dir: Path, *, use_piper: bool, piper_voice: str
) -> list[Path]:
    """(Re)generate the full corpus; returns the written WAV paths."""
    if corpus_dir.exists():
        for entry in corpus_dir.iterdir():
            entry.unlink()
    corpus_dir.mkdir(parents=True)

    written: list[Path] = []
    for fix in FIXTURES:
        wav_path = corpus_dir / f"{fix.stem}.wav"
        txt_path = corpus_dir / f"{fix.stem}.txt"
        if use_piper:
            generate_piper(fix, corpus_dir, piper_voice)
        else:
            synthesize_wav(fix, wav_path)
        txt_path.write_text(fix.transcript + "\n", encoding="utf-8")
        written.append(wav_path)
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--piper",
        action="store_true",
        help="use Piper TTS (fi_FY) instead of the built-in deterministic synthesis",
    )
    parser.add_argument(
        "--voice",
        default="fi_FY",
        help="Piper voice name (default: fi_FY); only used with --piper",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=CORPUS_DIR,
        help=f"corpus directory (default: {CORPUS_DIR})",
    )
    args = parser.parse_args(argv)

    written = generate_corpus(args.out, use_piper=args.piper, piper_voice=args.voice)
    for path in written:
        print(f"wrote {path}")
    print(f"corpus: {len(written)} fixtures under {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
