"""Generate the Finnish fixture corpus under ``tests/fixtures/corpus/``.

The checked-in corpus is **real synthesized Finnish speech** from the Piper
voice ``fi_FI-harri-low`` (dataset licensed CC0; the voice's Ryan lineage is
unencumbered, unlike ``harri-medium``'s Blizzard lineage — and unlike macOS
``say`` output, which Apple's SLA forbids redistributing). Piper is a
dev-time tool only, never a runtime dependency::

    uv pip install piper-tts
    uv run python scripts/gen_fixtures.py

The voice model is downloaded revision-pinned from ``rhasspy/piper-voices``
(project invariant #4 applied to dev assets) and cached in the HF cache.
Output is resampled to the 16 kHz mono 16-bit contract via ffmpeg and
paired by stem with a ``.txt`` reference transcript::

    tests/fixtures/corpus/<stem>.wav  <->  tests/fixtures/corpus/<stem>.txt

``--tones`` instead writes the deterministic sine-tone corpus (no speech,
no downloads) — useful only for exercising the audio contract and the
corpus-walking machinery, never for WER numbers.

After regenerating, listen-check the WAVs and re-measure the WER baseline
(``vemoizer eval --backend all --update-baseline``) in a dedicated commit.
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

from vemoizer.audio_contract import SAMPLE_RATE  # single home for the 16 kHz contract

SAMPLE_WIDTH = 2  # 16-bit PCM

#: Piper voice, revision-pinned inside rhasspy/piper-voices.
PIPER_REPO = "rhasspy/piper-voices"
PIPER_REVISION = "142ef8f267e1904d9da7cde9df3d7237ac809b1e"
PIPER_VOICE_PATH = "fi/fi_FI/harri/low/fi_FI-harri-low.onnx"


@dataclass(frozen=True)
class Fixture:
    stem: str
    transcript: str


#: Transcripts intentionally cover the project's real profile (AGENTS.md
#: invariant #3): mostly Finnish with English technical terms seeping in,
#: plus pure-Finnish and pure-English controls.
FIXTURES: tuple[Fixture, ...] = (
    Fixture(
        stem="fi_code_switch_short",
        transcript=(
            "Hei, tänään testataan uusi API endpoint ja se toimii "
            "toivomusten mukaisesti."
        ),
    ),
    Fixture(
        stem="fi_english_mixed",
        transcript=(
            "Meidän deployment pipeline käyttää GitHub Actions ja "
            "Kamal kun deployaamme productioniin."
        ),
    ),
    Fixture(
        stem="fi_technical_terms",
        transcript=(
            "Backendissa on useita API rate limit ja se pitää "
            "huomioida load testingissä."
        ),
    ),
    Fixture(
        stem="fi_pure_short",
        transcript="Muista ostaa maitoa ja leipää kaupasta kotimatkalla.",
    ),
    Fixture(
        stem="en_pure_short",
        transcript="The quarterly review meeting is scheduled for next Thursday.",
    ),
    Fixture(
        stem="fi_code_switch_dense",
        transcript=(
            "Meidän backlog on täynnä, mutta sprint planning siirtyy "
            "koska product owner on lomalla."
        ),
    ),
    Fixture(
        stem="fi_numbers",
        transcript=(
            "Julkaisu siirtyy maanantaille kello neljätoista ja "
            "kokous alkaa viisitoista yli."
        ),
    ),
    Fixture(
        stem="fi_product_names",
        transcript=(
            "Käytämme Kubernetesta ja PostgreSQL-tietokantaa, mutta "
            "Redis-cache pitää vielä konfiguroida."
        ),
    ),
)

CORPUS_DIR = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "corpus"


def resolve_corpus_dir(out: Path) -> Path:
    """Resolve *out* and verify it stays under the project root.

    Guards against a symlinked corpus dir (or its parent) pointing outside
    the project — ffmpeg would otherwise follow the symlink and overwrite an
    arbitrary file when resampling in place.
    """
    resolved = out.resolve()
    project_root = Path(__file__).resolve().parent.parent
    if not (resolved == project_root or project_root in resolved.parents):
        raise RuntimeError(
            f"refusing to write corpus outside the project root: {resolved} "
            f"(project root: {project_root})"
        )
    return resolved


def synthesize_tones(fix: Fixture, out_path: Path) -> None:
    """Write a deterministic sine-tone WAV standing in for the transcript.

    Not speech: one short tone burst per transcript word, frequencies varied
    deterministically per word index. Satisfies the 16 kHz mono contract so
    the harness and contract tests can run without any download; useless for
    WER (every model rightly transcribes silence-with-beeps as nothing).
    """
    words = fix.transcript.split()
    seg = 0.28
    gap = 0.06
    total_frames = int(math.ceil(len(words) * (seg + gap) * SAMPLE_RATE))
    frames = np.zeros(total_frames, dtype=np.float64)
    for i, _word in enumerate(words):
        start = int(i * (seg + gap) * SAMPLE_RATE)
        end = min(start + int(seg * SAMPLE_RATE), total_frames)
        t = np.arange(end - start) / SAMPLE_RATE
        freq = 280.0 + (i % 5) * 12.0
        tone = 0.35 * np.sin(2.0 * math.pi * freq * t)
        env = np.ones_like(t)
        env_len = min(8, len(t) // 4)
        if env_len > 0:
            env[:env_len] *= np.linspace(0.0, 1.0, env_len)
            env[-env_len:] *= np.linspace(1.0, 0.0, env_len)
        frames[start:end] += tone * env
    pcm = np.clip(frames * 32767, -32768, 32767).astype(np.int16)
    with wave.open(str(out_path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(SAMPLE_WIDTH)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(pcm.tobytes())


def resample_to_contract(src: Path, dst: Path) -> None:
    """Resample any WAV to the 16 kHz mono 16-bit contract via ffmpeg."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg not found on PATH; required for resampling.")
    try:
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
            capture_output=True,
            text=True,
            timeout=300,
        )
    except subprocess.TimeoutExpired as e:
        raise RuntimeError(f"ffmpeg timed out resampling {src} -> {dst}") from e
    except subprocess.CalledProcessError as e:
        stderr_tail = (e.stderr or "").strip()[-2000:]
        raise RuntimeError(
            f"ffmpeg failed (returncode {e.returncode}) resampling {src} -> {dst}:\n"
            f"{stderr_tail}"
        ) from e


def _load_piper_voice():
    """Download (revision-pinned) and load the fi_FI-harri-low Piper voice."""
    try:
        from piper import PiperVoice
    except ImportError as e:
        raise RuntimeError(
            "piper-tts is not installed. Install it with `uv pip install "
            "piper-tts` and re-run, or pass --tones for the no-download "
            "sine-tone corpus."
        ) from e
    from huggingface_hub import hf_hub_download

    model = hf_hub_download(PIPER_REPO, PIPER_VOICE_PATH, revision=PIPER_REVISION)
    config = hf_hub_download(
        PIPER_REPO, f"{PIPER_VOICE_PATH}.json", revision=PIPER_REVISION
    )
    return PiperVoice.load(model, config_path=config)


def generate_piper(voice, fix: Fixture, out_path: Path) -> None:
    """Synthesize one fixture with Piper and resample to the contract.

    Piper output lands in a ``.tmp`` file and is moved into place only
    after resampling succeeds, so a mid-write failure cannot leave a
    corrupt ``<stem>.wav`` behind.
    """
    tmp_path = out_path.with_suffix(".wav.tmp")
    try:
        with wave.open(str(tmp_path), "wb") as w:
            voice.synthesize_wav(fix.transcript, w)
        resample_to_contract(tmp_path, out_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def generate_corpus(corpus_dir: Path, *, tones: bool) -> list[Path]:
    """(Re)generate the full corpus; returns the written WAV paths."""
    if corpus_dir.exists():
        for entry in corpus_dir.iterdir():
            entry.unlink()
    corpus_dir.mkdir(parents=True, exist_ok=True)

    voice = None if tones else _load_piper_voice()
    written: list[Path] = []
    for fix in FIXTURES:
        wav_path = corpus_dir / f"{fix.stem}.wav"
        if tones:
            synthesize_tones(fix, wav_path)
        else:
            generate_piper(voice, fix, wav_path)
        (corpus_dir / f"{fix.stem}.txt").write_text(
            fix.transcript + "\n", encoding="utf-8"
        )
        written.append(wav_path)
    return written


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tones",
        action="store_true",
        help="write the deterministic sine-tone corpus instead of Piper speech "
        "(contract tests only; never a WER corpus)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=CORPUS_DIR,
        help=f"corpus directory (default: {CORPUS_DIR})",
    )
    args = parser.parse_args(argv)

    corpus_dir = resolve_corpus_dir(args.out)
    written = generate_corpus(corpus_dir, tones=args.tones)
    for path in written:
        print(f"wrote {path}")
    mode = "sine tones" if args.tones else "Piper fi_FI-harri-low"
    print(f"corpus: {len(written)} fixtures ({mode}) under {corpus_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
