"""Contract tests for the checked-in fixture corpus under ``tests/fixtures/corpus/``.

The eval harness (issue #11) consumes the corpus as stem-paired
``<stem>.wav`` ↔ ``<stem>.txt`` files. The tests here pin the **audio
contract** (16 kHz mono, 16-bit PCM, sane duration) using only the
standard-library ``wave`` module, so they run on any Python without a
network or model download. They are the regression guard against
fixtures silently drifting from the contract.
"""

from __future__ import annotations

import wave
from pathlib import Path

import pytest

from vemoizer.audio_contract import SAMPLE_RATE

CORPUS_DIR = Path(__file__).resolve().parent / "fixtures" / "corpus"

SAMPLE_WIDTH = 2  # 16-bit PCM

#: Duration bounds (seconds) for a single fixture.
#:
#: Upper bound is generous enough for a "few sentences" utterance; lower
#: bound is a short silence-tail guard that catches empty or truncated
#: WAVs. Tunes with the fixtures in ``scripts/gen_fixtures.py``.
MIN_SECONDS = 1.0
MAX_SECONDS = 10.0


def _read_headers(fixture: Path) -> wave.Wave_read:
    return wave.open(str(fixture), "rb")


def _assert_audio_contract(w: wave.Wave_read, fixture: Path) -> None:
    """Pin the 16 kHz mono 16-bit PCM contract on a single WAV file."""
    assert w.getnchannels() == 1, (
        f"{fixture.name}: expected mono, got {w.getnchannels()}"
    )
    assert w.getsampwidth() == SAMPLE_WIDTH, (
        f"{fixture.name}: expected {SAMPLE_WIDTH * 8}-bit PCM, "
        f"got {w.getsampwidth() * 8}-bit"
    )
    assert w.getframerate() == SAMPLE_RATE, (
        f"{fixture.name}: expected {SAMPLE_RATE} Hz, got {w.getframerate()} Hz"
    )


@pytest.mark.parametrize(
    "fixture",
    sorted(p for p in CORPUS_DIR.glob("*.wav")),
    ids=lambda p: p.stem,
)
def test_wav_audio_contract(fixture: Path) -> None:
    """Every checked-in WAV must satisfy the 16 kHz mono 16-bit contract."""
    with _read_headers(fixture) as w:
        _assert_audio_contract(w, fixture)


def test_fixture_corpus_exists() -> None:
    """The corpus directory must exist and contain at least one WAV."""
    assert CORPUS_DIR.is_dir(), f"missing corpus directory: {CORPUS_DIR}"
    wavs = list(CORPUS_DIR.glob("*.wav"))
    assert wavs, f"no .wav fixtures under {CORPUS_DIR}"


def test_every_wav_has_a_stem_paired_transcript() -> None:
    """The consumer contract: every WAV must have a same-stem .txt."""
    missing: list[str] = []
    for w in sorted(CORPUS_DIR.glob("*.wav")):
        paired = CORPUS_DIR / f"{w.stem}.txt"
        if not paired.is_file():
            missing.append(w.name)
    assert not missing, f"WAV fixtures without a stem-paired .txt: {missing}"


def test_every_transcript_is_nonempty() -> None:
    """Every .txt must hold a non-whitespace reference transcript."""
    empty: list[str] = []
    for t in sorted(CORPUS_DIR.glob("*.txt")):
        body = t.read_text(encoding="utf-8")
        if not body.strip():
            empty.append(t.name)
    assert not empty, f"empty or whitespace-only transcripts: {empty}"


def test_no_stray_files_in_corpus_dir() -> None:
    """Only ``.wav`` and ``.txt`` (stem-paired) files belong in the corpus."""
    stray: list[str] = []
    wavs = {p.stem for p in CORPUS_DIR.glob("*.wav")}
    txts = {p.stem for p in CORPUS_DIR.glob("*.txt")}
    for p in CORPUS_DIR.iterdir():
        if not p.is_file():
            stray.append(p.name)
            continue
        if p.suffix not in {".wav", ".txt"}:
            stray.append(p.name)
        elif p.suffix == ".wav" and p.stem not in txts:
            stray.append(f"{p.name} (no same-stem .txt)")
        elif p.suffix == ".txt" and p.stem not in wavs:
            stray.append(f"{p.name} (no same-stem .wav)")
    assert not stray, f"stray or mismatched files in corpus: {stray}"


def test_fixture_duration_within_bounds() -> None:
    """Each WAV must be a sane 'few seconds' length — not empty, not a full memo."""
    for w in sorted(CORPUS_DIR.glob("*.wav")):
        with wave.open(str(w), "rb") as f:
            frames = f.getnframes()
            rate = f.getframerate()
            duration = frames / rate if rate else 0.0
        assert MIN_SECONDS <= duration <= MAX_SECONDS, (
            f"{w.name}: duration {duration:.2f}s outside "
            f"[{MIN_SECONDS}, {MAX_SECONDS}]s bounds"
        )


def test_corpus_contains_code_switched_content() -> None:
    """The corpus is the point: Finnish + English technical terms must appear.

    AGENTS.md invariant #3 — the whole reason vemoizer exists. If a future
    fixture drops the English terms, this test fails so the regression is
    caught in CI, not in a WER regression.
    """
    body = "".join(t.read_text(encoding="utf-8") for t in CORPUS_DIR.glob("*.txt"))
    lower = body.lower()
    assert any(
        t in lower
        for t in (
            "api",
            "github",
            "deployment",
            "pipeline",
            "production",
            "backend",
            "rate",
        )
    ), (
        "corpus transcripts contain no English technical terms — "
        "code-switching contract broken (see AGENTS.md invariant #3)"
    )


def test_corpus_is_stem_pairable_for_eval() -> None:
    """Smoke-check: eval's stem-pair walk sees every WAV exactly once."""
    wavs = {p.stem for p in CORPUS_DIR.glob("*.wav")}
    txts = {p.stem for p in CORPUS_DIR.glob("*.txt")}
    assert wavs == txts, (
        f"stem sets differ: wav-only={wavs - txts}, txt-only={txts - wavs}"
    )
    assert len(wavs) >= 3, (
        f"corpus has only {len(wavs)} fixtures; expect at least 3 for a "
        f"meaningful WER regression (see scripts/gen_fixtures.py)"
    )
