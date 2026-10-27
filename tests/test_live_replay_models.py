"""Live replay through real silero-vad + whisper-turbo (opt-in: ``-m models``).

Builds one ~4-5 minute stream from the 37 corpus clips with seeded 1-3 s gaps,
replays it at 1x pace through :class:`StreamingVad` and :class:`WhisperTranscriber`,
and asserts the utterance-end lag p90 budget. The WER of the live draft against
the whisper-turbo backend on the same stream is printed for information only.
The stream lives in ``tmp_path`` and is never committed.
"""

from __future__ import annotations

import wave
from pathlib import Path

import huggingface_hub.constants as hf_constants
import numpy as np
import pytest
from _live_helpers import replay_lag

from vemoizer.audio_contract import SAMPLE_RATE
from vemoizer.eval_backends import transcribe_meeting_only
from vemoizer.eval_harness import wer
from vemoizer.ingest import ingest_audio
from vemoizer.pipeline import glossary_prompt, load_glossary
from vemoizer.vad import load_model
from vemoizer.vad_stream import StreamingVad
from vemoizer.whisper_transcriber import WhisperTranscriber

CORPUS = Path(__file__).resolve().parent / "fixtures" / "corpus"
SEED = 181
GAP_RANGE_S = (1.0, 3.0)
#: Utterance-end lag budget (issue #181): p90 of emit - (capture_t0 + end).
MAX_UTTERANCE_LAG_P90_S = 6.0

pytestmark = pytest.mark.models

# conftest repoints HF_HUB_CACHE at an empty dir per test; this module needs the
# warm cache that holds the pinned weights, so the real value is captured at import.
_REAL_HF_HUB_CACHE = hf_constants.HF_HUB_CACHE


@pytest.fixture(autouse=True)
def _warm_hf_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(hf_constants, "HF_HUB_CACHE", _REAL_HF_HUB_CACHE)


def _gap(rng: np.random.Generator) -> np.ndarray:
    seconds = rng.uniform(*GAP_RANGE_S)
    return np.zeros(int(round(seconds * SAMPLE_RATE)), dtype=np.float32)


def _build_stream(stream_wav: Path) -> tuple[np.ndarray, str]:
    """Concatenate the corpus with seeded gaps; return the audio and its reference.

    The stream is quantised to 16-bit once and written to *stream_wav* so the
    live path and the whisper-turbo backend decode the same samples.
    """
    rng = np.random.default_rng(SEED)
    parts: list[np.ndarray] = []
    refs: list[str] = []
    for wav in sorted(CORPUS.glob("*.wav")):
        txt = wav.with_suffix(".txt")
        if not txt.is_file():
            continue
        parts.extend([_gap(rng), ingest_audio(wav)])
        refs.append(txt.read_text(encoding="utf-8").strip())
    parts.append(_gap(rng))
    pcm = np.clip(np.round(np.concatenate(parts) * 32768.0), -32768, 32767)
    pcm16 = pcm.astype(np.int16)
    with wave.open(str(stream_wav), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(SAMPLE_RATE)
        out.writeframes(pcm16.tobytes())
    return pcm16.astype(np.float32) / 32768.0, " ".join(refs)


def test_live_replay_utterance_lag_p90_within_budget(tmp_path: Path) -> None:
    stream_wav = tmp_path / "stream.wav"
    audio, reference = _build_stream(stream_wav)
    prompt = glossary_prompt(load_glossary(None))

    replay = replay_lag(
        audio,
        vad=StreamingVad(load_model()),
        transcriber=WhisperTranscriber(language=None, initial_prompt=prompt),
    )

    assert replay.lag.utterance.count > 0, "live replay produced no utterances"
    assert replay.lag.utterance.p90 <= MAX_UTTERANCE_LAG_P90_S, replay.lag

    # Information only: the live draft versus the whisper-turbo backend on the
    # same stream. Run after the replay so MLX is torn down before reloading.
    live_text = " ".join(
        d.text for d in sorted(replay.decoded, key=lambda d: d.start_s)
    )
    turbo_text = transcribe_meeting_only(stream_wav)["text"]
    live_wer = wer(reference, live_text)
    turbo_wer = wer(reference, turbo_text)
    print(
        f"\nlive-replay WER {live_wer:.3f} vs whisper-turbo WER {turbo_wer:.3f} "
        f"(delta {live_wer - turbo_wer:+.3f}); lag p50 {replay.lag.utterance.p50:.2f}s "
        f"p90 {replay.lag.utterance.p90:.2f}s max {replay.lag.utterance.max:.2f}s"
    )
