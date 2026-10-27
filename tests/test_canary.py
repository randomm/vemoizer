"""Canary-1b-v2 audio contract, tokenizer and feature-extraction tests (issue #6).

All offline: tiny in-memory tensors and a small sentencepiece model; no weights.
"""

from __future__ import annotations

import base64
from typing import Any

import numpy as np
import pytest
from _canary_helpers import mx_to_np, tiny_canary_config_dict

from vemoizer import canary_mlx as cm
from vemoizer.audio_contract import SAMPLE_RATE as CONTRACT_SAMPLE_RATE


@pytest.fixture
def canary_config_dict() -> dict[str, Any]:
    return tiny_canary_config_dict()


@pytest.fixture
def sample_audio() -> np.ndarray:
    # 1 second of 16 kHz mono float32 — a short, deterministic signal.
    sr = 16_000
    t = np.arange(sr) / sr
    return (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)


# ---------------------------------------------------------------------------
# Audio contract (project invariant #6): 16 kHz mono float32
# ---------------------------------------------------------------------------


def test_sample_rate_is_16khz() -> None:
    assert CONTRACT_SAMPLE_RATE == 16_000


def test_canary_module_sample_rate_matches_audio_contract() -> None:
    # The port must speak the same 16 kHz contract as the rest of the pipeline.
    assert cm.SAMPLE_RATE == CONTRACT_SAMPLE_RATE


# ---------------------------------------------------------------------------
# SentencePiece parsing + tokenizer (pure Python — no sentencepiece dependency)
# ---------------------------------------------------------------------------


def test_parse_sentencepiece_model_roundtrip(
    canary_config_dict: dict[str, Any],
) -> None:
    raw = base64.b64decode(canary_config_dict["tokenizer"]["model_base64"])
    pieces, special = cm.parse_sentencepiece_model(raw)
    assert len(pieces) == 6
    # Special tokens (those starting with "<|") are collected by id.
    assert special["<|startoftranscript|>"] == 2
    assert special["<|startofcontext|>"] == 1
    assert special["<|nospeech|>"] == 5
    # "<pad>" does not start with "<|" so it is not in the special map.
    assert "<pad>" not in special


def test_canary_tokenizer_from_config_resolves_special_ids(
    canary_config_dict: dict[str, Any],
) -> None:
    tok = cm.CanaryTokenizer.from_config(canary_config_dict)
    assert tok.vocab_size == 6
    assert tok.bos_id == 2  # <|startoftranscript|> at id 2


def test_canary_tokenizer_from_config_missing_base64_raises() -> None:
    with pytest.raises(ValueError):
        cm.CanaryTokenizer.from_config({"tokenizer": {}})


def test_canary_tokenizer_decode_replaces_space_marker() -> None:
    # Build a tokenizer whose vocab includes the SentencePiece space marker ▁.
    # A single piece that IS the space marker decodes to a literal space.
    pieces = ["a", "▁", "b"]
    tok = cm.CanaryTokenizer(pieces, {})
    # decode ids [0,1,2] -> "a" + "▁" + "b" = "a▁b" -> "a b"
    assert tok.decode([0, 1, 2]) == "a b"


def test_canary_tokenizer_decode_strips_special_tokens(
    canary_config_dict: dict[str, Any],
) -> None:
    tok = cm.CanaryTokenizer.from_config(canary_config_dict)
    # Special token ids are dropped during decode.
    assert tok.decode([2]) == ""  # id 2 is <|startoftranscript|> (special)
    # Regular pieces decode to plain text.
    assert tok.decode([3, 4]) == "hi"  # "h" + "i"


def test_canary_tokenizer_build_prompt_encodes_canary2_prompt(
    canary_config_dict: dict[str, Any],
) -> None:
    tok = cm.CanaryTokenizer.from_config(canary_config_dict)
    prompt = tok.build_prompt("fi", "fi")
    # The canary2 transcribe prompt starts with startofcontext + startoftranscript.
    assert prompt[0] == tok.special_tokens["<|startofcontext|>"]
    assert prompt[1] == tok.special_tokens["<|startoftranscript|>"]


def test_canary_tokenizer_encode_rejects_non_special_lang(
    canary_config_dict: dict[str, Any],
) -> None:
    tok = cm.CanaryTokenizer.from_config(canary_config_dict)
    with pytest.raises(ValueError):
        tok.encode("hello", lang_id="fi")


def test_canary_tokenizer_eos_id_missing_vocab_returns_sentinel() -> None:
    """The real Canary-1b-v2 vocab has no end-of-transcript token; eos_id
    must degrade to the -1 sentinel (generation stops on max_tokens) instead
    of raising KeyError — the old chr(0x03) constant crashed on first use."""
    pieces = [
        "<pad>",
        "<|startofcontext|>",
        "<|startoftranscript|>",
        "h",
        "i",
    ]
    tok = cm.CanaryTokenizer(pieces, {})
    assert tok.eos_id == -1


def test_canary_tokenizer_eos_id_resolves_endoftranscript_when_present() -> None:
    """If a vocab does contain an end-of-transcript token, eos_id resolves it."""
    pieces = [
        "<pad>",
        "<|startoftranscript|>",
        "h",
        "<|endoftranscript|>",
    ]
    special = {p: i for i, p in enumerate(pieces) if p.startswith("<|")}
    tok = cm.CanaryTokenizer(pieces, special)
    assert tok.eos_id == 3


# ---------------------------------------------------------------------------
# Feature extraction (16 kHz mono float32 -> 128-dim log-mel, per-frame norm)
# ---------------------------------------------------------------------------


def test_compute_features_shape_is_batch_time_128(sample_audio: np.ndarray) -> None:
    feats = cm.compute_features(sample_audio)
    # (batch=1, time, n_mels=128)
    assert feats.ndim == 3
    assert feats.shape[0] == 1
    assert feats.shape[2] == 128
    # 1 s at hop=160 -> ~101 frames; allow a small margin.
    assert 90 <= feats.shape[1] <= 110


def test_compute_features_short_audio_returns_empty_frame() -> None:
    tiny = np.zeros(10, dtype=np.float32)
    feats = cm.compute_features(tiny)
    assert feats.shape[0] == 1
    assert feats.shape[2] == 128


def test_compute_features_is_per_feature_normalized() -> None:
    # The checkpoint's preprocessor sets normalize="per_feature": each mel bin
    # is zero-mean / unit-var *over time*, matching NeMo's normalize_batch.
    # Normalizing each time frame across bins instead measurably degrades the
    # transcript, so the axis is load-bearing rather than cosmetic.
    rng = np.random.default_rng(0)
    audio = (rng.standard_normal(16_000) * 0.1).astype(np.float32)
    frames = mx_to_np(cm.compute_features(audio))[0]  # (T, 128)
    means = frames.mean(axis=0)  # one mean per mel bin
    stds = frames.std(axis=0)

    # 128 mel filters over 257 rfft bins leaves some low-frequency filters
    # empty; those bins are constant, so (x - mean) / (std + 1e-5) divides
    # float32 rounding noise by the guard instead of normalizing. Assert on
    # the bins that actually carry signal.
    active = stds > 0.5
    assert active.sum() > 64, f"only {active.sum()} of 128 mel bins carry signal"
    assert np.allclose(means[active], 0.0, atol=1e-2)
    assert np.allclose(stds[active], 1.0, atol=1e-1)


def test_compute_features_dtype(sample_audio: np.ndarray) -> None:
    import mlx.core as mx

    feats = cm.compute_features(sample_audio, dtype=mx.float32)
    assert feats.dtype == mx.float32


def test_compute_features_bfloat16_for_inference(sample_audio: np.ndarray) -> None:
    import mlx.core as mx

    # The transcriber calls compute_features with dtype=bfloat16 for the model.
    feats = cm.compute_features(sample_audio, dtype=mx.bfloat16)
    assert feats.dtype == mx.bfloat16
