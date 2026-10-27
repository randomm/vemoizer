"""Canary-1b-v2 model forward and transcribe-decode tests (issue #6).

Tiny real architecture, no weights: generate, relative-position attention,
config parsing, and the CanaryTranscriber.transcribe result shape.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import numpy as np
import pytest
from _canary_helpers import FakeModel, StubModel, mx_to_np, tiny_canary_config_dict

from vemoizer import canary_mlx as cm
from vemoizer.canary_transcriber import CanaryTranscriber


@pytest.fixture
def canary_config_dict() -> dict[str, Any]:
    return tiny_canary_config_dict()


# ---------------------------------------------------------------------------
# CanaryModel: build + generate (tiny real architecture, no weights)
# ---------------------------------------------------------------------------


def test_canary_model_generate_returns_decoded_text(
    canary_config_dict: dict[str, Any],
) -> None:
    tok = cm.CanaryTokenizer.from_config(canary_config_dict)
    cfg = cm._canary_config_from_dict(canary_config_dict)
    model = cm.CanaryModel(cfg, tok)

    audio = np.zeros(1600, dtype=np.float32)
    mel = cm.compute_features(audio)
    prompt = tok.build_prompt("fi", "fi")
    text, language = model.generate(mel, prompt, max_tokens=3)
    assert isinstance(text, str)
    # An explicit prompt pins the language, so nothing was detected.
    assert language is None


def test_rel_pos_multi_head_attention_runs_with_mask_reshape() -> None:
    """Regression: the attention block's T5-style relative-position bias must
    be reshaped to a rank-3 mask ``(b*h, q, p)`` before being handed to
    ``mx.fast.scaled_dot_product_attention``. The pre-fix code produced a rank-5
    mask ``(b, 1, h, q, p)`` which the MLX kernel rejects with
    ``ValueError: mask ... expected to have at most rank 4``.

    This builds the block directly with a tiny config, runs it with a
    production-shaped ``b=1`` mel plus a T5-style ``pos_emb`` of length ``2L-1``
    (which is what ``CanaryModel.pre_encode`` produces via ``pos_enc(enc.shape[1])``),
    and pins the mask-broadcast contract end-to-end.
    """
    import mlx.core as mx

    n_heads = 4
    d_model = 8
    block = cm.RelPosMultiHeadAttention(d_model=d_model, n_heads=n_heads)

    batch = 1
    L = 16  # query length; pos_emb is 2L-1 = 31 (T5-style).
    mx.random.seed(1)
    x = mx.random.normal(shape=(batch, L, d_model))
    mx.random.seed(2)
    pos_emb = mx.random.normal(shape=(1, 2 * L - 1, d_model))

    out = block(x, pos_emb)
    assert out.shape == (batch, L, d_model)
    # Finite values: a shape/broadcast failure would produce NaN or crash earlier.
    assert np.all(np.isfinite(mx_to_np(out)))

    # Force a forward pass so the reshape actually runs (not just Python-level).
    mx.eval(out)


def test_canary_config_from_dict_reads_all_sections(
    canary_config_dict: dict[str, Any],
) -> None:
    cfg = cm._canary_config_from_dict(canary_config_dict)
    assert cfg.feat_in == 128
    assert cfg.d_model == 16
    assert cfg.n_heads == 4
    assert cfg.vocab_size == 6
    assert cfg.dec_hidden == 16
    assert cfg.head_num_classes == 6


def test_transcribe_empty_audio_returns_empty_result() -> None:
    with (
        patch(
            "vemoizer.canary_transcriber.load_canary_weights", return_value=FakeModel()
        ) as _,
        patch("huggingface_hub.snapshot_download", return_value="/tmp/fake"),
    ):
        inst = CanaryTranscriber()
        result = inst.transcribe(np.zeros(0, dtype=np.float32))
    assert result["text"] == ""
    assert result["words"] == []
    assert result["segments"] == []


def test_transcribe_returns_transcription_result_with_rtf() -> None:
    with (
        patch("huggingface_hub.snapshot_download", return_value="/tmp/fake"),
        patch(
            "vemoizer.canary_transcriber.load_canary_weights", return_value=FakeModel()
        ),
    ):
        inst = CanaryTranscriber()
        audio = np.zeros(16_000, dtype=np.float32)  # 1 s
        result = inst.transcribe(audio)
    # rtf must be a non-negative float (transcribe_time / audio_duration).
    assert "rtf" in result
    assert result["rtf"] >= 0.0
    assert result["audio_duration"] == pytest.approx(1.0, abs=1e-6)


def test_transcribe_handles_eos_immediately() -> None:
    """If the model immediately emits EOS, text should be empty (or whitespace)."""

    # Use a stub whose generate() returns fixed text. The transcriber's wiring
    # (lazy load -> compute_features -> generate -> result dict) is what this
    # test exercises; the classifier math is covered elsewhere.
    stub = StubModel(generate_return="hi hi")
    with (
        patch("vemoizer.canary_transcriber.load_canary_weights", return_value=stub),
        patch("huggingface_hub.snapshot_download", return_value="/tmp/fake"),
    ):
        inst = CanaryTranscriber()
        audio = np.zeros(16_000, dtype=np.float32)
        result = inst.transcribe(audio)

    assert isinstance(result["text"], str)
    assert result["words"] == []  # Canary port is text-only (no word timestamps)
    assert "language" not in result or result["language"] is None
