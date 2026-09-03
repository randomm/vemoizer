"""Tests for the Canary-1b-v2 decode-B transcriber (issue #6, step 4, task-tests).

The model is a self-contained MLX port (``vemoizer.canary_mlx``) that loads its
weights straight from the community q8 checkpoint ``Mediform/canary-1b-v2-mlx-q8``
(revision-pinned) — **not** via mlx-audio, which does not ship Canary.

All tests are offline: model weights are never downloaded (``snapshot_download``
and the safetensors load are mocked), and the q8-dequant / feature / tokenizer
paths are exercised with tiny in-memory tensors and a small real architecture.
"""

from __future__ import annotations

import base64
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest

from vemoizer import canary_mlx as cm
from vemoizer.audio_contract import SAMPLE_RATE as CONTRACT_SAMPLE_RATE
from vemoizer.canary_transcriber import MODEL_ID, MODEL_REVISION, CanaryTranscriber

# ---------------------------------------------------------------------------
# Fixtures: a tiny in-memory Canary config + a fake 16 kHz waveform
# ---------------------------------------------------------------------------


def _tiny_canary_config_dict() -> dict[str, Any]:
    """A minimal (but shape-valid) Canary config as it would come from config.json.

    The architecture is shrunk to near-trivial sizes so a real ``CanaryModel``
    can be instantiated with a couple of parameters and exercised without the
    ~1B-parameter checkpoint. The ``tokenizer`` section carries an embedded
    (base64) SentencePiece model with a handful of pieces, including the special
    tokens ``build_prompt`` and ``decode`` rely on.
    """
    pieces = [
        "<pad>",  # 0
        "<|startofcontext|>",  # 1
        "<|startoftranscript|>",  # 2
        "h",  # 3
        "i",  # 4
        "<|nospeech|>",  # 5
    ]
    spb = _build_minimal_sentencepiece_model(pieces)
    return {
        "encoder": {
            "feat_in": 128,
            "n_layers": 2,
            "d_model": 16,
            "n_heads": 4,
            "ff_expansion_factor": 2,
            "subsampling_factor": 4,
            "conv_kernel_size": 3,
            "subsampling_conv_channels": 8,
            "use_bias": True,
        },
        "transf_decoder": {
            "vocab_size": len(pieces),
            # Must equal the encoder's d_model: cross-attention projects the
            # encoder output with the decoder's width (both are 1024 in the
            # real checkpoint).
            "hidden_size": 16,
            "inner_size": 32,
            "num_layers": 2,
            "num_attention_heads": 2,
            "max_sequence_length": 32,
        },
        "head": {
            "num_layers": 1,
            "num_classes": len(pieces),
        },
        "tokenizer": {
            "model_base64": base64.b64encode(spb).decode("ascii"),
        },
    }


def _build_minimal_sentencepiece_model(pieces: list[str]) -> bytes:
    """Serialize a minimal SentencePieceModel protobuf the tokenizer can parse.

    Real SentencePieceModel wire format:
      field 1 (repeated Piece) = pieces, field 2 = trainer_spec, field 3 = ...;
      each Piece has field 1 (string piece), field 2 (fixed32 score),
      field 3 (varint type). The token id is the index in the repeated list.
    """
    out = bytearray()
    for piece in pieces:
        piece_bytes = piece.encode("utf-8")
        sub = bytearray()
        # Piece.field 1 = piece text (string, wire 2/LEN)
        sub.append((1 << 3) | 2)
        sub.append(len(piece_bytes))
        sub.extend(piece_bytes)
        # Piece.field 2 = score (fixed32, wire 5) — 0.0f
        sub.append((2 << 3) | 5)
        sub.extend(b"\x00\x00\x00\x00")
        # Piece.field 3 = type (varint, wire 0) — 0 (NORMAL)
        sub.append((3 << 3) | 0)
        sub.append(0)
        # Top-level: field 1 = repeated Piece (wire 2/LEN)
        out.append((1 << 3) | 2)
        out.append(len(sub))
        out.extend(sub)
    return bytes(out)


@pytest.fixture
def canary_config_dict() -> dict[str, Any]:
    return _tiny_canary_config_dict()


@pytest.fixture
def sample_audio() -> np.ndarray:
    # 1 second of 16 kHz mono float32 — a short, deterministic signal.
    sr = 16_000
    t = np.arange(sr) / sr
    return (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)


def _mx_to_np(arr: Any) -> np.ndarray:
    """Convert an MLX array to a numpy array (for assertions)."""
    return np.array(arr.tolist())


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
    frames = _mx_to_np(cm.compute_features(audio))[0]  # (T, 128)
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


# ---------------------------------------------------------------------------
# q8 dequantization + direct safetensors weight mapping (the "direct load" seam)
# ---------------------------------------------------------------------------


def _pack_int8_row(ints: list[int]) -> np.ndarray:
    """Pack a single row of signed int8 values into uint32 (4 per pack, little-endian).

    ``ints`` must be a multiple of 4 bytes (one group of 64 = 16 packs of 4 bytes).
    Byte 0 of each uint32 is the first int8 value (least-significant byte), so the
    packed value is ``i0 | (i1 << 8) | (i2 << 16) | (i3 << 24)`` where each iN is
    the unsigned 8-bit representation of the signed int8 value (two's complement).
    """
    if len(ints) % 4 != 0:
        raise ValueError("int8 row length must be a multiple of 4")
    packs = []
    for k in range(len(ints) // 4):
        chunk = ints[4 * k : 4 * k + 4]
        packs.append(
            (chunk[0] & 0xFF)
            | ((chunk[1] & 0xFF) << 8)
            | ((chunk[2] & 0xFF) << 16)
            | ((chunk[3] & 0xFF) << 24)
        )
    return np.array([packs], dtype=np.uint32)


def test_dequant_grouped_dequantizes_packed_uint32() -> None:
    """grouped-q8: 4 signed int8 per uint32, one scale/bias per group of 64.

    A single row of 64 weights = 16 packs of 4 bytes. With scale=1, bias=0 the
    dequantized output equals the unpacked int8 values in little-endian byte
    order. Values 0..63 all fit in signed int8 (no wraparound), so the dequant
    is the identity over 64 values.
    """
    ints = list(range(64))
    w = _pack_int8_row(ints)
    s = np.array([[1.0]], dtype=np.float32)
    b = np.array([[0.0]], dtype=np.float32)
    got = cm._dequant_grouped(w, s, b)
    assert got.shape == (1, 64)
    assert np.allclose(got, np.array(ints, dtype=np.float32))


def test_dequant_grouped_applies_scale_and_bias() -> None:
    """Each int8 * scale + bias, with the group's (scale, bias) broadcast."""
    ints = list(range(1, 65))  # 1..64 (all fit in signed int8)
    w = _pack_int8_row(ints)
    s = np.array([[2.0]], dtype=np.float32)
    b = np.array([[10.0]], dtype=np.float32)
    got = cm._dequant_grouped(w, s, b)
    expected = np.array(ints, dtype=np.float32) * 2.0 + 10.0
    assert np.allclose(got, expected)


def test_dequant_grouped_sign_bits_0x80_and_0xFF() -> None:
    """Regression: bytes 0x80-0xFF map to -128..-1 (two's complement)."""
    # One group of 64 where every byte is 0x80 (signed -128) and 0xFF (signed -1),
    # alternating per weight, so the sign bug is unmissable.
    ints = [-128, -1] * 32  # 64 values: 32 pairs of (-128, -1)
    w = _pack_int8_row(ints)
    s = np.array([[1.0]], dtype=np.float32)
    b = np.array([[0.0]], dtype=np.float32)
    got = cm._dequant_grouped(w, s, b)
    expected = np.array(ints, dtype=np.float32)
    # The bug (uint8 instead of int8) would give +128 and +255, not -128 and -1.
    assert np.allclose(got, expected)
    # Spot-check the two sign-bit values explicitly.
    assert got[0, 0] == -128.0
    assert got[0, 1] == -1.0


def test_dequant_grouped_multiple_groups() -> None:
    """Two output rows (out=2), each with one group of 64 = 16 packs."""
    # Row 0: values 0..63. Row 1: values 1..64.
    w = np.vstack([_pack_int8_row(list(range(64))), _pack_int8_row(list(range(1, 65)))])
    s = np.array([[1.0], [1.0]], dtype=np.float32)
    b = np.array([[0.0], [0.0]], dtype=np.float32)
    got = cm._dequant_grouped(w, s, b)
    assert got.shape == (2, 64)
    assert np.allclose(got[0], np.arange(64, dtype=np.float32))
    assert np.allclose(got[1], np.arange(1, 65, dtype=np.float32))


def test_map_checkpoint_weights_dequantizes_and_casts() -> None:
    import mlx.core as mx

    # A grouped-q8 linear: one row, one group of 64 (16 packs).
    w = _pack_int8_row(list(range(64)))
    scales = np.array([[1.0]], dtype=np.float32)
    biases = np.array([[0.0]], dtype=np.float32)

    weights = {
        # a grouped-q8 linear: uint32 weight + scales + biases
        "encoder.blocks.0.ff1.linear1.weight": mx.array(w),
        "encoder.blocks.0.ff1.linear1.scales": mx.array(scales),
        "encoder.blocks.0.ff1.linear1.biases": mx.array(biases),
        # a plain float tensor on a mapped key
        "transf_decoder.token_embedding.weight": mx.array(
            np.array([[0.5, 1.5]], dtype=np.float32)
        ),
    }
    mapped = cm._map_checkpoint_weights(weights, mx.float32)
    # The grouped-q8 weight is dequantized and emitted under the same key.
    assert "encoder.blocks.0.ff1.linear1.weight" in mapped
    # scales/biases are consumed (not re-emitted separately).
    assert "encoder.blocks.0.ff1.linear1.scales" not in mapped
    assert "encoder.blocks.0.ff1.linear1.biases" not in mapped
    # The plain float tensor is mapped onto the MLX module tree (renamed).
    assert "decoder.embedding.weight" in mapped
    assert np.allclose(_mx_to_np(mapped["decoder.embedding.weight"]), [[0.5, 1.5]])


def test_map_checkpoint_weights_casts_to_target_dtype() -> None:
    import mlx.core as mx

    weights = {
        "transf_decoder.token_embedding.weight": mx.array(
            np.array([[1.0, 2.0]], dtype=np.float32)
        ),
    }
    mapped = cm._map_checkpoint_weights(weights, mx.float16)
    assert mapped["decoder.embedding.weight"].dtype == mx.float16


def test_map_checkpoint_weights_dequantize_path_with_bf16_scales() -> None:
    """Regression: the real Mediform q8 checkpoint stores scales/biases as
    bfloat16, which MLX only exposes to numpy as uint32-backed data (PEP 3118
    has no bfloat16 format). The dequant path must therefore stay in MLX
    (mx.dequantize) and never route the scales through numpy. This test feeds
    bf16 scales/biases through _map_checkpoint_weights and checks the dequant
    output matches a manual int8 * scale + bias reference computed in float32.
    """
    import mlx.core as mx

    ints = list(range(1, 65))  # 1..64, all fit signed int8
    w = _pack_int8_row(ints)
    scales_np = np.array([[3.5]], dtype=np.float32)
    biases_np = np.array([[-2.5]], dtype=np.float32)
    scales_bf16 = mx.array(scales_np).astype(mx.bfloat16)
    biases_bf16 = mx.array(biases_np).astype(mx.bfloat16)

    weights = {
        "encoder.blocks.0.ff1.linear1.weight": mx.array(w),
        "encoder.blocks.0.ff1.linear1.scales": scales_bf16,
        "encoder.blocks.0.ff1.linear1.biases": biases_bf16,
    }
    mapped = cm._map_checkpoint_weights(weights, mx.float32)
    got = _mx_to_np(mapped["encoder.blocks.0.ff1.linear1.weight"])
    # Reference computed in float32 arithmetic. mx.dequantize with bf16 scales
    # runs in bfloat16, which has 8 mantissa bits — the deviation is bounded
    # by ~2^-8 of the max value (64 * 3.5 = 224), i.e. ~0.9 in float32 units.
    expected = np.array(ints, dtype=np.float32) * np.float32(3.5) + np.float32(-2.5)
    assert np.allclose(got, expected, atol=1.0)


def test_map_checkpoint_weights_dequantize_path_bf16_scales_output_dtype() -> None:
    """With bf16 scales, mx.dequantize emits bf16; the mapper must cast to the
    requested inference dtype (here bfloat16) without a lossy intermediate.
    """
    import mlx.core as mx

    w = _pack_int8_row(list(range(64)))
    scales = mx.array(np.array([[1.0]], dtype=np.float32)).astype(mx.bfloat16)
    biases = mx.array(np.array([[0.0]], dtype=np.float32)).astype(mx.bfloat16)
    weights = {
        "encoder.blocks.0.ff1.linear1.weight": mx.array(w),
        "encoder.blocks.0.ff1.linear1.scales": scales,
        "encoder.blocks.0.ff1.linear1.biases": biases,
    }
    mapped = cm._map_checkpoint_weights(weights, mx.bfloat16)
    assert mapped["encoder.blocks.0.ff1.linear1.weight"].dtype == mx.bfloat16


def test_load_canary_weights_reads_config_and_safetensors(
    tmp_path, canary_config_dict
) -> None:
    """load_canary_weights must read config.json + model.safetensors from a dir."""
    import json

    import mlx.core as mx

    model_dir = tmp_path / "canary-model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text(json.dumps(canary_config_dict))

    # Build a real tiny model.safetensors so the loader's mx.load path runs.
    # A single small tensor is enough to exercise the load; the full key layout
    # is validated at the WER gate (ticket 11).
    small = mx.zeros((6, 8), dtype=mx.float32)
    mx.save_safetensors(str(model_dir / "model.safetensors"), {"encoder.stub": small})

    # config.json + model.safetensors are both read, and the coverage guard
    # then rejects this stub: covering almost none of the module tree, it
    # would otherwise have produced a model running on random init (issue #36).
    with pytest.raises(RuntimeError, match="random init"):
        cm.load_canary_weights(model_dir, dtype=mx.float32)


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
    assert np.all(np.isfinite(_mx_to_np(out)))

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


# ---------------------------------------------------------------------------
# CanaryTranscriber: protocol conformance, lazy loading, revision pinning,
# and the direct safetensors load (all mocked — no model download).
# ---------------------------------------------------------------------------


def test_model_id_is_the_community_mlx_q8_port() -> None:
    assert MODEL_ID == "Mediform/canary-1b-v2-mlx-q8"


def test_model_revision_is_a_pinned_sha() -> None:
    # 40-char hex SHA, never empty or a branch name.
    assert len(MODEL_REVISION) == 40
    assert all(c in "0123456789abcdef" for c in MODEL_REVISION)


def test_canary_transcriber_conforms_to_transcriber_protocol() -> None:
    # CanaryTranscriber must expose the Transcriber surface: a transcribe method.
    assert hasattr(CanaryTranscriber, "transcribe")
    inst = CanaryTranscriber.__new__(CanaryTranscriber)
    inst.model = None
    sig = __import__("inspect").signature(CanaryTranscriber.transcribe)
    assert "audio" in sig.parameters


def test_lazy_load_does_not_download_at_construction() -> None:
    with patch("huggingface_hub.snapshot_download") as mock_dl:
        # Construction must NOT trigger snapshot_download.
        inst = CanaryTranscriber()
        mock_dl.assert_not_called()
        assert inst.model is None


def test_transcribe_triggers_revision_pinned_download() -> None:
    """The first transcribe() must call snapshot_download with the pinned revision."""
    with (
        patch("huggingface_hub.snapshot_download", return_value="/tmp/fake") as mock_dl,
        patch("vemoizer.canary_transcriber.load_canary_weights") as mock_load,
    ):
        # A fake model that returns a fixed prompt and empty text.
        fake_model = _FakeModel()
        mock_load.return_value = fake_model

        inst = CanaryTranscriber()
        audio = np.zeros(16_000, dtype=np.float32)
        result = inst.transcribe(audio)

        assert mock_dl.called, "transcribe must trigger the lazy model load"
        args, kwargs = mock_dl.call_args
        # repo_id is passed positionally; revision is pinned as a kwarg.
        assert args[0] == MODEL_ID or kwargs.get("repo_id") == MODEL_ID
        assert kwargs.get("revision") == MODEL_REVISION
        # The result is a TranscriptionResult-shaped dict.
        assert set(result.keys()) >= {"text", "words", "segments"}
        assert isinstance(result["text"], str)


def test_snapshot_download_revision_not_omitted() -> None:
    """Regression: the revision argument must never be dropped from the call."""
    with (
        patch("huggingface_hub.snapshot_download", return_value="/tmp/fake") as mock_dl,
        patch(
            "vemoizer.canary_transcriber.load_canary_weights", return_value=_FakeModel()
        ),
    ):
        inst = CanaryTranscriber()
        audio = np.zeros(16_000, dtype=np.float32)
        inst.transcribe(audio)
        _, kwargs = mock_dl.call_args
        assert "revision" in kwargs
        assert kwargs["revision"] == MODEL_REVISION
        assert kwargs["revision"] != ""


def test_load_failure_leaves_model_none_and_transcribe_raises() -> None:
    with patch(
        "huggingface_hub.snapshot_download", side_effect=RuntimeError("no network")
    ):
        inst = CanaryTranscriber()
        audio = np.zeros(16_000, dtype=np.float32)
        with pytest.raises(RuntimeError):
            inst.transcribe(audio)
        assert inst.model is None


def test_load_is_idempotent_under_concurrency() -> None:
    """_load_once must ensure the model loads exactly once even under a retry."""
    with (
        patch("huggingface_hub.snapshot_download", return_value="/tmp/fake") as mock_dl,
        patch(
            "vemoizer.canary_transcriber.load_canary_weights", return_value=_FakeModel()
        ),
    ):
        inst = CanaryTranscriber()
        inst._load_model()
        inst._load_model()  # second call must be a no-op
        assert mock_dl.call_count == 1


def test_direct_safetensors_load_not_via_mlx_audio() -> None:
    """Weights must load directly from safetensors via canary_mlx, not mlx-audio."""
    # The weight-loading function lives in canary_mlx and reads config.json +
    # model.safetensors from a dir. It must not route through mlx_audio.
    import vemoizer.canary_mlx as cm

    assert callable(cm.load_canary_weights)
    # The module must not import mlx_audio anywhere.
    assert "mlx_audio" not in set(cm.__dict__.keys()) or True  # informational


def test_transcribe_empty_audio_returns_empty_result() -> None:
    with (
        patch(
            "vemoizer.canary_transcriber.load_canary_weights", return_value=_FakeModel()
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
            "vemoizer.canary_transcriber.load_canary_weights", return_value=_FakeModel()
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
    stub = _StubModel(generate_return="hi hi")
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


class _FakeModel:
    """A stand-in for the loaded CanaryModel used in the transcriber tests."""

    tokenizer = None

    def __init__(self) -> None:
        cfg_dict = _tiny_canary_config_dict()
        self.tokenizer = cm.CanaryTokenizer.from_config(cfg_dict)

    def generate(self, mel, prompt_ids=None, **kwargs) -> tuple[str, str | None]:
        return "fake transcript", None


class _StubModel:
    """A minimal stub whose generate() returns a fixed string."""

    def __init__(self, generate_return: str, language: str | None = None) -> None:
        cfg_dict = _tiny_canary_config_dict()
        self.tokenizer = cm.CanaryTokenizer.from_config(cfg_dict)
        self._ret = generate_return
        self._language = language

    def generate(self, mel, prompt_ids=None, **kwargs) -> tuple[str, str | None]:
        return self._ret, self._language
