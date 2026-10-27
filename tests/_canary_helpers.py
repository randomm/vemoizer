"""Shared tiny-model builders and fake models for the Canary test modules.

Builds a minimal in-memory Canary config (real sentencepiece bytes, tiny
architecture) plus the fake model objects the transcriber tests patch in.
"""

from __future__ import annotations

import base64
from typing import Any

import numpy as np

from vemoizer import canary_mlx as cm


def tiny_canary_config_dict() -> dict[str, Any]:
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
    spb = build_minimal_sentencepiece_model(pieces)
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


def build_minimal_sentencepiece_model(pieces: list[str]) -> bytes:
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


def mx_to_np(arr: Any) -> np.ndarray:
    """Convert an MLX array to a numpy array (for assertions)."""
    return np.array(arr.tolist())


class FakeModel:
    """A stand-in for the loaded CanaryModel used in the transcriber tests."""

    tokenizer = None

    def __init__(self) -> None:
        cfg_dict = tiny_canary_config_dict()
        self.tokenizer = cm.CanaryTokenizer.from_config(cfg_dict)

    def generate(self, mel, prompt_ids=None, **kwargs) -> tuple[str, str | None]:
        return "fake transcript", None


class StubModel:
    """A minimal stub whose generate() returns a fixed string."""

    def __init__(self, generate_return: str, language: str | None = None) -> None:
        cfg_dict = tiny_canary_config_dict()
        self.tokenizer = cm.CanaryTokenizer.from_config(cfg_dict)
        self._ret = generate_return
        self._language = language

    def generate(self, mel, prompt_ids=None, **kwargs) -> tuple[str, str | None]:
        return self._ret, self._language
