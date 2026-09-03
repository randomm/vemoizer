"""Canary end-of-transcript and language identification.

Two prompt-slot behaviours that decide whether decode B produces anything at
all:

* ``eos_id`` — Canary spells the end token ``<|endoftext|>``. Looking for the
  Whisper spelling returned -1, which no token id can equal, so the generation
  loop's early exit was dead and every slice ran the full ``max_tokens``
  budget no matter how short the utterance.
* language — the model names the source language in a prompt slot, so its
  next-token distribution there is a language classifier. Reading it keeps
  language a per-utterance property (project invariant #3). ``<|unklang|>``
  works for the *source* slot but not the *target*: asked to write in an
  unspecified language the model emits an immediate end-of-transcript.

Pure logic — no model download, no network.
"""

from __future__ import annotations

import mlx.core as mx

import vemoizer.canary_mlx as cm


def _tokenizer(pieces: list[str]) -> cm.CanaryTokenizer:
    """A tokenizer over *pieces*, with every ``<|...|>`` piece marked special."""
    specials = {p: i for i, p in enumerate(pieces) if p.startswith("<|")}
    return cm.CanaryTokenizer(pieces, specials)


def _model(tok: cm.CanaryTokenizer) -> cm.CanaryModel:
    """A near-trivial CanaryModel over *tok* — small enough to run untrained."""
    cfg = cm.CanaryConfig(
        feat_in=128,
        n_layers=1,
        d_model=16,
        n_heads=2,
        ff_expansion_factor=2,
        subsampling_factor=4,
        conv_kernel_size=3,
        subsampling_conv_channels=8,
        use_bias=True,
        vocab_size=tok.vocab_size,
        # Cross-attention projects the encoder output with the decoder's
        # width, so these must match (both 1024 in the real checkpoint).
        dec_hidden=16,
        dec_inner=32,
        dec_num_layers=1,
        dec_num_heads=2,
        max_sequence_length=32,
        head_num_layers=1,
        head_num_classes=tok.vocab_size,
    )
    return cm.CanaryModel(cfg, tok)


def test_eos_id_finds_endoftext() -> None:
    tok = _tokenizer(["<pad>", "<|startoftranscript|>", "<|endoftext|>", "a"])
    assert tok.eos_id == 2


def test_eos_id_accepts_the_whisper_spelling() -> None:
    tok = _tokenizer(["<pad>", "<|endoftranscript|>", "a"])
    assert tok.eos_id == 1


def test_eos_id_is_minus_one_without_an_end_token() -> None:
    """-1 still means "no end token"; callers stop on the budget instead."""
    tok = _tokenizer(["<pad>", "<|startoftranscript|>", "a"])
    assert tok.eos_id == -1


def test_language_tokens_exclude_flag_tokens() -> None:
    """``<|pnc|>`` and ``<|itn|>`` share the tag shape but are not languages.

    A language argmax that could land on a flag token would pin a nonsense
    "language" for the decode.
    """
    tok = _tokenizer(
        ["<pad>", "<|fi|>", "<|en|>", "<|pnc|>", "<|itn|>", "<|unklang|>", "a"]
    )
    assert tok.language_tokens == {1: "fi", 2: "en"}


def test_language_tokens_reads_three_letter_codes() -> None:
    tok = _tokenizer(["<pad>", "<|fi|>", "<|swh|>"])
    assert tok.language_tokens == {1: "fi", 2: "swh"}


def test_detect_prefix_is_a_strict_prefix_of_the_full_prompt() -> None:
    """The prefix must stop exactly where the source-language token goes."""
    tok = _tokenizer(
        [
            "<pad>",
            "<|startofcontext|>",
            "<|startoftranscript|>",
            "<|emo:undefined|>",
            "<|fi|>",
            "<|pnc|>",
            "<|noitn|>",
            "<|notimestamp|>",
            "<|nodiarize|>",
        ]
    )
    prefix = tok.detect_prefix()
    full = tok.build_prompt("fi", "fi")

    assert prefix == full[: len(prefix)]
    assert len(prefix) < len(full)
    # The very next slot in the full prompt is the source language.
    assert full[len(prefix)] == tok.special_tokens["<|fi|>"]


def test_detect_language_returns_none_without_language_tokens() -> None:
    """No language tokens in the vocab means there is no detection to report."""
    tok = _tokenizer(["<pad>", "<|startofcontext|>", "<|startoftranscript|>", "a"])
    model = _model(tok)
    assert tok.language_tokens == {}
    assert model.detect_language(mx.zeros((1, 4, 16))) is None


def test_detect_language_only_ever_returns_a_language_code() -> None:
    """The argmax is restricted to language tokens.

    The model here is untrained, so *which* code wins is arbitrary — the
    invariant under test is that the result is always a real language code and
    never a flag token like ``<|pnc|>`` that shares the same tag shape.
    """
    tok = _tokenizer(
        [
            "<pad>",
            "<|startofcontext|>",
            "<|startoftranscript|>",
            "<|emo:undefined|>",
            "<|fi|>",
            "<|en|>",
            "<|pnc|>",
            "<|noitn|>",
            "a",
        ]
    )
    model = _model(tok)
    detected = model.detect_language(mx.zeros((1, 4, 16)))
    assert detected in {"fi", "en"}
