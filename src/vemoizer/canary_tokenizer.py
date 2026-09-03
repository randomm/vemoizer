"""Canary-1b-v2 tokenizer + SentencePiece parser (issue #6).

Sliced out of ``canary_mlx`` to keep each module under the 500-line limit.
Pure-Python SentencePiece model parser (no ``sentencepiece`` dependency)
and the ``CanaryTokenizer`` class used by the transcriber and model.
"""

from __future__ import annotations

import base64
from typing import Any

# Canary special-token strings (stable across the Canary model family).
CANARY_BOS = "<|startoftranscript|>"
CANARY_PAD = "<pad>"
CANARY_NOSPEECH = "<|nospeech|>"
CANARY_PNC = "<|pnc|>"
CANARY_NOPNC = "<|nopnc|>"
CANARY2_BOCTX = "<|startofcontext|>"

# ---------------------------------------------------------------------------
# SentencePiece model parsing (pure Python — no `sentencepiece` dependency)
# ---------------------------------------------------------------------------


def _parse_uvarint(buf: bytes, pos: int) -> tuple[int, int]:
    """Parse an LEB128 (unsigned varint) value from *buf* at *pos*."""
    result = 0
    shift = 0
    while True:
        b = buf[pos]
        pos += 1
        result |= (b & 0x7F) << shift
        if not b & 0x80:
            break
        shift += 7
    return result, pos


def parse_sentencepiece_model(data: bytes) -> tuple[list[str], dict[str, int]]:
    """Parse a serialized SentencePieceModel protobuf.

    Wire format (SentencePiece proto): top-level field 1 = repeated Piece
    (the vocabulary); top-level field 2 = trainer_spec; field 3 = params.
    Each Piece sub-message: field 1 = piece text (string, wire 2),
    field 2 = score (fixed32, wire 5), field 3 = type (varint, wire 0).
    The token id is the 0-based index in the repeated ``pieces`` list.

    Returns ``(pieces, special_ids)`` where *pieces* is the id-ordered list
    of vocab pieces and *special_ids* maps special-token strings (those
    starting with ``<|``) to their ids. This is the minimal reader needed to
    encode prompts and decode output; it does not expose the (unused) LM /
    trainer metadata.
    """
    pieces: list[str] = []
    special_ids: dict[str, int] = {}
    pos = 0
    n = len(data)
    while pos < n:
        tag, pos = _parse_uvarint(data, pos)
        field_no = tag >> 3
        wire = tag & 7
        if field_no == 1 and wire == 2:  # repeated Piece pieces
            length, pos = _parse_uvarint(data, pos)
            end = pos + length
            piece = ""
            q = pos
            while q < end:
                ptag, q = _parse_uvarint(data, q)
                pfield = ptag >> 3
                pwire = ptag & 7
                if pwire == 0:
                    # VARINT field — skip (type enum is field 3)
                    _, q = _parse_uvarint(data, q)
                elif pwire == 2:
                    plen, q = _parse_uvarint(data, q)
                    payload = data[q : q + plen]
                    q += plen
                    if pfield == 1:
                        piece = payload.decode("utf-8", errors="replace")
                elif pwire == 5:
                    q += 4
                elif pwire == 1:
                    q += 8
                else:
                    break
            pieces.append(piece)
            if piece.startswith("<|"):
                special_ids[piece] = len(pieces) - 1
            pos = end
        else:
            # Skip other fields (we only need pieces).
            if wire == 0:
                _, pos = _parse_uvarint(data, pos)
            elif wire == 2:
                length, pos = _parse_uvarint(data, pos)
                pos += length
            elif wire == 5:
                pos += 4
            elif wire == 1:
                pos += 8
            else:
                break
    return pieces, special_ids


class CanaryTokenizer:
    """Pure-Python Canary tokenizer backed by a parsed SentencePiece model.

    Canary-1b-v2 uses a single unified 16k-vocabulary model (not the
    per-language aggregate of the NeMo reference), so encoding a prompt is
    just a lookup of special tokens, and decoding joins pieces and turns the
    SentencePiece space marker into a space.
    """

    def __init__(self, pieces: list[str], special_ids: dict[str, int]) -> None:
        self._pieces = pieces
        self.special_tokens: dict[str, int] = dict(special_ids)
        self.vocab_size = len(pieces)

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> CanaryTokenizer:
        tok = config.get("tokenizer") or {}
        b64 = tok.get("model_base64")
        if not b64:
            raise ValueError("Canary config has no embedded tokenizer model_base64")
        raw = base64.b64decode(b64)
        pieces, special = parse_sentencepiece_model(raw)
        return cls(pieces, special)

    def encode(self, text: str, lang_id: str = "spl_tokens") -> list[int]:
        """Encode a special-token prompt into token ids."""
        if lang_id != "spl_tokens":
            raise ValueError("only spl_tokens prompts are supported")
        import re

        tokens = re.findall(r"<\|[^|]+\|>", text)
        return [self.special_tokens[t] for t in tokens if t in self.special_tokens]

    def decode(self, token_ids: list[int], strip: bool = True) -> str:
        """Decode regular token ids into text (special tokens are dropped)."""
        special_set = set(self.special_tokens.values())
        regular = [
            t for t in token_ids if t not in special_set and 0 <= t < self.vocab_size
        ]
        if not regular:
            return ""
        text = "".join(self._pieces[t] for t in regular).replace("▁", " ")
        return text.strip() if strip else text

    @property
    def eos_id(self) -> int:
        """Id of the transcript end token, or -1 if the vocab has none.

        The real Canary-1b-v2 SentencePiece vocab (16k pieces) has no explicit
        end-of-transcript token — its special tokens are only ``<|...|>``
        prompts, so decoding terminates on the ``max_tokens`` budget.
        """
        for token, token_id in self.special_tokens.items():
            if "endoftranscript" in token:
                return token_id
        return -1

    @property
    def bos_id(self) -> int:
        return self.special_tokens[CANARY_BOS]

    @property
    def nospeech_id(self) -> int:
        return self.special_tokens.get(CANARY_NOSPEECH, 0)

    @property
    def pad_id(self) -> int:
        return self.special_tokens.get(CANARY_PAD, 1)

    def build_prompt(
        self,
        source_lang: str,
        target_lang: str,
        task: str = "transcribe",
        pnc: bool = True,
        prompt_format: str = "canary2",
    ) -> list[int]:
        """Build the canary2 transcribe prompt token list."""
        if prompt_format != "canary2":
            raise ValueError("only canary2 prompt_format is supported")
        if task != "transcribe":
            raise ValueError("only the transcribe task is supported")
        src = f"<|{source_lang}|>"
        tgt = f"<|{target_lang}|>"
        pnc_token = CANARY_PNC if pnc else CANARY_NOPNC
        prompt = (
            f"{CANARY2_BOCTX}{CANARY_BOS}"
            f"<|emo:undefined|>{src}{tgt}{pnc_token}"
            f"<|noitn|><|notimestamp|><|nodiarize|>"
        )
        return self.encode(prompt, "spl_tokens")
