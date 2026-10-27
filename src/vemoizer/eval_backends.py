"""Eval decode backends for ``vemoizer eval`` (issue #175).

Each backend maps a WAV path to a hypothesis string for the harness. The CLI
module (:mod:`vemoizer.eval_cli`) only wires the command and reports; the
registry and the decode helpers live here so they can be exercised without
the Typer app. Tests patch :data:`BACKENDS` on this module.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path


def transcribe_decode_only(path: Path | str, *, backend: str) -> dict:
    """Run ingest -> VAD -> one single decode; no consensus, no LLM.

    Lives here because eval is its only caller: the harness scores each
    decode backend on its own so the consensus gain is a measured number
    (invariant #7). The stage chain mirrors ``transcribe_file``'s decode
    stage exactly — same VAD slicing, same merge.
    """
    from contextlib import suppress
    from pathlib import Path as _Path
    from typing import Any

    import vemoizer.pipeline as pipeline_module
    from vemoizer.decode_stage import decode_all
    from vemoizer.ingest import IngestError

    logger = pipeline_module.logger
    # Resolved through the pipeline module so the same seams (and test
    # monkeypatching) govern eval decodes and pipeline decodes alike.
    backends = {
        "parakeet": pipeline_module.ParakeetTranscriber,
        "canary": pipeline_module.CanaryTranscriber,
    }
    if backend not in backends:
        known = ", ".join(sorted(backends))
        raise ValueError(f"unknown backend {backend!r} (known: {known})")
    try:
        audio = pipeline_module.ingest_audio(_Path(path))
    except IngestError as e:
        logger.error("ingest failed for %s: %s", path, e)
        return {"text": "", "segments": [], "error": str(e)}
    if len(audio) == 0:
        return {"text": "", "segments": []}
    slices, _vad_found_speech = pipeline_module._speech_slices(audio)
    transcriber: Any = None
    result: dict[str, Any] | None = None
    try:
        transcriber = backends[backend]()
        result = decode_all(transcriber, slices, f"decode ({backend})")
    except Exception as e:  # noqa: BLE001 - fail-open stage boundary
        logger.warning("decode (%s) failed: %s", backend, e)
    finally:
        if transcriber is not None:
            with suppress(Exception):  # cleanup is best-effort (fail-open)
                transcriber.cleanup()
    if result is None:
        return {"text": "", "segments": []}
    return {
        "text": str(result.get("text", "")).strip(),
        "segments": list(result.get("segments") or []),
    }


def _decode_only(backend: str) -> Callable[[Path], str]:
    def _transcribe(wav: Path) -> str:
        return transcribe_decode_only(wav, backend=backend)["text"]

    return _transcribe


def _consensus(wav: Path) -> str:
    from vemoizer.pipeline import transcribe_file

    # os.devnull parses as empty TOML -> no [llm] -> adjudication skipped:
    # the eval number reflects the local consensus, not a network endpoint.
    return transcribe_file(wav, config_path=os.devnull)["text"]


def consensus_llm(wav: Path) -> str:
    from vemoizer.pipeline import transcribe_file

    return transcribe_file(wav)["text"]


#: name -> (wav path -> hypothesis). Tests monkeypatch this registry.
BACKENDS: dict[str, Callable[[Path], str]] = {
    "parakeet": _decode_only("parakeet"),
    "canary": _decode_only("canary"),
    "consensus": _consensus,
}
