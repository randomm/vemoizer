"""Eval decode backends for ``vemoizer eval`` (issue #175).

Each backend maps a WAV path to a hypothesis string for the harness. The CLI
module (:mod:`vemoizer.eval_cli`) only wires the command and reports; the
registry and the decode helpers live here so they can be exercised without
the Typer app. Tests patch :data:`BACKENDS` on this module.
"""

from __future__ import annotations

import os
import threading
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

#: C stack for each eval decode. MLX's graph compiler recurses once per graph
#: node; after the consensus arm has run in the same process, the whisper
#: decode's graph overflows macOS's 8 MB main-thread stack (SIGSEGV in
#: ``compile_dfs``). Passing the same decode with a 64 MB stack completes.
DECODE_STACK_BYTES = 256 * 1024 * 1024


def on_large_stack(decode: Callable[[Path], str]) -> Callable[[Path], str]:
    """Run each *decode* call on a worker thread with :data:`DECODE_STACK_BYTES`.

    ``threading.stack_size`` only affects threads created while it is set, so
    it is restored right after the worker is spawned. Exceptions from *decode*
    re-raise on the caller's thread via ``Future.result``.
    """

    def _run(wav: Path) -> str:
        previous = threading.stack_size(DECODE_STACK_BYTES)
        try:
            pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="eval-decode")
            future = pool.submit(decode, wav)
        finally:
            threading.stack_size(previous)
        with pool:
            return future.result()

    return _run


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


def transcribe_meeting_only(path: Path | str) -> dict:
    """Run the ``--profile meeting`` decode A alone: ingest -> VAD -> decode_meeting.

    Mirrors the pipeline's meeting branch (same VAD speech signal, same
    glossary prompt, ``language=None`` so per-window detection stays on,
    invariant #3). Deliberately not ``transcribe_file``: that call runs the
    preflight gate (which demands the HF token for the meeting profile) and
    falls back to the dictation consensus when decode A fails, which would
    score a different system under the whisper-turbo name.
    """
    import vemoizer.pipeline as pipeline_module
    from vemoizer.ingest import IngestError

    logger = pipeline_module.logger
    try:
        audio = pipeline_module.ingest_audio(Path(path))
    except IngestError as e:
        logger.error("ingest failed for %s: %s", path, e)
        return {"text": "", "segments": [], "error": str(e)}
    if len(audio) == 0:
        return {"text": "", "segments": []}
    slices, vad_found_speech = pipeline_module._speech_slices(audio)
    vad_slices = [(s, s + len(a)) for s, a in slices] if vad_found_speech else None
    prompt = pipeline_module.glossary_prompt(pipeline_module.load_glossary(None))
    result = pipeline_module.decode_meeting(
        audio,
        slices,
        initial_prompt=prompt,
        language=None,
        vad_slices=vad_slices,
    )
    if result is None:
        return {"text": "", "segments": []}
    return {
        "text": str(result.get("text", "")).strip(),
        "segments": list(result.get("segments") or []),
    }


def _whisper_turbo(wav: Path) -> str:
    return transcribe_meeting_only(wav)["text"]


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
    "whisper-turbo": _whisper_turbo,
}
