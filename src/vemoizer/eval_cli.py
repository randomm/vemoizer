"""``vemoizer eval`` Typer command (issues #11, #51).

Scores one or more decode backends over the stem-paired fixture corpus and
gates against a committed WER baseline:

    vemoizer eval --backend all --check

Backends: ``parakeet`` and ``canary`` are single decodes through
:func:`vemoizer.pipeline.transcribe_decode_only`; ``consensus`` is the full
pipeline with the LLM forced off (``config_path=os.devnull``) so eval runs
are deterministic and local — pass ``--llm`` to adjudicate with the user's
configured endpoint instead.

The baseline file records the corpus fingerprint alongside the numbers, so
``--check`` refuses to compare against a silently changed corpus (exit 2,
same as a regression). Accuracy claims in PRs come from this command's
output (AGENTS.md invariant #7).
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from pathlib import Path

import typer

from vemoizer.eval_harness import (
    AGGREGATE_KEY,
    compare_to_baseline,
    corpus_fingerprint,
    run_eval,
)

#: Gate tolerance: greedy decodes are deterministic in principle, but Metal
#: reductions are not bit-stable across MLX versions; a small slack keeps
#: the gate meaningful without flaking.
DEFAULT_TOLERANCE = 0.02

DEFAULT_BASELINE = Path("tests/fixtures/wer_baseline.json")


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
    slices = pipeline_module._speech_slices(audio)
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


def _consensus_llm(wav: Path) -> str:
    from vemoizer.pipeline import transcribe_file

    return transcribe_file(wav)["text"]


#: name -> (wav path -> hypothesis). Tests monkeypatch this registry.
BACKENDS: dict[str, Callable[[Path], str]] = {
    "parakeet": _decode_only("parakeet"),
    "canary": _decode_only("canary"),
    "consensus": _consensus,
}


def register_eval(app) -> None:
    """Attach the ``eval`` command to *app* (the main Typer instance)."""

    @app.command("eval")
    def eval(  # noqa: A001, A002 - mirrors vemoizer CLI subcommand name
        corpus: Path = typer.Option(  # noqa: B008
            Path("tests/fixtures/corpus"),
            "--corpus",
            help="Corpus directory of stem-paired .wav/.txt fixtures.",
        ),
        backend: str = typer.Option(
            "all",
            "--backend",
            help="Backend to score: parakeet, canary, consensus, or all.",
        ),
        baseline: Path = typer.Option(  # noqa: B008
            DEFAULT_BASELINE,
            "--baseline",
            help="Committed WER baseline file for --check/--update-baseline.",
        ),
        check: bool = typer.Option(
            False,
            "--check",
            help="Gate against the baseline; exit 2 on any regression.",
        ),
        update_baseline: bool = typer.Option(
            False,
            "--update-baseline",
            help="Write the measured numbers as the new baseline.",
        ),
        llm: bool = typer.Option(
            False,
            "--llm",
            help="Let the consensus backend adjudicate with the configured LLM "
            "(default: LLM off so eval stays local and deterministic).",
        ),
    ) -> None:
        """Score decode backends over the fixture corpus (WER)."""
        if not corpus.is_dir():
            typer.echo(f"error: corpus directory not found: {corpus}", err=True)
            raise typer.Exit(code=1)

        names = list(BACKENDS) if backend == "all" else [backend]
        unknown = [n for n in names if n not in BACKENDS]
        if unknown:
            known = ", ".join([*BACKENDS, "all"])
            typer.echo(
                f"error: unknown backend(s): {', '.join(unknown)} (known: {known})",
                err=True,
            )
            raise typer.Exit(code=2)

        measured: dict[str, dict[str, float]] = {}
        for name in names:
            transcribe = BACKENDS[name]
            if name == "consensus" and llm:
                transcribe = _consensus_llm
            results = run_eval(corpus, transcribe)
            measured[name] = results
            typer.echo(f"[{name}]")
            for sample, value in results.items():
                if sample == AGGREGATE_KEY:
                    continue
                typer.echo(f"{sample}\t{value:.4f}")
            typer.echo(f"{AGGREGATE_KEY}\t{results[AGGREGATE_KEY]:.4f}")

        fingerprint = corpus_fingerprint(corpus)
        if update_baseline:
            _write_baseline(baseline, fingerprint, measured)
            typer.echo(f"baseline updated: {baseline}")
        if check:
            _check_baseline(baseline, fingerprint, measured)


def _write_baseline(
    path: Path, fingerprint: str, measured: dict[str, dict[str, float]]
) -> None:
    existing = _read_baseline(path)
    backends = dict(existing.get("backends", {})) if existing else {}
    backends.update(measured)  # keep other backends' numbers when re-measuring one
    payload = {
        "corpus_fingerprint": fingerprint,
        "tolerance": (existing or {}).get("tolerance", DEFAULT_TOLERANCE),
        "note": (existing or {}).get(
            "note",
            "WER over tests/fixtures/corpus; update only in a dedicated commit.",
        ),
        "backends": backends,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1, sort_keys=True), encoding="utf-8")


def _read_baseline(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _check_baseline(
    path: Path, fingerprint: str, measured: dict[str, dict[str, float]]
) -> None:
    data = _read_baseline(path)
    if data is None:
        typer.echo(
            f"error: no baseline at {path}; run with --update-baseline first",
            err=True,
        )
        raise typer.Exit(code=2)
    if data.get("corpus_fingerprint") != fingerprint:
        typer.echo(
            "error: corpus fingerprint does not match the baseline — the corpus "
            "changed; re-measure with --update-baseline in a dedicated commit",
            err=True,
        )
        raise typer.Exit(code=2)
    tolerance = float(data.get("tolerance", DEFAULT_TOLERANCE))
    failed = False
    for name, results in measured.items():
        base = data.get("backends", {}).get(name)
        if base is None:
            typer.echo(f"error: baseline has no entry for backend {name}", err=True)
            failed = True
            continue
        for reg in compare_to_baseline(results, base, tolerance=tolerance):
            failed = True
            was = "missing" if reg.baseline is None else f"{reg.baseline:.4f}"
            typer.echo(
                f"regression [{name}] {reg.name}: baseline {was} -> "
                f"measured {reg.measured:.4f}",
                err=True,
            )
    if failed:
        raise typer.Exit(code=2)
    typer.echo("baseline check passed")
