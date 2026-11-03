"""Shared bench and pipeline fakes for the ``vemoizer accept`` test modules
(issue #229, epic #220).

Kept in its own module rather than conftest so the helpers are an explicit
import (the test modules use them directly) and the conftest stays a pure
test-suite configuration file.

The bench is synthetic: real excerpts come from the owner's private
recordings, so the helpers build one with ``freeze_excerpt`` over generated
WAVs with the ffmpeg cut monkeypatched (the same seam as
``tests/test_accept_bundle.py``). ``vemoizer.pipeline.transcribe_file`` is
monkeypatched with canned pipeline results — no models, no network, no audio,
and no real transcript text anywhere.

Every assertion built on these fixtures is ALSO a privacy assertion: excerpts
are named E1..Ek, and the real excerpt ids, the reference text and the
hypothesis text must never reach the printed output.
"""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from typer.testing import CliRunner

import vemoizer.pipeline as pipeline_module
from vemoizer import accept_bundle
from vemoizer import glossary as glossary_module
from vemoizer.accept_bundle import freeze_excerpt
from vemoizer.accept_metrics import score_excerpt
from vemoizer.audio_contract import SAMPLE_RATE, write_wav
from vemoizer.cli import app

runner = CliRunner()

# Ids that would leak straight from the owner's bench; distinctive so a
# leak into the printed output is unambiguous.
MIX_ID = "mix-excerpt-quokka"
TRACK_ID = "twotrack-excerpt-wombat"
REF_TOKEN = "reftokenzzyz"  # a word of the reviewed reference text
HYP_TOKEN = "hyptokenplugh"  # a word of the fresh hypothesis text
TITLE_TOKEN = "titlemarkertitle"  # the generated meeting title
GLOSSARY_TERM = "glossarytermxyz"

_WINDOW = (0.0, 2.0)


def _wav(path: Path) -> Path:
    """A 2 s 440 Hz tone as 16 kHz mono WAV (synthetic, tmp_path only)."""
    t = np.linspace(0, 2.0, int(SAMPLE_RATE * 2.0), endpoint=False)
    return write_wav(path, (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32))


def _sidecar(path: Path, segments: list[dict[str, Any]], names: dict[str, str]) -> Path:
    path.write_text(
        json.dumps({"segments": segments, "speaker_names": names}), encoding="utf-8"
    )
    return path


@pytest.fixture
def bench(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A two-excerpt bench in a project layer, with the cwd inside that layer.

    The cut is monkeypatched (deterministic bytes, no ffmpeg); the project
    layer is ``<tmp>/proj/.vemoizer/bench`` and the cwd is a subdirectory of
    it, so the default ``--bench`` discovery has something to find.
    """

    def _fake_cut(src: Path, dst: Path, start: float, end: float) -> None:
        # Same contract as the real cut: dst exists with content afterwards,
        # so the sha256 bookkeeping under test stays honest.
        Path(dst).parent.mkdir(parents=True, exist_ok=True)
        Path(dst).write_bytes(b"fake-flac")

    monkeypatch.setattr(accept_bundle, "_cut_audio", _fake_cut)
    proj = tmp_path / "proj"
    (proj / "sub").mkdir(parents=True)
    bench_dir = proj / ".vemoizer" / "bench"

    mix = _wav(tmp_path / "mix.wav")
    _sidecar(
        tmp_path / "mix.json",
        [
            {"start": 0.0, "end": 1.0, "speaker": "S1", "text": f"{REF_TOKEN} eilen"},
            {"start": 1.0, "end": 2.0, "speaker": "S2", "text": "toinen osa"},
        ],
        {"S1": "Anna", "S2": "Ben"},
    )
    freeze_excerpt(
        sidecar_path=tmp_path / "mix.json",
        audio={"mix": mix},
        window=_WINDOW,
        bench_dir=bench_dir,
        excerpt_id=MIX_ID,
    )
    _sidecar(
        tmp_path / "call.json",
        [
            {
                "start": 0.0,
                "end": 1.0,
                "speaker": "S1",
                "text": REF_TOKEN,
                "track": "me",
            },
            {"start": 1.0, "end": 2.0, "speaker": "S2", "text": "se oli joo"},
        ],
        {"S1": "Anna", "S2": "Ben"},
    )
    freeze_excerpt(
        sidecar_path=tmp_path / "call.json",
        audio={"me": _wav(tmp_path / "me.wav"), "them": _wav(tmp_path / "them.wav")},
        window=_WINDOW,
        bench_dir=bench_dir,
        excerpt_id=TRACK_ID,
        owner="Maija",
    )
    monkeypatch.chdir(proj / "sub")
    return bench_dir


def _mark_reviewed(bench: Path, *ids: str) -> None:
    for excerpt_id in ids:
        path = bench / "reference" / f"{excerpt_id}.json"
        obj = json.loads(path.read_text(encoding="utf-8"))
        obj["reviewed_by"] = "Owner"
        obj["reviewed_on"] = "2026-02-02"
        path.write_text(json.dumps(obj), encoding="utf-8")


def _result(
    text: str,
    *,
    speaker: str = "Anna",
    title: str = TITLE_TOKEN,
    segments: list[dict[str, Any]] | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    """A canned ``transcribe_file`` result: one paragraph, notes, segments.

    The paragraphs carry the post-repair text the user sees (issue #229),
    deliberately WITHOUT ``compression_ratio`` — only raw segments do.
    """
    para: dict[str, Any] = {"start": 0.0, "end": 2.0, "text": text, "speaker": speaker}
    result: dict[str, Any] = {
        "text": text,
        "segments": segments if segments is not None else [dict(para)],
        "paragraphs": [para],
        "notes": {"title": title},
    }
    if error is not None:
        result["error"] = error
    return result


@pytest.fixture
def fake_pipeline(
    monkeypatch: pytest.MonkeyPatch,
) -> SimpleNamespace:
    """Patch ``transcribe_file`` through its module; record every call.

    ``state.per_excerpt`` maps a bench excerpt id to a canned result, so a
    test can measure one run, then inject a regression for the next.
    """
    calls: list[dict[str, Any]] = []
    state: dict[str, Any] = {"default": _result(HYP_TOKEN), "per_excerpt": {}}

    def fake(path: str | Path, **kwargs: Any) -> dict[str, Any]:
        p = Path(path)
        # The temp glossary is unlinked after the run, so its content has to
        # be sampled at call time; the cwd is recorded so a test can check
        # the run happened inside the bench's project layer.
        gp = kwargs.get("glossary_path")
        calls.append(
            {
                "path": p,
                "cwd": Path(os.getcwd()),
                "glossary_content": (Path(gp).read_text("utf-8") if gp else None),
                **kwargs,
            }
        )
        # bench/audio/<excerpt id>/<slot>.flac -> the excerpt id
        canned = state["per_excerpt"].get(p.parts[-2], state["default"])
        return copy.deepcopy(canned)

    monkeypatch.setattr(pipeline_module, "transcribe_file", fake)
    return SimpleNamespace(calls=calls, state=state)


@pytest.fixture
def fake_glossary(monkeypatch: pytest.MonkeyPatch) -> list[Any]:
    """Patch the glossary loader the pipeline stages use; record its paths.

    The run's own terms come from the layered composition, but the temp
    file handed to ``transcribe_file`` still flows through this loader
    downstream, so the seam stays patched: no test reads ``~/.vemoizer``.
    """
    seen: list[Any] = []

    def fake_load(path: str | Path | None) -> list[str]:
        seen.append(path)
        return []

    monkeypatch.setattr(glossary_module, "load_glossary", fake_load)
    return seen


@pytest.fixture(autouse=True)
def isolated_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Redirect HOME into tmp_path: the real ``~/.vemoizer`` is never read."""
    home = tmp_path / "fakehome"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    return home


def _invoke(
    tmp_path: Path,
    *flags: str,
    bench: Path | None = None,
    baseline: Path | None = None,
) -> Any:
    args = ["accept"]
    if bench is not None:
        args += ["--bench", str(bench)]
    args += ["--baseline", str(baseline or (tmp_path / "accept_baseline.json"))]
    return runner.invoke(app, args + list(flags))


def _metric_keys() -> list[str]:
    """The metric key set ``score_excerpt`` publishes (the stable contract)."""
    return list(
        score_excerpt(
            [{"start": 0.0, "end": 1.0, "speaker": "Anna", "text": "a"}],
            [{"start": 0.0, "end": 1.0, "text": "a"}],
            title="",
            title_expect={"must_contain": [], "forbid": []},
            glossary_terms=[],
        )
    )
