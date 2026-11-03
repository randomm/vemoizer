"""The per-excerpt bench run: the real meeting path, and its scoring (#229).

``vemoizer accept`` exists to gate the pipeline a user actually runs, so these
tests pin the contract of one excerpt's measurement: the kwargs handed to
``transcribe_file`` are the ones ``vemoizer meeting`` hands it, the run happens
in the bench's project layer with the layered glossary, the hypothesis scored
is the post-repair text, and a crashing excerpt is a FAILED row instead of an
aborted bench. Flags, the report and the gate are
``tests/test_accept_cli.py``; the shared synthetic bench and the pipeline fakes
are ``tests/_accept_helpers.py``. No models, no network, no audio.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest
from _accept_helpers import (
    GLOSSARY_TERM,
    HYP_TOKEN,
    MIX_ID,
    TRACK_ID,
    _invoke,
    _result,
)

# The shared fixtures are imported into THIS module's namespace, where pytest
# looks them up; the redundant alias is the re-export form ruff accepts.
from _accept_helpers import bench as bench
from _accept_helpers import fake_glossary as fake_glossary
from _accept_helpers import fake_pipeline as fake_pipeline
from _accept_helpers import isolated_home as isolated_home

import vemoizer.pipeline as pipeline_module
from vemoizer import batch as batch_module
from vemoizer import batch_output, batch_preset
from vemoizer.loops import loop_seconds
from vemoizer.presets import MEETING_SPEAKERS, RunOptions, replace
from vemoizer.tracks import TrackPaths


def _expected_kwargs(options: RunOptions) -> dict[str, Any]:
    """The kwargs ``batch_guard._transcribe_one`` passes for *options*.

    Written out from the real meeting path (``vemoizer meeting`` ->
    ``run_preset`` -> ``_transcribe_one``), so the accept gate is compared
    against the user's pipeline instead of against itself.
    """
    return {
        "diarize": options.diarize,
        "config_path": options.config_path,
        "profile": options.profile,
        "repair": options.repair,
        "glossary_path": options.glossary_path,
        "speakers": options.speakers,
        "display": None,
        "language": None if options.language == "auto" else options.language,
        "preprocess": options.preprocess,
        "tracks": options.tracks,
        "owner": options.owner,
    }


def test_run_passes_exactly_the_meeting_pipeline_arguments(
    tmp_path: Path, bench: Path, fake_pipeline: Any, fake_glossary: list[Any]
) -> None:
    """``accept`` runs the SAME pipeline a user's ``vemoizer meeting`` does.

    Regression (issue #229 review): the bench called ``transcribe_file`` with
    a hand-picked kwarg subset, so ``repair`` — the stage that invents
    glossary terms — and the meeting speaker bounds were silently off, and
    the gate could never see the thing it exists to measure.
    """
    proj = bench.parent.parent
    (proj / ".vemoizer" / "glossary.txt").write_text(_PROJECT_GLOSSARY, "utf-8")

    res = _invoke(tmp_path, bench=bench)
    assert res.exit_code == 0, res.output

    # The very options a meeting run resolves from this project layer, with
    # the per-excerpt fields (tracks/owner) and the composed temp glossary.
    resolved = batch_preset.resolve_preset_glossary_options("meeting", {})
    glossary = fake_pipeline.calls[0]["glossary_path"]
    assert glossary is not None
    mix_options = replace(
        resolved, glossary_path=glossary, diarize=True, tracks=None, owner=None
    )
    track_options = replace(
        resolved,
        glossary_path=glossary,
        diarize=True,
        tracks=TrackPaths(
            me=bench / "audio" / TRACK_ID / "me.flac",
            them=bench / "audio" / TRACK_ID / "them.flac",
        ),
        owner="Maija",
    )

    mix, track = fake_pipeline.calls
    assert mix["path"] == bench / "audio" / MIX_ID / "mix.flac"
    assert track["path"].name == "me.flac"
    for call, options in ((mix, mix_options), (track, track_options)):
        recorded = {
            k: v
            for k, v in call.items()
            if k not in ("path", "cwd", "glossary_content")
        }
        assert recorded == _expected_kwargs(options)
    # The stage contract, spelled out: the gate is blind without repair.
    for call in (mix, track):
        assert call["repair"] is True
        assert call["speakers"] == MEETING_SPEAKERS
        assert call["profile"] == "meeting"
        assert call["diarize"] is True
        # Default layered config resolution: no explicit config path.
        assert call["config_path"] is None


# ---------------------------------------------------------------------------
# The run uses the BENCH's project layer: glossary, config and cwd all
# resolve as they would in a normal run inside the bench's project.
# ---------------------------------------------------------------------------

_PROJECT_GLOSSARY = f"{GLOSSARY_TERM}\n@{GLOSSARY_TERM}-llm\nkonto => kontexti\n"


def test_project_layer_glossary_reaches_transcribe_and_scoring(
    tmp_path: Path,
    bench: Path,
    fake_pipeline: Any,
    fake_glossary: list[Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A term and a correction pair in the bench's project glossary must
    reach the pipeline (temp glossary file) AND the scoring (terms), and
    the run must execute from the project root even when invoked elsewhere.
    """
    proj = bench.parent.parent
    (proj / ".vemoizer" / "glossary.txt").write_text(_PROJECT_GLOSSARY, "utf-8")
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)  # NOT inside the bench's project layer
    fake_pipeline.state["per_excerpt"][MIX_ID] = _result(
        f"{HYP_TOKEN} {GLOSSARY_TERM} ja {GLOSSARY_TERM}"
    )

    res = _invoke(tmp_path, bench=bench)

    assert res.exit_code == 0, res.output
    # Every per-excerpt pipeline work item runs inside the project layer.
    assert {c["cwd"] for c in fake_pipeline.calls} == {proj.resolve()}
    mix = fake_pipeline.calls[0]
    assert mix["glossary_path"] is not None
    # The temp file is unlinked after the run, so its content was sampled
    # at call time: the project term and the correction pair must be in the
    # file the pipeline gets (the lines a vemoizer meeting run writes).
    content = mix["glossary_content"]
    assert GLOSSARY_TERM in content
    assert "konto => kontexti" in content
    # The @-prefixed term is written with its @ marker (LLM-only).
    assert f"@{GLOSSARY_TERM}-llm" in content
    # Temp glossary deleted after the run.
    assert not Path(mix["glossary_path"]).exists()
    # Scoring saw the project term: the hypothesis inserts it twice where
    # the reference never says it.
    line = next(line for line in res.output.splitlines() if line.startswith("E1"))
    assert "glossary_insertions=2" in line
    # The cwd is restored afterwards.
    assert Path(os.getcwd()) == elsewhere.resolve()


def test_run_restores_cwd_when_a_pipeline_run_raises(
    tmp_path: Path,
    bench: Path,
    fake_glossary: list[Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An exception inside the chdir'd work must still restore the cwd."""

    def boom(path: str | Path, **kwargs: Any) -> dict[str, Any]:
        raise RuntimeError("decode exploded")

    monkeypatch.setattr(pipeline_module, "transcribe_file", boom)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)

    res = _invoke(tmp_path, bench=bench)

    assert res.exit_code == 0  # a FAILED row, not a crashed command
    assert Path(os.getcwd()) == elsewhere.resolve()


# ---------------------------------------------------------------------------
# One shared code path: batch_preset and accept compose the meeting
# glossary through the SAME helpers.
# ---------------------------------------------------------------------------


def test_batch_preset_and_accept_share_the_glossary_composition(
    tmp_path: Path,
    bench: Path,
    fake_pipeline: Any,
    fake_glossary: list[Any],
    isolated_home: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``meeting_glossary_lines`` is the single code path for the temp file.

    batch_preset writes exactly the lines the helper returns (byte-identical
    behaviour, one function, two callers), and accept passes those same
    lines to the pipeline.
    """
    proj = bench.parent.parent
    (proj / ".vemoizer" / "glossary.txt").write_text(_PROJECT_GLOSSARY, "utf-8")
    home = isolated_home / ".vemoizer"
    home.mkdir(parents=True)
    (home / "glossary.txt").write_text("vanha\nkonto => koto\n", "utf-8")

    options = batch_preset.resolve_preset_glossary_options("meeting", {})
    lines = batch_preset.meeting_glossary_lines(options)
    assert lines == [
        GLOSSARY_TERM,
        "vanha",
        f"@{GLOSSARY_TERM}-llm",
        "konto => kontexti",
    ]

    # accept writes the helper's lines through the same temp-file seam the
    # meeting path uses (vemoizer.batch._write_temp_glossary).
    seen: list[list[str]] = []

    def spy(ls: list[str]) -> str:
        seen.append(list(ls))
        return batch_output._write_temp_glossary(ls)

    monkeypatch.setattr(batch_module, "_write_temp_glossary", spy)
    fake_pipeline.state["per_excerpt"][MIX_ID] = _result(f"{HYP_TOKEN} {GLOSSARY_TERM}")

    res = _invoke(tmp_path, bench=bench)

    assert res.exit_code == 0, res.output
    # One composed temp glossary for the run, handed to both excerpts.
    assert seen == [lines]
    assert fake_pipeline.calls[0]["glossary_content"] == "\n".join(lines) + "\n"
    assert (
        fake_pipeline.calls[0]["glossary_path"]
        == fake_pipeline.calls[1]["glossary_path"]
    )

    # One code path, not two copies: patching the shared helper changes what
    # accept writes, because accept calls exactly that function.
    monkeypatch.setattr(
        batch_preset, "meeting_glossary_lines", lambda opts: ["shared-only"]
    )
    seen.clear()
    assert _invoke(tmp_path, bench=bench).exit_code == 0
    assert seen == [["shared-only"]]


def test_loop_seconds_comes_from_raw_segments(
    tmp_path: Path, bench: Path, fake_pipeline: Any, fake_glossary: list[Any]
) -> None:
    # Only raw segments carry compression_ratio; paragraphs do not, so the
    # wall is invisible to the metric unless the CLI overrides it.
    wall = {
        "start": 0.0,
        "end": 7.0,
        "text": "a " * 10,
        "compression_ratio": 3.0,
    }
    fake_pipeline.state["per_excerpt"][MIX_ID] = _result(HYP_TOKEN, segments=[wall])

    res = _invoke(tmp_path, bench=bench)

    assert res.exit_code == 0, res.output
    assert loop_seconds([wall]) == pytest.approx(7.0)
    line = next(line for line in res.output.splitlines() if line.startswith("E1"))
    assert "loop_seconds=7.0000" in line
    # The other excerpt has no wall at all: seconds always read as decimals.
    other = next(line for line in res.output.splitlines() if line.startswith("E2"))
    assert "loop_seconds=0.0000" in other


def test_pipeline_exception_is_a_failed_row_not_a_traceback(
    tmp_path: Path,
    bench: Path,
    fake_glossary: list[Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A crash in one excerpt is that excerpt's FAILED row (fail-open bench).

    The pipeline is fail-open by contract and the bench must REPORT a crash,
    so a raising ``transcribe_file`` neither aborts the run nor hides which
    stage died: the exception type goes to stderr, the row reads FAILED, and
    the other excerpts are still measured.
    """

    def boom(path: str | Path, **kwargs: Any) -> dict[str, Any]:
        if Path(path).parts[-2] == MIX_ID:
            raise RuntimeError("decode exploded")
        return _result(HYP_TOKEN)

    monkeypatch.setattr(pipeline_module, "transcribe_file", boom)

    res = _invoke(tmp_path, bench=bench)

    assert res.exit_code == 0, res.output
    assert "E1\tFAILED" in res.stdout
    assert "E2\t" in res.stdout
    assert "RuntimeError" in res.stderr
    assert "Traceback" not in res.stderr


def test_keyboard_interrupt_is_not_a_failed_row(
    tmp_path: Path,
    bench: Path,
    fake_glossary: list[Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ctrl-C aborts the bench: the fail-open boundary is for crashes only."""

    def interrupt(path: str | Path, **kwargs: Any) -> dict[str, Any]:
        raise KeyboardInterrupt

    monkeypatch.setattr(pipeline_module, "transcribe_file", interrupt)

    res = _invoke(tmp_path, bench=bench)

    assert res.exit_code == 130  # click's Abort path, not a measured run
    assert "FAILED" not in res.stdout and "aggregate" not in res.stdout
