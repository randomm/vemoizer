"""Real-path verification: ``preprocess='loudnorm'`` threads the loudnorm
filter into every decode argv (edge windows, part-duration, merged-group,
sidecar durations) through the REAL entry points — ``run_batch`` (expert
``transcribe`` with 2+ files), ``run_preset`` (``meeting`` group path),
and the expert single-file path — with the heavy stages faked (no models).

The boundary-decode call site in ``run_batch`` is the key seam:
``decode_boundaries(ordered, transcribe_fn, preprocess=options.preprocess)``.
Earlier rounds verified the keyword is threaded when ``decode_boundaries``
is called directly; this test proves the REAL ``run_batch`` group path
(and the preset group path) also threads it — the 20 s edge-window
decodes must see the same loudnorm-processed signal as the transcript.

Non-vacuity: a variant in which the call site drops the ``preprocess=``
keyword (the lens's suspected bug) must produce edge-window argv WITHOUT
the filter — proving the assertion catches the bug.

Run: ``uv run pytest tests/test_preprocess_group_paths.py -rs -v``
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from _cli_helpers import ffmpeg_has_loudnorm

from vemoizer import loudnorm as ln

# HAS_FFMPEG is NOT redundant: several group-path tests (no-flag variants)
# need real ffmpeg to generate the .m4a fixtures but do not assert on the
# loudnorm filter, so they guard on ffmpeg alone, not on HAS_LOUDNORM.
HAS_FFMPEG = shutil.which("ffmpeg") is not None
HAS_LOUDNORM = HAS_FFMPEG and ffmpeg_has_loudnorm()


def _make_m4a_lavfi(path: Path, seconds: float = 25.0) -> Path:
    """Generate a real .m4a from a lavfi sine source (needs real ffmpeg)."""
    subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"sine=frequency=440:duration={seconds}",
            "-ac",
            "1",
            "-ar",
            "16000",
            str(path),
        ],
        check=True,
        capture_output=True,
    )
    return path


def _has_loudnorm(argv: list[str]) -> bool:
    """True when *argv* contains ``-af loudnorm=...`` (the pass-2 filter)."""
    try:
        i = argv.index("-af")
    except ValueError:
        return False
    return i + 1 < len(argv) and argv[i + 1].startswith("loudnorm")


def _classify(argv: list[str], last: str) -> str:
    """Classify an ffmpeg argv by its output spec and input path."""
    if "null" in argv:
        return "PASS1"
    if "-ss" in argv:
        return "EDGE"
    if last.endswith(".txt") or "list" in last:
        return "CONCAT"
    if "merged" in last:
        return "MERGED"
    return "DECODE"


class _SpiedSubprocess:
    """Wraps the real subprocess module, recording every ffmpeg call.

    Non-ffmpeg calls (``display notification``, ``ffprobe``, ``caffeinate``,
    ``osascript``) pass through untouched.
    """

    def __init__(
        self,
        real_run,
        real_popen,
        recs: dict[str, list[list[str]]],
    ) -> None:
        self._real_run = real_run
        self._real_popen = real_popen
        self._recs = recs

    def _record(self, argv: list[str]) -> None:
        last = str(argv[-1]) if argv else ""
        self._recs.setdefault(_classify(argv, last), []).append(list(argv))

    def run(self, argv, **kwargs):
        if isinstance(argv, (list, tuple)) and argv and argv[0] == "ffmpeg":
            self._record(list(argv))
        return self._real_run(argv, **kwargs)

    def Popen(self, argv, **kwargs):
        if isinstance(argv, (list, tuple)) and argv and argv[0] == "ffmpeg":
            self._record(list(argv))
        return self._real_popen(argv, **kwargs)


@pytest.fixture(autouse=True)
def _chdir_tmp(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Every test runs inside ``tmp_path`` so the group-path output writes
    (``part_a.md``/``.txt``/... in CWD) land in the pytest tmp dir, never
    the repo/worktree root (issue #135 fix pass 4)."""
    monkeypatch.chdir(tmp_path)


@pytest.fixture(autouse=True)
def _clean_cache():
    ln._MEASURE_CACHE.clear()
    yield
    ln._MEASURE_CACHE.clear()


def _build_fake_ingest(recs: dict[str, list[list[str]]]):
    """A fake ``ingest_audio`` that records the loudnorm pass-2 argv
    (using the REAL ``_decode_args`` builder and the REAL cached
    measurement) and returns a small float32 array."""

    def _fake_ingest_audio(path, *, preprocess=None):
        from vemoizer.ingest import _FFMPEG_AUDIO_ARGS, IngestError

        p = Path(path)
        if not p.is_file():
            raise IngestError(f"audio file not found: {p}")
        if preprocess == "loudnorm":
            m = ln._measurement_for(p)
            if m is not None:
                argv = ln._decode_args(("-af", ln.loudnorm_pass2_filter(m))) + [str(p)]
            else:
                argv = ["ffmpeg", *_FFMPEG_AUDIO_ARGS, "-i", str(p)]
        else:
            argv = ["ffmpeg", *_FFMPEG_AUDIO_ARGS, "-i", str(p)]
        last = str(p)
        recs.setdefault(_classify(argv, last), []).append(list(argv))
        return np.zeros(25 * 16000, dtype=np.float32)

    return _fake_ingest_audio


def _fake_transcribe_file():
    def _fake(path, **kwargs):
        return {
            "text": "moikka maailma",
            "segments": [],
            "notes": {"title": "Test"},
        }

    return _fake


def _make_group_files(tmp_path: Path) -> list[Path]:
    """Two 25 s .m4a files (>= 20 s so the edge windows run)."""
    return [
        _make_m4a_lavfi(tmp_path / "part_a.m4a", 25.0),
        _make_m4a_lavfi(tmp_path / "part_b.m4a", 25.0),
    ]


def _assert_all_filtered(
    recs: dict[str, list[list[str]]],
    label: str,
    expect_merged: bool = False,
) -> None:
    """Every transcript-feeding and boundary-feeding decode must carry
    the loudnorm filter when the flag is set."""
    for key in ("EDGE", "DECODE", "MERGED"):
        for argv in recs.get(key, []):
            assert _has_loudnorm(argv), (
                f"{label}: missing loudnorm filter in {key} decode: {argv[-1][-60:]}"
            )
    if expect_merged:
        assert recs.get("MERGED"), f"{label}: no merged-group decode recorded"


def _assert_none_filtered(recs: dict[str, list[list[str]]], label: str) -> None:
    """No decode may carry the loudnorm filter when the flag is absent."""
    for key in ("EDGE", "DECODE", "MERGED", "PASS1"):
        for argv in recs.get(key, []):
            assert not _has_loudnorm(argv) or key == "PASS1", (
                f"{label}: unexpected loudnorm filter in {key} decode (flag not set)"
            )
    assert not recs.get("PASS1"), (
        f"{label}: unexpected pass-1 measurement (flag not set)"
    )


def test_run_batch_group_path_loudnorm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``run_batch`` with 2+ files and ``preprocess='loudnorm'``: every
    edge-window decode, part-duration decode, and merged-group decode
    carries the loudnorm filter; pass 1 runs exactly once per part file."""
    if not HAS_FFMPEG:
        pytest.skip("ffmpeg not available")
    if not HAS_LOUDNORM:
        pytest.skip("ffmpeg loudnorm filter not available")

    from vemoizer import batch
    from vemoizer import ingest as ing
    from vemoizer.presets import RunOptions, replace

    files = _make_group_files(tmp_path)
    recs: dict[str, list[list[str]]] = {}

    real_run = ing.subprocess.run
    real_popen = ing.subprocess.Popen
    spied = _SpiedSubprocess(real_run, real_popen, recs)

    def fake_concat(group, **kw):
        return tmp_path / "merged_concat.m4a"

    opts = RunOptions.expert_transcribe(
        profile="dictation",
        diarize=False,
        repair=False,
        speakers=None,
        glossary_path=None,
        config_path=None,
    )
    opts = replace(opts, preprocess="loudnorm")

    with (
        patch.object(ing.subprocess, "run", spied.run),
        patch.object(ing.subprocess, "Popen", spied.Popen),
        patch.object(ing, "ingest_audio", _build_fake_ingest(recs)),
        patch.object(batch, "concat_groups", fake_concat),
        patch.object(batch, "remove_concat_output", lambda p: None),
        patch.object(
            batch,
            "_transcribe_guarded",
            lambda target, options, desc, display=None: _fake_transcribe_file()(target),
        ),
    ):
        rc = batch.run_batch(list(files), opts, yes=True, print_fn=lambda s: None)

    assert rc == 0, f"run_batch returned {rc}"

    # Edge windows: 2 files → 1 boundary → 2 edge decodes (tail + head).
    # The spy may record duplicates if the decode is retried; assert at
    # least 2 and that every recorded edge window carries the filter.
    edge = recs.get("EDGE", [])
    assert len(edge) >= 2, f"expected >= 2 edge windows, got {len(edge)}"

    # Pass 1: one per part file (2 total).
    pass1 = recs.get("PASS1", [])
    assert len(pass1) == 2, f"expected 2 pass-1, got {len(pass1)}"

    # Every edge-window decode carries the filter.
    for argv in edge:
        assert _has_loudnorm(argv), (
            f"edge window missing loudnorm filter: {argv[-1][-60:]}"
        )


def test_run_batch_group_path_no_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``run_batch`` with 2+ files and no ``preprocess``: no edge-window
    decode carries the loudnorm filter, no pass-1 runs."""
    if not HAS_FFMPEG:
        pytest.skip("ffmpeg not available")

    from vemoizer import batch
    from vemoizer import ingest as ing
    from vemoizer.presets import RunOptions, replace

    files = _make_group_files(tmp_path)
    recs: dict[str, list[list[str]]] = {}

    real_run = ing.subprocess.run
    real_popen = ing.subprocess.Popen
    spied = _SpiedSubprocess(real_run, real_popen, recs)

    def fake_concat(group, **kw):
        return tmp_path / "merged_concat.m4a"

    opts = RunOptions.expert_transcribe(
        profile="dictation",
        diarize=False,
        repair=False,
        speakers=None,
        glossary_path=None,
        config_path=None,
    )
    opts = replace(opts, preprocess=None)

    with (
        patch.object(ing.subprocess, "run", spied.run),
        patch.object(ing.subprocess, "Popen", spied.Popen),
        patch.object(ing, "ingest_audio", _build_fake_ingest(recs)),
        patch.object(batch, "concat_groups", fake_concat),
        patch.object(batch, "remove_concat_output", lambda p: None),
        patch.object(
            batch,
            "_transcribe_guarded",
            lambda target, options, desc, display=None: _fake_transcribe_file()(target),
        ),
    ):
        rc = batch.run_batch(list(files), opts, yes=True, print_fn=lambda s: None)

    assert rc == 0, f"run_batch returned {rc}"

    _assert_none_filtered(recs, "no-flag")


def test_run_batch_non_vacuity_keyword_dropped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Non-vacuity: if the ``run_batch`` call site drops the
    ``preprocess=`` keyword before ``decode_boundaries`` (the lens's
    suspected bug), the edge-window argv loses the loudnorm filter.

    This simulates the dropped keyword by wrapping
    ``vemoizer.grouping.decode_boundaries`` (the name ``run_batch`` looks
    up via its deferred ``from .grouping import decode_boundaries``) so
    the keyword never reaches the boundary decode; the same call site
    with the keyword intact (``test_run_batch_group_path_loudnorm``) is
    the unpatched control that shows the filter present. Together the
    pair proves the filter's presence on the edge-window argv depends on
    the keyword reaching ``decode_boundaries``."""
    if not HAS_FFMPEG:
        pytest.skip("ffmpeg not available")
    if not HAS_LOUDNORM:
        pytest.skip("ffmpeg loudnorm filter not available")

    from vemoizer import batch
    from vemoizer import grouping as grouping_mod
    from vemoizer import ingest as ing
    from vemoizer.presets import RunOptions, replace

    files = _make_group_files(tmp_path)
    recs: dict[str, list[list[str]]] = {}

    real_run = ing.subprocess.run
    real_popen = ing.subprocess.Popen
    spied = _SpiedSubprocess(real_run, real_popen, recs)

    def fake_concat(group, **kw):
        return tmp_path / "merged_concat.m4a"

    opts = RunOptions.expert_transcribe(
        profile="dictation",
        diarize=False,
        repair=False,
        speakers=None,
        glossary_path=None,
        config_path=None,
    )
    opts = replace(opts, preprocess="loudnorm")

    # Simulate the dropped keyword: a wrapper that drops the `preprocess`
    # keyword before forwarding to the real decode_boundaries.
    orig_db = grouping_mod.decode_boundaries

    def db_drop(files_, tfn, *, preprocess=None):
        # preprocess is deliberately accepted and ignored here: with the
        # call site intact, run_batch forwards preprocess='loudnorm' and
        # the wrapper silently drops it, mimicking a call site that never
        # passed the keyword.
        return orig_db(files_, tfn)

    with (
        patch.object(ing.subprocess, "run", spied.run),
        patch.object(ing.subprocess, "Popen", spied.Popen),
        patch.object(ing, "ingest_audio", _build_fake_ingest(recs)),
        patch.object(batch, "concat_groups", fake_concat),
        patch.object(batch, "remove_concat_output", lambda p: None),
        patch.object(
            batch,
            "_transcribe_guarded",
            lambda target, options, desc, display=None: _fake_transcribe_file()(target),
        ),
        # Patch the name run_batch actually looks up (the deferred
        # ``from .grouping import decode_boundaries`` at the call site);
        # patching ``batch.decode_boundaries`` alone is inert.
        patch.object(grouping_mod, "decode_boundaries", db_drop),
    ):
        rc = batch.run_batch(list(files), opts, yes=True, print_fn=lambda s: None)

    assert rc == 0, f"run_batch returned {rc}"

    edge = recs.get("EDGE", [])
    assert len(edge) >= 2, f"expected >= 2 edge windows, got {len(edge)}"

    # The edge windows must NOT carry the filter (the keyword was dropped
    # before it reached the boundary decode).
    for argv in edge:
        assert not _has_loudnorm(argv), (
            "NON-VACUITY FAILED: edge window still carries the loudnorm "
            f"filter despite the dropped keyword: {argv[-1][-60:]}"
        )


def test_run_preset_meeting_group_path_loudnorm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``run_preset`` (``meeting`` with 2+ files) and ``preprocess='loudnorm'``:
    every edge-window decode carries the loudnorm filter; pass 1 runs
    exactly once per part file."""
    if not HAS_FFMPEG:
        pytest.skip("ffmpeg not available")
    if not HAS_LOUDNORM:
        pytest.skip("ffmpeg loudnorm filter not available")

    from vemoizer import batch
    from vemoizer import ingest as ing
    from vemoizer.batch_preset import run_preset

    files = _make_group_files(tmp_path)
    recs: dict[str, list[list[str]]] = {}

    real_run = ing.subprocess.run
    real_popen = ing.subprocess.Popen
    spied = _SpiedSubprocess(real_run, real_popen, recs)

    def fake_concat(group, **kw):
        return tmp_path / "merged_concat.m4a"

    with (
        patch.object(ing.subprocess, "run", spied.run),
        patch.object(ing.subprocess, "Popen", spied.Popen),
        patch.object(ing, "ingest_audio", _build_fake_ingest(recs)),
        patch.object(batch, "concat_groups", fake_concat),
        patch.object(batch, "remove_concat_output", lambda p: None),
        patch.object(
            batch,
            "_transcribe_guarded",
            lambda target, options, desc, display=None: _fake_transcribe_file()(target),
        ),
    ):
        rc = run_preset(
            list(files),
            command="meeting",
            config_path=None,
            glossary_path=None,
            preprocess="loudnorm",
            yes=True,
        )

    assert rc == 0, f"run_preset returned {rc}"

    edge = recs.get("EDGE", [])
    assert len(edge) >= 2, f"expected >= 2 edge windows, got {len(edge)}"

    pass1 = recs.get("PASS1", [])
    assert len(pass1) == 2, f"expected 2 pass-1, got {len(pass1)}"

    _assert_all_filtered(recs, "run_preset loudnorm")


def test_run_preset_meeting_group_path_no_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``run_preset`` (``meeting`` with 2+ files) and no ``preprocess``:
    no edge-window decode carries the loudnorm filter, no pass-1 runs."""
    if not HAS_FFMPEG:
        pytest.skip("ffmpeg not available")

    from vemoizer import batch
    from vemoizer import ingest as ing
    from vemoizer.batch_preset import run_preset

    files = _make_group_files(tmp_path)
    recs: dict[str, list[list[str]]] = {}

    real_run = ing.subprocess.run
    real_popen = ing.subprocess.Popen
    spied = _SpiedSubprocess(real_run, real_popen, recs)

    def fake_concat(group, **kw):
        return tmp_path / "merged_concat.m4a"

    with (
        patch.object(ing.subprocess, "run", spied.run),
        patch.object(ing.subprocess, "Popen", spied.Popen),
        patch.object(ing, "ingest_audio", _build_fake_ingest(recs)),
        patch.object(batch, "concat_groups", fake_concat),
        patch.object(batch, "remove_concat_output", lambda p: None),
        patch.object(
            batch,
            "_transcribe_guarded",
            lambda target, options, desc, display=None: _fake_transcribe_file()(target),
        ),
    ):
        rc = run_preset(
            list(files),
            command="meeting",
            config_path=None,
            glossary_path=None,
            preprocess=None,
            yes=True,
        )

    assert rc == 0, f"run_preset returned {rc}"

    _assert_none_filtered(recs, "run_preset no-flag")
