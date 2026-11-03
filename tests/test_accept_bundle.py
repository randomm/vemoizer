"""Tests for the private acceptance bench bundle (issue #227, epic #220).

All audio is synthetic (a 2 s tone written with the repo's own WAV
writer) into tmp_path — no models, no network, no real memos. The ffmpeg
cut is monkeypatched for the format tests; one end-to-end test drives the
real ffmpeg when it is on PATH.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from vemoizer import accept_bundle
from vemoizer.accept_bundle import (
    BundleError,
    freeze_excerpt,
    load_bundle,
    unreviewed,
)
from vemoizer.audio_contract import SAMPLE_RATE, write_wav

_FFMPEG = shutil.which("ffmpeg")


def _make_wav(path: Path, seconds: float = 2.0) -> Path:
    """A 2 s 440 Hz tone as 16 kHz mono WAV (synthetic, tmp_path only)."""
    t = np.linspace(0, seconds, int(SAMPLE_RATE * seconds), endpoint=False)
    return write_wav(path, (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32))


def _sidecar(
    path: Path,
    segments: list[dict[str, Any]],
    speaker_names: dict[str, str] | None = None,
) -> Path:
    obj: dict[str, Any] = {"segments": segments}
    if speaker_names is not None:
        obj["speaker_names"] = speaker_names
    path.write_text(json.dumps(obj), encoding="utf-8")
    return path


@pytest.fixture()
def fake_cut(monkeypatch: pytest.MonkeyPatch) -> list[tuple[Path, Path, float, float]]:
    """Replace the ffmpeg cut with deterministic byte writes (no ffmpeg needed).

    Mirrors the real contract: *dst* must end up existing with content, so
    the sha256 bookkeeping under test stays honest.
    """
    calls: list[tuple[Path, Path, float, float]] = []

    def _cut(src: Path, dst: Path, start: float, end: float) -> None:
        calls.append((Path(src), Path(dst), start, end))
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(f"fake-flac:{Path(src).name}:{start}:{end}".encode())

    monkeypatch.setattr(accept_bundle, "_cut_audio", _cut)
    return calls


def _overlap_segments() -> list[dict[str, Any]]:
    """Segments around a (0.5, 1.5) window: one straddling each edge, one outside."""
    return [
        {"start": 0.0, "end": 0.8, "speaker": "SPEAKER_01", "text": "moi"},
        {"start": 1.0, "end": 1.6, "speaker": "SPEAKER_02", "text": "terve"},
        {"start": 5.0, "end": 6.0, "speaker": "SPEAKER_01", "text": "outside"},
    ]


def _freeze_mix(
    tmp_path: Path,
    fake_cut: list[Any],
    excerpt_id: str = "ex1",
    **overrides: Any,
) -> tuple[Any, Path]:
    """Freeze a mix excerpt with a standard sidecar; returns (excerpt, bench)."""
    src = _make_wav(tmp_path / "orig.wav")
    sidecar = overrides.pop("sidecar", None) or _sidecar(
        tmp_path / "orig.json", _overlap_segments(), {"SPEAKER_01": "Anna"}
    )
    args: dict[str, Any] = {
        "sidecar_path": sidecar,
        "audio": {"mix": src},
        "window": (0.5, 1.5),
        "bench_dir": tmp_path / "bench",
        "excerpt_id": excerpt_id,
    }
    args.update(overrides)
    return freeze_excerpt(**args), Path(args["bench_dir"])


def _mark_reviewed(bench: Path, excerpt_id: str, reviewer: str = "Janni") -> None:
    ref_path = bench / "reference" / f"{excerpt_id}.json"
    obj = json.loads(ref_path.read_text(encoding="utf-8"))
    obj["reviewed_by"] = reviewer
    obj["reviewed_on"] = "2026-02-02"
    ref_path.write_text(json.dumps(obj), encoding="utf-8")


def test_freeze_then_load_roundtrip(tmp_path: Path, fake_cut: list[Any]) -> None:
    exc, bench = _freeze_mix(tmp_path, fake_cut)
    excerpts, refs = load_bundle(bench)

    assert [e.id for e in excerpts] == ["ex1"]
    got = excerpts[0]
    assert got.kind == "mix"
    assert got.audio == {"mix": "audio/ex1/mix.flac"}
    assert got.window == (0.5, 1.5)
    assert got.owner is None
    assert got.glossary_sha256 is None
    assert got.source == str(tmp_path / "orig.json")
    # The recorded digest must match the bytes actually on disk.
    on_disk = hashlib.sha256((bench / "audio/ex1/mix.flac").read_bytes()).hexdigest()
    assert got.sha256 == {"audio/ex1/mix.flac": on_disk}

    ref = refs["ex1"]
    assert ref.id == "ex1"
    assert ref.reviewed_by is None
    assert ref.reviewed_on is None
    assert ref.title_expect["must_contain"] == []
    assert ref.title_expect["forbid"] == []
    assert [s.text for s in ref.segments] == ["moi", "terve"]


def test_load_detects_sha_mismatch(tmp_path: Path, fake_cut: list[Any]) -> None:
    _, bench = _freeze_mix(tmp_path, fake_cut)
    (bench / "audio" / "ex1" / "mix.flac").write_bytes(b"tampered")
    with pytest.raises(BundleError, match="sha256 mismatch"):
        load_bundle(bench)


def test_load_detects_missing_audio_file(tmp_path: Path, fake_cut: list[Any]) -> None:
    _, bench = _freeze_mix(tmp_path, fake_cut)
    (bench / "audio" / "ex1" / "mix.flac").unlink()
    with pytest.raises(BundleError, match="missing"):
        load_bundle(bench)


def test_excerpt_without_reference_is_error(
    tmp_path: Path, fake_cut: list[Any]
) -> None:
    _, bench = _freeze_mix(tmp_path, fake_cut)
    (bench / "reference" / "ex1.json").unlink()
    with pytest.raises(ValueError, match="reference"):
        load_bundle(bench)


def test_manifest_bad_kind_is_value_error(tmp_path: Path, fake_cut: list[Any]) -> None:
    _, bench = _freeze_mix(tmp_path, fake_cut)
    man_path = bench / "manifest.json"
    man = json.loads(man_path.read_text(encoding="utf-8"))
    man["excerpts"][0]["kind"] = "bogus"
    man_path.write_text(json.dumps(man), encoding="utf-8")
    with pytest.raises(ValueError, match="kind"):
        load_bundle(bench)


def test_unreviewed_lists_drafts_until_reviewed(
    tmp_path: Path, fake_cut: list[Any]
) -> None:
    _, bench = _freeze_mix(tmp_path, fake_cut)
    _, refs = load_bundle(bench)
    assert unreviewed(refs) == ["ex1"]

    _mark_reviewed(bench, "ex1")
    _, refs = load_bundle(bench)
    assert unreviewed(refs) == []


def test_window_clipping_and_time_shift(tmp_path: Path, fake_cut: list[Any]) -> None:
    _, bench = _freeze_mix(tmp_path, fake_cut)
    _, refs = load_bundle(bench)
    segs = refs["ex1"].segments
    # Segment 1 straddles the window start; segment 2 the end; segment 3 is out.
    assert len(segs) == 2
    assert segs[0].start == pytest.approx(0.0)
    assert segs[0].end == pytest.approx(0.3)  # 0.8 clipped to window, shifted
    assert segs[1].start == pytest.approx(0.5)  # 1.0 - 0.5 window start
    assert segs[1].end == pytest.approx(1.0)  # 1.6 clipped to 1.5, shifted


def test_speaker_name_mapping_and_unclear(tmp_path: Path, fake_cut: list[Any]) -> None:
    sidecar = _sidecar(
        tmp_path / "orig.json",
        [
            {"start": 0.5, "end": 0.7, "speaker": "SPEAKER_01", "text": "a"},
            {"start": 0.7, "end": 0.9, "speaker": "SPEAKER_02", "text": "b"},
            {"start": 0.9, "end": 1.1, "speaker": "SPEAKER_03", "text": "c"},
        ],
        # SPEAKER_02 is "named" with an empty value; SPEAKER_03 is absent.
        {"SPEAKER_01": "Anna", "SPEAKER_02": ""},
    )
    _, bench = _freeze_mix(tmp_path, fake_cut, sidecar=sidecar)
    _, refs = load_bundle(bench)
    assert [s.speaker for s in refs["ex1"].segments] == [
        "Anna",
        "unclear",
        "unclear",
    ]


def test_owner_applied_on_me_track(tmp_path: Path, fake_cut: list[Any]) -> None:
    sidecar = _sidecar(
        tmp_path / "orig.json",
        [
            {
                "start": 0.5,
                "end": 0.9,
                "speaker": "SPEAKER_01",
                "text": "i",
                "track": "me",
            },
            {
                "start": 0.9,
                "end": 1.2,
                "speaker": "SPEAKER_01",
                "text": "you",
                "track": "them",
            },
        ],
        {"SPEAKER_01": "Anna"},
    )
    exc, bench = _freeze_mix(tmp_path, fake_cut, sidecar=sidecar, owner="Janni")
    assert exc.owner == "Janni"
    _, refs = load_bundle(bench)
    # The owner's own track wins over the diarization label's name.
    assert [s.speaker for s in refs["ex1"].segments] == ["Janni", "Anna"]


def test_refuses_to_overwrite_reviewed_reference(
    tmp_path: Path, fake_cut: list[Any]
) -> None:
    _, bench = _freeze_mix(tmp_path, fake_cut)
    _mark_reviewed(bench, "ex1")
    with pytest.raises(ValueError, match="reviewed"):
        _freeze_mix(tmp_path, fake_cut)


def test_two_track_layout(tmp_path: Path, fake_cut: list[Any]) -> None:
    me = _make_wav(tmp_path / "me.wav")
    them = _make_wav(tmp_path / "them.wav")
    sidecar = _sidecar(
        tmp_path / "call.json",
        [
            {
                "start": 0.2,
                "end": 0.9,
                "speaker": "SPEAKER_01",
                "text": "moi",
                "track": "me",
            }
        ],
    )
    bench = tmp_path / "bench"
    exc = freeze_excerpt(
        sidecar_path=sidecar,
        audio={"me": me, "them": them},
        window=(0.0, 1.5),
        bench_dir=bench,
        excerpt_id="call1",
    )
    assert exc.kind == "two-track"
    assert exc.audio == {"me": "audio/call1/me.flac", "them": "audio/call1/them.flac"}
    assert (bench / "audio" / "call1" / "me.flac").is_file()
    assert (bench / "audio" / "call1" / "them.flac").is_file()

    excerpts, refs = load_bundle(bench)
    assert excerpts[0].audio == exc.audio
    assert refs["call1"].segments[0].text == "moi"


def test_refreeze_replaces_manifest_entry(tmp_path: Path, fake_cut: list[Any]) -> None:
    _, bench = _freeze_mix(tmp_path, fake_cut)
    exc2, _ = _freeze_mix(tmp_path, fake_cut, window=(0.0, 1.0))
    excerpts, _ = load_bundle(bench)
    assert [e.id for e in excerpts] == ["ex1"]
    assert excerpts[0].window == exc2.window == (0.0, 1.0)


def test_glossary_sha256_recorded(tmp_path: Path, fake_cut: list[Any]) -> None:
    gloss = tmp_path / "glossary.txt"
    gloss.write_text("kuiskaus vemoizer\n", encoding="utf-8")
    exc, _ = _freeze_mix(tmp_path, fake_cut, glossary_path=gloss)
    assert exc.glossary_sha256 == hashlib.sha256(gloss.read_bytes()).hexdigest()


# --- review fixes (issue #227 follow-up) ---------------------------------


def test_freeze_rejects_traversal_ids_and_writes_nothing(
    tmp_path: Path, fake_cut: list[Any]
) -> None:
    for bad in ("../../escaped", "../up", "a/b", "", "-lead", ".hidden", "x" * 65):
        with pytest.raises(ValueError, match="excerpt id"):
            _freeze_mix(tmp_path, fake_cut, excerpt_id=bad)
    # Nothing may land outside the bench dir — not even before validation fails.
    assert not (tmp_path / "escaped.flac").exists()
    assert list(tmp_path.glob("*.flac")) == []
    assert not (tmp_path / "bench").exists()


def test_loader_rejects_traversal_id_in_manifest(
    tmp_path: Path, fake_cut: list[Any]
) -> None:
    _, bench = _freeze_mix(tmp_path, fake_cut)
    man_path = bench / "manifest.json"
    man = json.loads(man_path.read_text(encoding="utf-8"))
    man["excerpts"][0]["id"] = "../../escaped"
    man_path.write_text(json.dumps(man), encoding="utf-8")
    with pytest.raises(ValueError, match="excerpt id"):
        load_bundle(bench)


def test_id_pattern_accepts_reasonable_ids(tmp_path: Path, fake_cut: list[Any]) -> None:
    exc, _ = _freeze_mix(tmp_path, fake_cut, excerpt_id="Memo-2026_01_A")
    assert exc.id == "Memo-2026_01_A"


def test_draft_skips_non_str_and_blank_text(
    tmp_path: Path, fake_cut: list[Any]
) -> None:
    sidecar = _sidecar(
        tmp_path / "orig.json",
        [
            {"start": 0.6, "end": 0.8, "speaker": "S1", "text": None},
            {"start": 0.6, "end": 0.8, "speaker": "S1", "text": 123},
            {"start": 0.6, "end": 0.8, "speaker": "S1", "text": "   "},
            {"start": 0.6, "end": 0.8, "speaker": "S1", "text": "keep"},
        ],
    )
    _, bench = _freeze_mix(tmp_path, fake_cut, sidecar=sidecar)
    _, refs = load_bundle(bench)
    assert [s.text for s in refs["ex1"].segments] == ["keep"]


def test_loader_rejects_non_str_segment_text(
    tmp_path: Path, fake_cut: list[Any]
) -> None:
    _, bench = _freeze_mix(tmp_path, fake_cut)
    ref_path = bench / "reference" / "ex1.json"
    ref = json.loads(ref_path.read_text(encoding="utf-8"))
    ref["segments"][0]["text"] = None
    ref_path.write_text(json.dumps(ref), encoding="utf-8")
    with pytest.raises(ValueError, match="text must be a string"):
        load_bundle(bench)


def _frozen_snapshot(bench: Path) -> dict[str, str]:
    """Path -> sha256 for every file in the bench (failure-atomicity checks)."""
    return {
        str(p.relative_to(bench)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(bench.rglob("*"))
        if p.is_file()
    }


def test_refreeze_with_bad_sidecar_is_all_or_nothing(
    tmp_path: Path, fake_cut: list[Any]
) -> None:
    _, bench = _freeze_mix(tmp_path, fake_cut)
    _mark_reviewed(bench, "ex1", reviewer="")  # unreviewed, so re-freeze is legal
    before = _frozen_snapshot(bench)
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"segments": [{"start": 0.0}]}), encoding="utf-8")
    with pytest.raises(ValueError, match="start/end/text"):
        _freeze_mix(tmp_path, fake_cut, sidecar=bad, window=(0.0, 1.0))
    assert _frozen_snapshot(bench) == before


def test_refreeze_with_failing_cut_is_all_or_nothing(
    tmp_path: Path, fake_cut: list[Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    _, bench = _freeze_mix(tmp_path, fake_cut)
    before = _frozen_snapshot(bench)

    def _boom(src: Path, dst: Path, start: float, end: float) -> None:
        Path(dst).unlink(missing_ok=True)  # a half-cut temp must not survive
        raise BundleError("cut exploded")

    monkeypatch.setattr(accept_bundle, "_cut_audio", _boom)
    with pytest.raises(BundleError, match="cut exploded"):
        _freeze_mix(tmp_path, fake_cut, window=(0.0, 1.0))
    assert _frozen_snapshot(bench) == before


@pytest.mark.parametrize(
    "patch, match",
    [
        ({"window": [float("nan"), 1.5]}, "window"),
        ({"window": [0.5, float("inf")]}, "window"),
        ({"window": [True, 1.5]}, "window"),
        ({"window": [-0.5, 1.5]}, "window"),
        ({"window": [1.5, 0.5]}, "window"),
    ],
)
def test_manifest_window_rejects_nonfinite_bool_negative(
    tmp_path: Path, fake_cut: list[Any], patch: dict[str, Any], match: str
) -> None:
    _, bench = _freeze_mix(tmp_path, fake_cut)
    man_path = bench / "manifest.json"
    man = json.loads(man_path.read_text(encoding="utf-8"))
    man["excerpts"][0].update(patch)
    man_path.write_text(json.dumps(man), encoding="utf-8")
    with pytest.raises(ValueError, match=match):
        load_bundle(bench)


@pytest.mark.parametrize("bad", [None, True, float("nan"), float("inf"), -0.5, 0.0])
def test_reference_rejects_bad_segment_times(
    tmp_path: Path, fake_cut: list[Any], bad: Any
) -> None:
    _, bench = _freeze_mix(tmp_path, fake_cut)
    ref_path = bench / "reference" / "ex1.json"
    ref = json.loads(ref_path.read_text(encoding="utf-8"))
    ref["segments"][0]["end"] = bad
    ref_path.write_text(json.dumps(ref), encoding="utf-8")
    with pytest.raises(ValueError, match="segment"):
        load_bundle(bench)


def test_freeze_rejects_bad_window(tmp_path: Path, fake_cut: list[Any]) -> None:
    for bad in ((float("nan"), 1.5), (0.5, float("inf")), (-0.5, 1.5), (1.5, 0.5)):
        with pytest.raises(ValueError, match="window"):
            _freeze_mix(tmp_path, fake_cut, window=bad)


def test_straddling_segments_marked_edge_not_fully_contained(
    tmp_path: Path, fake_cut: list[Any]
) -> None:
    sidecar = _sidecar(
        tmp_path / "orig.json",
        [
            {"start": 0.0, "end": 0.8, "speaker": "S1", "text": "straddle-left"},
            {"start": 0.6, "end": 0.8, "speaker": "S1", "text": "inside"},
            {"start": 1.0, "end": 2.0, "speaker": "S1", "text": "straddle-right"},
        ],
    )
    _, bench = _freeze_mix(tmp_path, fake_cut, sidecar=sidecar)
    raw = json.loads((bench / "reference" / "ex1.json").read_text(encoding="utf-8"))
    assert [d.get("edge", False) for d in raw["segments"]] == [True, False, True]
    # Full text is kept (the owner trims it), and the loader preserves edge.
    _, refs = load_bundle(bench)
    assert [s.text for s in refs["ex1"].segments] == [
        "straddle-left",
        "inside",
        "straddle-right",
    ]
    assert [s.edge for s in refs["ex1"].segments] == [True, False, True]


def test_loader_accepts_segments_without_edge_key(
    tmp_path: Path, fake_cut: list[Any]
) -> None:
    _, bench = _freeze_mix(tmp_path, fake_cut)
    ref_path = bench / "reference" / "ex1.json"
    ref = json.loads(ref_path.read_text(encoding="utf-8"))
    for d in ref["segments"]:
        d.pop("edge", None)
    ref_path.write_text(json.dumps(ref), encoding="utf-8")
    _, refs = load_bundle(bench)
    assert all(s.edge is False for s in refs["ex1"].segments)


def test_missing_ffmpeg_raises_bundle_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _no_ffmpeg(*args: Any, **kwargs: Any) -> Any:
        raise FileNotFoundError("ffmpeg")

    monkeypatch.setattr(accept_bundle.subprocess, "run", _no_ffmpeg)
    with pytest.raises(BundleError, match="ffmpeg"):
        _freeze_mix(tmp_path, [])


def test_refreeze_rejects_malformed_existing_manifest_entry(
    tmp_path: Path, fake_cut: list[Any]
) -> None:
    _, bench = _freeze_mix(tmp_path, fake_cut)
    man_path = bench / "manifest.json"
    man = json.loads(man_path.read_text(encoding="utf-8"))
    man["excerpts"].append({"id": "broken"})  # not a valid excerpt entry
    man_path.write_text(json.dumps(man), encoding="utf-8")
    with pytest.raises(ValueError, match="broken"):
        _freeze_mix(tmp_path, fake_cut, excerpt_id="ex2")


def test_malformed_manifest_aborts_freeze_before_any_write(
    tmp_path: Path, fake_cut: list[Any]
) -> None:
    _, bench = _freeze_mix(tmp_path, fake_cut)
    man_path = bench / "manifest.json"
    man = json.loads(man_path.read_text(encoding="utf-8"))
    man["excerpts"].append({"id": "broken"})
    man_path.write_text(json.dumps(man), encoding="utf-8")
    with pytest.raises(ValueError, match="broken"):
        _freeze_mix(tmp_path, fake_cut, excerpt_id="ex2")
    assert not (bench / "audio" / "ex2").exists()
    assert not (bench / "reference" / "ex2.json").exists()


@pytest.mark.skipif(_FFMPEG is None, reason="ffmpeg not on PATH")
def test_real_ffmpeg_cut_is_16k_mono(tmp_path: Path) -> None:
    """End-to-end: the real ffmpeg produces a 16 kHz mono FLAC of the window."""
    import soundfile as sf

    # A stereo 44.1 kHz source: "-ac 1" must be an output option to downmix.
    t = np.arange(int(3.0 * 44100)) / 44100
    tone = (0.1 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    src = tmp_path / "orig.wav"
    sf.write(str(src), np.stack([tone, tone], axis=1), 44100)
    sidecar = _sidecar(
        tmp_path / "orig.json",
        [{"start": 0.0, "end": 3.0, "speaker": "SPEAKER_01", "text": "x"}],
    )
    bench = tmp_path / "bench"
    freeze_excerpt(
        sidecar_path=sidecar,
        audio={"mix": src},
        window=(0.5, 1.5),
        bench_dir=bench,
        excerpt_id="real",
    )
    data, sr = sf.read(str(bench / "audio" / "real" / "mix.flac"))
    assert sr == SAMPLE_RATE  # audio contract: 16 kHz
    assert data.ndim == 1  # mono
    assert len(data) / sr == pytest.approx(1.0, abs=0.05)
