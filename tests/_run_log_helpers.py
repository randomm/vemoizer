"""Shared helpers for the run_log CLI and path test modules.

Kept in its own module so the helpers are an explicit import and each
test file stays focused on its own concern.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import vemoizer.grouping as grouping
import vemoizer.grouping_concat as grouping_concat
import vemoizer.grouping_probe as grouping_probe
import vemoizer.ingest as ingest_module


def fake_ingest(monkeypatch: pytest.MonkeyPatch) -> None:
    """The zero-byte touch-files are not real audio: skip the real ffmpeg
    probe/duration calls (the transcribe seam is faked anyway)."""
    monkeypatch.setattr(grouping_probe, "_probe_stream", lambda p: "wav,16000,1")
    monkeypatch.setattr(grouping, "_probe_stream", lambda p: "wav,16000,1")
    monkeypatch.setattr(
        ingest_module, "pcm_duration_seconds", lambda p, timeout=300.0: 1.0
    )
    monkeypatch.setattr(grouping_probe, "probe_duration_seconds", lambda p: 1.0)


def fake_continuation_seams(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch the grouping seams so 2 files become ONE multi-part group."""
    from vemoizer.grouping import GroupProposal

    fake_ingest(monkeypatch)
    monkeypatch.setattr(
        grouping,
        "decode_boundaries",
        lambda files, transcribe_fn=None: (
            ["ja tässä ollaan nyt"],
            ["tässä jatketaan"],
        ),
    )
    import vemoizer.batch as batch

    monkeypatch.setattr(batch, "concat_groups", lambda files: files[0])
    monkeypatch.setattr(batch, "part_offsets", lambda files: [])
    monkeypatch.setattr(grouping, "concat_groups", lambda files: files[0])
    monkeypatch.setattr(grouping, "part_offsets", lambda files: [])
    monkeypatch.setattr(grouping_concat, "concat_groups", lambda files: files[0])
    monkeypatch.setattr(grouping_concat, "part_offsets", lambda files: [])
    monkeypatch.setattr(
        grouping,
        "propose_groups",
        lambda files, t, h: [
            GroupProposal(
                parts=(str(files[0].name), str(files[1].name)),
                is_continuation=True,
                evidence=("", ""),
            )
        ],
    )


def fake_breaks(monkeypatch: pytest.MonkeyPatch) -> None:
    """Patch the grouping seams so 2 files stay TWO single-part groups
    (explicit break proposal at the only boundary)."""
    fake_ingest(monkeypatch)
    monkeypatch.setattr(
        grouping,
        "decode_boundaries",
        lambda files, transcribe_fn=None: ("", ""),
    )
    from vemoizer.grouping import GroupProposal

    monkeypatch.setattr(
        grouping,
        "propose_groups",
        lambda files, t, h: [
            GroupProposal(
                parts=(files[0].name, files[1].name),
                is_continuation=False,
                evidence=("", ""),
            )
        ],
    )


def logs_dir(tmp_path: Path) -> Path:
    return tmp_path / ".vemoizer" / "logs"
