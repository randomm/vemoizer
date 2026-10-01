"""Relative-input-path regression tests for ``concat_groups`` (issue #77).

Real-audio smoke test found: ``vemoizer transcribe a.m4a b.m4a --yes`` with
RELATIVE input paths failed with
``error: concat: ffmpeg failed for ... (exit 254): Impossible to open
'<tmpdir>/vemoizer-concat-xxxx/parts/Uusi äänitys 901.m4a'`` — the concat
demuxer list file was written with the parts' paths exactly as given
(relative), and the concat demuxer resolves relative list-file paths against
the LIST FILE's directory (the 0o700 temp dir), not the process CWD.

These tests pin the fix: the list file must contain ABSOLUTE paths
(``os.path.abspath`` — absolute, not necessarily resolved; NFC/NFD names
byte-for-byte as on disk) while error messages keep naming the parts as the
user typed them.

Unit level: ``monkeypatch.chdir`` + RELATIVE part paths + a fake
``subprocess.run`` that captures the list file content inside the fake (the
file is unlinked after the call, so it must be read there). Real-ffmpeg level
(skip when ffmpeg/ffprobe are absent, same ``skipif`` pattern as
``test_grouping_fixtures.py`` / ``test_ingest.py``): copy the two small
checked-in fixtures into ``tmp_path``, chdir, concat the RELATIVE names
(including one with a space, an apostrophe, and a non-ASCII ``ä``), and
check the merged decoded-PCM duration against the sum of the parts'
``pcm_duration_seconds`` (tolerance 0.1 s).
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import pytest
from test_grouping_bugs import _make_wav

from vemoizer.grouping import GroupingError, concat_groups, remove_concat_output
from vemoizer.ingest import pcm_duration_seconds

GROUPING_DIR = Path(__file__).resolve().parent / "fixtures" / "grouping"

FFMPEG = shutil.which("ffmpeg") is not None
FFPROBE = shutil.which("ffprobe") is not None
requires_ffmpeg = pytest.mark.skipif(
    not (FFMPEG and FFPROBE), reason="ffmpeg/ffprobe not on PATH"
)

# Duplicated rather than imported: the pair's decoded-PCM durations are
# pinned in test_grouping_fixtures.py (DURATION_A / DURATION_B) and importing
# a private-ish test constant across files grows the import surface for two
# floats.
DURATION_A = 2.5
DURATION_B = 3.0


@dataclass
class _ListCapture:
    """Record of the concat list file content(s) and ffmpeg argv(s)."""

    pinned: Path
    list_contents: list[str]
    argvs: list[list[str]]


def _pin_and_capture(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> _ListCapture:
    """Pin the concat temp dir into *tmp_path*; fake ffmpeg and ffprobe.

    The fake ``subprocess.run`` reads the concat list file from disk INSIDE
    the call (the real code unlinks it right after ``subprocess.run``
    returns, so the content must be captured there) and returns success.
    """
    pinned = tmp_path / "vemoizer-concat-pin"
    pinned.mkdir()
    cap = _ListCapture(pinned, [], [])

    def fake_mkdtemp(**kwargs: str) -> str:
        return str(pinned)

    def fake_run(cmd, *args, **kwargs):
        cap.argvs.append(list(cmd))
        idx = cmd.index("-i")
        cap.list_contents.append(Path(cmd[idx + 1]).read_text(encoding="utf-8"))
        return subprocess.CompletedProcess(cmd, 0, b"", b"")

    import vemoizer.grouping as grouping

    monkeypatch.setattr(tempfile, "mkdtemp", fake_mkdtemp)
    monkeypatch.setattr(grouping, "_probe_stream", lambda p: "aac,48000,1")
    monkeypatch.setattr(subprocess, "run", fake_run)
    return cap


def _list_paths(content: str) -> list[str]:
    """The un-escaped paths of the ``file '...'`` lines in *content*."""
    return [
        line.removeprefix("file '").removesuffix("'")
        for line in content.splitlines()
        if line.strip().startswith("file '")
    ]


# ---------------------------------------------------------------------------
# Unit: the concat list file must contain ABSOLUTE paths that exist,
# even when the parts were passed RELATIVE to the process CWD.
# ---------------------------------------------------------------------------


def test_concat_list_file_contains_absolute_paths_for_relative_inputs(
    tmp_path, monkeypatch
) -> None:
    """Relative part paths (the CLI case) must be written into the list
    file as ABSOLUTE paths — the concat demuxer resolves list entries
    against the list file's temp dir, not the CWD — and every entry must
    exist. Also pins: the list file itself lives in the pinned temp dir,
    and the merged output path handed back is the temp dir's group file."""
    workdir = tmp_path / "workdir"
    workdir.mkdir()
    a = _make_wav(workdir / "Uusi äänitys 901.m4a", 0.25)
    b = _make_wav(workdir / "Uusi äänitys 902.m4a", 0.25)

    cap = _pin_and_capture(monkeypatch, tmp_path)
    monkeypatch.chdir(workdir)

    # Relative names exactly as the CLI would pass them.
    out = concat_groups([Path("Uusi äänitys 901.m4a"), Path("Uusi äänitys 902.m4a")])

    assert out == cap.pinned / "group.m4a"
    assert len(cap.list_contents) == 1
    paths = _list_paths(cap.list_contents[0])
    assert len(paths) == 2
    for inner in paths:
        assert os.path.isabs(inner), f"list entry is not absolute: {inner!r}"
        assert os.path.isfile(inner), f"list entry does not exist: {inner!r}"
    # Both parts, in order, resolved against the CWD the CLI ran in.
    assert paths[0] == str(a)
    assert paths[1] == str(b)
    # The -i arg is the temp list file itself (absolute, temp-dir based).
    i_arg = cap.argvs[0][cap.argvs[0].index("-i") + 1]
    assert os.path.isabs(i_arg)
    assert i_arg.startswith(str(cap.pinned))


def test_concat_list_absolute_paths_do_not_resolve_symlinks(
    tmp_path, monkeypatch
) -> None:
    """The list entries must be ``os.path.abspath`` — not ``Path.resolve``:
    the requirement is absolute, byte-for-byte names as on disk (NFC/NFD
    untouched), not symlink-resolved paths. A part reached through a
    symlinked directory must appear in the list via the symlink's path."""
    real_dir = tmp_path / "realdir"
    real_dir.mkdir()
    _make_wav(real_dir / "osa_1.m4a", 0.25)
    _make_wav(real_dir / "osa_2.m4a", 0.25)
    link_dir = tmp_path / "linkdir"
    link_dir.symlink_to(real_dir)

    cap = _pin_and_capture(monkeypatch, tmp_path)
    monkeypatch.chdir(tmp_path)
    concat_groups([link_dir / "osa_1.m4a", link_dir / "osa_2.m4a"])

    paths = _list_paths(cap.list_contents[0])
    assert len(paths) == 2
    for p in paths:
        assert os.path.isabs(p)
        assert str(link_dir) in p
        assert str(real_dir) not in p


def test_concat_list_escapes_apostrophe_space_and_unicode(
    tmp_path, monkeypatch
) -> None:
    """A name with a space, an apostrophe, and a non-ASCII character
    (``Uusi äänitys 9'1.m4a``) must be written through the existing
    ``_escape_concat_path`` quoting — apostrophe escaped ``'\\''``, space
    and ``ä`` byte-for-byte — as an absolute path."""
    workdir = tmp_path / "workdir"
    workdir.mkdir()
    weird = workdir / "Uusi äänitys 9'1.m4a"
    _make_wav(weird, 0.25)
    _make_wav(workdir / "osa_2.m4a", 0.25)

    cap = _pin_and_capture(monkeypatch, tmp_path)
    monkeypatch.chdir(workdir)
    concat_groups([Path("Uusi äänitys 9'1.m4a"), Path("osa_2.m4a")])

    content = cap.list_contents[0]
    lines = [ln for ln in content.splitlines() if ln.strip().startswith("file '")]
    assert len(lines) == 2
    # ``_escape_concat_path``: the wrapper quotes stay; apostrophes INSIDE
    # the path become ``'\\''``. The name contains exactly one apostrophe.
    inner_abs = os.path.abspath(weird)
    assert lines[0] == "file '" + inner_abs.replace("'", "'\\''") + "'"
    # Only the escaped form of the name appears in the file.
    assert "Uusi äänitys 9'1.m4a" not in content.replace("'\\''", "")


def test_concat_error_message_names_parts_as_typed(tmp_path, monkeypatch) -> None:
    """Even though the list file now carries absolute paths, the failure
    message must still name the parts as the user typed them — file names
    only, as today."""
    workdir = tmp_path / "workdir"
    workdir.mkdir()
    _make_wav(workdir / "Uusi äänitys 901.m4a", 0.25)
    _make_wav(workdir / "Uusi äänitys 902.m4a", 0.25)

    pinned = tmp_path / "vemoizer-concat-pin"
    pinned.mkdir()
    import vemoizer.grouping as grouping

    monkeypatch.setattr(tempfile, "mkdtemp", lambda **k: str(pinned))
    monkeypatch.setattr(grouping, "_probe_stream", lambda p: "aac,48000,1")

    def fake_run(cmd, *args, **kwargs):
        return subprocess.CompletedProcess(cmd, 254, b"", b"Impossible to open")

    monkeypatch.setattr(subprocess, "run", fake_run)
    monkeypatch.chdir(workdir)

    with pytest.raises(GroupingError) as exc:
        concat_groups([Path("Uusi äänitys 901.m4a"), Path("Uusi äänitys 902.m4a")])
    msg = str(exc.value)
    assert "Uusi äänitys 901.m4a" in msg
    assert "Uusi äänitys 902.m4a" in msg
    # Only file names — no directory prefix of any kind in the message.
    assert str(workdir) not in msg


def test_concat_groups_missing_part_error_unchanged(tmp_path, monkeypatch) -> None:
    """A missing part (in a multi-part group) still raises naming the file
    as given — pre-concat, no ffmpeg involved. (A single part is a
    passthrough: no existence check, the input itself is returned.)"""
    workdir = tmp_path / "workdir"
    workdir.mkdir()
    _make_wav(workdir / "existing.m4a", 0.25)
    monkeypatch.chdir(workdir)
    with pytest.raises(GroupingError, match="part file not found"):
        concat_groups([Path("missing_901.m4a"), Path("existing.m4a")])


# ---------------------------------------------------------------------------
# Real ffmpeg: relative input paths end-to-end, including a space +
# apostrophe + non-ASCII name in the list.
# ---------------------------------------------------------------------------


def _copy_pair(workdir: Path) -> tuple[Path, Path]:
    a = workdir / "pair_a.m4a"
    b = workdir / "pair_b.m4a"
    shutil.copy(GROUPING_DIR / "pair_a.m4a", a)
    shutil.copy(GROUPING_DIR / "pair_b.m4a", b)
    return a, b


@requires_ffmpeg
def test_concat_groups_relative_paths_real_ffmpeg(tmp_path) -> None:
    """Copy the two small fixtures into tmp_path, chdir, concat the
    RELATIVE names: the merged file must exist and its decoded-PCM
    duration must be the sum of the parts' decoded-PCM durations
    (tolerance 0.1 s)."""
    workdir = tmp_path / "rel"
    workdir.mkdir()
    a, b = _copy_pair(workdir)
    os.chdir(workdir)

    merged = concat_groups([Path("pair_a.m4a"), Path("pair_b.m4a")])
    try:
        assert merged.is_file()
        merged_dur = pcm_duration_seconds(merged)
        parts_dur = pcm_duration_seconds(a) + pcm_duration_seconds(b)
        assert abs(merged_dur - parts_dur) < 0.1
        assert abs(parts_dur - (DURATION_A + DURATION_B)) < 0.1
    finally:
        remove_concat_output(merged)
        assert not merged.exists()


@requires_ffmpeg
def test_concat_groups_relative_paths_special_name_real_ffmpeg(tmp_path) -> None:
    """Same real path, but the second file is named with a space, an
    apostrophe, and a non-ASCII character: ``Uusi äänitys 9'1.m4a`` — the
    concat list must survive the ``_escape_concat_path`` quoting and ffmpeg
    must merge both."""
    workdir = tmp_path / "rel"
    workdir.mkdir()
    a, _ = _copy_pair(workdir)
    weird = workdir / "Uusi äänitys 9'1.m4a"
    shutil.copy(GROUPING_DIR / "pair_b.m4a", weird)
    os.chdir(workdir)

    merged = concat_groups([Path("pair_a.m4a"), Path("Uusi äänitys 9'1.m4a")])
    try:
        assert merged.is_file()
        merged_dur = pcm_duration_seconds(merged)
        parts_dur = pcm_duration_seconds(a) + pcm_duration_seconds(weird)
        assert abs(merged_dur - parts_dur) < 0.1
    finally:
        remove_concat_output(merged)
        assert not merged.exists()
