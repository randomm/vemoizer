"""Regression tests for issue #143: the rich ProgressDisplay must be
closed before the end-of-run "wrote <file>" lines and before the naming-hook
prompts in both ``batch_preset`` paths (the meeting 2+ files group path and
the single-file / ``--no-group`` plain path).

The bug is TTY-only: with non-TTY stderr (piped, CI, ``--quiet``) the
display never renders and the bug is silent. These tests install a real PTY
via ``os.openpty()`` as ``sys.stderr`` and construct a REAL
:class:`~vemoizer.progress.ProgressDisplay` afterwards, so the display is not
disabled (``disable=False``) when the transcribe seam
starts it — a plain file object would work the same way (``start()`` flips
the state flag regardless of whether rich actually renders), the PTY just
makes the setup realistic. The transcribe seam is stubbed to call
``display.start()`` exactly like the production pipeline does, and the
injected ``input_fn`` spy records the display's ``is_live`` state each time
it is called — i.e. at every interactive prompt (the hook's y/N and
run_names' downstream prompts).

The assertion is on the display's lifecycle state (``is_live``), not on
rendered bytes, so the test stays deterministic. Break-and-fail: removing
the display close in ``batch_preset.py`` leaves the display live when
``input_fn`` fires, so the tests fail.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest
from _cli_helpers import isolate_home, touch_files

from vemoizer.batch_preset import run_preset
from vemoizer.progress import ProgressDisplay


def _two_label_paragraphs() -> list[dict[str, Any]]:
    return [
        {"start": 0.0, "end": 10.0, "text": "Moikka.", "speaker": "SPEAKER_1"},
        {"start": 12.0, "end": 20.0, "text": "Kyll\u00e4.", "speaker": "SPEAKER_2"},
    ]


@pytest.fixture
def pty_display(monkeypatch: pytest.MonkeyPatch):
    """A real :class:`ProgressDisplay` bound to a fake-PTY stderr.

    The PTY gives the display a non-disabled stderr so ``start()`` flips
    its ``is_live`` state; the master fd stays open for the test duration
    (writing to an orphaned pty slave raises EIO) and both fds are closed
    in teardown even if construction fails.
    """
    import sys

    master, slave = os.openpty()
    buf = os.fdopen(slave, "w")
    try:
        monkeypatch.setattr(sys, "stderr", buf)
        monkeypatch.setattr(sys.stderr, "isatty", lambda: True, raising=False)
        display = ProgressDisplay()
        assert display.disable is False, "display must be live on a fake-PTY stderr"
    except Exception:
        buf.close()
        os.close(master)
        raise
    try:
        yield display
    finally:
        display.close()
        buf.close()
        os.close(master)


def _stub_seams(
    display: ProgressDisplay | None,
    *,
    group: bool = False,
):
    """Patch pipeline.transcribe_file (starts the display, returns
    two-label paragraphs so the run's own sidecar is hook-eligible),
    the notify seams, the stdout-TTY gate, and optionally the grouping
    seams (for the 2-file group path). Returns a context manager that
    patches all of them."""
    from unittest import mock

    import vemoizer.naming_hook as naming_hook_module
    import vemoizer.notify as notify_module
    import vemoizer.pipeline as pipeline_module

    patches = [
        mock.patch.object(
            pipeline_module, "transcribe_file", _make_fake_transcribe(display)
        ),
        mock.patch.object(naming_hook_module, "_stdout_isatty", lambda: True),
        mock.patch.object(notify_module, "notify_write", lambda *a, **kw: None),
        mock.patch.object(notify_module, "notify_result", lambda *a, **kw: None),
    ]
    if group:
        import vemoizer.batch as batch
        import vemoizer.grouping as grouping

        patches.extend(
            [
                mock.patch.object(
                    grouping,
                    "decode_boundaries",
                    lambda *a, **kw: ["b", "c"],
                ),
                mock.patch.object(grouping, "concat_groups", lambda files: files[0]),
                mock.patch.object(grouping, "part_offsets", lambda files, **kw: []),
                mock.patch.object(batch, "concat_groups", lambda files: files[0]),
                mock.patch.object(batch, "part_offsets", lambda files, **kw: []),
            ]
        )

    class _Ctx:
        def __enter__(self):
            for p in patches:
                p.start()
            return self

        def __exit__(self, *exc):
            for p in reversed(patches):
                p.stop()

    return _Ctx()


def _make_fake_transcribe(display: ProgressDisplay | None):
    """Fake transcribe that starts the display (like production) and returns
    a result whose paragraphs make the run's own sidecar hook-eligible."""

    def fake_transcribe(path, **kwargs):
        if display is not None:
            display.start()
        return {
            "text": "moikka",
            "segments": [],
            "notes": {"title": Path(path).stem},
            "paragraphs": _two_label_paragraphs(),
        }

    return fake_transcribe


def test_single_file_path_closes_display_before_hook(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, pty_display: ProgressDisplay
) -> None:
    """run_preset plain path (meeting, single file): the display must be
    closed before the wrote lines and the naming-hook prompt."""
    isolate_home(monkeypatch, tmp_path, tmp_path)
    [f] = touch_files(["2025-01-01 M.m4a"], tmp_path)

    record: dict[str, Any] = {}

    def input_fn(prompt: str) -> str:
        record["prompt"] = prompt
        record["display_started"] = pty_display.is_live
        return ""  # "no" — do not enter run_names

    ctx = _stub_seams(pty_display)
    with ctx:
        code = run_preset(
            [f],
            command="meeting",
            config_path=None,
            glossary_path=None,
            quiet=False,
            yes=False,
            no_group=True,
            input_fn=input_fn,
            tty_isatty=lambda: True,
            display=pty_display,
        )
    assert code == 0
    assert record.get("prompt"), "naming-hook input_fn was never called"
    assert record["prompt"].startswith("Name the speakers now?")
    # The display must NOT be live at the moment of the prompt.
    assert record["display_started"] is False
    # And it must remain closed (no later stage re-opened it).
    assert pty_display.is_live is False


def test_group_path_closes_display_before_hook(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, pty_display: ProgressDisplay
) -> None:
    """run_preset group path (meeting, 2+ files): the display must be
    closed before the wrote lines and the naming-hook prompt."""
    isolate_home(monkeypatch, tmp_path, tmp_path)
    files = touch_files(["2025-01-01 A.m4a", "2025-01-01 B.m4a"], tmp_path)

    record: dict[str, Any] = {}

    def input_fn(prompt: str) -> str:
        record["prompt"] = prompt
        record["display_started"] = pty_display.is_live
        return ""

    ctx = _stub_seams(pty_display, group=True)
    with ctx:
        code = run_preset(
            files,
            command="meeting",
            config_path=None,
            glossary_path=None,
            quiet=False,
            yes=False,
            input_fn=input_fn,
            tty_isatty=lambda: True,
            display=pty_display,
        )
    assert code == 0
    assert record.get("prompt"), "naming-hook input_fn was never called"
    assert record["prompt"].startswith("Name the speakers now?")
    # The display must NOT be live at the moment of the prompt.
    assert record["display_started"] is False
    assert pty_display.is_live is False


def test_names_cli_prompts_run_after_display_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, pty_display: ProgressDisplay
) -> None:
    """The downstream run_names prompts ("Name for ...", "Add ... to
    people") run through the SAME input_fn the hook receives — so
    observing the display closed on every input_fn call proves those
    prompts also run on a plain (closed-display) terminal."""
    isolate_home(monkeypatch, tmp_path, tmp_path)
    [f] = touch_files(["2025-01-01 M.m4a"], tmp_path)

    calls: list[tuple[bool, str]] = []

    def input_fn(prompt: str) -> str:
        calls.append((pty_display.is_live, prompt))
        if prompt.startswith("Name the speakers now?"):
            return "y"
        if prompt.startswith("Name for "):
            return "Janni"
        if prompt.startswith("Add "):
            return "n"
        return ""

    ctx = _stub_seams(pty_display)
    with ctx:
        code = run_preset(
            [f],
            command="meeting",
            config_path=None,
            glossary_path=None,
            quiet=False,
            yes=False,
            no_group=True,
            input_fn=input_fn,
            tty_isatty=lambda: True,
            display=pty_display,
        )
    assert code == 0
    prompts = [p for _, p in calls]
    assert any(p.startswith("Name for ") for p in prompts), (
        f"run_names naming prompt never reached: {prompts!r}"
    )
    # Every prompt — the hook's y/N AND run_names' "Name for ..." — ran
    # with the display closed.
    assert all(not started for started, _ in calls), (
        f"some prompts ran with the display live: {calls!r}"
    )
