"""Guard: the autouse ``system_effects`` stub keeps the suite from firing
REAL macOS side effects (``osascript`` notifications, ``caffeinate`` spawns,
``afplay`` playback).

This test runs under the autouse fixture (which is active for every test)
and proves the stub is in the call path end-to-end:

- ``vemoizer.notify.notify("t", "m")`` (darwin-simulated) is recorded by the
  stub — no real ``subprocess.run`` is reached.
- ``caffeinate_context`` (darwin-simulated) records the spawn and uses a
  harmless fake process — no real ``subprocess.Popen`` is reached.
- ``speaker_clips.play`` (darwin-simulated) records the player call — no
  real ``afplay`` is executed.
- A representative end-to-end CLI invocation (``transcribe`` through the
  Typer app with a faked ``transcribe_file``) records a notification with
  the expected message form, proving the stub is in the call path.

The sentinels do NOT globally patch ``subprocess`` for the whole suite (that
would break the unrelated ffmpeg/ingest tests); they are installed only
inside the individual seam tests. The production seams never reach real
``subprocess`` because the autouse fixture stubs the seams themselves.
"""

from __future__ import annotations

from typing import NoReturn
from unittest.mock import patch

from _cli_helpers import fake_transcribe, isolate_home, touch_files
from typer.testing import CliRunner

from vemoizer.caffeinate import caffeinate_context
from vemoizer.cli import app
from vemoizer.notify import notify
from vemoizer.speaker_clips import play

runner = CliRunner()


def _sentinel_run(*args: object, **kwargs: object) -> NoReturn:
    raise AssertionError(
        f"real subprocess.run called: {args!r} {kwargs!r} — the autouse "
        f"system-effects stub did not intercept this call."
    )


def _sentinel_popen(*args: object, **kwargs: object) -> NoReturn:
    raise AssertionError(
        f"real subprocess.Popen called: {args!r} {kwargs!r} — the autouse "
        f"system-effects stub did not intercept this call."
    )


def test_notify_is_recorded_not_executed(system_effects):
    """``notify`` under the autouse stub records argv; no real osascript."""
    with (
        patch("vemoizer.notify.sys.platform", "darwin"),
        patch("subprocess.run", side_effect=_sentinel_run),
    ):
        notify("vemoizer", "a: done")
    # The recorder captured the osascript argv (proving the stub is in the
    # call path), and the sentinel subprocess.run was never reached.
    assert len(system_effects["notify"]) == 1
    argv = system_effects["notify"][0]
    assert argv[0] == "osascript"
    assert argv[1] == "-e"
    assert 'display notification "a: done" with title "vemoizer"' in argv[2]


def test_caffeinate_context_is_recorded_not_spawned(system_effects):
    """``caffeinate_context`` under the autouse stub records argv and uses a
    harmless fake process; no real Popen is spawned."""
    with (
        patch("vemoizer.caffeinate.sys.platform", "darwin"),
        patch("subprocess.Popen", side_effect=_sentinel_popen),
        caffeinate_context(),
    ):
        pass
    assert len(system_effects["caffeinate"]) == 1
    argv = system_effects["caffeinate"][0]
    assert argv[0] == "caffeinate"
    assert "-ims" in argv


def test_play_is_recorded_not_executed(system_effects, tmp_path):
    """``play`` under the autouse stub records argv and returns success; no
    real afplay is executed."""
    f = tmp_path / "c.wav"
    f.write_bytes(b"RIFFxxxx")
    with (
        patch("vemoizer.speaker_clips.sys.platform", "darwin"),
        patch("subprocess.run", side_effect=_sentinel_run),
    ):
        assert play(f) is True
    assert len(system_effects["afplay"]) == 1
    argv = system_effects["afplay"][0]
    assert argv[0] == "afplay"
    assert argv[1] == str(f)


def test_end_to_end_cli_invocation_records_notify(
    tmp_path, monkeypatch, system_effects
):
    """A representative end-to-end ``transcribe`` invocation records a
    notification with the expected message form (proving the stub is
    demonstrably in the call path), enters the caffeinate seam (recorded,
    not spawned), and never plays audio.

    The stubs are what prevent any real ``osascript``/``caffeinate``/
    ``afplay`` execution: if the stubs were absent (or bypassed), the real
    seams would execute those system calls. The recorder therefore doubles
    as the proof that no real system call ran — a recorded argv means the
    stub intercepted it, and an empty ``afplay`` list means no playback.
    """
    touch_files(["meeting.m4a"], tmp_path)
    record: list[str] = []
    fake_transcribe(monkeypatch, record)
    isolate_home(monkeypatch, tmp_path)
    result = runner.invoke(app, ["transcribe", "meeting.m4a"])
    assert result.exit_code == 0
    assert len(record) == 1
    # The notify recorder captured exactly one osascript call with the
    # expected message form ("meeting: done").
    assert len(system_effects["notify"]) == 1
    script = system_effects["notify"][0][2]
    assert 'display notification "meeting: done"' in script
    assert 'with title "vemoizer"' in script
    # The caffeinate seam was entered during the run (recorded, not spawned).
    assert len(system_effects["caffeinate"]) >= 1
    # No afplay in a plain transcribe run.
    assert system_effects["afplay"] == []
