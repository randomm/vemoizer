"""Issue #148 promise, state-based: per-stage markers, single final
"complete" line, display lifecycle around the wrote lines / naming prompt.

A real :class:`~vemoizer.progress.ProgressDisplay` is constructed with
``isatty`` patched BEFORE construction (the fake-stderr pattern from
``test_meeting_display_close`` — no real pty). The pipeline seam is stubbed
to drive the display exactly like production: it ``add_stage``s the four
stages (``decode``, ``diarize``, ``repair``, ``notes``) and ``finish``es
each. The assertions are on rendered task descriptions (state, not
terminal bytes) plus the echoed final line:

* each rendered stage description ends in ``✓ <stage>`` — none contains
  the word ``complete``;
* the word ``complete`` appears exactly once overall and only on the
  final line, printed AFTER the wrote lines;
* the display is closed (``is_live`` False) before the wrote lines and
  before the naming prompt;
* ``--quiet`` prints none of it.

Break-and-fail: removing the ``_close_run_display`` call (or moving the
final line above the wrote lines in ``batch_preset.run_preset``) fails the
matching assertion.
"""

from __future__ import annotations

import io
from collections.abc import Callable
from pathlib import Path
from typing import Any
from unittest import mock

import pytest
from _cli_helpers import isolate_home, touch_files

from vemoizer.batch_preset import run_preset
from vemoizer.progress import ProgressDisplay

_EXPECTED_STAGES = ["decode", "diarize", "repair", "notes"]


def _two_label_paragraphs() -> list[dict[str, Any]]:
    return [
        {"start": 0.0, "end": 10.0, "text": "Moikka.", "speaker": "SPEAKER_1"},
        {"start": 12.0, "end": 20.0, "text": "Kyll\u00e4.", "speaker": "SPEAKER_2"},
    ]


@pytest.fixture
def tty_display(monkeypatch: pytest.MonkeyPatch):
    """A real, live ProgressDisplay with isatty patched BEFORE construction."""
    import sys

    buf = io.StringIO()
    monkeypatch.setattr(sys, "stderr", buf)
    monkeypatch.setattr(buf, "isatty", lambda: True, raising=False)
    display = ProgressDisplay()
    assert display.disable is False, "display must be live on a TTY stderr"
    yield display
    display.close()


def _echo_recorder(record: dict[str, Any]):
    def fake_echo(msg, err: bool = False, **kw: Any) -> None:
        if not err:
            record.setdefault("echoes", []).append(str(msg))

    return fake_echo


def _final_line_recorder(record: dict[str, Any]):
    def fake_final_line(n_files: int, *, quiet: bool) -> None:
        record["final_line_calls"] = record.get("final_line_calls", 0) + 1
        if quiet:
            return
        record.setdefault("echoes", []).append(f"✓ complete — wrote {n_files} file(s)")

    return fake_final_line


def _wrote_lines_recorder(record: dict[str, Any]):
    def fake_wrote_lines(written: list[str], quiet: bool) -> None:
        record["wrote_lines_calls"] = record.get("wrote_lines_calls", 0) + 1
        if not quiet:
            for name in written:
                record.setdefault("echoes", []).append(f"wrote {name}")

    return fake_wrote_lines


def _notify_recorder(record: dict[str, Any]):
    """Record the per-file notifications instead of swallowing them.

    The loop reaches both posters through ``vemoizer.notify``, so that is
    where they are patched (issue #151 review — a name is patched where it is
    looked up). Without a recorder here the whole ``notify_write`` /
    ``notify_result`` call in the moved loop could be deleted — or the deferred
    import reverted to a module-level binding, which no ``mock.patch.object``
    on ``vemoizer.notify`` could then reach — and this file would still pass.

    ``notify_result`` is the only consumer of ``check_failure_reason()`` whose
    value flows into a desktop notification, so its recorded ``reason`` is the
    privacy-relevant part of the contract (issue #151 review).
    """

    def fake_notify_write(
        label: Path | str, written: int, expected: int, reason: str = ""
    ) -> None:
        record.setdefault("notify_write", []).append(
            {
                "label": str(label),
                "written": written,
                "expected": expected,
                "reason": reason,
            }
        )

    def fake_notify_result(label: Path | str, outcome: str, reason: str = "") -> None:
        record.setdefault("notify_result", []).append((str(label), outcome, reason))

    return fake_notify_write, fake_notify_result


def _gate_recorder(
    record: dict[str, Any], real_gate: Callable[..., bool]
) -> Callable[..., bool]:
    def fake_gate(written: list[str], expected_pairs: int, exit_code: int) -> bool:
        record["gate"] = {
            "written": written,
            "expected_pairs": expected_pairs,
            "exit_code": exit_code,
        }
        return real_gate(written, expected_pairs, exit_code)

    return fake_gate


def _assert_epilogue_fired(record: dict[str, Any]) -> None:
    """The run's epilogue contract, shared by both tests here.

    ``--quiet`` silences the terminal, not the run: both helpers are still
    called (each suppresses its own output) and the notification still posts.
    The counters are what prove the three ``preset_plain_loop`` patches are
    live rather than vacuous — the echoed strings alone do not, since the real
    helpers reach ``typer.echo`` through ``preset_final`` and the spy records
    the identical text either way (issue #151 review).
    """
    assert record.get("wrote_lines_calls") == 1, "wrote-lines patch never fired"
    assert record.get("final_line_calls") == 1, "final-line patch never fired"
    # One success notification for the one file: full pair, no reason. The
    # label is the file the loop iterated (an absolute path under tmp_path).
    posted = record.get("notify_write")
    assert posted is not None and len(posted) == 1, posted
    assert Path(posted[0]["label"]).name == "2025-01-01 M.m4a", posted
    assert posted[0]["written"] == 2
    assert posted[0]["expected"] == 2
    assert posted[0]["reason"] == ""
    # A fully written pair posts no failure notification.
    assert record.get("notify_result", []) == []


def _run_meeting(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    display: ProgressDisplay,
    *,
    quiet: bool,
    yes: bool,
    record: dict[str, Any],
    result_overrides: dict[str, Any] | None = None,
) -> int:
    """Run one single-file meeting with the pipeline seam stubbed to drive
    the display through the four stage names, and record the display state
    at the first interactive prompt."""
    import vemoizer.naming_hook as naming_hook_module
    import vemoizer.notify as notify_module
    import vemoizer.pipeline as pipeline_module
    import vemoizer.preset_plain_loop as plain_loop_module

    fake_notify_write, fake_notify_result = _notify_recorder(record)

    def fake_transcribe(path, **kwargs):
        for name in _EXPECTED_STAGES:
            task_id = display.add_stage(name)
            display.finish(task_id)
        return {
            "text": "Moikka. Kyll.",
            "segments": [],
            "notes": {"title": Path(path).stem},
            "paragraphs": _two_label_paragraphs(),
            **(result_overrides or {}),
        }

    def input_fn(prompt: str) -> str:
        record["prompt"] = prompt
        record["display_live_at_prompt"] = display.is_live
        return "n"

    [f] = touch_files(["2025-01-01 M.m4a"], tmp_path)
    with (
        mock.patch.object(pipeline_module, "transcribe_file", fake_transcribe),
        mock.patch.object(naming_hook_module, "_stdout_isatty", lambda: True),
        mock.patch.object(notify_module, "notify_write", fake_notify_write),
        mock.patch.object(notify_module, "notify_result", fake_notify_result),
        mock.patch.object(naming_hook_module.typer, "echo", _echo_recorder(record)),
        mock.patch("vemoizer.batch_preset.typer.echo", _echo_recorder(record)),
        # All three are looked up in ``preset_plain_loop`` (the issue #151 move
        # of the plain per-file loop), so they are patched THERE —
        # ``batch_preset`` deliberately does not re-export them, and patching
        # them there would leave the records empty while the real helpers ran.
        # Each patch is proven live below: the gate record in (2b), the two
        # recorder call counts in (2c). The echoed strings alone do not prove
        # it — the real helpers reach ``typer.echo`` through ``preset_final``
        # and the ``batch_preset.typer.echo`` spy records the identical text
        # either way (issue #151 review).
        mock.patch.object(
            plain_loop_module, "print_wrote_lines", _wrote_lines_recorder(record)
        ),
        mock.patch.object(
            plain_loop_module,
            "run_went_full",
            _gate_recorder(record, plain_loop_module.run_went_full),
        ),
        mock.patch.object(
            plain_loop_module, "print_final_line", _final_line_recorder(record)
        ),
    ):
        return run_preset(
            [f],
            command="meeting",
            config_path=None,
            glossary_path=None,
            quiet=quiet,
            yes=yes,
            input_fn=input_fn,
            tty_isatty=lambda: True,
            display=display,
        )


def test_meeting_stage_markers_and_single_complete_line(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tty_display: ProgressDisplay,
) -> None:
    """The rendered stage descriptions are exactly decode/diarize/repair/
    notes each with a per-stage ✓ marker; "complete" appears exactly once
    and only on the final line, after the wrote lines; the display is
    closed before both the wrote lines and the naming prompt."""
    isolate_home(monkeypatch, tmp_path, tmp_path)
    record: dict[str, Any] = {}

    code = _run_meeting(
        tmp_path, monkeypatch, tty_display, quiet=False, yes=False, record=record
    )
    assert code == 0

    # (1) Rendered task descriptions: exactly the four stage names, each
    # finished with its own ✓ marker, none containing "complete".
    descriptions = [t.description for t in tty_display._progress.tasks]
    assert len(descriptions) == len(_EXPECTED_STAGES)
    for name, desc in zip(_EXPECTED_STAGES, descriptions, strict=True):
        assert desc == f"[green]\u2713 {name}", f"got {desc!r} for stage {name!r}"
        assert "complete" not in desc

    # (2) "complete" exactly once, only on the final line, after the wrote.
    echoes = record.get("echoes", [])
    assert sum(s.count("complete") for s in echoes) == 1
    final_lines = [s for s in echoes if "complete" in s]
    assert len(final_lines) == 1
    wrote_idx = echoes.index(next(s for s in echoes if s.startswith("wrote ")))
    final_idx = echoes.index(final_lines[0])
    assert final_idx > wrote_idx
    # The final line is PLAIN text (issue #148 FIX 2): rich markup like
    # ``[green]`` would print verbatim through ``typer.echo``. The count
    # and position are what matter; no square brackets may leak.
    assert final_lines[0] == "\u2713 complete — wrote 2 file(s)"
    assert "[" not in final_lines[0]
    # (2b) The gate spy actually fired: this is what makes the three
    # ``preset_plain_loop`` patches above live rather than vacuous (issue #151
    # review — a patch target where the name is NOT looked up leaves the record
    # empty while the real helper runs). The plain loop gates the final line on
    # ONE independent expected pair per file, so the recorder must have seen
    # exactly the full pair, the file count, and a clean exit.
    assert record.get("gate") == {
        "written": [
            name.removeprefix("wrote ") for name in echoes if name.startswith("wrote ")
        ],
        "expected_pairs": 1,
        "exit_code": 0,
    }, f"plain-loop gate patch never fired or saw wrong args: {record.get('gate')!r}"
    assert all(s.startswith("wrote ") for s in echoes[: wrote_idx + 1])
    # (2c) The epilogue contract: both helper patches fired, and the loop
    # posted the success notification for the full pair.
    _assert_epilogue_fired(record)

    # (3) Display closed before the wrote lines and before the prompt.
    # The prompt spy proves the close predates the last interactive output;
    # the wrote lines fire before the hook in run_preset, so
    # closed-at-prompt implies closed-at-wrote.
    assert record.get("prompt"), "naming-hook input_fn was never called"
    assert record["display_live_at_prompt"] is False
    assert tty_display.is_live is False


def test_meeting_quiet_prints_none_of_it(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tty_display: ProgressDisplay,
) -> None:
    """--quiet suppresses the wrote lines, the final line, and any
    skip-reason narration."""
    isolate_home(monkeypatch, tmp_path, tmp_path)
    record: dict[str, Any] = {}

    code = _run_meeting(
        tmp_path, monkeypatch, tty_display, quiet=True, yes=True, record=record
    )
    assert code == 0
    # The claim specific to --quiet: nothing reaches the terminal at all.
    assert sum(s.count("complete") for s in record.get("echoes", [])) == 0
    assert record.get("echoes", []) == []
    assert tty_display.is_live is False
    # Quiet is not silent-to-the-operator: the epilogue still runs (issue #151
    # review — the same invariant the non-quiet test pins, not a copy of it).
    _assert_epilogue_fired(record)


def test_meeting_check_failure_notifies_with_reason(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tty_display: ProgressDisplay,
) -> None:
    """The failure-path ``notify_result`` in the moved loop is live, and its
    reason is exactly ``check_failure_reason()`` (issue #151 review).

    This is the only consumer of that reason which flows into a desktop
    notification, so it is the path a widened notification payload would leak
    on. Previously stubbed with a bare no-op lambda, so deleting the call — or
    reverting the deferred ``from vemoizer.notify import notify_result`` to a
    module-level binding, which ``mock.patch.object`` on ``vemoizer.notify``
    could no longer reach — left this file green. The epilogue assertion also
    pins that a check-failed file posts no *success* notification.
    """
    isolate_home(monkeypatch, tmp_path, tmp_path)
    record: dict[str, Any] = {}

    code = _run_meeting(
        tmp_path,
        monkeypatch,
        tty_display,
        quiet=True,
        yes=True,
        record=record,
        result_overrides={"error": "api_key_env is required"},
    )
    assert code == 1
    posted = record.get("notify_result")
    assert posted is not None and len(posted) == 1, (
        f"plain-loop notify_result patch never fired: {posted!r}"
    )
    label, outcome, reason = posted[0]
    assert Path(label).name == "2025-01-01 M.m4a", posted
    assert outcome == "failed"
    assert reason == "api_key_env is required", (
        "the notification reason drifted from check_failure_reason()"
    )
    assert record.get("notify_write", []) == []
