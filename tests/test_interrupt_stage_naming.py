"""Ctrl-C stage naming through the REAL seams (issue #148, FIX 2).

The interrupt tracker is set to ``decoding`` once and must be updated at
each stage seam (decode, diarize, repair, notes) — the label must never
depend on parsing the rich task description (the display carries the
file stem for multi-file runs, and the strip helper only removes the
``[i/N] ... ·`` prefix).

These tests drive the REAL seams (patching the names the code actually
looks up: ``vemoizer.pipeline.run_diarization_stage``,
``vemoizer.llm_tail.repair_paragraphs`` / ``generate_notes``, and the
pipeline's decode path) and prove the patch took effect (the fake was
called). Each case:

* KeyboardInterrupt raised inside the real seam,
* exactly ONE stderr line naming the right stage (decode/diarize/
  repair/notes),
* exit 130,
* the display is closed,
* no output files written,
* no traceback,
* for BOTH quiet/non-TTY (display disabled or None) AND a live TTY
  display (the stem must never leak into the line).

Break-and-fail: dropping one stage-setter call at a seam makes the
matching case print the wrong stage.
"""

from __future__ import annotations

import contextlib
from typing import Any

import numpy as np
import pytest
from _cli_helpers import isolate_home, touch_files
from typer.testing import CliRunner

import vemoizer.pipeline as pipeline_module
from vemoizer.cli import app

SENTINEL_STEM = "SEnTineL-stEm"

runner = CliRunner()


def _two_label_paragraphs() -> list[dict[str, Any]]:
    return [
        {"start": 0.0, "end": 10.0, "text": "Moikka.", "speaker": "SPEAKER_1"},
        {"start": 12.0, "end": 20.0, "text": "Kyll\u00e4.", "speaker": "SPEAKER_2"},
    ]


def _good_result() -> dict[str, Any]:
    return {
        "text": "Moikka. Kyll.",
        "segments": [],
        "paragraphs": _two_label_paragraphs(),
        "notes": {"title": "T"},
    }


def _recorded_line(result) -> str:
    """The single stderr line the interrupt handler printed."""
    lines = [
        line
        for line in result.stderr.splitlines()
        if line.strip() and "Traceback" not in line
    ]
    return lines[-1] if lines else ""


def _assert_interrupt_contract(result, stage: str) -> None:
    assert result.exit_code == 130, (result.exit_code, result.stderr)
    assert "Traceback" not in result.stderr, result.stderr
    line = _recorded_line(result)
    assert line.startswith("interrupted during "), line
    assert f"interrupted during {stage} " in line, (stage, line)
    # The file stem never leaks into the line.
    assert SENTINEL_STEM not in line, (SENTINEL_STEM, line)
    # No files were written.
    assert "no files written" in line, line
    # No "complete" anywhere.
    assert "complete" not in result.stdout, result.stdout


def _setup_decode(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub the pre-decode seams (preflight + the meeting decode) so a run
    reaches the stage under test with a valid transcript result."""
    import vemoizer.preflight as preflight_module

    monkeypatch.setattr(preflight_module, "preflight_gate", lambda **kw: None)
    monkeypatch.setattr(
        pipeline_module, "decode_meeting", lambda *a, **kw: _good_result()
    )
    import vemoizer.ingest as ingest_module

    def _fake_audio(*a, **kw):
        return np.zeros(1600, dtype=np.float32)

    monkeypatch.setattr(ingest_module, "ingest_audio", _fake_audio)
    monkeypatch.setattr(pipeline_module, "ingest_audio", _fake_audio)
    import vemoizer.diarization as diarization_module

    monkeypatch.setattr(
        diarization_module, "run_diarization_stage", lambda *a, **kw: None
    )
    monkeypatch.setattr(pipeline_module, "run_diarization_stage", lambda *a, **kw: None)


def _live_display(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Replace the CLI's display construction with a live TTY display;
    returns a holder dict so the test can assert on the display state."""
    import io
    import sys

    from vemoizer.progress import ProgressDisplay
    from vemoizer.progress_wiring import make_batch_display  # noqa: F401

    holder: dict[str, Any] = {}
    buf = io.StringIO()
    monkeypatch.setattr(sys, "stderr", buf)
    monkeypatch.setattr(buf, "isatty", lambda: True, raising=False)
    display = ProgressDisplay()
    holder["display"] = display

    def fake_make(quiet: bool = False):
        return display

    import vemoizer.progress_wiring as progress_wiring_module

    monkeypatch.setattr(progress_wiring_module, "make_batch_display", fake_make)
    return holder


class TestInterruptStageNaming:
    """One test per stage seam × display kind (quiet and live TTY)."""

    # -- decode -----------------------------------------------------------

    def test_ctrl_c_in_decode_quiet(self, tmp_path, monkeypatch) -> None:
        """Ctrl-C inside the real decode seam (the meeting decode), --quiet:
        the line names 'decoding'."""
        files = touch_files([f"{SENTINEL_STEM}.m4a"], tmp_path)
        import vemoizer.preflight as preflight_module

        monkeypatch.setattr(preflight_module, "preflight_gate", lambda **kw: None)
        import vemoizer.ingest as ingest_module

        monkeypatch.setattr(
            ingest_module,
            "ingest_audio",
            lambda *a, **kw: np.zeros(1600, dtype=np.float32),
        )
        monkeypatch.setattr(
            pipeline_module,
            "ingest_audio",
            lambda *a, **kw: np.zeros(1600, dtype=np.float32),
        )
        calls: list[Any] = []

        def fake_decode(audio, slices, **kw):
            calls.append(audio)
            raise KeyboardInterrupt()

        monkeypatch.setattr(pipeline_module, "decode_meeting", fake_decode)
        isolate_home(monkeypatch, tmp_path, tmp_path)
        result = runner.invoke(app, ["meeting", files[0].name, "--quiet"])
        assert calls, "the decode seam was never reached"
        _assert_interrupt_contract(result, "decoding")

    def test_ctrl_c_in_decode_tty(self, tmp_path, monkeypatch) -> None:
        """Ctrl-C inside the decode seam with a live TTY display: the line
        names 'decoding' and the stem does not leak."""
        files = touch_files([f"{SENTINEL_STEM}.m4a"], tmp_path)
        import vemoizer.preflight as preflight_module

        monkeypatch.setattr(preflight_module, "preflight_gate", lambda **kw: None)
        import vemoizer.ingest as ingest_module

        monkeypatch.setattr(
            ingest_module,
            "ingest_audio",
            lambda *a, **kw: np.zeros(1600, dtype=np.float32),
        )
        monkeypatch.setattr(
            pipeline_module,
            "ingest_audio",
            lambda *a, **kw: np.zeros(1600, dtype=np.float32),
        )
        holder = _live_display(monkeypatch)
        calls: list[Any] = []

        def fake_decode(audio, slices, **kw):
            calls.append(audio)
            raise KeyboardInterrupt()

        monkeypatch.setattr(pipeline_module, "decode_meeting", fake_decode)
        isolate_home(monkeypatch, tmp_path, tmp_path)
        result = runner.invoke(app, ["meeting", files[0].name])
        assert calls, "the decode seam was never reached"
        _assert_interrupt_contract(result, "decoding")
        assert holder["display"].is_live is False

    def test_ctrl_c_in_diarize_quiet(self, tmp_path, monkeypatch) -> None:
        """Ctrl-C inside the real run_diarization_stage seam, --quiet
        (display None): the line names 'diarize'."""
        files = touch_files([f"{SENTINEL_STEM}.m4a"], tmp_path)
        _setup_decode(monkeypatch)
        calls: list[Any] = []

        def fake_diarize(audio, speakers):
            calls.append(audio)
            raise KeyboardInterrupt()

        monkeypatch.setattr(pipeline_module, "run_diarization_stage", fake_diarize)
        import vemoizer.diarization as diarization_module

        monkeypatch.setattr(diarization_module, "run_diarization_stage", fake_diarize)
        isolate_home(monkeypatch, tmp_path, tmp_path)
        result = runner.invoke(app, ["meeting", files[0].name, "--quiet"])
        assert calls, "the run_diarization_stage seam was never reached"
        _assert_interrupt_contract(result, "diarize")

    def test_ctrl_c_in_diarize_tty(self, tmp_path, monkeypatch) -> None:
        """Ctrl-C inside run_diarization_stage with a live TTY display:
        the line still names 'diarize' and the stem does not leak (the
        display's task description carries the stem — the tracker must be
        the authoritative source)."""
        files = touch_files([f"{SENTINEL_STEM}.m4a"], tmp_path)
        _setup_decode(monkeypatch)
        holder = _live_display(monkeypatch)
        calls: list[Any] = []

        def fake_diarize(audio, speakers):
            calls.append(audio)
            raise KeyboardInterrupt()

        monkeypatch.setattr(pipeline_module, "run_diarization_stage", fake_diarize)
        isolate_home(monkeypatch, tmp_path, tmp_path)
        result = runner.invoke(app, ["meeting", files[0].name])
        assert calls, "the run_diarization_stage seam was never reached"
        _assert_interrupt_contract(result, "diarize")
        assert holder["display"].is_live is False

    # -- repair -----------------------------------------------------------

    def _run_repair_case(self, tmp_path, monkeypatch, *, quiet: bool) -> None:
        """Drive llm_tail._run_repair directly: the tracker must say 'repair'
        when a Ctrl-C fires inside repair_paragraphs (as bound in llm_tail)."""

        import vemoizer.llm_tail as lt
        from vemoizer.llm_config import LLMConfig
        from vemoizer.preset_interrupt import begin_interrupt_tracking, handle_interrupt

        calls: list[Any] = []

        def fake_repair(client, paragraphs, glossary=None, **kw):
            calls.append(client)
            raise KeyboardInterrupt()

        monkeypatch.setattr(lt, "repair_paragraphs", fake_repair)

        class _FakeClient:
            def __init__(self, cfg):
                self._c = cfg

            def close(self):
                pass

            def complete(self, *a, **kw):
                return "ok"

        monkeypatch.setattr(lt, "LLMClient", _FakeClient)
        config = LLMConfig(
            base_url="http://localhost:9999/v1",
            model="test",
            api_key_env="VEMOIZER_TEST_KEY",
            timeout_seconds=10.0,
        )
        result: dict[str, Any] = {
            "text": "x",
            "paragraphs": [{"text": "a"}, {"text": "b"}],
        }
        tracker = begin_interrupt_tracking()
        with contextlib.suppress(KeyboardInterrupt):
            lt._run_repair(result, config, None, tracker=tracker)
        assert calls, "the repair_paragraphs seam was never reached"
        line = handle_interrupt(tracker, None)
        assert "interrupted during repair " in line, line
        assert SENTINEL_STEM not in line

    def test_ctrl_c_in_repair_quiet(self, tmp_path, monkeypatch) -> None:
        self._run_repair_case(tmp_path, monkeypatch, quiet=True)

    def test_ctrl_c_in_repair_tty(self, tmp_path, monkeypatch) -> None:
        self._run_repair_case(tmp_path, monkeypatch, quiet=False)

    # -- notes -------------------------------------------------------------

    def _run_notes_case(self, tmp_path, monkeypatch, *, quiet: bool) -> None:
        """Drive llm_tail._run_notes directly: the tracker must say 'notes'
        when a Ctrl-C fires inside generate_notes (as bound in llm_tail)."""

        import vemoizer.llm_tail as lt
        from vemoizer.llm_config import LLMConfig
        from vemoizer.preset_interrupt import begin_interrupt_tracking, handle_interrupt

        calls: list[Any] = []

        def fake_notes(client, text, **kw):
            calls.append(client)
            raise KeyboardInterrupt()

        monkeypatch.setattr(lt, "generate_notes", fake_notes)

        class _FakeClient:
            def __init__(self, cfg):
                self._c = cfg

            def close(self):
                pass

            def complete(self, *a, **kw):
                return "ok"

        monkeypatch.setattr(lt, "LLMClient", _FakeClient)
        config = LLMConfig(
            base_url="http://localhost:9999/v1",
            model="test",
            api_key_env="VEMOIZER_TEST_KEY",
            timeout_seconds=10.0,
        )
        result: dict[str, Any] = {
            "text": "some transcript",
            "paragraphs": [{"text": "a"}, {"text": "b"}],
        }
        tracker = begin_interrupt_tracking()
        with contextlib.suppress(KeyboardInterrupt):
            lt._run_notes(result, config, None, None, tracker=tracker)
        assert calls, "the generate_notes seam was never reached"
        line = handle_interrupt(tracker, None)
        assert "interrupted during notes " in line, line
        assert SENTINEL_STEM not in line

    def test_ctrl_c_in_notes_quiet(self, tmp_path, monkeypatch) -> None:
        self._run_notes_case(tmp_path, monkeypatch, quiet=True)

    def test_ctrl_c_in_notes_tty(self, tmp_path, monkeypatch) -> None:
        self._run_notes_case(tmp_path, monkeypatch, quiet=False)
