"""Tests for stored-name prompt defaults (issue #93, decision 4).

For a label that already has a stored ``speaker_names`` entry, the prompt
shows the stored name as a bracketed default, e.g. ``Name for SPEAKER_1
[Mikko]:``. An empty answer keeps the stored name (no change); a non-empty
answer replaces it. For a label with no stored name the prompt is unchanged
and an empty answer skips it. A kept stored name never triggers the
Add-to-people prompt.

Reuses the ``_sidecar`` / ``_write_sidecar`` / ``_capture_stderr`` helpers
from ``test_names_cli`` (by import) and the ``isolate_home`` fixture.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from _cli_helpers import isolate_home
from test_names_cli import _sidecar, _write_sidecar

from vemoizer.names_cli import run_names


def _sidecar_with_stored(stored: dict[str, str]) -> dict[str, Any]:
    """A two-speaker sidecar carrying the given stored ``speaker_names``."""
    data = _sidecar()
    data.pop("source", None)
    data["speaker_names"] = stored
    return data


class TestStoredNameDefault:
    def test_default_shown_in_prompt(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Stored name is shown as a bracketed default in the prompt."""
        isolate_home(monkeypatch, tmp_path, tmp_path)
        sc = _write_sidecar(tmp_path, _sidecar_with_stored({"SPEAKER_1": "Mikko"}))

        prompts: list[str] = []

        def recording_input(prompt: str) -> str:
            prompts.append(prompt)
            if "Name for" in prompt:
                return "Mikko"
            return "n"

        rc = run_names(
            sc,
            no_play=True,
            input_fn=recording_input,
            tty_isatty=lambda: True,
        )
        assert rc == 0
        assert any("[Mikko]" in p for p in prompts)

    def test_empty_answer_keeps_stored_name(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Empty answer keeps the stored name (no change)."""
        isolate_home(monkeypatch, tmp_path, tmp_path)
        sc = _write_sidecar(tmp_path, _sidecar_with_stored({"SPEAKER_1": "Mikko"}))

        # SPEAKER_1 (stored): empty -> keep. SPEAKER_2 (no stored): empty -> skip.
        inputs = iter(["", ""])
        rc = run_names(
            sc,
            no_play=True,
            input_fn=lambda prompt: next(inputs),
            tty_isatty=lambda: True,
        )
        assert rc == 0

        updated = json.loads(sc.read_text(encoding="utf-8"))
        assert updated["speaker_names"].get("SPEAKER_1") == "Mikko"
        assert "SPEAKER_2" not in updated["speaker_names"]

    def test_non_empty_answer_replaces_stored(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A non-empty answer overwrites the stored name."""
        isolate_home(monkeypatch, tmp_path, tmp_path)
        sc = _write_sidecar(tmp_path, _sidecar_with_stored({"SPEAKER_1": "Mikko"}))

        # SPEAKER_1 (stored Mikko): replace with Aino + decline add.
        # SPEAKER_2 (no stored): empty -> skip.
        inputs = iter(["Aino", "n", ""])
        rc = run_names(
            sc,
            no_play=True,
            input_fn=lambda prompt: next(inputs),
            tty_isatty=lambda: True,
        )
        assert rc == 0

        updated = json.loads(sc.read_text(encoding="utf-8"))
        assert updated["speaker_names"].get("SPEAKER_1") == "Aino"

    def test_no_stored_name_empty_skips(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A label with no stored name: the prompt is unchanged and an empty
        answer skips it (nothing persisted)."""
        isolate_home(monkeypatch, tmp_path, tmp_path)
        sc = _write_sidecar(tmp_path, _sidecar_with_stored({}))

        prompts: list[str] = []

        def recording_input(prompt: str) -> str:
            prompts.append(prompt)
            if "Name for" in prompt:
                return ""
            return "n"

        rc = run_names(
            sc,
            no_play=True,
            input_fn=recording_input,
            tty_isatty=lambda: True,
        )
        assert rc == 0
        # No-stored-name prompt is the standard "empty to skip" form, not a
        # bracketed default.
        assert any("empty to skip" in p for p in prompts)
        assert not any("[" in p and "Name for" in p for p in prompts)

    def test_kept_stored_name_no_add_to_people_prompt(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A kept stored name must not trigger the Add-to-people prompt."""
        isolate_home(monkeypatch, tmp_path, tmp_path)
        # A project config so a config path exists (rules out "no config"
        # masking); empty people list.
        vemoizer_dir = tmp_path / ".vemoizer"
        vemoizer_dir.mkdir()
        config = vemoizer_dir / "config.toml"
        config.write_text("people = []\n", encoding="utf-8")

        sc = _write_sidecar(tmp_path, _sidecar_with_stored({"SPEAKER_1": "Mikko"}))

        add_prompts: list[str] = []
        consumed = [0]

        def recording_input(prompt: str) -> str:
            consumed[0] += 1
            if "Add" in prompt:
                add_prompts.append(prompt)
                return "n"
            if "Name for" in prompt:
                return ""  # empty -> keep stored for both labels
            return ""

        rc = run_names(
            sc,
            no_play=True,
            input_fn=recording_input,
            tty_isatty=lambda: True,
        )
        assert rc == 0
        # Only SPEAKER_1 is kept; no Add-to-people prompt at all.
        assert add_prompts == []
        # Exactly one name prompt was issued (SPEAKER_2 also empty, skipped);
        # the kept stored name produced no additional prompt.
        assert consumed[0] == 2


class TestStoredNameMergeStillApplies:
    def test_same_name_merge_still_applies(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Replacing SPEAKER_2's stored name with the same name as SPEAKER_1
        still collapses to the earliest label via the render merge rule."""
        isolate_home(monkeypatch, tmp_path, tmp_path)
        sc = _write_sidecar(
            tmp_path, _sidecar_with_stored({"SPEAKER_1": "Mikko", "SPEAKER_2": "Old"})
        )

        # SPEAKER_1 (stored Mikko): empty -> keep.
        # SPEAKER_2 (stored Old): replace with Mikko + decline add.
        prompts: list[str] = []

        def recording_input(prompt: str) -> str:
            prompts.append(prompt)
            if "Name for" in prompt:
                if "SPEAKER_1" in prompt:
                    return ""
                return "Mikko"
            return "n"

        rc = run_names(
            sc,
            no_play=True,
            input_fn=recording_input,
            tty_isatty=lambda: True,
        )
        assert rc == 0

        updated = json.loads(sc.read_text(encoding="utf-8"))
        assert updated["speaker_names"].get("SPEAKER_1") == "Mikko"
        assert updated["speaker_names"].get("SPEAKER_2") == "Mikko"

        md_files = list(tmp_path.glob("*.md"))
        assert len(md_files) == 1
        md_content = md_files[0].read_text(encoding="utf-8")
        # Same-name collapse: SPEAKER_2 does not appear as a separate speaker.
        assert "SPEAKER_2" not in md_content
