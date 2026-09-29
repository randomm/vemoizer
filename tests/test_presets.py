"""The meeting/memo presets: ``presets.resolve_options`` (issue #82).

Pins the pure core of the new commands: ``resolve_options(command,
layers, cli_overrides) -> RunOptions`` takes already-parsed layer dicts
and CLI overrides and does no file I/O and no printing. The tests cover
the field contents for both presets, the layer-merge fallback (project
first, case-insensitive dedupe keeping the project spelling, correction
union with the project right side winning), the ``@``-line LLM-only
split, the memo seam (whisper prompt always empty), and the CLI
override precedence (CLI > layers > preset defaults).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from vemoizer.presets import MEETING_SPEAKERS, RunOptions, resolve_options


def _opts(
    command: str,
    layers: dict[str, dict] | None = None,
    cli: dict[str, Any] | None = None,
    capsys: pytest.CaptureFixture[str] | None = None,
) -> RunOptions:
    opts = resolve_options(command, layers, cli)
    # resolve_options is pure: no output on stdout or stderr.
    if capsys is not None:
        captured = capsys.readouterr()
        assert captured.out == ""
        assert captured.err == ""
    return opts


class TestMeetingPreset:
    def test_defaults_are_meeting_profile_with_diarize_and_repair(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        opts = _opts("meeting", None, None, capsys)
        assert opts.profile == "meeting"
        assert opts.diarize is True
        assert opts.repair is True
        assert opts.speakers == MEETING_SPEAKERS == (2, 6)

    def test_no_layers_no_glossary_no_corrections(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        opts = _opts("meeting", None, None, capsys)
        assert opts.glossary_path is None
        assert opts.whisper_prompt == []
        assert opts.llm_terms == []
        assert opts.corrections == {}

    def test_config_path_is_none_so_the_layered_search_runs(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # Without --config the presets must NOT short-circuit the search:
        # config_path=None so llm.load_default_config runs the layered
        # search (~/.vemoizer + nearest ./.vemoizer + legacy). Only an
        # explicit caller (e.g. eval) passes "os.devnull".
        opts = _opts("meeting", None, None, capsys)
        assert opts.config_path is None

    def test_meeting_terms_seed_whisper_prompt_and_llm(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        layers = {"project": {"terms": ["Flagship", "@Nordea"]}}
        opts = _opts("meeting", layers, None, capsys)
        # Bare terms seed the whisper prompt; @ terms are LLM-only.
        assert opts.whisper_prompt == ["Flagship"]
        assert opts.llm_terms == ["Nordea"]

    def test_meeting_layers_merge_project_first(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        layers = {
            "home": {
                "terms": ["Nordea", "flagship", "RFID"],
                "corrections": {"Blacksit": "Flagship"},
            },
            "project": {"terms": ["Flagship", "Vemoizer"]},
        }
        opts = _opts("meeting", layers, None, capsys)
        # Project terms first (whisper priority: last-listed wins), then
        # the home terms the project did not cover (case-insensitive:
        # home's "flagship" is dropped, "Nordea"/"RFID" survive).
        assert opts.whisper_prompt == ["Flagship", "Vemoizer", "Nordea", "RFID"]
        # Correction union: project right side wins on the same wrong side.
        assert opts.corrections == {"Blacksit": "Flagship"}

    def test_meeting_correction_union_project_right_side_wins(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        layers = {
            "home": {"terms": [], "corrections": {"Blacksit": "Oldname"}},
            "project": {"terms": [], "corrections": {"Blacksit": "Flagship"}},
        }
        opts = _opts("meeting", layers, None, capsys)
        assert opts.corrections == {"Blacksit": "Flagship"}


class TestMemoPreset:
    def test_defaults_are_whisper_decode_without_diarization(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        opts = _opts("memo", None, None, capsys)
        assert opts.profile == "meeting"  # the whisper meeting decode seam
        assert opts.diarize is False
        assert opts.repair is False
        assert opts.speakers is None

    def test_whisper_prompt_is_always_empty_even_with_terms(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # Memo seam: a 30-minute memo must not seed recognition with
        # prompt terms; the temp glossary file the batch runner writes
        # carries corrections only, so glossary_prompt yields None.
        layers = {"project": {"terms": ["Flagship", "@Nordea", "Riihimäki"]}}
        opts = _opts("memo", layers, None, capsys)
        assert opts.whisper_prompt == []

    def test_llm_terms_still_flow_for_memo(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # The LLM stages (notes) are still grounded on the @ terms.
        layers = {"project": {"terms": ["@Nordea"]}}
        opts = _opts("memo", layers, None, capsys)
        assert opts.llm_terms == ["Nordea"]

    def test_memo_corrections_come_from_layers(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # The batch runner writes the project layer's correction pairs to
        # the temp file so apply_corrections still fires; resolve_options
        # surfaces the merged pairs.
        layers = {"project": {"terms": [], "corrections": {"Blacksit": "Flagship"}}}
        opts = _opts("memo", layers, None, capsys)
        assert opts.corrections == {"Blacksit": "Flagship"}

    def test_config_path_is_none_so_the_layered_search_runs(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # Same as meeting: None means the layered config search runs.
        opts = _opts("memo", None, None, capsys)
        assert opts.config_path is None


class TestCliOverrides:
    def test_explicit_glossary_replaces_layers(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # --glossary REPLACES both layers entirely (no merging): the path
        # is passed straight through and the layer terms/corrections are
        # ignored (the batch runner passes the single file to
        # transcribe_file, glossary_path unchanged).
        glossary_file = Path("/tmp/some/glossary.txt")
        layers = {"project": {"terms": ["Flagship"]}}
        opts = _opts(
            "meeting",
            layers,
            {"glossary": glossary_file},
            capsys,
        )
        assert opts.glossary_path == str(glossary_file)
        assert opts.whisper_prompt == []
        assert opts.corrections == {}

    def test_explicit_config_wins_over_the_default_search(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        cfg = Path("/tmp/some/config.toml")
        opts = _opts("memo", None, {"config": cfg}, capsys)
        assert opts.config_path == str(cfg)

    def test_cli_speakers_override_the_preset_default(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        opts = _opts("meeting", None, {"speakers": (3, 5)}, capsys)
        assert opts.speakers == (3, 5)

    def test_cli_diarize_off_wins(self, capsys: pytest.CaptureFixture[str]) -> None:
        opts = _opts("meeting", None, {"diarize": False}, capsys)
        assert opts.diarize is False

    def test_cli_repair_off_wins(self, capsys: pytest.CaptureFixture[str]) -> None:
        opts = _opts("meeting", None, {"repair": False}, capsys)
        assert opts.repair is False

    def test_cli_profile_override(self, capsys: pytest.CaptureFixture[str]) -> None:
        opts = _opts("memo", None, {"profile": "dictation"}, capsys)
        assert opts.profile == "dictation"

    def test_merged_terms_override_the_layer_merge(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        # The batch runner pre-merges the layers (glossary_layers.merge)
        # and passes the result; the layer dicts are then ignored.
        layers = {"project": {"terms": ["FromLayers"]}}
        opts = _opts(
            "meeting",
            layers,
            {"glossary_terms": ["FromMerge", "@X"], "glossary_corrections": {"a": "b"}},
            capsys,
        )
        assert opts.whisper_prompt == ["FromMerge"]
        assert opts.llm_terms == ["X"]
        assert opts.corrections == {"a": "b"}


class TestInvalidCommand:
    def test_unknown_command_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match=r"unknown preset command"):
            resolve_options("podcast")

    def test_error_names_the_known_commands(self) -> None:
        with pytest.raises(ValueError, match=r"meeting, memo"):
            resolve_options("nope")


def test_no_file_reads_during_resolve(monkeypatch: pytest.MonkeyPatch) -> None:
    """resolve_options is pure: no open(), no Path.read_* calls."""
    import builtins as _b

    real_open = _b.open
    reads: list[str] = []

    def _forbidden_open(*args: Any, **kwargs: Any) -> Any:
        reads.append(f"open{args!r}")
        return real_open(*args, **kwargs)

    def _forbidden_read_text(self: Path, *a: Any, **kw: Any) -> str:
        reads.append(f"read_text{self!r}")
        return ""

    def _forbidden_read_bytes(self: Path, *a: Any, **kw: Any) -> bytes:
        reads.append(f"read_bytes{self!r}")
        return b""

    monkeypatch.setattr(_b, "open", _forbidden_open)
    monkeypatch.setattr(Path, "read_text", _forbidden_read_text)
    monkeypatch.setattr(Path, "read_bytes", _forbidden_read_bytes)
    resolve_options(
        "meeting",
        {"project": {"terms": ["X"], "corrections": {"a": "b"}}},
        {"glossary": Path("/tmp/g.txt")},
    )
    resolve_options("memo", None, {"config": Path("/tmp/c.toml")})
    assert reads == []
