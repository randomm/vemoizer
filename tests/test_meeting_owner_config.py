"""Owner name UX for two-track calls (issue #171, P5).

Covers the ``[meeting] owner`` key (known to the validator, falls through
from project to home config), the first-run prompt that saves to the home
config, and the atomic ``save_owner`` writer. Every test runs under a tmp
HOME (``isolate_home``); the real home is never read or written.
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

import pytest
from _cli_helpers import isolate_home, touch_files

import vemoizer.pipeline as pipeline_module
from vemoizer.batch_preset import run_preset
from vemoizer.llm_config import _validate_sections
from vemoizer.owner_config import OWNER_PROMPT, resolve_run_owner, save_owner
from vemoizer.tracks import TrackPaths


def _home_config(tmp_path: Path) -> Path:
    return tmp_path / "home" / ".vemoizer" / "config.toml"


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


class _Prompt:
    """A scripted ``input_fn`` that records every prompt it was asked."""

    def __init__(self, *answers: str) -> None:
        self.answers = list(answers)
        self.prompts: list[str] = []

    def __call__(self, text: str) -> str:
        self.prompts.append(text)
        return self.answers.pop(0) if self.answers else ""


def _no_prompt(text: str) -> str:
    raise AssertionError(f"unexpected prompt: {text!r}")


@pytest.fixture
def home(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Path:
    isolate_home(monkeypatch, tmp_path)
    return tmp_path / "home"


def test_meeting_owner_key_is_known_to_the_validator(capsys) -> None:
    _validate_sections({"meeting": {"owner": "Janni"}}, Path("c.toml"))
    assert capsys.readouterr().err == ""


def test_project_config_without_owner_falls_through_to_home(
    home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write(_home_config(tmp_path), '[meeting]\nowner = "Janni"\n')
    project = tmp_path / "project"
    _write(project / ".vemoizer" / "config.toml", '[meeting]\nlanguage = "en"\n')
    monkeypatch.chdir(project)

    name = resolve_run_owner(None, prompt=False, input_fn=_no_prompt, echo=print)

    assert name == "Janni"


def test_owner_only_project_file_beats_home_owner(
    home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression (PR #193 review): a project layer holding only [meeting] owner
    is still a layer, so it wins over the home owner (project before home)."""
    _write(_home_config(tmp_path), '[meeting]\nowner = "HomeName"\n')
    project = tmp_path / "project"
    _write(project / ".vemoizer" / "config.toml", '[meeting]\nowner = "ProjectName"\n')
    monkeypatch.chdir(project)

    name = resolve_run_owner(None, prompt=False, input_fn=_no_prompt, echo=print)

    assert name == "ProjectName"


def test_preferences_only_project_file_is_still_the_project_layer(
    home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression (PR #193 review): only the home file is skipped as
    preferences-only; a project [meeting]-only file is found by walk-up, so its
    [meeting] language is not silently dropped."""
    from vemoizer.llm_config import _find_nearest_vemoizer_config

    _write(_home_config(tmp_path), '[meeting]\nowner = "HomeName"\n')
    project = tmp_path / "project"
    project_file = _write(
        project / ".vemoizer" / "config.toml", '[meeting]\nlanguage = "en"\n'
    )

    assert _find_nearest_vemoizer_config(project) == project_file


def test_flag_beats_project_and_home(
    home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write(_home_config(tmp_path), '[meeting]\nowner = "Home"\n')
    project = tmp_path / "project"
    _write(project / ".vemoizer" / "config.toml", '[meeting]\nowner = "Project"\n')
    monkeypatch.chdir(project)

    name = resolve_run_owner("Flag", prompt=False, input_fn=_no_prompt, echo=print)

    assert name == "Flag"


def test_first_interactive_run_prompts_once_and_saves(
    home: Path, tmp_path: Path
) -> None:
    asked = _Prompt("Janni")
    echoed: list[str] = []

    first = resolve_run_owner(None, prompt=True, input_fn=asked, echo=echoed.append)
    second = resolve_run_owner(None, prompt=True, input_fn=asked, echo=echoed.append)

    assert first == second == "Janni"
    assert asked.prompts == [OWNER_PROMPT]
    saved = tomllib.loads(_home_config(tmp_path).read_text(encoding="utf-8"))
    assert saved["meeting"]["owner"] == "Janni"


def test_blank_answer_uses_fallback_and_saves_nothing(
    home: Path, tmp_path: Path
) -> None:
    asked = _Prompt("   ")

    name = resolve_run_owner(None, prompt=True, input_fn=asked, echo=print)

    assert name == "Minä"
    assert not _home_config(tmp_path).exists()


def test_non_interactive_never_prompts_and_prints_a_hint(
    home: Path, tmp_path: Path
) -> None:
    echoed: list[str] = []

    name = resolve_run_owner(
        None, prompt=False, input_fn=_no_prompt, echo=echoed.append
    )

    assert name == "Minä"
    assert not _home_config(tmp_path).exists()
    assert len(echoed) == 1
    assert "--me" in echoed[0] and "owner" in echoed[0]


def test_english_config_falls_back_to_me(
    home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write(_home_config(tmp_path), 'language = "en"\n')

    name = resolve_run_owner(None, prompt=False, input_fn=_no_prompt, echo=print)

    assert name == "Me"


def test_owner_only_home_file_is_not_an_llm_config(home: Path, tmp_path: Path) -> None:
    """The saved owner alone must not break the layered config search."""
    from vemoizer.llm_config import _default_search

    _write(_home_config(tmp_path), '[meeting]\nowner = "Janni"\n')
    llm, raw = _default_search(
        home=lambda: tmp_path / "home",
        cwd=lambda: tmp_path,
        legacy_paths=(tmp_path / "absent.toml",),
    )

    assert llm is None
    assert raw is None


def test_save_owner_inserts_under_an_existing_meeting_table(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    original = '# keep me\n[llm]\nmodel = "x"\n\n[meeting]\nlanguage = "fi"  # note\n'
    path.write_text(original, encoding="utf-8")

    save_owner(path, "Janni")

    text = path.read_text(encoding="utf-8")
    assert text == original.replace("[meeting]\n", '[meeting]\nowner = "Janni"\n', 1)
    parsed = tomllib.loads(text)
    assert parsed["meeting"] == {"owner": "Janni", "language": "fi"}


def test_save_owner_appends_a_table_when_meeting_is_absent(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    original = '[llm]\nmodel = "x"\n'
    path.write_text(original, encoding="utf-8")

    save_owner(path, "Janni")

    text = path.read_text(encoding="utf-8")
    assert text.startswith(original)
    parsed = tomllib.loads(text)
    assert parsed["llm"] == {"model": "x"}
    assert parsed["meeting"] == {"owner": "Janni"}


def test_save_owner_creates_missing_file_and_directory(tmp_path: Path) -> None:
    path = tmp_path / "new" / ".vemoizer" / "config.toml"

    save_owner(path, "Janni")

    assert tomllib.loads(path.read_text(encoding="utf-8")) == {
        "meeting": {"owner": "Janni"}
    }


def test_save_owner_escapes_quotes_and_backslashes(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    name = 'Ja"nni\\K'

    save_owner(path, name)

    assert tomllib.loads(path.read_text(encoding="utf-8"))["meeting"]["owner"] == name


def test_save_owner_refuses_to_write_invalid_toml(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    original = '[meeting\nlanguage = "fi"\n'
    path.write_text(original, encoding="utf-8")

    with pytest.raises(ValueError):
        save_owner(path, "Janni")
    assert path.read_text(encoding="utf-8") == original


def test_run_preset_prompts_once_on_first_interactive_tracks_run(
    home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: list[Any] = []

    def fake_transcribe(path, **kwargs):
        seen.append(kwargs.get("owner"))
        return {"text": "moikka", "segments": [], "notes": {"title": "Call"}}

    monkeypatch.setattr(pipeline_module, "transcribe_file", fake_transcribe)
    call = touch_files(["call.m4a"], tmp_path)
    tracks = TrackPaths(me=tmp_path / "me.wav", them=tmp_path / "them.wav")
    asked = _Prompt("Janni")

    def run() -> int:
        return run_preset(
            call,
            command="meeting",
            config_path=None,
            glossary_path=None,
            tracks=tracks,
            quiet=True,
            input_fn=asked,
            tty_isatty=lambda: True,
        )

    assert run() == 0
    assert run() == 0

    assert asked.prompts == [OWNER_PROMPT]
    assert seen == ["Janni", "Janni"]


def test_run_preset_yes_never_prompts_and_uses_fallback(
    home: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys
) -> None:
    seen: list[Any] = []

    def fake_transcribe(path, **kwargs):
        seen.append(kwargs.get("owner"))
        return {"text": "moikka", "segments": [], "notes": {"title": "Call"}}

    monkeypatch.setattr(pipeline_module, "transcribe_file", fake_transcribe)
    call = touch_files(["call.m4a"], tmp_path)
    tracks = TrackPaths(me=tmp_path / "me.wav", them=tmp_path / "them.wav")

    code = run_preset(
        call,
        command="meeting",
        config_path=None,
        glossary_path=None,
        tracks=tracks,
        quiet=True,
        yes=True,
        input_fn=_no_prompt,
        tty_isatty=lambda: True,
    )

    assert code == 0
    assert seen == ["Minä"]
    assert "--me" in capsys.readouterr().err
    assert not _home_config(tmp_path).exists()


_VALID_LLM = (
    '[llm]\nbase_url = "http://localhost:1234/v1"\nmodel = "m"\n'
    'api_key_env = "K"\ntimeout_seconds = 5\n'
)


def test_owner_only_home_file_does_not_shadow_the_llm_config_from_under_home(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Regression (PR #193 review): cwd under the fake HOME must not walk up
    into the owner-only home file and fail loud on the missing [llm]."""
    from vemoizer.llm_config import load_default_config

    recordings = tmp_path / "home" / "Recordings"
    recordings.mkdir(parents=True)
    _write(_home_config(tmp_path), '[meeting]\nowner = "Janni"\n')
    isolate_home(monkeypatch, tmp_path, chdir_to=recordings)

    llm, raw = load_default_config()

    assert llm is None
    assert raw is None


def test_owner_only_home_file_does_not_shadow_legacy_people_or_language(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A legacy-only user keeps their people list and section language when a
    saved-owner home file sits under the walk-up (PR #193 review)."""
    import vemoizer.llm_config as llm_config
    from vemoizer.llm_config import _default_search
    from vemoizer.people_config import find_people_config_path

    home = tmp_path / "home"
    recordings = home / "Recordings"
    recordings.mkdir(parents=True)
    _write(_home_config(tmp_path), '[meeting]\nowner = "Janni"\n')
    legacy = tmp_path / "legacy.toml"
    _write(legacy, 'language = "en"\npeople = ["Eero"]\n' + _VALID_LLM)
    isolate_home(monkeypatch, tmp_path, chdir_to=recordings)
    monkeypatch.setattr(llm_config, "_LEGACY_CONFIG_PATHS", (legacy,))

    assert (
        find_people_config_path(home=home, cwd=recordings, legacy_paths=(legacy,))
        == legacy
    )
    llm, raw = _default_search(
        home=lambda: home, cwd=lambda: recordings, legacy_paths=(legacy,)
    )
    assert llm is not None and llm.model == "m"
    assert raw is not None and raw.get("people") == ["Eero"]
    assert llm_config.load_language() == "en"
