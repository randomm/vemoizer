"""Language handling for the ``meeting``/``memo`` presets (issue #108).

Moved from ``test_meeting_memo_cli.py`` (AGENTS.md 800-line test-file cap):
the ``--language`` flag plumbing, the shared memo/meeting exit-2 error
contract, and the config-layer coverage: the REAL layered config search
plus the REAL ``pipeline.transcribe_file`` body (heavy stages faked at
the same seams ``test_pipeline.py`` uses) feeding the ``[meeting]
language`` key into the ``decode_meeting`` ``language`` kwarg.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from _cli_helpers import isolate_home
from test_pipeline import (  # noqa: F401 - shared orchestrator fixtures
    _patch_ingest,
    _patch_preflight_pass,
    _patch_vad,
)
from typer.testing import CliRunner

from vemoizer.cli import app

runner = CliRunner()


# -- flag forwarding and the shared error contract ------------------------


@pytest.mark.parametrize(
    ("flag_args", "expected"),
    [
        ([], None),
        (["--language", "fi"], "fi"),
        (["--language", "EN"], "en"),
        (["--language", "auto"], None),
    ],
    ids=[
        "default-auto-passes-none",
        "flag-pins-fi",
        "flag-case-insensitive-en",
        "auto-flag-passes-none",
    ],
)
def test_meeting_language_flag_forwarding(
    tmp_path, monkeypatch, flag_args: list[str], expected: str | None
) -> None:
    """Issue #108 options A/B: the language flag is forwarded verbatim to
    transcribe_file (normalized, with ``auto``/no-flag coercing to None so
    per-window detection runs), or None for the default run."""
    import vemoizer.pipeline as pipeline_module

    seen: dict = {}

    def fake_transcribe(path, **kwargs):
        seen.update(kwargs)
        return {
            "text": "moikka",
            "segments": [],
            "notes": {"title": "Test"},
        }

    monkeypatch.setattr(pipeline_module, "transcribe_file", fake_transcribe)
    isolate_home(monkeypatch, tmp_path, tmp_path)
    result = runner.invoke(app, ["meeting", "a.m4a", *flag_args])
    assert result.exit_code == 0
    assert "language" in seen
    assert seen["language"] == expected


def test_meeting_language_invalid_value_fails_closed_before_transcription(
    tmp_path, monkeypatch
) -> None:
    """An unknown --language value must fail in milliseconds, not after
    a decode."""
    import vemoizer.pipeline as pipeline_module

    def fake_transcribe(path, **kwargs):
        raise AssertionError("transcribe_file must not run on a bad --language")

    monkeypatch.setattr(pipeline_module, "transcribe_file", fake_transcribe)
    isolate_home(monkeypatch, tmp_path, tmp_path)
    result = runner.invoke(app, ["meeting", "a.m4a", "--language", "xx"])
    assert result.exit_code == 2
    assert "unknown language" in result.stderr


def test_memo_language_error_contract_matches_meeting(tmp_path, monkeypatch) -> None:
    """A bad recognition-language value produces a clean exit 2 on memo
    exactly as on meeting — never a raw traceback (issue #108: the two
    preset commands share the error contract; memo has no --language
    flag, so the ValueError arrives via run_preset, which the memo
    command's handler must catch the same way)."""
    import vemoizer.batch as batch_module

    def fake_run_preset(*args, **kwargs):
        raise ValueError("unknown language 'xx' (known: auto, fi, en)")

    monkeypatch.setattr(batch_module, "run_preset", fake_run_preset)
    isolate_home(monkeypatch, tmp_path, tmp_path)
    result = runner.invoke(app, ["memo", "a.m4a"])
    assert result.exit_code == 2
    assert "unknown language" in result.stderr
    assert "Traceback" not in result.stderr


# -- [meeting] language via the real config search + transcribe_file ------


def _write_config(root: Path, sub: str, body: str) -> Path:
    path = root / sub / "config.toml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    return path


#: A valid ``[llm]`` section pointing at an unroutable host — the LLM
#: config is present (the strict layered search accepts the file) but
#: every LLM call times out and the tail stages fail open, so no network
#: is ever needed and the run still succeeds.
_NO_LLM_CONFIG_BODY = (
    "[llm]\n"
    'base_url = "https://llm.invalid/v1"\n'
    'model = "no-network"\n'
    'api_key_env = "K"\n'
    "timeout_seconds = 0.05\n"
)


def _config_with_meeting_language(
    tmp_path: Path, language: str | None, *, extra_meeting_keys: str = ""
) -> Path:
    """``~/.vemoizer/config.toml`` (home layer) with ``[meeting] language``."""
    home = tmp_path / "home"
    body = _NO_LLM_CONFIG_BODY
    if language is not None or extra_meeting_keys:
        body += "\n[meeting]\n"
        if language is not None:
            body += f'language = "{language}"\n'
        body += extra_meeting_keys
    return _write_config(home, ".vemoizer", body)


def _spy_decode_meeting(monkeypatch, seen: dict) -> None:
    """Record every ``decode_meeting`` call's kwargs and return a
    single-segment Finnish-ish transcript the run checks accept."""
    import vemoizer.pipeline as pipeline_module

    def fake_decode_meeting(audio, slices, initial_prompt=None, **kwargs):
        seen.setdefault("calls", []).append(kwargs)
        return {
            "text": "hei maailma",
            "words": [{"word": "hei", "start": 0.0, "end": 0.4}],
            "segments": [{"start": 0.0, "end": 1.0, "text": "hei maailma"}],
            "slices": [],
        }

    monkeypatch.setattr(pipeline_module, "decode_meeting", fake_decode_meeting)


def _run_real_pipeline(monkeypatch, tmp_path: Path, extra_args: list[str] | None):
    """Invoke ``meeting`` with the REAL ``transcribe_file`` body.

    The heavy stages are faked at the module-level seams exactly as
    ``test_pipeline.py`` does: preflight forced green, ingest and VAD
    patched, ``decode_meeting`` a spy (the meeting profile is
    whisper-only, so decode B / re-decode / adjudication never run), and
    HOME + CWD isolated so the layered config search only sees the
    config this test wrote. Returns the recorded ``decode_meeting``
    kwargs (the first window).
    """
    import vemoizer.batch as batch_module
    import vemoizer.pipeline as pipeline_module

    _patch_preflight_pass(monkeypatch)
    _patch_ingest(monkeypatch)
    _patch_vad(monkeypatch)
    # Import the HF token helper BEFORE the test's logging snapshot (the
    # autouse guard in conftest) so the StreamHandler the import attaches
    # to the huggingface_hub logger predates the snapshot and cannot look
    # like a leak. The real transcribe_file body imports it mid-test via
    # preflight's hf_token_present (a meeting run is always gated); the
    # fake keeps the token absent without the import's side effects.
    from huggingface_hub import get_token  # noqa: F401

    import vemoizer.preflight as preflight_module

    monkeypatch.setattr(preflight_module, "hf_token_present", lambda: True)
    seen: dict = {}
    _spy_decode_meeting(monkeypatch, seen)

    def fake_pcm_duration_seconds(path, **kw):
        return 2.0

    # The post-transcribe sidecar duration probe is an ffmpeg subprocess;
    # fail it open so the run never shells out.
    import vemoizer.ingest as ingest_module

    monkeypatch.setattr(
        ingest_module, "pcm_duration_seconds", fake_pcm_duration_seconds
    )

    def fake_diarize(audio, speakers=None):
        return [(0.0, 2.0, "SPEAKER_00")]

    # The meeting preset diarizes by default. Patch the stage seam in the
    # pipeline module's namespace (where the transcribe_file body calls
    # run_diarization_stage) — not diarization.diarize, which the real
    # stage resolves in its own module. The fake avoids the real pyannote
    # load and returns one segment so the fail-loud no-labels check passes.
    monkeypatch.setattr(pipeline_module, "run_diarization_stage", fake_diarize)

    # The LLM config points at an unroutable host, but keep the tail
    # (repair/notes) off the wire entirely: the stage seams resolve in
    # the pipeline module's namespace, so a fake generate_notes (and the
    # client constructor) means zero network and a faster, deterministic
    # run — the LLM tail is not under test here.
    def fake_notes(client, text, paragraphs=None, glossary=None, budget=None):
        return {"title": "Test", "summary": "s", "key_points": [], "action_items": []}

    class _Client:
        def complete(self, *a, **kw):
            return ""

        def close(self) -> None:
            pass

    monkeypatch.setattr(pipeline_module, "generate_notes", fake_notes)
    monkeypatch.setattr(pipeline_module, "LLMClient", lambda cfg: _Client())
    # The per-file loop writes through batch_output's seam; keep the write
    # off the real filesystem (the output pair is not under test here).
    monkeypatch.setattr(batch_module, "_write_preset_output", lambda *a, **k: [])
    isolate_home(monkeypatch, tmp_path, tmp_path)
    result = runner.invoke(app, ["meeting", "a.m4a", *(extra_args or [])])
    assert result.exit_code == 0, result.stderr
    assert len(seen["calls"]) == 1
    return seen["calls"][0]


def test_config_meeting_language_en_reaches_decode(tmp_path, monkeypatch) -> None:
    """[meeting] language = "en" -> the language kwarg on decode_meeting is "en"."""
    _config_with_meeting_language(tmp_path, "en")
    kw = _run_real_pipeline(monkeypatch, tmp_path, None)
    assert kw["language"] == "en"


def test_config_meeting_language_fi_reaches_decode(tmp_path, monkeypatch) -> None:
    """[meeting] language = "fi" -> the language kwarg is "fi"."""
    _config_with_meeting_language(tmp_path, "fi")
    kw = _run_real_pipeline(monkeypatch, tmp_path, None)
    assert kw["language"] == "fi"


def test_config_meeting_language_auto_reaches_decode_as_none(
    tmp_path, monkeypatch
) -> None:
    """[meeting] language = "auto" -> None (per-window detection)."""
    _config_with_meeting_language(tmp_path, "auto")
    kw = _run_real_pipeline(monkeypatch, tmp_path, None)
    assert kw["language"] is None


def test_config_without_meeting_section_reaches_decode_as_none(
    tmp_path, monkeypatch
) -> None:
    """No [meeting] table at all -> None (auto-detect, the default)."""
    _config_with_meeting_language(tmp_path, None)
    kw = _run_real_pipeline(monkeypatch, tmp_path, None)
    assert kw["language"] is None


def test_config_meeting_language_invalid_fails_open_to_none(
    tmp_path, monkeypatch
) -> None:
    """[meeting] language = "xx" -> None (fail-open: the parse falls back
    to auto-detect; a typo must not kill the run)."""
    _config_with_meeting_language(tmp_path, "xx")
    kw = _run_real_pipeline(monkeypatch, tmp_path, None)
    assert kw["language"] is None


def test_cli_language_en_beats_config_fi(tmp_path, monkeypatch) -> None:
    """The run option beats the config value: --language en with
    [meeting] language = "fi" pins the decode to "en"."""
    _config_with_meeting_language(tmp_path, "fi")
    kw = _run_real_pipeline(monkeypatch, tmp_path, ["--language", "en"])
    assert kw["language"] == "en"


def test_config_unknown_meeting_key_warns_not_fails(tmp_path, monkeypatch) -> None:
    """Unknown [meeting] keys are warned (not fatal) — the section is
    cosmetic, a typo must not trade a harmless typo for a failing run."""
    _config_with_meeting_language(
        tmp_path, "fi", extra_meeting_keys='langugae = "xx"\n'
    )
    kw = _run_real_pipeline(monkeypatch, tmp_path, None)
    # The run still succeeds and the valid key still pins.
    assert kw["language"] == "fi"
