"""Review-finding hardening tests for the meeting/memo presets (issue #82).

Six fixes from the 6-lens review of PR #85:

1. ``@``-prefixed glossary terms are LLM-only: never in the whisper prompt,
   delivered to repair/notes with the ``@`` stripped.
2. ``vemoizer transcribe`` (``transcribe_batch``) must turn the strict
   config search's ``ConfigError`` into a clean ``error: ...`` line and
   exit 1 — same as ``run_preset``.
3. The M1 fail-loud per-file checks live in one helper shared by
   ``transcribe_batch`` and ``run_preset``.
4. The temp glossary file in ``run_preset`` is created inside the protected
   (finally) region; a write failure is a clean error, no leaked file.
5. The ``.md``/``.json`` output pair is probed as a unit: both files always
   share one stem (never ``X.md`` + ``X (2).json``).
6. An overflowing ``timeout_seconds`` (``1e400``) is a clean ``None`` in
   ``load_config`` (fail-open) and a ``ConfigError`` in the strict path —
   not an escaping ``OverflowError``.

All stages mocked — no models, no network, no ffmpeg. ``HOME`` is
monkeypatched and the CWD moved into a tmp dir so the layered config/glossary
search starts from a clean root (the dev machine has a live LLM config).
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from test_pipeline import _llm_config, _patch_ingest, _patch_preflight_pass, _patch_vad

import vemoizer.batch as batch
import vemoizer.llm_config as llm
import vemoizer.output.naming as naming
import vemoizer.pipeline as pipeline
from vemoizer.output.naming import collision_free_paths

# -- shared plumbing --------------------------------------------------------


def _clean_env(monkeypatch, tmp_path, *, chdir_to: str | os.PathLike | None = None):
    """Isolate the layered config/glossary search from the dev machine.

    HOME points at an empty tmp dir; CWD moves into a clean tmp root (or
    *chdir_to*) so the ``./.vemoizer`` walk-up never finds a real layer.
    """
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.chdir(chdir_to or tmp_path)


def _patch_whisper_meeting(monkeypatch, seen: dict, text: str = "hei maailma") -> None:
    """The meeting-profile decode seam; records the initial_prompt it got."""

    def fake_decode_meeting(audio, slices, initial_prompt=None):
        seen["initial_prompt"] = initial_prompt
        return {
            "text": text,
            "words": [{"word": "hei", "start": 0.0, "end": 0.4}],
            "segments": [{"start": 0.0, "end": 1.0, "text": text}],
            "slices": [],
        }

    monkeypatch.setattr(pipeline, "decode_meeting", fake_decode_meeting)


def _patch_llm_stage(monkeypatch, seen: dict) -> None:
    """The repair/notes seams (the ``pipeline`` namespace is what
    ``apply_llm_tail`` receives via the ``*_fn`` / ``llm_client_cls`` args)."""

    def fake_repair(client, paragraphs, glossary=None):
        seen.setdefault("repair_calls", []).append(glossary)
        return paragraphs

    def fake_notes(client, text, paragraphs=None, glossary=None):
        seen.setdefault("notes_calls", []).append(glossary)
        return {
            "title": "Kokeilutilanne",
            "summary": "x",
            "key_points": [],
            "action_items": [],
        }

    class _Client:
        def close(self) -> None:
            pass

    monkeypatch.setattr(pipeline, "repair_paragraphs", fake_repair)
    monkeypatch.setattr(pipeline, "generate_notes", fake_notes)
    monkeypatch.setattr(pipeline, "LLMClient", lambda cfg: _Client())


def _fake_result() -> dict:
    return {
        "text": "hei maailma",
        "segments": [{"start": 0.0, "end": 1.0, "text": "hei maailma"}],
        "paragraphs": [{"text": "hei maailma"}],
    }


def _patch_transcribe(monkeypatch, fake) -> None:
    """Patch ``vemoizer.pipeline.transcribe_file`` (the name both batch
    entry points resolve through their local imports).  Patches the
    preflight pass too so the patched ``transcribe_file``'s import is
    consistent (issue #79 inline preflight)."""
    from test_pipeline import _patch_preflight_pass

    _patch_preflight_pass(monkeypatch)
    monkeypatch.setattr(pipeline, "transcribe_file", fake)


# -- item 1: @-terms are LLM-only ------------------------------------------


def test_at_terms_reach_llm_tail_not_whisper_prompt(tmp_path, monkeypatch) -> None:
    """transcribe_file with an @-term glossary: the whisper prompt never
    contains the @-term; repair and notes receive it with the @ stripped."""
    seen: dict = {}
    _patch_preflight_pass(monkeypatch)
    _patch_ingest(monkeypatch)
    _patch_vad(monkeypatch)
    _patch_whisper_meeting(monkeypatch, seen)
    _patch_llm_stage(monkeypatch, seen)
    # Seam: the glossary load returns the file's terms as the M2 contract
    # says — @-term kept (LLM-only), bare term as prompt seed.
    monkeypatch.setattr(
        pipeline, "load_glossary", lambda path: ["@Janni Peltola", "Nordea"]
    )
    cfg = _llm_config(tmp_path)

    result = pipeline.transcribe_file(
        "/nonexistent.m4a",
        config_path=str(cfg),
        profile="meeting",
        glossary_path=str(tmp_path / "glossary.txt"),
        repair=True,
    )

    prompt = seen["initial_prompt"] or ""
    assert "Janni Peltola" not in prompt
    assert "Nordea" in prompt
    for glossary in [*seen["notes_calls"], *seen["repair_calls"]]:
        assert glossary == ["Janni Peltola", "Nordea"]
    # the transcript itself is unaffected
    assert result["text"]


def test_at_only_glossary_yields_empty_whisper_prompt(tmp_path, monkeypatch) -> None:
    """A glossary of @-terms only must leave the whisper prompt None while
    still delivering the @-stripped terms to repair and notes."""
    seen: dict = {}
    _patch_preflight_pass(monkeypatch)
    _patch_ingest(monkeypatch)
    _patch_vad(monkeypatch)
    _patch_whisper_meeting(monkeypatch, seen)
    _patch_llm_stage(monkeypatch, seen)
    # Seam: an @-only glossary — no bare prompt terms at all.
    monkeypatch.setattr(pipeline, "load_glossary", lambda path: ["@Janni Peltola"])
    cfg = _llm_config(tmp_path)

    pipeline.transcribe_file(
        "/nonexistent.m4a",
        config_path=str(cfg),
        profile="meeting",
        glossary_path=str(tmp_path / "glossary.txt"),
        repair=True,
    )

    assert seen["initial_prompt"] is None
    for glossary in [*seen["notes_calls"], *seen["repair_calls"]]:
        assert glossary == ["Janni Peltola"]


def test_at_term_with_space_is_stripped_for_llm_tail(tmp_path, monkeypatch) -> None:
    """``@Janni`` (no space) and ``@ Janni`` (space after @) both arrive at
    the repair/notes stages as ``Janni`` — the strip is ``t[1:].lstrip()``.
    """
    seen: dict = {}
    _patch_preflight_pass(monkeypatch)
    _patch_ingest(monkeypatch)
    _patch_vad(monkeypatch)
    _patch_whisper_meeting(monkeypatch, seen)
    _patch_llm_stage(monkeypatch, seen)
    # Seam: a layered-glossary file whose @-term carries a space.
    monkeypatch.setattr(pipeline, "load_glossary", lambda path: ["@ Janni Peltola"])
    cfg = _llm_config(tmp_path)

    pipeline.transcribe_file(
        "/nonexistent.m4a",
        config_path=str(cfg),
        profile="meeting",
        glossary_path=str(tmp_path / "glossary.txt"),
        repair=True,
    )

    assert seen["initial_prompt"] is None
    for glossary in [*seen["notes_calls"], *seen["repair_calls"]]:
        assert glossary == ["Janni Peltola"]


# -- item 2: transcribe_batch fails clean on a strict ConfigError -----------


def test_transcribe_batch_clean_config_error_exit_1(
    tmp_path, monkeypatch, capsys
) -> None:
    """A malformed ./.vemoizer/config.toml is a clean `error: ...` line and
    exit 1 for `vemoizer transcribe` — not a raw traceback (issue #78)."""
    (tmp_path / ".vemoizer").mkdir()
    (tmp_path / ".vemoizer" / "config.toml").write_text(
        '[llm]\nbase_url = "http://localhost"\nmodel = "m"\n',
        encoding="utf-8",
    )  # missing api_key_env + timeout_seconds -> strict ConfigError

    called: list[str] = []

    def fake_transcribe_file(path, **kwargs):
        called.append(str(path))
        return _fake_result()

    _patch_transcribe(monkeypatch, fake_transcribe_file)
    _clean_env(monkeypatch, tmp_path)

    code = batch.transcribe_batch(
        [tmp_path / "a.m4a"],
        formats=["txt"],
        config_path=None,
        profile="dictation",
        repair=False,
        glossary_path=None,
        speakers=None,
        diarize=False,
    )
    out = capsys.readouterr()
    assert code == 1
    assert "error:" in out.err
    assert "api_key_env" in out.err
    assert "Traceback" not in out.err
    # no transcript was attempted: the error happens before the file loop
    assert called == []


def test_transcribe_batch_stops_after_config_error(
    tmp_path, monkeypatch, capsys
) -> None:
    """Consistent with run_preset (stop): a config error aborts the batch
    before any file is transcribed."""
    (tmp_path / ".vemoizer").mkdir()
    (tmp_path / ".vemoizer" / "config.toml").write_text(
        '[llm]\nbase_url = "http://localhost"\nmodel = "m"\n', encoding="utf-8"
    )

    called: list[str] = []

    def fake_transcribe_file(path, **kwargs):
        called.append(str(path))
        return _fake_result()

    _patch_transcribe(monkeypatch, fake_transcribe_file)
    _clean_env(monkeypatch, tmp_path)

    code = batch.transcribe_batch(
        [tmp_path / "a.m4a", tmp_path / "b.m4a"],
        formats=["txt"],
        config_path=None,
        profile="dictation",
        repair=False,
        glossary_path=None,
        speakers=None,
        diarize=False,
    )
    assert code == 1
    assert called == []
    assert "wrote transcript for" not in capsys.readouterr().out


# -- item 3: one shared fail-loud helper ------------------------------------


def _run_with_result(result: dict) -> int:
    """Drive the shared per-file checks through the batch helper."""
    return batch._check_result(Path("x.m4a"), result, diarize=False)


def test_shared_helper_error_key(tmp_path, monkeypatch, capsys) -> None:
    code = _run_with_result(
        {"text": "", "segments": [], "error": "decode A produced no output"}
    )
    assert code == 1
    assert "error: decode A produced no output" in capsys.readouterr().err


def test_shared_helper_empty_transcript_fails_loud(
    tmp_path, monkeypatch, capsys
) -> None:
    code = _run_with_result({"text": "", "segments": []})
    assert code == 1
    assert "no transcript produced for x.m4a (empty transcript)" in (
        capsys.readouterr().err
    )


def test_shared_helper_diarize_without_labels_fails_loud(capsys) -> None:
    """Default label keeps the transcribe path's M1-pinned ``--diarize``
    wording; the preset path passes ``diarize_label="diarize"`` (also the
    run_preset M1 wording — each entry point keeps its own message)."""
    result = {
        "text": "hei",
        "segments": [{"start": 0.0, "end": 1.0, "text": "hei"}],
    }
    code = batch._check_result(Path("x.m4a"), result, diarize=True)
    assert code == 1
    err = capsys.readouterr().err
    assert "--diarize requested but no speaker labels returned for x.m4a" in err
    code = batch._check_result(
        Path("x.m4a"), result, diarize=True, diarize_label="diarize"
    )
    assert code == 1
    assert "diarize requested but no speaker labels returned for x.m4a" in (
        capsys.readouterr().err
    )


def test_shared_helper_success_is_silent(capsys) -> None:
    code = _run_with_result(_fake_result())
    assert code == 0
    assert "error:" not in capsys.readouterr().err


def test_shared_helper_echoes_warnings(capsys) -> None:
    result = _fake_result()
    result["warnings"] = ["diarization failed; continuing without speaker labels"]
    code = _run_with_result(result)
    assert code == 0
    assert (
        "diarization failed; continuing without speaker labels"
        in capsys.readouterr().err
    )


# -- item 4: temp glossary inside the protected region ----------------------


def test_glossary_write_failure_is_clean_and_no_leak(
    tmp_path, monkeypatch, capsys
) -> None:
    """An OSError writing the temp glossary is a clean `error: ...` line, a
    non-zero exit code, and no leaked temp file."""
    (tmp_path / ".vemoizer").mkdir()
    (tmp_path / ".vemoizer" / "glossary.txt").write_text("Nordea\n", encoding="utf-8")

    def boom(lines: list[str]) -> str:
        raise OSError("disk full")

    monkeypatch.setattr(batch, "_write_temp_glossary", boom)
    _patch_transcribe(monkeypatch, lambda path, **kw: _fake_result())
    _clean_env(monkeypatch, tmp_path)

    code = batch.run_preset(
        [tmp_path / "a.m4a"],
        command="meeting",
        config_path=str(tmp_path / "missing.toml"),  # explicit: no config search
        glossary_path=None,
        repair=True,
        quiet=True,
    )
    err = capsys.readouterr().err
    assert code == 1
    assert "error: could not write glossary" in err
    assert "Traceback" not in err
    # nothing leaked: no vemoizer-glossary-* file anywhere under tmp
    assert [p.name for p in tmp_path.iterdir() if p.name.startswith("vemoizer-")] == []


# -- item 5: .md/.json pair probed as a unit ---------------------------------


def test_pair_unit_only_json_taken(tmp_path) -> None:
    """Only the .json name is taken -> BOTH files get the ` (2)` suffix."""
    (tmp_path / "2026-01-15 Tapaaminen.json").write_text("{}", encoding="utf-8")
    md, js = collision_free_paths(tmp_path, "2026-01-15 Tapaaminen", [".md", ".json"])
    assert md.name == "2026-01-15 Tapaaminen (2).md"
    assert js.name == "2026-01-15 Tapaaminen (2).json"


def test_pair_unit_only_md_taken(tmp_path) -> None:
    (tmp_path / "2026-01-15 Tapaaminen.md").write_text("x", encoding="utf-8")
    md, js = collision_free_paths(tmp_path, "2026-01-15 Tapaaminen", [".md", ".json"])
    assert md.name == "2026-01-15 Tapaaminen (2).md"
    assert js.name == "2026-01-15 Tapaaminen (2).json"


def test_pair_unit_free_base_untouched(tmp_path) -> None:
    md, js = collision_free_paths(tmp_path, "2026-01-15 Tapaaminen", [".md", ".json"])
    assert md.name == "2026-01-15 Tapaaminen.md"
    assert js.name == "2026-01-15 Tapaaminen.json"


def test_pair_unit_existing_suffixes_step_to_next_free(tmp_path) -> None:
    (tmp_path / "2026-01-15 Tapaaminen (2).md").write_text("x", encoding="utf-8")
    (tmp_path / "2026-01-15 Tapaaminen (2).json").write_text("{}", encoding="utf-8")
    (tmp_path / "2026-01-15 Tapaaminen.md").write_text("x", encoding="utf-8")
    (tmp_path / "2026-01-15 Tapaaminen.json").write_text("{}", encoding="utf-8")
    md, js = collision_free_paths(tmp_path, "2026-01-15 Tapaaminen", [".md", ".json"])
    assert md.name == "2026-01-15 Tapaaminen (3).md"
    assert js.name == "2026-01-15 Tapaaminen (3).json"


def test_preset_output_pair_shares_stem(tmp_path, monkeypatch) -> None:
    """The .md/.json pair gets one shared stem even when the .json name
    was taken beforehand (batch-module names are monkeypatched, since
    ``_write_preset_output`` resolves them through the batch namespace)."""
    (tmp_path / "2026-01-15 Kokeilu.json").write_text("{}", encoding="utf-8")

    def fixed_base(title: str, **kw) -> str:
        return "2026-01-15 Kokeilu"

    monkeypatch.setattr("vemoizer.batch_output.dated_basename", fixed_base)
    monkeypatch.setattr(
        "vemoizer.batch_output.collision_free_paths", naming.collision_free_paths
    )
    written = batch._write_preset_output(
        {"notes": {"title": "Kokeilu"}, "text": "x", "segments": []},
        "fallback",
        tmp_path,
    )
    assert len(written) == 2
    stems = {n.rsplit(".", 1)[0] for n in written}
    assert stems == {"2026-01-15 Kokeilu (2)"}


# -- item 6: overflow timeout is fail-open / strict --------------------------


def test_load_config_overflow_timeout_returns_none(tmp_path) -> None:
    cfg = tmp_path / "config.toml"
    cfg.write_text(
        "[llm]\n"
        'base_url = "http://localhost"\n'
        'model = "m"\n'
        'api_key_env = "K"\n'
        "timeout_seconds = 1e400\n",
        encoding="utf-8",
    )
    # The fail-open contract: a bad value is None, never an exception.
    assert llm.load_config(cfg) is None


def test_strict_path_overflow_timeout_is_config_error(tmp_path) -> None:
    cfg = tmp_path / "config.toml"
    cfg.write_text(
        "[llm]\n"
        'base_url = "http://localhost"\n'
        'model = "m"\n'
        'api_key_env = "K"\n'
        "timeout_seconds = 1e400\n",
        encoding="utf-8",
    )
    with pytest.raises(llm.ConfigError):
        llm._strict_load(cfg)


def test_parse_llm_section_overflow_timeout_is_none() -> None:
    section = {
        "base_url": "http://localhost",
        "model": "m",
        "api_key_env": "K",
        "timeout_seconds": 1e400,
    }
    assert llm._parse_llm_section(section) is None
