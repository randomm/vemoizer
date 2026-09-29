"""End-to-end (unmocked ``load_glossary``) tests for ``@``-prefixed glossary
terms (issue #82, M2).

The M2 contract: ``@``-prefixed lines are LLM-only names — they must never
seed the whisper ``initial_prompt`` (that enforcement lives in
``glossary_prompt``), but they DO reach the repair and notes stages via
``load_glossary`` + the ``llm_tail`` boundary (with the ``@`` stripped).
The previous suite (``test_batch_hardening`` item 1) monkeypatched
``pipeline.load_glossary`` to return ``@``-terms, which masked the defect
that ``load_glossary`` itself filtered them out — so in production the
temp file ``run_preset`` writes (bare terms + ``@`` lines + correction
pairs) lost every ``@``-name at the loader.

These tests read the REAL glossary file through the REAL ``load_glossary``;
only the decoders, the LLM client, and the LLM stage functions are mocked.
``HOME`` is monkeypatched and the CWD moved into a tmp dir so the layered
config/glossary search never sees the dev machine's live
``~/.config/vemoizer/config.toml``.
"""

from __future__ import annotations

from test_pipeline import _llm_config, _patch_ingest, _patch_vad

import vemoizer.batch as batch
import vemoizer.pipeline as pipeline
from vemoizer.glossary import glossary_prompt, load_corrections, load_glossary

# -- shared plumbing --------------------------------------------------------


def _clean_env(monkeypatch, tmp_path, *, chdir_to: str | None = None) -> None:
    """Point HOME at tmp (empty of vemoizer config) and chdir into a clean
    tmp root so the layered search starts from scratch."""
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    (tmp_path / "home").mkdir(exist_ok=True)
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


# -- transcribe_file with a REAL glossary file -----------------------------


def test_transcribe_file_real_glossary_at_terms_end_to_end(
    tmp_path, monkeypatch
) -> None:
    """transcribe_file reads the glossary through the REAL load_glossary:
    the whisper prompt holds the bare term and never the @-name; the @-name
    (with the ``@`` stripped) reaches repair AND notes; the correction pair
    still applies."""
    glossary_file = tmp_path / "glossary.txt"
    glossary_file.write_text(
        "@Janni Peltola\nNordea\nBlacksit => Flagship\n", encoding="utf-8"
    )

    seen: dict = {}
    _clean_env(monkeypatch, tmp_path)
    _patch_ingest(monkeypatch)
    _patch_vad(monkeypatch)
    _patch_whisper_meeting(monkeypatch, seen)
    _patch_llm_stage(monkeypatch, seen)
    cfg = _llm_config(tmp_path)

    result = pipeline.transcribe_file(
        "/nonexistent.m4a",
        config_path=str(cfg),
        profile="meeting",
        glossary_path=str(glossary_file),
        repair=True,
    )

    prompt = seen["initial_prompt"] or ""
    # Whisper prompt: bare term in, @-name out (glossary_prompt filter).
    assert "Nordea" in prompt
    assert "Janni Peltola" not in prompt
    # LLM stages: the @-term arrives with the @ stripped, alongside Nordea.
    for glossary in [*seen["notes_calls"], *seen["repair_calls"]]:
        assert glossary == ["Janni Peltola", "Nordea"]
    # Correction pairs are independent of the @-filter and still apply.
    assert load_corrections(glossary_file) == {"Blacksit": "Flagship"}
    # The transcript itself is unaffected.
    assert result["text"]


def test_transcribe_file_real_glossary_at_only_gives_no_prompt(
    tmp_path, monkeypatch
) -> None:
    """An @-only real glossary file leaves the whisper prompt None while
    still delivering the @-stripped name to repair and notes."""
    glossary_file = tmp_path / "glossary.txt"
    glossary_file.write_text("@Janni Peltola\n", encoding="utf-8")

    seen: dict = {}
    _clean_env(monkeypatch, tmp_path)
    _patch_ingest(monkeypatch)
    _patch_vad(monkeypatch)
    _patch_whisper_meeting(monkeypatch, seen)
    _patch_llm_stage(monkeypatch, seen)
    cfg = _llm_config(tmp_path)

    pipeline.transcribe_file(
        "/nonexistent.m4a",
        config_path=str(cfg),
        profile="meeting",
        glossary_path=str(glossary_file),
        repair=True,
    )

    assert seen["initial_prompt"] is None
    for glossary in [*seen["notes_calls"], *seen["repair_calls"]]:
        assert glossary == ["Janni Peltola"]


# -- run_preset meeting: the real layered-glossary seam --------------------


def test_run_preset_meeting_real_layered_at_terms(
    tmp_path, monkeypatch, capsys
) -> None:
    """run_preset(meeting) end to end: the real layered glossary (project
    layer with an @-name) is written to a temp file, that file is read back
    through the REAL load_glossary the way the pipeline does, and the
    prompt terms vs. the LLM term list split correctly. transcribe_file is
    mocked at its seam to capture the glossary_path while the temp file
    still exists (it is deleted only in the batch runner's finally)."""
    proj = tmp_path / "proj"
    proj.mkdir()
    home = tmp_path / "home"
    (home / ".vemoizer").mkdir(parents=True)
    (home / ".vemoizer" / "glossary.txt").write_text("Movescount\n", encoding="utf-8")
    (proj / ".vemoizer").mkdir()
    (proj / ".vemoizer" / "glossary.txt").write_text(
        "@Janni Peltola\nNordea\nBlacksit => Flagship\n", encoding="utf-8"
    )

    captured: dict = {}

    def fake_transcribe(path, **kwargs):
        gp = kwargs.get("glossary_path")
        # The temp file still exists at the time transcribe_file is called.
        assert gp is not None
        captured["terms"] = load_glossary(gp)
        captured["prompt"] = glossary_prompt(captured["terms"])
        captured["corrections"] = load_corrections(gp)
        return {
            "text": "hei maailma",
            "segments": [
                {
                    "start": 0.0,
                    "end": 1.0,
                    "text": "hei maailma",
                    "speaker": "SPEAKER_00",
                }
            ],
            "notes": {"title": "T"},
        }

    monkeypatch.setattr(pipeline, "transcribe_file", fake_transcribe)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.chdir(proj)

    code = batch.run_preset(
        [proj / "a.m4a"],
        command="meeting",
        config_path=None,
        glossary_path=None,
        quiet=True,
    )
    assert code == 0
    # Whisper prompt: bare terms in, @-name out — even though load_glossary
    # now returns the @-line, glossary_prompt is the single enforcement point.
    prompt = captured["prompt"]
    assert prompt is not None
    assert "Nordea" in prompt
    assert "Movescount" in prompt
    assert "Janni Peltola" not in prompt
    # The LLM term list: the @-line is present (with its prefix, stripped
    # later by llm_tail) — the production gap this ticket closes.
    assert "@Janni Peltola" in captured["terms"]
    # Corrections survived the temp-file round trip.
    assert captured["corrections"] == {"Blacksit": "Flagship"}


# -- run_preset memo: @-names must stay out AND prompt must stay empty ------


def test_run_preset_memo_real_layers_no_at_names_no_prompt(
    tmp_path, monkeypatch, capsys
) -> None:
    """run_preset(memo) end to end with an @-name in the layers: the memo
    seam must keep the whisper prompt empty AND must not deliver the
    @-name through the temp file (correction pairs only)."""
    proj = tmp_path / "proj"
    proj.mkdir()
    (proj / ".vemoizer").mkdir()
    (proj / ".vemoizer" / "glossary.txt").write_text(
        "@Janni Peltola\nNordea\nBlacksit => Flagship\n", encoding="utf-8"
    )

    captured: dict = {}

    def fake_transcribe(path, **kwargs):
        gp = kwargs.get("glossary_path")
        from pathlib import Path

        assert gp is not None
        captured["terms"] = load_glossary(gp)
        captured["prompt"] = glossary_prompt(captured["terms"])
        captured["content"] = Path(gp).read_text(encoding="utf-8")
        return {
            "text": "hei maailma",
            "segments": [{"start": 0.0, "end": 1.0, "text": "hei maailma"}],
            "notes": {"title": "T"},
        }

    monkeypatch.setattr(pipeline, "transcribe_file", fake_transcribe)
    _clean_env(monkeypatch, tmp_path, chdir_to=proj)

    code = batch.run_preset(
        [proj / "a.m4a"],
        command="memo",
        config_path=None,
        glossary_path=None,
        quiet=True,
    )
    assert code == 0
    # Memo seam: correction pairs only — no prompt terms at all.
    assert captured["terms"] == []
    assert captured["prompt"] is None
    assert "Janni Peltola" not in captured["content"]
    assert "Nordea" not in captured["content"]
    assert "Blacksit => Flagship" in captured["content"]
