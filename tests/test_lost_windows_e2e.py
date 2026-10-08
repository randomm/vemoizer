"""FIX 3 (issue #152): prove the lost-windows claim end-to-end through the
REAL seams — real ``format_json`` writer, real ``render_report`` /
``format_md`` / ``render_markdown`` — with a fake ``mlx_whisper.transcribe``
only. No mocks of the output layers.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from _whisper_helpers import _speech_audio

from vemoizer.output.formatters import format_json, format_transcript
from vemoizer.render import render_markdown
from vemoizer.report import render_report
from vemoizer.whisper_transcriber import WhisperTranscriber


def _transcribe_with_empty_window(speech: bool = True):
    """Drive WhisperTranscriber.transcribe with a mocked mlx_whisper whose
    first (and only) decode of the 10 s window returns 0 segments.

    Returns (mock, result) with the audio's energy matching *speech* so the
    retry gate (issue #152, FIX 2) takes the intended path.
    """
    import numpy as np

    empty_raw = {"text": "", "language": "fi", "segments": []}
    mock = MagicMock()
    mock.transcribe = MagicMock(return_value=empty_raw)
    audio = _speech_audio(10.0) if speech else np.zeros(160_000, dtype=np.float32)
    with (
        patch.dict("sys.modules", {"mlx_whisper": mock}),
        patch("huggingface_hub.snapshot_download", return_value="/tmp/turbo"),
    ):
        t = WhisperTranscriber(initial_prompt="Sanasto: Flagship.")
        t._model_path = "/tmp/turbo"
        t._mlx_whisper = mock
        result = t.transcribe(audio)
    return mock, result


def _format_md_safe(transcript: dict) -> str:
    """Real format_md via the format_transcript dispatch (the same seam the
    writer uses)."""
    return format_transcript(transcript, "md")


# -- (a) the sidecar JSON carries the key ----------------------------------


def test_sidecar_json_carries_lost_windows_pair() -> None:
    """(a) A lost window survives the real format_json writer: the key is a
    list of [start, end] pairs with correct floats when parsed back."""
    mock, result = _transcribe_with_empty_window()
    assert result.get("lost_windows") == [(0.0, 30.0)]
    assert mock.transcribe.call_count == 2  # initial + prompt-free retry

    out = format_json(result)
    parsed = json.loads(out)
    assert "lost_windows" in parsed
    assert parsed["lost_windows"] == [[0.0, 30.0]]
    for pair in parsed["lost_windows"]:
        assert isinstance(pair, list) and len(pair) == 2
        for v in pair:
            assert isinstance(v, float)
    assert parsed["lost_windows"][0][0] == 0.0
    assert parsed["lost_windows"][0][1] == 30.0


# -- (b) the Markdown/quality report shows the section ----------------------


def test_md_report_shows_lost_windows_with_clock_zero() -> None:
    """(b) The Markdown/quality report carries a visible
    'windows with speech but no text' section with correct clock strings,
    including start 0.0 -> '00:00:00'."""
    _, result = _transcribe_with_empty_window()
    report = render_report(result)
    assert "Häviäkkäiset ikkunat" in report
    assert "puhetta, ei tekstiä" in report
    assert "[00:00:00]" in report
    assert "[00:00:30]" in report
    # The quality report also flows into the md document's <details> block.
    result["quality_report"] = report
    md = _format_md_safe(result)
    assert "Häviäkkäiset ikkunat" in md
    assert "[00:00:00]" in md


def test_md_report_omits_section_when_no_lost_windows() -> None:
    """(b) Empty decode with no lost windows (silent audio): no section,
    no blank block."""
    _, result = _transcribe_with_empty_window(speech=False)
    assert "lost_windows" not in result
    report = render_report(result)
    assert "Häviäkkäiset ikkunat" not in report
    assert "Lost windows" not in report
    result["quality_report"] = report
    md = _format_md_safe(result)
    assert "Häviäkkäiset ikkunat" not in md


# -- (c) the re-render path over a stored sidecar ---------------------------


def test_render_rerender_path_shows_lost_windows(tmp_path: Path) -> None:
    """(c) The re-render path still shows the lost-window section: the real
    render_markdown over the sidecar JSON written by the real writer, with
    the quality report computed the way a run computes it (from the stored
    lost_windows)."""
    _, result = _transcribe_with_empty_window()
    data = json.loads(format_json(result))
    assert "lost_windows" in data  # the sidecar carried it
    # A real run computes the quality report before the md write; the
    # re-render must reproduce it from the stored lost_windows.
    data["quality_report"] = render_report(data)
    rendered = render_markdown(data, corrections={}, speaker_names={})
    assert "Häviäkkäiset ikkunat" in rendered
    assert "[00:00:00]" in rendered
    assert "[00:00:30]" in rendered


def test_render_rerender_empty_has_no_section() -> None:
    """(c) And a clean sidecar renders with no lost-window section."""
    _, result = _transcribe_with_empty_window(speech=False)
    data = json.loads(format_json(result))
    assert "lost_windows" not in data
    rendered = render_markdown(data, corrections={}, speaker_names={})
    assert "Häviäkkäiset ikkunat" not in rendered
    assert "Lost windows" not in rendered


# -- kwargs proof (issue #152, option 1; invariant 3) ----------------------


def test_per_window_kwargs_unchanged_except_hallucination_removed() -> None:
    """The per-window decode kwargs changed EXACTLY one thing vs main:
    hallucination_silence_threshold (2.0) is gone; everything else —
    including ``language`` (invariant 3) — is unchanged."""
    raw = {"text": "", "language": "fi", "segments": []}
    mock = MagicMock()
    mock.transcribe = MagicMock(return_value=raw)
    with (
        patch.dict("sys.modules", {"mlx_whisper": mock}),
        patch("huggingface_hub.snapshot_download", return_value="/tmp/turbo"),
    ):
        t = WhisperTranscriber(initial_prompt="Sanasto: Flagship.")
        t._model_path = "/tmp/turbo"
        t._mlx_whisper = mock
        t.transcribe(_speech_audio(30.0))

    kwargs = mock.transcribe.call_args_list[0].kwargs
    # The removal (issue option 1):
    assert "hallucination_silence_threshold" not in kwargs
    # Everything else identical to main's per-window kwargs:
    assert kwargs["path_or_hf_repo"] == "/tmp/turbo"
    assert kwargs["word_timestamps"] is True
    assert kwargs["task"] == "transcribe"
    assert kwargs["temperature"] == (0.0, 0.2, 0.4)
    assert kwargs["condition_on_previous_text"] is True
    assert kwargs["compression_ratio_threshold"] == 2.4
    assert kwargs["logprob_threshold"] == -1.0
    assert kwargs["no_speech_threshold"] == 0.6
    assert kwargs["initial_prompt"] == "Sanasto: Flagship."
    assert kwargs["verbose"] is False
    # Invariant 3: language handling untouched — None default, per-window
    # detection, never a file-level pin.
    assert kwargs["language"] is None
    # The full kwarg surface, nothing extra:
    assert set(kwargs) == {
        "path_or_hf_repo",
        "word_timestamps",
        "task",
        "temperature",
        "condition_on_previous_text",
        "compression_ratio_threshold",
        "logprob_threshold",
        "no_speech_threshold",
        "initial_prompt",
        "verbose",
        "language",
    }


def test_per_window_kwargs_language_override_pins_every_call() -> None:
    """A language override (issue #108 option B) still pins every window."""
    raw = {"text": "", "language": "fi", "segments": []}
    mock = MagicMock()
    mock.transcribe = MagicMock(return_value=raw)
    with (
        patch.dict("sys.modules", {"mlx_whisper": mock}),
        patch("huggingface_hub.snapshot_download", return_value="/tmp/turbo"),
    ):
        t = WhisperTranscriber(language="fi")
        t._model_path = "/tmp/turbo"
        t._mlx_whisper = mock
        t.transcribe(_speech_audio(60.0))
    for call in mock.transcribe.call_args_list[:2]:
        assert call.kwargs["language"] == "fi"


def test_render_report_list_entries_round_trip_from_json() -> None:
    """The report layer accepts the list form that JSON round-tripping
    produces (tuples become lists); a [start, end] list renders like the
    tuple form."""
    data = {
        "text": "x",
        "language": "fi",
        "segments": [],
        "lost_windows": [[0.0, 30.0], [180.0, 210.0]],
    }
    report = render_report(data)
    assert "Häviäkkäiset ikkunat: 2" in report
    assert "[00:00:00]" in report
    assert "[00:03:00]" in report
