"""Type-guarded ``format_json`` mirrors for ``duration_s`` / ``glossary_source``
(issue #107, fix pass 4, LENS MEDIUM #1) and the render-side tolerance for a
hand-edited sidecar that carries a wrong-typed value.

``format_json`` persists the two M6 header keys into the sidecar JSON
present-only, matching the guard style of the neighbouring ``speaker_names``
mirror (``isinstance`` + non-empty): ``duration_s`` only when it is an
int/float (not a bool, not NaN/inf), ``glossary_source`` only when it is a
non-empty ``str``. A string ``duration_s`` (e.g. ``'9.15'``) must NOT be
mirrored — writing it would make a later ``render`` raise ``TypeError`` in
the header formatter instead of failing cleanly.

The render side already tolerates wrong types (``format_md``'s header checks
``isinstance``); these tests pin that a hand-edited/old sidecar with
``duration_s = '9.15'`` or a numeric ``glossary_source`` renders without a
crash and without the corresponding header line, and that the exit codes
stay 0.

Pure-stdlib + CliRunner: no model imports, no network, no ffmpeg.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from _cli_helpers import isolate_home
from typer.testing import CliRunner

from vemoizer.cli import app
from vemoizer.output.formatters import format_json

runner = CliRunner()

INF = float("inf")
NAN = float("nan")


def _parse(s: str) -> dict[str, Any]:
    return json.loads(s)


# ---------------------------------------------------------------------------
# format_json: duration_s mirror
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value",
    ["9.15", True, False, NAN, INF, -INF],
    ids=["string", "true", "false", "nan", "inf", "-inf"],
)
def test_format_json_omits_bad_duration_s(value: Any) -> None:
    """A non-numeric (or non-finite) duration_s is not mirrored."""
    out = _parse(format_json({"text": "hei", "duration_s": value}))
    assert "duration_s" not in out, f"mirrored {value!r}"


@pytest.mark.parametrize(
    "value",
    [9.15, 0, 0.0, 3600],
    ids=["float", "zero-int", "zero-float", "int-hours"],
)
def test_format_json_mirrors_numeric_duration_s(value: float) -> None:
    """A numeric (int/float, not bool) duration_s IS mirrored — a zero
    duration is numeric, so it is kept (present-only means None-only
    skipped)."""
    out = _parse(format_json({"text": "hei", "duration_s": value}))
    assert out.get("duration_s") == value


def test_format_json_omits_none_duration_s() -> None:
    """An explicit None duration_s is not mirrored (present-only)."""
    out = _parse(format_json({"text": "hei", "duration_s": None}))
    assert "duration_s" not in out


def test_format_json_omits_absent_duration_s() -> None:
    out = _parse(format_json({"text": "hei"}))
    assert "duration_s" not in out


# ---------------------------------------------------------------------------
# format_json: glossary_source mirror
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value",
    ["", 5, True, ["x"], NAN],
    ids=["empty-str", "int", "true", "list", "nan"],
)
def test_format_json_omits_bad_glossary_source(value: Any) -> None:
    """A non-str or empty-string glossary_source is not mirrored."""
    out = _parse(format_json({"text": "hei", "glossary_source": value}))
    assert "glossary_source" not in out, f"mirrored {value!r}"


def test_format_json_mirrors_non_empty_glossary_source() -> None:
    out = _parse(format_json({"text": "hei", "glossary_source": "x"}))
    assert out.get("glossary_source") == "x"


# ---------------------------------------------------------------------------
# render: a hand-edited sidecar with a wrong-typed header value
# ---------------------------------------------------------------------------


def _sidecar_path(tmp_path: Path, **extra: Any) -> Path:
    base: dict[str, Any] = {
        "text": "Puhuttiin Blacksit-hankkeesta.",
        "notes": {"title": "Alustus"},
        **extra,
    }
    path = tmp_path / "sidecar.json"
    path.write_text(json.dumps(base, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def test_render_sidecar_string_duration_s_no_crash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A hand-edited sidecar with ``duration_s = '9.15'`` renders without a
    crash and without a Kesto line (the render side tolerates the wrong
    type; exit code unchanged)."""
    isolate_home(monkeypatch, tmp_path, tmp_path)
    sc = _sidecar_path(tmp_path, duration_s="9.15")

    result = runner.invoke(app, ["render", str(sc)])
    assert result.exit_code == 0, result.stderr

    md = (tmp_path / "sidecar.md").read_text(encoding="utf-8")
    assert "Kesto" not in md
    assert "# Alustus" in md


def test_render_sidecar_wrong_typed_glossary_source_no_crash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A hand-edited sidecar with a NON-string ``glossary_source`` (a dict,
    the type ``str`` cannot coerce without ``AttributeError``) renders
    without a crash and without a Sanasto line — the render side coerces
    to a non-string sentinel instead of the raw value."""
    isolate_home(monkeypatch, tmp_path, tmp_path)
    sc = _sidecar_path(tmp_path, glossary_source={"not": "a string"})

    result = runner.invoke(app, ["render", str(sc)])
    assert result.exit_code == 0, result.stderr

    md = (tmp_path / "sidecar.md").read_text(encoding="utf-8")
    assert "Sanasto" not in md
    assert "# Alustus" in md


def test_render_sidecar_valid_header_lines_still_present(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The tolerance does not suppress a VALID header line: a sidecar with a
    numeric duration_s and a non-empty str glossary_source still renders the
    Kesto and Sanasto lines."""
    isolate_home(monkeypatch, tmp_path, tmp_path)
    sc = _sidecar_path(tmp_path, duration_s=9.15, glossary_source="g.txt (1)")

    result = runner.invoke(app, ["render", str(sc)])
    assert result.exit_code == 0, result.stderr

    md = (tmp_path / "sidecar.md").read_text(encoding="utf-8")
    assert "Kesto: [00:00:09]" in md
    assert "Sanasto: g.txt (1)" in md
