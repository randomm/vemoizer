"""Tests for ``vemoizer doctor`` (issue #79, M7).

Unit tests only: no network, no model downloads.  The doctor's private
check helpers (``ffmpeg_ok``, ``hf_token_present``, ``models_missing``,
``_config_load``) are monkeypatched; the LLM 1-token ping is injected via
the ``ping`` parameter so no real HTTP request is made.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest
from _listen_helpers import FakeClock, fake_hardware, tone, zeros
from typer.testing import CliRunner

from vemoizer import capture_sources, listen
from vemoizer import doctor as doctor_mod
from vemoizer.capture import Chunk
from vemoizer.cli import app
from vemoizer.llm import LLMConfig

runner = CliRunner()


def _mock_env(monkeypatch, *, ff=True, token=True, missing=(), parse_err=None):
    """Patch the four doctor check helpers (config, ping injected separately)."""
    monkeypatch.setattr(doctor_mod, "ffmpeg_ok", lambda: ff)
    monkeypatch.setattr(doctor_mod, "hf_token_present", lambda: token)
    monkeypatch.setattr(doctor_mod, "models_missing", lambda: list(missing))
    monkeypatch.setattr(
        doctor_mod,
        "_config_load",
        lambda: (None, parse_err) if parse_err else (_cfg(), None),
    )


def _cfg() -> LLMConfig:
    return LLMConfig(
        base_url="http://localhost",
        model="m",
        api_key_env="DOCTOR_TEST_KEY",
        timeout_seconds=1.0,
    )


def _run(monkeypatch, *, ping="auto", set_key=True, **env):
    if set_key:
        monkeypatch.setenv("DOCTOR_TEST_KEY", "k")
    else:
        monkeypatch.delenv("DOCTOR_TEST_KEY", raising=False)
    _mock_env(monkeypatch, **env)
    return doctor_mod.run_doctor(ping=ping)


# ---------------------------------------------------------------------------
# Registration
# ---------------------------------------------------------------------------


def test_doctor_help_is_registered() -> None:
    result = runner.invoke(app, ["doctor", "--help"])
    assert result.exit_code == 0
    assert "doctor" in result.stdout.lower()


def test_main_help_lists_doctor() -> None:
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "doctor" in result.stdout


# ---------------------------------------------------------------------------
# Exit semantics (report.ok)
# ---------------------------------------------------------------------------


def test_all_green_exits_zero(monkeypatch) -> None:
    report = _run(monkeypatch, ping=True)
    assert report.ok is True
    assert all(c.status == doctor_mod.GREEN for c in report.checks)


def test_ffmpeg_red_is_not_ok(monkeypatch) -> None:
    report = _run(monkeypatch, ff=False, ping=True)
    assert report.ok is False
    check = next(c for c in report.checks if c.name == "ffmpeg on PATH")
    assert check.status == doctor_mod.RED
    assert check.hint  # hint on each red line


def test_no_token_red_is_not_ok(monkeypatch) -> None:
    report = _run(monkeypatch, token=False, ping=True)
    assert report.ok is False
    check = next(c for c in report.checks if c.name == "HuggingFace token present")
    assert check.status == doctor_mod.RED
    assert "HF_TOKEN" in check.hint or "licence" in check.hint


def test_pyannote_licence_red_is_not_ok(monkeypatch) -> None:
    report = _run(monkeypatch, token=False, ping=True)
    check = next(
        c for c in report.checks if c.name == "pyannote licence (token presence only)"
    )
    assert check.status == doctor_mod.RED
    assert check.hint  # hint to accept the licence form + set HF_TOKEN


def test_missing_model_red_is_not_ok(monkeypatch) -> None:
    report = _run(monkeypatch, missing=("pyannote",), ping=True)
    assert report.ok is False
    check = next(c for c in report.checks if c.name == "model pyannote cached")
    assert check.status == doctor_mod.RED
    assert "models pull" in check.hint


def test_malformed_config_red_is_not_ok(monkeypatch) -> None:
    report = _run(monkeypatch, parse_err="bad [llm] section", ping=True)
    assert report.ok is False
    check = next(c for c in report.checks if c.name == "config parse")
    assert check.status == doctor_mod.RED
    assert "bad [llm] section" in check.hint


def test_no_config_is_green_and_skips_llm_checks(monkeypatch) -> None:
    """Absent config = no LLM = green; key/ping checks omitted."""
    monkeypatch.delenv("DOCTOR_TEST_KEY", raising=False)
    _mock_env(monkeypatch)
    monkeypatch.setattr(doctor_mod, "_config_load", lambda: (None, None))
    report = doctor_mod.run_doctor()
    assert report.ok is True
    names = [c.name for c in report.checks]
    assert any("no LLM configured" in n for n in names)
    assert not any("LLM key" in n for n in names)
    assert not any("LLM ping" in n for n in names)


def test_key_unset_with_config_is_red(monkeypatch) -> None:
    report = _run(monkeypatch, set_key=False)
    assert report.ok is False
    check = next(c for c in report.checks if c.name.startswith("LLM key env var"))
    assert check.status == doctor_mod.RED


def test_key_set_ping_none_is_warning_only(monkeypatch) -> None:
    report = _run(monkeypatch, ping=False)
    ping = next(c for c in report.checks if c.name == "LLM ping (1 token)")
    assert ping.status == doctor_mod.WARN
    assert ping.hint
    # Warning never affects the exit verdict
    assert report.ok is True


def test_key_set_ping_ok_is_green(monkeypatch) -> None:
    report = _run(monkeypatch, ping=True)
    ping = next(c for c in report.checks if c.name == "LLM ping (1 token)")
    assert ping.status == doctor_mod.GREEN
    assert report.ok is True


def test_get_token_raises_is_red_never_crash(monkeypatch) -> None:
    """get_token() may raise (version-dependent) — red with hint, no crash."""

    def _token_check() -> bool:
        try:
            from huggingface_hub import get_token

            token = get_token()
        except Exception:
            return False
        return bool(token)

    monkeypatch.setattr(doctor_mod, "ffmpeg_ok", lambda: True)
    monkeypatch.setattr(doctor_mod, "models_missing", lambda: [])
    monkeypatch.setattr(doctor_mod, "_config_load", lambda: (_cfg(), None))
    monkeypatch.setenv("DOCTOR_TEST_KEY", "k")
    monkeypatch.setattr(doctor_mod, "hf_token_present", _token_check)

    with patch("huggingface_hub.get_token", side_effect=RuntimeError("corrupted")):
        report = doctor_mod.run_doctor(ping=True)

    check = next(c for c in report.checks if c.name == "HuggingFace token present")
    assert check.status == doctor_mod.RED
    assert check.hint
    assert report.ok is False


def test_all_five_models_cached_gives_five_green_lines(monkeypatch) -> None:
    report = _run(monkeypatch, ping=True)
    model_checks = [c for c in report.checks if c.name.startswith("model ")]
    assert len(model_checks) == 5
    assert all(c.status == doctor_mod.GREEN for c in model_checks)


# ---------------------------------------------------------------------------
# CLI wiring (exit code)
# ---------------------------------------------------------------------------


def test_cli_doctor_exit_code_tracks_report(monkeypatch) -> None:
    """The CLI raises Exit(1) when the report is not ok, else exits 0."""
    _mock_env(monkeypatch)
    monkeypatch.setenv("DOCTOR_TEST_KEY", "k")
    # Patch run_doctor in the cli module namespace: it is imported lazily
    # inside the command, so patch the source module attribute.
    from vemoizer import doctor as dmod

    real = dmod.run_doctor

    def _ping_doctor(**kw):
        kw["ping"] = True
        return real(**kw)

    monkeypatch.setattr(dmod, "run_doctor", _ping_doctor)

    # Force a red via ffmpeg
    monkeypatch.setattr(dmod, "ffmpeg_ok", lambda: False)
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 1
    assert "[FAIL]" in result.output


def test_cli_doctor_green_exits_zero(monkeypatch) -> None:
    _mock_env(monkeypatch)
    monkeypatch.setenv("DOCTOR_TEST_KEY", "k")

    from vemoizer import doctor as dmod

    real = dmod.run_doctor

    def _green_doctor(**kw):
        kw["ping"] = True
        return real(**kw)

    monkeypatch.setattr(dmod, "run_doctor", _green_doctor)
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0
    assert "[ ok ]" in result.output


# ---------------------------------------------------------------------------
# doctor --listen (issue #174): the capture probe
# ---------------------------------------------------------------------------

TINK = "/System/Library/Sounds/Tink.aiff"


def _chunk(samples) -> Chunk:
    return Chunk(samples, 0.0)


@pytest.fixture
def capture_hw(monkeypatch):
    """Fake the catap install check and the two started sources (no hardware)."""
    monkeypatch.setattr(listen, "catap_installed", lambda: True)

    def install(mic_chunks, tap_chunks) -> None:
        monkeypatch.setattr(
            capture_sources,
            "_open_real_sources",
            fake_hardware(mic_chunks, tap_chunks),
        )

    return install


def _run_listen(monkeypatch) -> doctor_mod.DoctorReport:
    _mock_env(monkeypatch)
    monkeypatch.setattr(doctor_mod, "_config_load", lambda: (None, None))
    return doctor_mod.run_doctor(ping=True, listen=True)


def _check(report: doctor_mod.DoctorReport, prefix: str) -> doctor_mod.DoctorCheck:
    return next(c for c in report.checks if c.name.startswith(prefix))


def test_doctor_without_listen_flag_never_probes_capture(
    monkeypatch, capture_hw
) -> None:
    def must_not_run(**_kw):
        raise AssertionError("the capture probe must not run without --listen")

    monkeypatch.setattr(listen, "probe_capture", must_not_run)
    report = _run(monkeypatch, ping=True)
    assert report.ok is True
    assert not any(c.name.startswith("microphone") for c in report.checks)


def test_doctor_listen_probe_ok(monkeypatch, capture_hw, system_effects) -> None:
    capture_hw([_chunk(tone(2.5))], [_chunk(tone(2.5))])
    report = _run_listen(monkeypatch)
    assert report.ok is True
    for prefix in ("catap importable", "microphone", "system audio"):
        assert _check(report, prefix).status == doctor_mod.GREEN


def test_doctor_listen_silent_system_fails_with_permission_hint(
    monkeypatch, capture_hw, system_effects
) -> None:
    capture_hw([_chunk(tone(2.5))], [_chunk(zeros(2.5))])
    report = _run_listen(monkeypatch)
    check = _check(report, "system audio")
    assert report.ok is False
    assert check.status == doctor_mod.RED
    assert "System Audio Recording Only" in check.hint
    assert "restart Ghostty" in check.hint
    assert "output device" in check.hint


def test_doctor_listen_silent_mic_fails_with_mic_hint(
    monkeypatch, capture_hw, system_effects
) -> None:
    capture_hw([_chunk(zeros(2.5))], [_chunk(tone(2.5))])
    report = _run_listen(monkeypatch)
    check = _check(report, "microphone")
    assert report.ok is False
    assert check.status == doctor_mod.RED
    assert "Privacy & Security > Microphone" in check.hint


def test_doctor_listen_missing_catap_fails_with_install_hint(
    monkeypatch, system_effects
) -> None:
    monkeypatch.setattr(listen, "catap_installed", lambda: False)
    report = _run_listen(monkeypatch)
    check = _check(report, "catap importable")
    assert report.ok is False
    assert check.status == doctor_mod.RED
    assert "uv sync --extra listen" in check.hint


def test_doctor_listen_plays_tone_through_stubbed_player(
    monkeypatch, capture_hw, system_effects
) -> None:
    capture_hw([_chunk(tone(2.5))], [_chunk(tone(2.5))])
    _run_listen(monkeypatch)
    assert system_effects["afplay"] == [["afplay", TINK]]


def test_probe_mic_that_never_starts_is_reported_not_as_permission() -> None:
    clock = FakeClock()
    hardware = fake_hardware([], [_chunk(tone(2.5))])
    results = listen.probe_capture(hardware=hardware, clock=clock, sleep=clock.sleep)
    mic = next(r for r in results if r.name.startswith("microphone"))
    assert mic.ok is False
    assert "did not start" in mic.hint
    assert "Privacy" not in mic.hint


def test_probe_tap_that_never_starts_names_the_ten_second_window() -> None:
    """issue #205: the cold-start tap gets 10 s, and the probe says so."""
    clock = FakeClock()
    hardware = fake_hardware([_chunk(tone(2.5))], [])
    results = listen.probe_capture(hardware=hardware, clock=clock, sleep=clock.sleep)
    tap = next(r for r in results if r.name.startswith("system audio"))
    assert tap.ok is False
    assert "no audio within 10 s" in tap.hint


def test_cli_doctor_listen_flag_runs_the_probe(monkeypatch, capture_hw) -> None:
    capture_hw([_chunk(tone(2.5))], [_chunk(tone(2.5))])
    _mock_env(monkeypatch)
    monkeypatch.setattr(doctor_mod, "_config_load", lambda: (None, None))
    real = doctor_mod.run_doctor

    def _ping_doctor(**kw):
        kw["ping"] = True
        return real(**kw)

    monkeypatch.setattr(doctor_mod, "run_doctor", _ping_doctor)
    result = runner.invoke(app, ["doctor", "--listen"])
    assert result.exit_code == 0
    assert "system audio" in result.output


def test_probe_devices_that_cannot_open_is_a_red_line() -> None:
    def broken(_mic_name):
        raise RuntimeError("no default input device")

    results = listen.probe_capture(hardware=broken)
    devices = next(r for r in results if r.name == "capture devices")
    assert devices.ok is False
    assert "no default input device" in devices.hint


# ---------------------------------------------------------------------------
# --preprocess loudnorm (issue #135): informational, never fails the report
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("has_filter", [True, False])
def test_doctor_loudnorm_line_is_informational(monkeypatch, has_filter) -> None:
    from vemoizer import preflight

    monkeypatch.setattr(preflight, "loudnorm_filter_ok", lambda: has_filter)
    monkeypatch.setenv("DOCTOR_TEST_KEY", "k")
    _mock_env(monkeypatch)
    lines: list[str] = []
    report = doctor_mod.run_doctor(echo=lines.append, ping=True, preprocess="loudnorm")
    assert report.ok is True
    (line,) = [ln for ln in lines if "loudnorm" in ln]
    assert ("brew upgrade ffmpeg" in line) is not has_filter
