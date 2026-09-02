"""Tests for the ``models pull`` CLI surface (issue #3).

Unit tests only: no downloads, no network, no real HF cache.
``snapshot_download`` is mocked everywhere it could touch the hub, and the
cache-size walk is pointed at a ``tmp_path`` fake cache via ``HF_HOME``.

Covers:
- subcommand registration + help output
- pull: per-model ``snapshot_download`` call with the full-SHA revision
- pull: success report (per-model + total cache line), exit 0
- pull: idempotent warm cache (no re-download, no network)
- pull: failure surfacing — 404 / gated / offline (``HF_HUB_OFFLINE``) /
  incomplete snapshot / generic — with exit 1
- cache size reporting: per-model + total, ``format_size``
- pin guard: every registry entry is a full-SHA, and the transcriber
  constants (parakeet / canary) cannot drift from the registry
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import huggingface_hub
import pytest
from huggingface_hub.errors import (
    GatedRepoError,
    IncompleteSnapshotError,
    OfflineModeIsEnabled,
    RepositoryNotFoundError,
    RevisionNotFoundError,
)
from typer.testing import CliRunner

from vemoizer import models as models_mod
from vemoizer.canary_transcriber import MODEL_ID as CANARY_MODEL_ID
from vemoizer.canary_transcriber import MODEL_REVISION as CANARY_MODEL_REVISION
from vemoizer.cli import app
from vemoizer.parakeet_transcriber import MODEL_ID as PARAKEET_MODEL_ID
from vemoizer.parakeet_transcriber import MODEL_REVISION as PARAKEET_MODEL_REVISION

runner = CliRunner()


def _patch_download(*args, **kwargs):
    """Patch ``huggingface_hub.snapshot_download`` (module attribute).

    ``pull_models`` does ``from huggingface_hub import snapshot_download``
    inside the function body, so patching the module attribute works.
    """
    return patch.object(huggingface_hub, "snapshot_download", *args, **kwargs)


def _make_http_error(status_code: int, message: str):
    """Build a ``HfHubHTTPError`` subclass instance without a live response."""
    from huggingface_hub.errors import HfHubHTTPError

    def _make(cls, code, msg):
        inst = cls.__new__(cls)
        inst.message = msg
        inst.status_code = code
        return inst

    # Most-specific subclass first (GatedRepoError < RepositoryNotFoundError)
    if status_code == 401 and issubclass(GatedRepoError, HfHubHTTPError):
        return _make(GatedRepoError, status_code, message)
    if issubclass(RepositoryNotFoundError, HfHubHTTPError):
        return _make(RepositoryNotFoundError, status_code, message)
    if issubclass(RevisionNotFoundError, HfHubHTTPError):
        return _make(RevisionNotFoundError, status_code, message)
    return _make(HfHubHTTPError, status_code, message)


def _empty_sizes() -> dict[str, int]:
    return {"decode-a": 0, "decode-b": 0, "redecode": 0}


# ---------------------------------------------------------------------------
# Subcommand registration
# ---------------------------------------------------------------------------


def test_models_help_lists_pull() -> None:
    result = runner.invoke(app, ["models", "--help"])
    assert result.exit_code == 0
    assert "pull" in result.stdout


def test_models_pull_help_describes_download() -> None:
    result = runner.invoke(app, ["models", "pull", "--help"])
    assert result.exit_code == 0
    assert "Pre-download" in result.stdout


# ---------------------------------------------------------------------------
# Pull: pinned snapshot_download calls
# ---------------------------------------------------------------------------


def test_pull_calls_snapshot_download_per_model_with_full_sha() -> None:
    """Every model is pulled with its full-SHA revision as a kwarg."""
    calls: list[tuple[str, str | None]] = []

    def fake_download(repo_id, revision=None):
        calls.append((repo_id, revision))
        return f"/fake/cache/{repo_id}"

    with (
        _patch_download(side_effect=fake_download),
        patch.object(models_mod, "cache_size", return_value=_empty_sizes()),
    ):
        result = runner.invoke(app, ["models", "pull"])

    assert result.exit_code == 0
    assert len(calls) == 3
    for (repo_id, revision), spec in zip(calls, models_mod.MODELS, strict=True):
        assert repo_id == spec.repo_id
        assert revision == spec.revision
        assert len(revision) == 40
        assert all(c in "0123456789abcdef" for c in revision)


# ---------------------------------------------------------------------------
# Pull: success report
# ---------------------------------------------------------------------------


def test_pull_success_reports_all_models_and_cache_sizes() -> None:
    fake_cache = {spec.name: 1024 * 1024 * 100 for spec in models_mod.MODELS}

    with (
        _patch_download(return_value="/fake/cache"),
        patch.object(models_mod, "cache_size", return_value=fake_cache),
    ):
        result = runner.invoke(app, ["models", "pull"])

    assert result.exit_code == 0
    for spec in models_mod.MODELS:
        assert spec.name in result.stdout
        assert spec.repo_id in result.stdout
    assert "pulled" in result.stdout
    assert "cache:" in result.stdout
    assert "total" in result.stdout
    assert "100.0 MiB" in result.stdout
    assert "300.0 MiB" in result.stdout


# ---------------------------------------------------------------------------
# Pull: idempotent warm cache
# ---------------------------------------------------------------------------


def test_pull_warm_cache_no_redownload() -> None:
    with (
        _patch_download(return_value="/fake/cache") as mock_dl,
        patch.object(models_mod, "cache_size", return_value=_empty_sizes()),
    ):
        result = runner.invoke(app, ["models", "pull"])

    assert result.exit_code == 0
    assert mock_dl.call_count == 3


# ---------------------------------------------------------------------------
# Pull: failure surfacing
# ---------------------------------------------------------------------------


def test_pull_404_surfaces_repository_not_found() -> None:
    error = _make_http_error(404, "Repository Not Found: something/bad")
    with (
        _patch_download(side_effect=error),
        patch.object(models_mod, "cache_size", return_value=_empty_sizes()),
    ):
        result = runner.invoke(app, ["models", "pull"])

    assert result.exit_code == 1
    assert "FAILED" in result.stdout
    assert "not found" in result.stdout.lower()


def test_pull_gated_repo_surfaces_token_hint() -> None:
    error = _make_http_error(401, "gated repo")
    with (
        _patch_download(side_effect=error),
        patch.object(models_mod, "cache_size", return_value=_empty_sizes()),
    ):
        result = runner.invoke(app, ["models", "pull"])

    assert result.exit_code == 1
    assert "gated" in result.stdout.lower()
    assert "HF_TOKEN" in result.stdout


def test_pull_offline_surfaces_hf_hub_offline() -> None:
    error = OfflineModeIsEnabled("Cannot access the repo while offline")
    with (
        _patch_download(side_effect=error),
        patch.object(models_mod, "cache_size", return_value=_empty_sizes()),
    ):
        result = runner.invoke(app, ["models", "pull"])

    assert result.exit_code == 1
    assert "HF_HUB_OFFLINE" in result.stdout


def test_pull_incomplete_snapshot_surfaces_hint() -> None:
    error = IncompleteSnapshotError("repo", "/snapshot/path")
    with (
        _patch_download(side_effect=error),
        patch.object(models_mod, "cache_size", return_value=_empty_sizes()),
    ):
        result = runner.invoke(app, ["models", "pull"])

    assert result.exit_code == 1
    assert "incomplete" in result.stdout.lower()


def test_pull_generic_error_not_raw() -> None:
    error = RuntimeError("secret-internal-traceback")
    with (
        _patch_download(side_effect=error),
        patch.object(models_mod, "cache_size", return_value=_empty_sizes()),
    ):
        result = runner.invoke(app, ["models", "pull"])

    assert result.exit_code == 1
    assert "RuntimeError" in result.stdout
    assert "secret-internal-traceback" not in result.stdout


def test_pull_partial_failure_continues_to_remaining_models() -> None:
    call_count = {"n": 0}

    def flaky_download(repo_id, revision=None):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise _make_http_error(404, "boom")
        return f"/cache/{repo_id}"

    with (
        _patch_download(side_effect=flaky_download) as mock_dl,
        patch.object(models_mod, "cache_size", return_value=_empty_sizes()),
    ):
        result = runner.invoke(app, ["models", "pull"])

    assert mock_dl.call_count == 3
    assert result.exit_code == 1
    assert "FAILED" in result.stdout


# ---------------------------------------------------------------------------
# Cache size reporting
# ---------------------------------------------------------------------------


def test_cache_size_walks_tmp_cache(monkeypatch, tmp_path: Path) -> None:
    fake_home = tmp_path / "hf-home"
    fake_cache = fake_home / "hub"
    fake_cache.mkdir(parents=True)
    # Use the library's own naming so the fixture matches what
    # snapshot_download writes (the `models--` prefix is the exact bug
    # that this test must catch).
    from huggingface_hub.file_download import repo_folder_name

    for repo in ("org/model-a", "org/model-b"):
        model_dir = fake_cache / repo_folder_name(repo_id=repo, repo_type="model")
        model_dir.mkdir()
        (model_dir / "weights.bin").write_bytes(b"x" * 1024)

    monkeypatch.setenv("HF_HOME", str(fake_home))

    specs = (
        models_mod.ModelSpec("a", "org/model-a", "a" * 40),
        models_mod.ModelSpec("b", "org/model-b", "b" * 40),
    )
    sizes = models_mod.cache_size(specs)
    assert sizes == {"a": 1024, "b": 1024}


def test_cache_size_missing_dir_reports_zero(monkeypatch, tmp_path: Path) -> None:
    fake_home = tmp_path / "empty-home"
    fake_home.mkdir()
    monkeypatch.setenv("HF_HOME", str(fake_home))

    specs = (models_mod.ModelSpec("x", "org/missing", "c" * 40),)
    sizes = models_mod.cache_size(specs)
    assert sizes == {"x": 0}


def test_format_size_units() -> None:
    assert models_mod.format_size(0) == "0 B"
    assert models_mod.format_size(100) == "100 B"
    assert models_mod.format_size(1024) == "1.0 KiB"
    assert models_mod.format_size(1024 * 1024) == "1.0 MiB"
    assert models_mod.format_size(1024**3) == "1.0 GiB"


def test_format_size_large_values() -> None:
    assert models_mod.format_size(1024**3 * 5 + 1024**3 // 2) == "5.5 GiB"


# ---------------------------------------------------------------------------
# Pin guard
# ---------------------------------------------------------------------------


def test_all_revisions_are_full_sha() -> None:
    models_mod.assert_all_revisions_pinned()


def test_reject_non_full_sha_revision() -> None:
    bad = (models_mod.ModelSpec("bad", "org/bad", "main"),)
    with pytest.raises(ValueError, match="not pinned to a full-SHA"):
        models_mod.assert_all_revisions_pinned(bad)


def test_reject_short_sha_revision() -> None:
    bad = (models_mod.ModelSpec("short", "org/short", "abcd1234"),)
    with pytest.raises(ValueError, match="not pinned to a full-SHA"):
        models_mod.assert_all_revisions_pinned(bad)


def test_parakeet_constants_match_registry() -> None:
    registry = {s.name: s for s in models_mod.MODELS}["decode-a"]
    assert registry.repo_id == PARAKEET_MODEL_ID
    assert registry.revision == PARAKEET_MODEL_REVISION


def test_canary_constants_match_registry() -> None:
    registry = {s.name: s for s in models_mod.MODELS}["decode-b"]
    assert registry.repo_id == CANARY_MODEL_ID
    assert registry.revision == CANARY_MODEL_REVISION


# ---------------------------------------------------------------------------
# render_pull_report edge cases
# ---------------------------------------------------------------------------


def test_render_pull_report_no_sizes() -> None:
    spec = models_mod.MODELS[0]
    result = models_mod.PulledModel(spec, "/cache", None, 1.5)
    report = models_mod.render_pull_report([result], None)
    assert "pulled" in report
    assert "cache:" not in report


def test_render_pull_report_failure_line() -> None:
    spec = models_mod.MODELS[0]
    result = models_mod.PulledModel(spec, None, "some error", 0.1)
    report = models_mod.render_pull_report([result], {"decode-a": 0})
    assert "FAILED" in report
    assert "some error" in report


# ---------------------------------------------------------------------------
# Existing transcribe tests stay green (smoke check)
# ---------------------------------------------------------------------------


def test_transcribe_placeholder_still_works() -> None:
    result = runner.invoke(app, ["transcribe", "memo.m4a"])
    assert result.exit_code == 1
    assert "not implemented" in result.stderr
