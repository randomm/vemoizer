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

from vemoizer import model_pull_report
from vemoizer import models as models_mod
from vemoizer.canary_transcriber import MODEL_ID as CANARY_MODEL_ID
from vemoizer.canary_transcriber import MODEL_REVISION as CANARY_MODEL_REVISION
from vemoizer.cli import app
from vemoizer.diarization import DIARIZATION_REPO_ID, DIARIZATION_REVISION
from vemoizer.parakeet_transcriber import MODEL_ID as PARAKEET_MODEL_ID
from vemoizer.parakeet_transcriber import MODEL_REVISION as PARAKEET_MODEL_REVISION
from vemoizer.redecode import MODEL_ID as REDECODE_MODEL_ID
from vemoizer.redecode import MODEL_REVISION as REDECODE_MODEL_REVISION
from vemoizer.whisper_transcriber import MODEL_ID as WHISPER_TURBO_MODEL_ID
from vemoizer.whisper_transcriber import MODEL_REVISION as WHISPER_TURBO_MODEL_REVISION

runner = CliRunner()


def _patch_download(*args, **kwargs):
    """Patch ``huggingface_hub.snapshot_download`` (module attribute).

    ``pull_models`` does ``from huggingface_hub import snapshot_download``
    inside the function body, so patching the module attribute works.
    """
    return patch.object(huggingface_hub, "snapshot_download", *args, **kwargs)


def _make_http_error(
    status_code: int, message: str, *, cls: type[Exception] | None = None
):
    """Build a ``HfHubHTTPError`` subclass instance without a live response.

    *cls* selects the exception class explicitly. Omitting it keeps the
    long-standing resolution order (most-specific subclass first), which for
    every status code the suite had used lands on ``GatedRepoError`` or
    ``RepositoryNotFoundError`` — so a test that needs a *generic*
    ``HfHubHTTPError`` or a ``RevisionNotFoundError`` has to name the class,
    otherwise an earlier ``isinstance`` branch swallows it and the branch
    under test never runs.
    """
    from huggingface_hub.errors import HfHubHTTPError

    def _make(klass, code, msg):
        inst = klass.__new__(klass)
        inst.message = msg
        inst.status_code = code
        return inst

    if cls is not None:
        return _make(cls, status_code, message)
    # Most-specific subclass first (GatedRepoError < RepositoryNotFoundError)
    if status_code == 401 and issubclass(GatedRepoError, HfHubHTTPError):
        return _make(GatedRepoError, status_code, message)
    if issubclass(RepositoryNotFoundError, HfHubHTTPError):
        return _make(RepositoryNotFoundError, status_code, message)
    if issubclass(RevisionNotFoundError, HfHubHTTPError):
        return _make(RevisionNotFoundError, status_code, message)
    return _make(HfHubHTTPError, status_code, message)


def _empty_sizes() -> dict[str, int]:
    return {
        "parakeet": 0,
        "canary": 0,
        "whisper-finnish": 0,
        "whisper-turbo": 0,
        "pyannote": 0,
    }


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

    def fake_download(repo_id, revision=None, **_kwargs):
        calls.append((repo_id, revision))
        return f"/fake/cache/{repo_id}"

    with (
        _patch_download(side_effect=fake_download),
        patch.object(models_mod, "cache_size", return_value=_empty_sizes()),
    ):
        result = runner.invoke(app, ["models", "pull"])

    assert result.exit_code == 0
    # The conservative probe no longer calls snapshot_download: exactly one
    # real call per model.
    assert len(calls) == len(models_mod.MODELS)
    for i, spec in enumerate(models_mod.MODELS):
        real_repo, real_rev = calls[i]
        assert real_repo == spec.repo_id
        assert real_rev == spec.revision
        assert len(real_rev) == 40
        assert all(c in "0123456789abcdef" for c in real_rev)


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
        # Each model line carries its pipeline stage (issue #67)
        stage = models_mod.STAGES[spec.name]
        assert f"[{stage}]" in result.stdout
    assert "pulled" in result.stdout
    assert "cache:" in result.stdout
    assert "total" in result.stdout
    assert "100.0 MiB" in result.stdout
    assert "500.0 MiB" in result.stdout


def test_pull_report_includes_expected_size() -> None:
    """Each model line shows the expected on-disk size (issue #67)."""
    with (
        _patch_download(return_value="/fake/cache"),
        patch.object(models_mod, "cache_size", return_value=_empty_sizes()),
    ):
        result = runner.invoke(app, ["models", "pull"])

    assert result.exit_code == 0
    for spec in models_mod.MODELS:
        assert models_mod.expected_size_str(spec.name) in result.stdout


def test_render_pull_report_stages_and_sizes() -> None:
    """Unit test: render_pull_report includes stage and expected size per line."""
    for spec in models_mod.MODELS:
        result = models_mod.PulledModel(spec, "/cache", None, 1.0)
        report = models_mod.render_pull_report([result], None)
        stage = models_mod.STAGES[spec.name]
        assert f"[{stage}]" in report
        # Expected size is present (format varies by magnitude)
        assert "~" in report


# ---------------------------------------------------------------------------
# Pull: idempotent warm cache
# ---------------------------------------------------------------------------


def test_pull_warm_cache_no_redownload(
    tmp_path: Path, _hermetic_hf_cache: Path
) -> None:
    """A fully warm cache (laid out under the hermetic cache the probe
    actually reads, i.e. ``huggingface_hub.constants.HF_HUB_CACHE``) takes
    the SILENT warm path: exactly one ``snapshot_download`` call per model,
    and every call runs with HF's progress-bar switch OFF. Before the fix
    this test passed vacuously — the snapshots were laid out in the test's
    own ``tmp_path`` while the probe read the hermetic empty cache, so the
    cold path ran and still showed 5 calls. The ``seen == [True]*5``
    assertion is the proof the warm path was taken."""
    from huggingface_hub import utils as hf_utils

    seen: list[bool] = []

    def fake_download(repo_id, revision=None, **_kwargs):
        seen.append(hf_utils.are_progress_bars_disabled())
        return f"/fake/cache/{repo_id}"

    for spec in models_mod.MODELS:
        _make_cache_in(_hermetic_hf_cache, spec.repo_id, spec.revision)
    with (
        _patch_download(side_effect=fake_download) as mock_dl,
        patch.object(models_mod, "cache_size", return_value=_empty_sizes()),
    ):
        result = runner.invoke(app, ["models", "pull"])

    assert result.exit_code == 0
    # The conservative probe no longer calls snapshot_download: exactly one
    # (silent) real call per model.
    assert mock_dl.call_count == 5
    # Proof the warm path was taken: every call ran bars-disabled.
    assert seen == [True] * 5, (
        "warm-cache pull must run every download with HF's progress bars "
        f"disabled (the probe read the hermetic cache); saw disabled-states {seen}"
    )
    assert not hf_utils.are_progress_bars_disabled()  # no leaked disable


def _make_cache_in(cache: Path, repo_id: str, revision: str) -> None:
    from huggingface_hub.file_download import repo_folder_name

    from vemoizer import model_cache as _mc

    folder = cache / repo_folder_name(repo_id=repo_id, repo_type="model")
    snap = folder / "snapshots" / revision
    snap.mkdir(parents=True)
    for pattern in _mc._expected_weights_for(repo_id):
        weight = snap / pattern.replace("*", "weights")
        weight.parent.mkdir(parents=True, exist_ok=True)
        weight.write_bytes(b"0" * 16)


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


def test_pull_partial_failure_continues_to_remaining_models(
    tmp_path: Path, _hermetic_hf_cache: Path
) -> None:
    """A per-model failure (404 on the first model) must not stop the pull:
    all five models still get exactly one download call, exit 1, and the
    models whose snapshot was laid out complete under the hermetic cache
    were silenced (warm path) while the failing model was NOT — proof the
    probe read the hermetic cache, not the test's own tmp_path."""
    from huggingface_hub import utils as hf_utils

    # Layout: the first model (parakeet) is NOT in the cache (it fails with
    # a 404); the other four have complete snapshots in the hermetic cache.
    first_spec = models_mod.MODELS[0]
    for spec in models_mod.MODELS[1:]:
        _make_cache_in(_hermetic_hf_cache, spec.repo_id, spec.revision)

    seen: dict[str, bool] = {}

    def flaky_download(repo_id, revision=None):
        seen[repo_id] = hf_utils.are_progress_bars_disabled()
        if repo_id == first_spec.repo_id:
            raise _make_http_error(404, "boom")
        return f"/cache/{repo_id}"

    with (
        _patch_download(side_effect=flaky_download) as mock_dl,
        patch.object(models_mod, "cache_size", return_value=_empty_sizes()),
    ):
        result = runner.invoke(app, ["models", "pull"])

    # The conservative probe no longer calls snapshot_download: exactly one
    # real call per model, and the first model's failure did not stop the
    # remaining four.
    assert mock_dl.call_count == 5
    assert result.exit_code == 1
    assert "FAILED" in result.stdout
    # Warm-path proof: the four laid-out models ran bars-disabled; the
    # failing model (cold + error) did not.
    assert seen[first_spec.repo_id] is False
    for spec in models_mod.MODELS[1:]:
        assert seen[spec.repo_id] is True
    assert not hf_utils.are_progress_bars_disabled()  # no leaked disable


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
    assert model_pull_report.format_size(0) == "0 B"
    assert model_pull_report.format_size(100) == "100 B"
    assert model_pull_report.format_size(1024) == "1.0 KiB"
    assert model_pull_report.format_size(1024 * 1024) == "1.0 MiB"
    assert model_pull_report.format_size(1024**3) == "1.0 GiB"


def test_format_size_large_values() -> None:
    assert model_pull_report.format_size(1024**3 * 5 + 1024**3 // 2) == "5.5 GiB"


def test_report_helpers_reachable_from_the_registry_module() -> None:
    """``cli.py`` imports the report helpers via :mod:`vemoizer.models`
    (``from vemoizer.models import ... render_pull_report``), so those stay
    re-exported. ``format_size`` is deliberately NOT one of them: nothing
    outside this module's own report renders byte counts, so it lives only at
    its definition (issue #151 lens, SIMPLICITY — no public surface for a
    name no consumer looks up).
    """
    for name in (
        "STAGES",
        "_EXPECTED_SIZE_GIB",
        "expected_size_str",
        "render_pull_report",
        "format_pull_error",
        "_describe_hf_error",
    ):
        assert getattr(models_mod, name) is getattr(model_pull_report, name), name
    assert not hasattr(models_mod, "format_size"), (
        "format_size is not part of the registry's contract; it belongs to "
        "vemoizer.model_pull_report, where its only caller lives"
    )


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
    registry = {s.name: s for s in models_mod.MODELS}["parakeet"]
    assert registry.repo_id == PARAKEET_MODEL_ID
    assert registry.revision == PARAKEET_MODEL_REVISION


def test_canary_constants_match_registry() -> None:
    registry = {s.name: s for s in models_mod.MODELS}["canary"]
    assert registry.repo_id == CANARY_MODEL_ID
    assert registry.revision == CANARY_MODEL_REVISION


def test_whisper_turbo_constants_match_registry() -> None:
    """whisper_transcriber reads whisper-turbo from the registry (issue #79)."""
    registry = {s.name: s for s in models_mod.MODELS}["whisper-turbo"]
    assert registry.repo_id == WHISPER_TURBO_MODEL_ID
    assert registry.revision == WHISPER_TURBO_MODEL_REVISION


def test_whisper_finnish_redecode_constants_match_registry() -> None:
    """redecode reads whisper-finnish from the registry (issue #79)."""
    registry = {s.name: s for s in models_mod.MODELS}["whisper-finnish"]
    assert registry.repo_id == REDECODE_MODEL_ID
    assert registry.revision == REDECODE_MODEL_REVISION


def test_diarization_constants_match_registry() -> None:
    """diarization reads pyannote from the registry (issue #79)."""
    registry = {s.name: s for s in models_mod.MODELS}["pyannote"]
    assert registry.repo_id == DIARIZATION_REPO_ID
    assert registry.revision == DIARIZATION_REVISION


def test_stage_and_size_maps_cover_registry() -> None:
    """STAGES and _EXPECTED_SIZE_GIB stay in sync with the MODELS registry.

    Pinned at the module that *defines* the constants, so a rename or a
    deleted entry fails here rather than being hidden by the ``models.py``
    re-export shim (issue #151 lens, ARCHITECTURE).
    """
    names = {s.name for s in models_mod.MODELS}
    assert set(model_pull_report.STAGES) == names
    assert set(model_pull_report._EXPECTED_SIZE_GIB) == names
    # The re-export surface the CLI and the rest of the suite import from.
    assert models_mod.STAGES is model_pull_report.STAGES
    assert models_mod._EXPECTED_SIZE_GIB is model_pull_report._EXPECTED_SIZE_GIB


def test_readme_does_not_restate_canonical_size_total() -> None:
    """The README must point to the manifest table, not hard-code a total.

    The ``models pull`` report computes on-disk sizes from
    ``_EXPECTED_SIZE_GIB`` in the same ``models.py``. A hard-coded total
    (e.g. ``~8 GB``) in the README would drift as any model revision is
    bumped and contradict the spec's canonical table. See AGENTS.md
    "200-PR test": point to the canonical source, never restate it.
    """
    readme = (Path(__file__).resolve().parents[1] / "README.md").read_text(
        encoding="utf-8"
    )
    # The README may mention individual per-model sizes (fine), but must not
    # restate the *aggregate* total in a form that could drift from the
    # registry. The only legitimate total is in the spec's manifest paragraph
    # ("total roughly 7.8 GiB"); the README should not duplicate that.
    import re

    # Find any "~<num> GB" pattern (not GiB — the spec uses GiB).
    # The spec manifest uses GiB throughout; the README previously said
    # "~8 GB" (two places) which contradicted the spec's 7.8 GiB.
    matches = re.findall(r"~\d+(?:\.\d+)?\s+GB\b", readme)
    assert not matches, (
        f"README hard-codes a total GB figure {matches}; "
        "point to the Model manifest table in docs/pipeline-spec.md instead."
    )


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
    report = models_mod.render_pull_report([result], {"parakeet": 0})
    assert "FAILED" in report
    assert "some error" in report
    # Stage is present even on failure lines
    assert "[decode A]" in report


# ---------------------------------------------------------------------------
# Credential redaction on the stdout path (issue #151 review, SECURITY)
#
# ``models pull`` echoes library-owned exception text to the terminal. The
# run log scrubs ``hf_…``/``Bearer …`` via ``run_log_io.RedactingFormatter``,
# but that formatter only ever sees log records — stdout has no formatter, so
# the formatters themselves must scrub. Break-and-fail holds per branch: each
# of the four text-echoing sites below (``RepositoryNotFoundError``,
# ``RevisionNotFoundError``, generic ``HfHubHTTPError``, ``GatedRepoError``)
# has its own case, and removing that site's ``_scrub_credentials`` call fails
# that case.
# ---------------------------------------------------------------------------

_TOKEN = "hf_" + "T" * 34  # the HF token shape run_log_io matches


def test_describe_hf_error_scrubs_token_from_hub_message() -> None:
    """A token inside a hub-owned ``message`` never reaches the report.

    ``_describe_hf_error`` interpolates ``getattr(exc, 'message', ...)`` for
    ``RepositoryNotFoundError``.
    """
    exc = _make_http_error(404, f"Repository not found for token {_TOKEN}")
    report = models_mod._describe_hf_error(exc)
    assert _TOKEN not in report
    assert "hf_<redacted>" in report


def test_describe_hf_error_scrubs_token_from_revision_not_found() -> None:
    """The ``RevisionNotFoundError`` branch scrubs too (issue #151 lens).

    ``_make_http_error`` resolves 404 to ``RepositoryNotFoundError``, so this
    branch was never reached: removing its ``_scrub_credentials`` call kept
    the whole suite green. The class is named explicitly to land on it.
    Break-and-fail: deleting the call on that branch fails this test.
    """
    exc = _make_http_error(404, f"Revision {_TOKEN} missing", cls=RevisionNotFoundError)
    report = models_mod._describe_hf_error(exc)
    assert "pinned revision not found" in report
    assert _TOKEN not in report
    assert "hf_<redacted>" in report


def test_describe_hf_error_scrubs_token_from_generic_http_error() -> None:
    """The generic ``HfHubHTTPError`` branch scrubs — the branch that most
    needs it (issue #151 lens, SECURITY).

    ``hf_raise_for_status`` builds ``message`` from ``response.text`` and
    echoes ``response.url``, which under the query-param auth form carries
    the token. Every existing case resolved to a more specific subclass, so
    this ``isinstance`` arm ran with no test at all: mutating it back to the
    unscrubbed ``getattr(exc, "message", str(exc))`` left the full suite
    green (verified: 2273 passed). Build a real 429 through the constructor
    so the branch is reached with a response attached, the shape the hub
    actually raises.
    """
    import httpx
    from huggingface_hub.errors import HfHubHTTPError

    request = httpx.Request(
        "GET", "https://huggingface.co/api/models/x?token=Bearer%20query-param-secret"
    )
    response = httpx.Response(429, request=request, text="rate limited")
    exc = HfHubHTTPError(
        f"429 Too Many Requests: token {_TOKEN} in the hub's own body text",
        response=response,
    )
    # No ``status_code`` is assigned: the real constructor does not set one and
    # production reads it with ``getattr(exc, "status_code", None)``, so this
    # case pins the response-carrying shape that actually reaches the branch.
    report = models_mod._describe_hf_error(exc)
    assert "model download failed" in report, report
    assert _TOKEN not in report
    assert "hf_<redacted>" in report
    # the constructor-built exception really does reach the generic arm: an
    # earlier isinstance() would have returned one of the fixed strings
    assert "failed to download model" not in report


def test_format_pull_error_gated_scrubs_bearer_header() -> None:
    """``GatedRepoError`` is interpolated bare (``f"{exc}"``) — the one branch
    that echoes the whole exception, so it is the one that must scrub.

    Built through the real constructor (``str(exc)`` is ``args[0]``, which the
    no-response ``_make_http_error`` helper does not populate). The hub's own
    gated failure text carries the request ``Authorization`` header.
    """
    import httpx

    request = httpx.Request(
        "GET",
        "https://huggingface.co/api/models/x",
        headers={"Authorization": "Bearer secret-token-value"},
    )
    response = httpx.Response(403, request=request)
    exc = GatedRepoError(
        "403 Forbidden for url: https://huggingface.co/api/models/x\n"
        "Authorization: Bearer secret-token-value",
        response=response,
    )
    message = models_mod.format_pull_error(exc)
    assert "secret-token-value" not in message
    assert "Bearer <redacted>" in message


def test_pull_report_stdout_has_no_credential_shaped_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End to end on the CLI surface: a token in a hub failure does not reach
    stdout, where no log formatter runs."""

    def boom(repo_id, **kwargs):
        raise _make_http_error(404, f"no access with {_TOKEN}")

    monkeypatch.setenv("HF_HOME", str(Path("/nonexistent-vemoizer-cache")))
    with _patch_download(side_effect=boom):
        result = runner.invoke(app, ["models", "pull"])
    assert _TOKEN not in result.stdout
    assert "hf_<redacted>" in result.stdout


# ---------------------------------------------------------------------------
# Existing transcribe tests stay green (smoke check)
# ---------------------------------------------------------------------------


def test_transcribe_missing_file_fails_closed(monkeypatch) -> None:
    from test_pipeline import _patch_preflight_pass

    _patch_preflight_pass(monkeypatch)
    result = runner.invoke(app, ["transcribe", "memo.m4a"])
    assert result.exit_code == 1
    assert result.stdout == ""
    assert "not found" in result.stderr
