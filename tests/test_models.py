"""Tests for the central model registry (issue #3, workstream task-registry).

Offline only: ``snapshot_download`` and ``scan_cache_dir`` are mocked or
pointed at ``tmp_path``; nothing touches the real HF cache or the network.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

import vemoizer.models as _models_mod
from vemoizer.models import (
    MODEL_REGISTRY,
    MODELS,
    ModelEntry,
    cache_size_bytes,
    format_pull_error,
    get_model,
    pull_all,
    pull_model,
    resolve_model_path,
)

# The exact pins the spec (docs/pipeline-spec.md + plan) fixes.
EXPECTED = {
    "parakeet": (
        "mlx-community/parakeet-tdt-0.6b-v3",
        "ed2b7e8c15f9aaa0b5772e2efb986255eaef7e15",
    ),
    "canary": (
        "Mediform/canary-1b-v2-mlx-q8",
        "0b6b32ee10f30c89e3ead7249bb636445e3019ee",
    ),
    "whisper-finnish": (
        "FredrikKarlssonSpeech/whisper-large-finnish-v3-mlx",
        "f51f0310c1b2a3e5acb16905c1a7245bb9476846",
    ),
    "whisper-turbo": (
        "mlx-community/whisper-large-v3-turbo",
        "a4aaeec0636e6fef84abdcbe3544cb2bf7e9f6fb",
    ),
    "pyannote": (
        "pyannote/speaker-diarization-community-1",
        "3533c8cf8e369892e6b79ff1bf80f7b0286a54ee",
    ),
}

_HEX40 = "0123456789abcdef"


def _entry(name: str) -> ModelEntry:
    return MODEL_REGISTRY[name]


def _make_complete_cache(tmp_path: Path, repo_id: str, revision: str) -> Path:
    """Lay out a complete HF cache for *repo_id*@*revision* in *tmp_path*.

    Returns the snapshot directory (the path a warm cache must be served
    from without any hub call, issue #252).
    """
    from huggingface_hub.file_download import repo_folder_name

    from vemoizer import model_cache

    folder = tmp_path / repo_folder_name(repo_id=repo_id, repo_type="model")
    snap = folder / "snapshots" / revision
    snap.mkdir(parents=True)
    for pattern in model_cache._EXPECTED_WEIGHT_FILES[repo_id]:
        weight = snap / pattern.replace("*", "weights")
        weight.parent.mkdir(parents=True, exist_ok=True)
        weight.write_bytes(b"0" * 16)
    return snap


# ---------------------------------------------------------------------------
# Registry shape
# ---------------------------------------------------------------------------


def test_registry_has_exactly_five_models() -> None:
    assert sorted(MODEL_REGISTRY) == [
        "canary",
        "parakeet",
        "pyannote",
        "whisper-finnish",
        "whisper-turbo",
    ]
    assert len(MODELS) == 5


def test_models_tuple_is_in_pipeline_order() -> None:
    assert [e.name for e in MODELS] == [
        "parakeet",
        "canary",
        "whisper-finnish",
        "whisper-turbo",
        "pyannote",
    ]


def test_registry_pins_match_spec() -> None:
    for name, (repo_id, revision) in EXPECTED.items():
        assert _entry(name).repo_id == repo_id, name
        assert _entry(name).revision == revision, name


def test_revision_guard_every_entry_is_40_lower_hex() -> None:
    """Regression: no entry may carry a branch name, short SHA, or empty
    pin — a moving ref is the exact regression invariant #4 exists to kill."""
    for entry in MODELS:
        assert len(entry.revision) == 40, entry.name
        assert all(c in _HEX40 for c in entry.revision), entry.name
        assert entry.revision == entry.revision.lower(), entry.name


def test_entry_repo_ids_are_full_hf_paths() -> None:
    for entry in MODELS:
        assert "/" in entry.repo_id, entry.name
        assert len(entry.repo_id) > len(entry.name), entry.name


def test_entries_are_frozen() -> None:
    assert ModelEntry.__dataclass_params__.frozen is True


def test_registry_import_does_not_download() -> None:
    with patch("huggingface_hub.snapshot_download") as mock_dl:
        import vemoizer.models  # noqa: F401, S110 — re-import is a no-op

        mock_dl.assert_not_called()


# ---------------------------------------------------------------------------
# get_model
# ---------------------------------------------------------------------------


def test_get_model_returns_entry() -> None:
    assert get_model("canary").repo_id == "Mediform/canary-1b-v2-mlx-q8"


def test_get_model_unknown_raises_keyerror_with_known_names() -> None:
    with pytest.raises(KeyError) as exc:
        get_model("gpt-so-v4")
    message = str(exc.value)
    for name in ("canary", "parakeet", "pyannote", "whisper-finnish", "whisper-turbo"):
        assert name in message


# ---------------------------------------------------------------------------
# pull_model / pull_all — pin enforcement
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_pull_model_calls_snapshot_download_with_pinned_revision(
    name: str, tmp_path
) -> None:
    repo_id, revision = EXPECTED[name]
    # Cold cache (empty tmp_path): the pull must reach the pinned download.
    # A warm cache is served from disk with no hub call at all (issue #252).
    with patch("huggingface_hub.snapshot_download", return_value="/tmp/snap") as snap:
        result = pull_model(name, cache_dir=str(tmp_path))
    # The (single) real call uses the pinned revision.
    assert snap.call_count == 1
    for call in snap.call_args_list:
        _, kwargs = call
        assert kwargs.get("revision") == revision
    assert result == "/tmp/snap"


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_pull_model_revision_never_omitted(name: str, tmp_path) -> None:
    """The revision kwarg must be present and equal to the full SHA —
    a call without it is the 'moving ref' regression. Cold cache, so the
    real download runs (a warm cache makes no hub call, issue #252)."""
    with patch("huggingface_hub.snapshot_download", return_value="/tmp/snap") as snap:
        pull_model(name, cache_dir=str(tmp_path))
        assert snap.call_count == 1
        _, kwargs = snap.call_args
        assert "revision" in kwargs
        assert kwargs["revision"] == EXPECTED[name][1]
        assert len(kwargs["revision"]) == 40


def test_pull_model_accepts_cache_dir(tmp_path) -> None:
    # Cold cache: the download call must carry the pinned revision.
    with patch("huggingface_hub.snapshot_download", return_value="/tmp/snap") as snap:
        pull_model("parakeet", cache_dir=str(tmp_path))
    assert snap.call_count == 1
    for call in snap.call_args_list:
        _, kwargs = call
        assert kwargs.get("revision") == EXPECTED["parakeet"][1]


def test_pull_model_unknown_name_raises_before_any_download() -> None:
    with patch("huggingface_hub.snapshot_download") as snap:
        with pytest.raises(KeyError):
            pull_model("nope")
        snap.assert_not_called()


def test_pull_model_routes_through_resolve_model_path() -> None:
    """``pull_model`` delegates to the shared ``resolve_model_path`` seam —
    the warm-cache shortcut logic lives in exactly one place (issues #147,
    #252). If ``pull_model`` ever re-implements its own ``snapshot_download``
    call, this test fails: the loader-side fix stops covering ``models pull``."""
    with patch.object(
        _models_mod, "resolve_model_path", return_value="/tmp/snap"
    ) as seam:
        result = pull_model("parakeet")
    assert result == "/tmp/snap"
    seam.assert_called_once_with(
        EXPECTED["parakeet"][0], EXPECTED["parakeet"][1], cache_dir=None
    )


def test_pull_model_result_is_str(tmp_path) -> None:
    # snapshot_download returns a str; pull_model must not change its type.
    repo_id, revision = EXPECTED["canary"]
    _make_complete_cache(tmp_path, repo_id, revision)
    with patch("huggingface_hub.snapshot_download", return_value="/tmp/snap"):
        assert isinstance(pull_model("canary", cache_dir=str(tmp_path)), str)


def test_pull_all_warms_all_five_in_pipeline_order(tmp_path) -> None:
    paths = {
        "parakeet": "/p",
        "canary": "/c",
        "whisper-finnish": "/w",
        "whisper-turbo": "/t",
        "pyannote": "/d",
    }
    with patch("huggingface_hub.snapshot_download") as snap:
        snap.side_effect = lambda repo_id, **kw: {
            "mlx-community/parakeet-tdt-0.6b-v3": paths["parakeet"],
            "Mediform/canary-1b-v2-mlx-q8": paths["canary"],
            "FredrikKarlssonSpeech/whisper-large-finnish-v3-mlx": paths[
                "whisper-finnish"
            ],
            "mlx-community/whisper-large-v3-turbo": paths["whisper-turbo"],
            "pyannote/speaker-diarization-community-1": paths["pyannote"],
        }[repo_id]
        result = pull_all(cache_dir=str(tmp_path))

    assert result == paths
    # The probe no longer calls snapshot_download: exactly one real call per model.
    assert snap.call_count == 5
    order = [c.args[0] for c in snap.call_args_list]
    assert order == [
        EXPECTED[n][0]
        for n in ("parakeet", "canary", "whisper-finnish", "whisper-turbo", "pyannote")
    ]


def test_pull_all_revises_every_call_with_full_sha(tmp_path) -> None:
    with patch("huggingface_hub.snapshot_download", return_value="/tmp/snap") as snap:
        pull_all(cache_dir=str(tmp_path))
    for call in snap.call_args_list:
        _, kwargs = call
        assert len(kwargs["revision"]) == 40


# ---------------------------------------------------------------------------
# Offline + HF error surfacing
# ---------------------------------------------------------------------------


def test_pull_model_propagates_offline_mode_error() -> None:
    from huggingface_hub.errors import OfflineModeIsEnabled

    with (
        patch(
            "huggingface_hub.snapshot_download",
            side_effect=OfflineModeIsEnabled("HF_HUB_OFFLINE=1"),
        ),
        pytest.raises(OfflineModeIsEnabled),
    ):
        pull_model("parakeet")


def test_format_offline_error_names_hf_hub_offline() -> None:
    from huggingface_hub.errors import OfflineModeIsEnabled

    message = format_pull_error(OfflineModeIsEnabled("offline mode is enabled"))
    assert "offline" in message.lower()
    assert "HF_HUB_OFFLINE" in message


def test_format_gated_repo_error() -> None:
    from huggingface_hub.errors import GatedRepoError

    exc = GatedRepoError(
        "You are not allowed to download this model",
        response=_fake_response(401),
    )
    message = format_pull_error(exc)
    assert "license" in message.lower() or "token" in message.lower()


def test_format_http_401_error() -> None:
    from huggingface_hub.errors import HfHubHTTPError

    exc = HfHubHTTPError(
        "401 Client Error: Unauthorized",
        response=_fake_response(401),
    )
    message = format_pull_error(exc)
    assert "auth" in message.lower()
    assert "401" in message


def test_format_http_404_error() -> None:
    from huggingface_hub.errors import HfHubHTTPError

    exc = HfHubHTTPError(
        "404 Client Error: Not Found",
        response=_fake_response(404),
    )
    message = format_pull_error(exc)
    assert "404" in message


def test_format_generic_error_does_not_leak_raw_text() -> None:
    message = format_pull_error(Exception("something opaque"))
    assert "something opaque" not in message
    assert "Exception" in message
    assert "Model download failed" in message


def _fake_response(status: int) -> Any:
    import httpx

    request = httpx.Request("GET", "https://huggingface.co")
    return httpx.Response(status, request=request)


# ---------------------------------------------------------------------------
# Warm-cache quietness (issue #147: no HF progress bars when cached)
# ---------------------------------------------------------------------------


def test_pull_model_warm_cache_never_calls_snapshot_download(tmp_path) -> None:
    """Issue #252 (invariant #1): a locally-complete snapshot is served
    straight from the cache directory — ``huggingface_hub`` is not called
    at all, so no header-building HTTP request (e.g. the
    ``/api/agent-harnesses`` call huggingface_hub 1.29 makes while
    building headers) can leak on a warm cache."""
    repo_id, revision = EXPECTED["parakeet"]
    snap_dir = _make_complete_cache(tmp_path, repo_id, revision)

    with patch("huggingface_hub.snapshot_download") as snap:
        result = pull_model("parakeet", cache_dir=str(tmp_path))

    snap.assert_not_called()
    assert result == str(snap_dir)


def test_pull_model_cold_cache_keeps_progress_bars(tmp_path) -> None:
    """When the conservative probe fails (cold or incomplete cache) the real
    download runs with the progress bars untouched — a multi-GB silent
    download would look like a hang."""
    from huggingface_hub import utils as hf_utils

    seen: list[bool] = []

    def fake_snapshot(repo_id, **kwargs):
        seen.append(hf_utils.are_progress_bars_disabled())
        return str(tmp_path)

    with patch("huggingface_hub.snapshot_download", side_effect=fake_snapshot):
        result = pull_model("canary", cache_dir=str(tmp_path))
        assert not hf_utils.are_progress_bars_disabled()

    assert result == str(tmp_path)
    assert seen == [False], (
        "cold-cache download must run with HF's progress bars ENABLED; "
        f"saw disabled-states {seen}"
    )


def test_pull_models_warm_cache_never_calls_snapshot_download(tmp_path) -> None:
    """``pull_models`` (the ``models pull`` seam) serves every locally-
    complete model from disk: the probe short-circuits the hub call for
    each model individually (issue #252)."""
    from vemoizer import model_cache as _mc

    _mc.clear_memo()
    snap_dirs = {
        spec.name: _make_complete_cache(tmp_path, spec.repo_id, spec.revision)
        for spec in MODELS
    }

    with patch("huggingface_hub.snapshot_download") as snap:
        from vemoizer.models import pull_models

        results = pull_models(MODELS, cache_dir=str(tmp_path))

    snap.assert_not_called()
    assert all(r.error is None for r in results)
    for r in results:
        assert r.local_path == str(snap_dirs[r.spec.name])


# ---------------------------------------------------------------------------
# Shared resolve_model_path seam (issue #147: the silencing lives in one place)
# ---------------------------------------------------------------------------


def test_resolve_model_path_forwards_revision_and_extra_kwargs(tmp_path) -> None:
    """Break-and-fail: the real download carries the pinned revision plus the
    loader's extra kwargs (e.g. the gated repo's ``token``). The conservative
    probe does not call snapshot_download, so exactly one call is observed."""
    calls: list[tuple[str, dict[str, Any]]] = []

    def fake_snapshot(repo_id, **kwargs):
        calls.append((repo_id, kwargs))
        return str(tmp_path)

    with patch("huggingface_hub.snapshot_download", side_effect=fake_snapshot):
        path = resolve_model_path(
            "org/repo", "a" * 40, cache_dir=str(tmp_path), token="hf_token"
        )

    assert path == str(tmp_path)
    assert len(calls) == 1
    real = calls[0]
    assert real[1].get("revision") == "a" * 40
    assert real[1].get("token") == "hf_token"


def test_resolve_model_path_warm_cache_returns_local_path_without_download(
    tmp_path,
) -> None:
    """Issue #252: on a warm cache the seam hands out the snapshot directory
    and the fake ``snapshot_download`` is NEVER called — not even inside
    ``disable_progress_bars``, because the library call itself makes HTTP
    requests (header building) on a warm cache."""
    repo_id, revision = EXPECTED["whisper-turbo"]
    snap_dir = _make_complete_cache(tmp_path, repo_id, revision)

    with patch("huggingface_hub.snapshot_download") as snap:
        path = resolve_model_path(repo_id, revision, cache_dir=str(tmp_path))

    assert path == str(snap_dir)
    snap.assert_not_called()


def test_resolve_model_path_cold_cache_keeps_bars(tmp_path) -> None:
    """A failed conservative probe means 'not provably cached' — the real
    download runs once, with the bars on (a multi-GB silent fetch looks
    like a hang)."""
    from huggingface_hub import utils as hf_utils

    seen: list[bool] = []

    def fake_snapshot(repo_id, **kwargs):
        seen.append(hf_utils.are_progress_bars_disabled())
        return str(tmp_path)

    with patch("huggingface_hub.snapshot_download", side_effect=fake_snapshot):
        resolve_model_path("org/repo", "a" * 40, cache_dir=str(tmp_path))

    assert seen == [False]


# ---------------------------------------------------------------------------
# Cache size reporting
# ---------------------------------------------------------------------------


def test_cache_size_bytes_uses_scan_cache_dir() -> None:
    fake_repo = _fake_cache_repo("mlx-community/parakeet-tdt-0.6b-v3", 123_456)
    info = type("Info", (), {"repos": [fake_repo]})()
    with patch("huggingface_hub.scan_cache_dir", return_value=info) as scan:
        size = cache_size_bytes("parakeet", cache_dir="/tmp/fake-cache")
    scan.assert_called_once_with("/tmp/fake-cache")
    assert size == 123_456


def test_cache_size_bytes_absent_repo_returns_none() -> None:
    info = type("Info", (), {"repos": []})()
    with patch("huggingface_hub.scan_cache_dir", return_value=info):
        assert cache_size_bytes("whisper-finnish") is None


def test_cache_size_bytes_unknown_model_raises() -> None:
    with pytest.raises(KeyError):
        cache_size_bytes("gpt-so-v4")


def _fake_cache_repo(repo_id: str, size: int) -> Any:
    return type("Repo", (), {"repo_id": repo_id, "size_on_disk": size})()
