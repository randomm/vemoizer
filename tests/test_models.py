"""Tests for the central model registry (issue #3, workstream task-registry).

Offline only: ``snapshot_download`` and ``scan_cache_dir`` are mocked or
pointed at ``tmp_path``; nothing touches the real HF cache or the network.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest

from vemoizer.models import (
    MODEL_REGISTRY,
    MODELS,
    ModelEntry,
    cache_size_bytes,
    format_pull_error,
    get_model,
    pull_all,
    pull_model,
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
}

_HEX40 = "0123456789abcdef"


def _entry(name: str) -> ModelEntry:
    return MODEL_REGISTRY[name]


# ---------------------------------------------------------------------------
# Registry shape
# ---------------------------------------------------------------------------


def test_registry_has_exactly_three_models() -> None:
    assert sorted(MODEL_REGISTRY) == ["canary", "parakeet", "whisper-finnish"]
    assert len(MODELS) == 3


def test_models_tuple_is_in_pipeline_order() -> None:
    assert [e.name for e in MODELS] == [
        "parakeet",
        "canary",
        "whisper-finnish",
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
    for name in ("canary", "parakeet", "whisper-finnish"):
        assert name in message


# ---------------------------------------------------------------------------
# pull_model / pull_all — pin enforcement
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_pull_model_calls_snapshot_download_with_pinned_revision(
    name: str,
) -> None:
    repo_id, revision = EXPECTED[name]
    with patch("huggingface_hub.snapshot_download", return_value="/tmp/snap") as snap:
        result = pull_model(name)
    snap.assert_called_once_with(repo_id, revision=revision)
    assert result == "/tmp/snap"


@pytest.mark.parametrize("name", sorted(EXPECTED))
def test_pull_model_revision_never_omitted(name: str) -> None:
    """The revision kwarg must be present and equal to the full SHA —
    a call without it is the 'moving ref' regression."""
    with patch("huggingface_hub.snapshot_download", return_value="/tmp/snap") as snap:
        pull_model(name)
        _, kwargs = snap.call_args
        assert "revision" in kwargs
        assert kwargs["revision"] == EXPECTED[name][1]
        assert len(kwargs["revision"]) == 40


def test_pull_model_accepts_cache_dir() -> None:
    with patch("huggingface_hub.snapshot_download", return_value="/tmp/snap") as snap:
        pull_model("parakeet", cache_dir="/tmp/custom")
    snap.assert_called_once_with(
        "mlx-community/parakeet-tdt-0.6b-v3",
        revision=EXPECTED["parakeet"][1],
        cache_dir="/tmp/custom",
    )


def test_pull_model_unknown_name_raises_before_any_download() -> None:
    with patch("huggingface_hub.snapshot_download") as snap:
        with pytest.raises(KeyError):
            pull_model("nope")
        snap.assert_not_called()


def test_pull_model_result_is_str() -> None:
    # snapshot_download returns a str; pull_model must not change its type.
    with patch("huggingface_hub.snapshot_download", return_value="/tmp/snap"):
        assert isinstance(pull_model("canary"), str)


def test_pull_all_warms_all_three_in_pipeline_order() -> None:
    paths = {
        "parakeet": "/p",
        "canary": "/c",
        "whisper-finnish": "/w",
    }
    with patch("huggingface_hub.snapshot_download") as snap:
        snap.side_effect = lambda repo_id, **kw: {
            "mlx-community/parakeet-tdt-0.6b-v3": paths["parakeet"],
            "Mediform/canary-1b-v2-mlx-q8": paths["canary"],
            "FredrikKarlssonSpeech/whisper-large-finnish-v3-mlx": paths[
                "whisper-finnish"
            ],
        }[repo_id]
        result = pull_all()

    assert result == paths
    assert snap.call_count == 3
    order = [c.args[0] for c in snap.call_args_list]
    assert order == [EXPECTED[n][0] for n in ("parakeet", "canary", "whisper-finnish")]


def test_pull_all_revises_every_call_with_full_sha() -> None:
    with patch("huggingface_hub.snapshot_download", return_value="/tmp/snap") as snap:
        pull_all()
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
