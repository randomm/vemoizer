"""Tests for the conservative local-cache completeness probe (issue #147).

The probe decides whether a pinned HF snapshot can be served from the local
cache (warm-cache silencing). It must fail toward *keeping the progress
bars*: a partial snapshot that once counted as "complete" let
``resolve_model_path`` run a real multi-GB download with ``disable_progress_bars()``
— a silent fetch that looks like a hang.

Every layout here is a hand-built tmp HF cache directory:
``<cache>/models--<org>--<repo>/{blobs,refs,snapshots,<sha>}`` — no
network, no ``snapshot_download``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from vemoizer import model_cache

REPO_ID = "org/repo"
PINNED_REVISION = "a" * 40
OTHER_REVISION = "b" * 40


def _folder(tmp_path: Path, repo_id: str = REPO_ID) -> Path:
    from huggingface_hub.file_download import repo_folder_name

    return tmp_path / repo_folder_name(repo_id=repo_id, repo_type="model")


def _snapshot_dir(tmp_path: Path, revision: str, repo_id: str = REPO_ID) -> Path:
    snap = _folder(tmp_path, repo_id) / "snapshots" / revision
    snap.mkdir(parents=True)
    return snap


def _blobs_dir(tmp_path: Path, repo_id: str = REPO_ID) -> Path:
    blobs = _folder(tmp_path, repo_id) / "blobs"
    blobs.mkdir(parents=True, exist_ok=True)
    return blobs


def _write_file(dirpath: Path, name: str, size: int = 16) -> Path:
    path = dirpath / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"0" * size)
    return path


def _complete_snapshot(tmp_path: Path, repo_id: str = REPO_ID) -> Path:
    """A pinned-revision snapshot that satisfies every probe expectation."""
    snap = _snapshot_dir(tmp_path, PINNED_REVISION, repo_id)
    for pattern in model_cache._expected_weights_for(repo_id):
        _write_file(snap, pattern.replace("*", "weights"))
    return snap


# -- the five pinned models' real-cache layout expectations ------------------


@pytest.mark.parametrize("repo_id", sorted(model_cache._EXPECTED_WEIGHT_FILES))
def test_expected_patterns_satisfy_real_cached_layout(
    tmp_path: Path, repo_id: str
) -> None:
    """A cache laid out like the operator's real snapshot (a real file per
    expected pattern) must count as complete — an always-noisy probe would
    defeat the feature."""
    _complete_snapshot(tmp_path, repo_id)
    assert model_cache.snapshot_locally_complete(repo_id, PINNED_REVISION, tmp_path)


# -- completeness matrix ------------------------------------------------------


def _probe(repo_id: str, revision: str, cache_dir: Path | None) -> bool:
    return model_cache.snapshot_locally_complete(repo_id, revision, str(cache_dir))


def test_complete_snapshot_is_complete(tmp_path: Path) -> None:
    _complete_snapshot(tmp_path)
    assert _probe(REPO_ID, PINNED_REVISION, tmp_path) is True


def test_missing_revision_snapshot_dir_is_incomplete(tmp_path: Path) -> None:
    # Repo folder + blobs exist, but no snapshot dir for the pinned SHA.
    _folder(tmp_path).mkdir(parents=True)
    _blobs_dir(tmp_path)
    assert _probe(REPO_ID, PINNED_REVISION, tmp_path) is False


def test_snapshot_with_only_config_json_is_incomplete(tmp_path: Path) -> None:
    """The adversarial-reproduced bug: refs file + a snapshot dir holding
    only config.json must NOT count as complete."""
    snap = _snapshot_dir(tmp_path, PINNED_REVISION)
    (snap / "config.json").write_text("{}")
    assert _probe(REPO_ID, PINNED_REVISION, tmp_path) is False


def test_weights_present_but_incomplete_blob_is_incomplete(tmp_path: Path) -> None:
    """A ``*.incomplete`` blob (an interrupted download) makes the snapshot
    incomplete even when every expected weight file is present."""
    _complete_snapshot(tmp_path)
    (_blobs_dir(tmp_path) / "deadbeef.incomplete").write_bytes(b"partial")
    assert _probe(REPO_ID, PINNED_REVISION, tmp_path) is False


def test_dangling_symlink_is_incomplete(tmp_path: Path) -> None:
    _complete_snapshot(tmp_path)
    # A snapshot entry whose blob target has vanished (removed / truncated
    # blob): the entry must not resolve.
    (_folder(tmp_path) / "snapshots" / PINNED_REVISION / "config.json").symlink_to(
        "/nonexistent/blob-target"
    )
    assert _probe(REPO_ID, PINNED_REVISION, tmp_path) is False


def test_other_revision_cached_is_incomplete(tmp_path: Path) -> None:
    """Caching a different revision than the pinned SHA must not count:
    the pinned snapshot dir must exist, not just any snapshot."""
    _write_file(_snapshot_dir(tmp_path, OTHER_REVISION), "model.safetensors")
    assert _probe(REPO_ID, PINNED_REVISION, tmp_path) is False


def test_cold_cache_is_incomplete(tmp_path: Path) -> None:
    # Empty cache dir (nothing downloaded yet).
    assert _probe(REPO_ID, PINNED_REVISION, tmp_path) is False


def test_default_cache_dir_uses_hf_hub_cache(tmp_path: Path, monkeypatch) -> None:
    """When ``cache_dir`` is None the probe reads ``HF_HUB_CACHE`` from the
    installed huggingface_hub constants (no ``cache_dir`` argument)."""
    monkeypatch.setenv("HF_HUB_CACHE", str(tmp_path))
    import huggingface_hub.constants as constants

    monkeypatch.setattr(constants, "HF_HUB_CACHE", str(tmp_path), raising=False)
    _complete_snapshot(tmp_path)
    assert model_cache.snapshot_locally_complete(REPO_ID, PINNED_REVISION) is True


# -- memo semantics -----------------------------------------------------------


def test_negative_memo_is_per_repo_revision(tmp_path: Path) -> None:
    """A ``False`` probe result is memoized per ``(repo_id, revision)``: a
    different revision is probed independently."""
    assert _probe(REPO_ID, PINNED_REVISION, tmp_path) is False
    _complete_snapshot(tmp_path)
    # The pinned key was memoized False BEFORE the snapshot appeared: the
    # negative memo keeps the bars on (conservative).
    assert _probe(REPO_ID, PINNED_REVISION, tmp_path) is False
    # A different revision is probed fresh and sees the new snapshot dir.
    assert _probe(REPO_ID, OTHER_REVISION, tmp_path) is False
    _write_file(_folder(tmp_path) / "snapshots" / OTHER_REVISION, "model.safetensors")
    assert _probe(REPO_ID, OTHER_REVISION, tmp_path) is False


def test_true_is_not_memoized_so_download_failure_cannot_poison(tmp_path: Path) -> None:
    """Break-and-fail guard: the memo never caches ``True``. A failed real
    download (which happens OUTSIDE the probe) must be able to re-probe —
    if the probe's ``True`` had been memoized, the next call would skip the
    re-verification and keep silencing bars for a now-broken cache."""
    _complete_snapshot(tmp_path)
    model_cache._COMPLETE_SNAPSHOTS.clear()
    first = model_cache.snapshot_locally_complete(REPO_ID, PINNED_REVISION, tmp_path)
    assert first is True
    # No True may be cached: the memo holds only False entries.
    assert model_cache._COMPLETE_SNAPSHOTS.get((REPO_ID, PINNED_REVISION)) is None
    # Simulate the cache being damaged by a failed real download: remove all
    # weight files so the probe re-checks and finds the damage.
    snap_dir = _folder(tmp_path) / "snapshots" / PINNED_REVISION
    for f in (
        list(snap_dir.rglob("*.safetensors"))
        + list(snap_dir.rglob("*.bin"))
        + list(snap_dir.rglob("*.npz"))
        + list(snap_dir.rglob("*.pt"))
    ):
        f.unlink()
    # The snapshot dir may now be empty or hold only non-weight files.
    second = model_cache.snapshot_locally_complete(REPO_ID, PINNED_REVISION, tmp_path)
    assert second is False, (
        "after the cache was damaged, the probe must see the damage — a "
        "memoized True would have kept the bars silenced"
    )


def test_clear_memo_forgets_negatives() -> None:
    model_cache._COMPLETE_SNAPSHOTS[("x/y", "z")] = False
    model_cache.clear_memo()
    assert model_cache._COMPLETE_SNAPSHOTS == {}


# -- probe exception -> not complete -----------------------------------------


def test_probe_exception_reports_not_complete(monkeypatch) -> None:
    """Any exception inside the probe (e.g. an unreadable snapshot dir)
    means "not provably complete" — the bars stay on."""

    def _raise(*_args: Any, **_kwargs: Any) -> bool:
        raise OSError("boom")

    monkeypatch.setattr(model_cache, "_snapshot_is_complete", _raise)
    assert model_cache.snapshot_locally_complete(REPO_ID, PINNED_REVISION) is False


# -- resolve_model_path integration -------------------------------------------


def test_resolve_model_path_partial_snapshot_keeps_bars(tmp_path: Path) -> None:
    """End-to-end on the real seam: a partial snapshot (config.json only,
    plus a refs file) must NOT trigger the silenced path — the real download
    runs with the progress bars ENABLED (a silent multi-GB fetch looks like
    a hang)."""
    from huggingface_hub import utils as hf_utils

    from vemoizer.models import resolve_model_path

    # refs/<pinned-sha> exists (as it would after a prior download attempt)
    # and the snapshot dir holds only config.json: the old
    # ``snapshot_download(local_files_only=True)`` probe accepted this as a
    # "resolvable snapshot" and silenced the real download.
    folder = _folder(tmp_path)
    refs = folder / "refs"
    refs.mkdir(parents=True)
    (refs / PINNED_REVISION).write_text(PINNED_REVISION)
    snap = _snapshot_dir(tmp_path, PINNED_REVISION)
    (snap / "config.json").write_text("{}")

    seen: list[bool] = []

    def fake_snapshot(repo_id, **kwargs):
        seen.append(hf_utils.are_progress_bars_disabled())
        return str(snap)

    with patch(
        "huggingface_hub.snapshot_download", side_effect=fake_snapshot
    ) as snap_mock:
        path = resolve_model_path(REPO_ID, PINNED_REVISION, cache_dir=str(tmp_path))

    # The conservative probe says "incomplete", so the real download ran
    # with the bars ENABLED (not silently inside disable_progress_bars).
    assert path == str(snap)
    assert seen == [False], (
        f"a partial snapshot must keep the progress bars on; saw disabled-states {seen}"
    )
    assert not hf_utils.are_progress_bars_disabled()
    # The probe no longer calls snapshot_download: exactly one (real) call.
    assert snap_mock.call_count == 1


def test_resolve_model_path_complete_snapshot_silences(tmp_path: Path) -> None:
    from huggingface_hub import utils as hf_utils

    from vemoizer.models import resolve_model_path

    _complete_snapshot(tmp_path)
    seen: list[bool] = []

    def fake_snapshot(repo_id, **kwargs):
        seen.append(hf_utils.are_progress_bars_disabled())
        return str(_folder(tmp_path) / "snapshots" / PINNED_REVISION)

    with patch("huggingface_hub.snapshot_download", side_effect=fake_snapshot):
        resolve_model_path(REPO_ID, PINNED_REVISION, cache_dir=str(tmp_path))

    # Complete snapshot: the (faked) local fetch ran inside the silencing
    # switch. The probe itself does not call snapshot_download anymore.
    assert seen == [True]
    assert not hf_utils.are_progress_bars_disabled()


def test_offline_mode_error_surfaces_same_message_as_main(tmp_path: Path) -> None:
    """Regression guard (FIX 3): with ``HF_HUB_OFFLINE=1`` and a cold cache,
    the user-facing error from the pull/load path must match the message
    that ``main`` produced (the friendly ``format_pull_error`` hint). The
    conservative probe must NOT swallow ``OfflineModeIsEnabled``: it only
    checks the filesystem (no network) and returns ``False`` for a cold
    cache; the real ``snapshot_download`` then raises ``OfflineModeIsEnabled``
    which propagates unchanged to ``_describe_hf_error``.
    """
    from huggingface_hub.errors import OfflineModeIsEnabled

    from vemoizer.models import _describe_hf_error, pull_model

    # Cold cache: the probe sees no snapshot dir and returns False.
    # The real snapshot_download then raises OfflineModeIsEnabled.
    with (
        patch(
            "huggingface_hub.snapshot_download",
            side_effect=OfflineModeIsEnabled("offline mode is enabled"),
        ),
        pytest.raises(OfflineModeIsEnabled),
    ):
        pull_model("parakeet", cache_dir=str(tmp_path))

    # The message from _describe_hf_error (used by pull_models) must be
    # the friendly offline hint, matching what main produced.
    exc = OfflineModeIsEnabled("offline mode is enabled")
    msg = _describe_hf_error(exc)
    assert "HF_HUB_OFFLINE" in msg
    assert "not cached" in msg or "not in the local cache" in msg
    # The message must NOT be a generic "failed to download" fallback.
    assert "failed to download model" not in msg
