"""Eval decode backends live in ``vemoizer.eval_backends`` (issue #175).

The CLI module only wires the Typer command; the registry and the decode
helpers are reached through ``eval_backends`` with no re-export, so tests
patch the module that owns the attribute.
"""

from __future__ import annotations

import vemoizer.eval_backends as eval_backends
import vemoizer.eval_cli as eval_cli


def test_backend_registry_lives_in_eval_backends() -> None:
    assert set(eval_backends.BACKENDS) == {"parakeet", "canary", "consensus"}


def test_decode_helper_lives_in_eval_backends() -> None:
    assert callable(eval_backends.transcribe_decode_only)


def test_eval_cli_does_not_define_or_reexport_backends() -> None:
    assert not hasattr(eval_cli, "BACKENDS")
    assert not hasattr(eval_cli, "transcribe_decode_only")
