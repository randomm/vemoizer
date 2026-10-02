"""Contract test for the mlx-whisper tqdm shim (issue #105).

Reads the installed ``mlx_whisper.transcribe`` source via ``inspect.getsource``
and FAILS LOUDLY if an mlx-whisper upgrade changes what the shim relies on:

- the module-level ``import tqdm`` (module attribute ``tqdm``)
- ``tqdm.tqdm(total=..., unit="frames", ...)`` call
- ``disable=verbose is not False``
- ``with ... as pbar`` + ``pbar.update(...)`` usage
- the per-segment ``print`` stays behind ``if verbose:``

No network, no model load — just ``inspect`` on the installed package.
"""

from __future__ import annotations

import re

REVISION = "0.4.3"


def _get_mlx_whisper_transcribe_source() -> str:
    """Get the source of the installed mlx_whisper/transcribe.py file.

    The package ``__init__`` re-exports the ``transcribe`` FUNCTION as
    ``mlx_whisper.transcribe``, shadowing the submodule of the same name.
    So ``inspect.getsource(mlx_whisper.transcribe)`` returns only the
    function body (which lacks the module-level ``import tqdm``). Read the
    file directly from the module's ``__file__`` instead.
    """
    import importlib
    from pathlib import Path

    tr_mod = importlib.import_module("mlx_whisper.transcribe")
    return Path(tr_mod.__file__).read_text()


def test_mlx_whisper_contract_tqdm_module_attribute() -> None:
    """The module-level `import tqdm` must exist in the installed source."""
    source = _get_mlx_whisper_transcribe_source()
    assert re.search(r"^import tqdm\s*$", source, re.MULTILINE), (
        "mlx-whisper contract violation: module-level 'import tqdm' not found. "
        "The shim in src/vemoizer/progress_shim.py (with_whisper_progress) "
        "must be revisited: it patches the module-level tqdm attribute of "
        "mlx_whisper.transcribe; if the import has moved, been renamed, or "
        "become a local import, the shim has nothing to patch and the decode "
        "will run without progress."
    )


def test_mlx_whisper_contract_tqdm_call_total_and_unit() -> None:
    """The tqdm.tqdm(total=..., unit="frames") call must exist."""
    source = _get_mlx_whisper_transcribe_source()
    assert re.search(r"tqdm\.tqdm\(\s*total=", source), (
        "mlx-whisper contract violation: 'tqdm.tqdm(total=...)' call not found. "
        "The shim in src/vemoizer/progress_shim.py (with_whisper_progress) "
        "wraps this call to convert the frame counter to minutes for the "
        "ProgressDisplay; if the call signature changes (e.g. total is renamed "
        "or the call is refactored), the shim must be updated."
    )
    assert re.search(r'unit\s*=\s*"frames"', source), (
        "mlx-whisper contract violation: 'unit=\"frames\"' not found in the "
        "tqdm.tqdm call. The shim converts frames to minutes (frames * "
        "HOP_LENGTH / SAMPLE_RATE / 60); if the unit changes to seconds or "
        "something else, the conversion constant must be updated."
    )


def test_mlx_whisper_contract_disable_verbose_inverted() -> None:
    """The 'disable=verbose is not False' condition must exist."""
    source = _get_mlx_whisper_transcribe_source()
    assert re.search(r"disable\s*=\s*verbose\s+is\s+not\s+False", source), (
        "mlx-whisper contract violation: 'disable=verbose is not False' not "
        "found. In mlx-whisper 0.4.3, verbose=False ENABLES the tqdm bar and "
        "SUPPRESSES the per-segment print (inverted vs upstream whisper). If "
        "this condition changes (e.g. to 'disable=not verbose'), the shim's "
        "verbose=False contract is broken: the bar will not activate with "
        "verbose=False, and the shim will patch a bar that never renders."
    )


def test_mlx_whisper_contract_with_pbar_and_update() -> None:
    """The 'with tqdm.tqdm(...) as pbar:' + 'pbar.update(...)' pattern must exist."""
    source = _get_mlx_whisper_transcribe_source()
    assert re.search(r"with tqdm\.tqdm\(", source), (
        "mlx-whisper contract violation: 'with tqdm.tqdm(...)' not found. "
        "The shim in src/vemoizer/progress_shim.py (with_whisper_progress) "
        "relies on the tqdm bar being used as a context manager; if the "
        "with-block is removed or the bar is used differently (e.g. as a "
        "plain object without __enter__), the shim's _ShimmedProgress will "
        "not intercept the bar's lifecycle."
    )
    assert re.search(r"as pbar\s*:", source), (
        "mlx-whisper contract violation: 'as pbar:' not found in the "
        "with-block. The shim identifies the bar variable via this name; if "
        "it is renamed (e.g. 'as progress'), the shim must be updated to "
        "match."
    )
    assert re.search(r"pbar\.update\(", source), (
        "mlx-whisper contract violation: 'pbar.update(...)' not found. "
        "The shim intercepts pbar.update() calls to convert the frame "
        "counter to file-level minutes and drive the ProgressDisplay; if the "
        "bar uses a different API (e.g. pbar.set_n(...)), the shim must be "
        "updated."
    )


def test_mlx_whisper_contract_verbose_print_guard() -> None:
    """The per-segment print must stay behind 'if verbose:'."""
    source = _get_mlx_whisper_transcribe_source()
    assert re.search(r"if verbose\s*:", source), (
        "mlx-whisper contract violation: 'if verbose:' guard not found. "
        "The per-segment print (printing each transcribed segment to stdout) "
        "must stay behind this guard. If the print moves to the verbose=False "
        "path (or the guard is removed), the shim's verbose=False contract "
        "(which suppresses per-segment output while enabling the tqdm bar) is "
        "broken: the user will see per-segment text on stdout during a TTY "
        "run, polluting the terminal alongside the progress display."
    )
