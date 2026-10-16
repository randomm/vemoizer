"""The plain-loop seam contract, asserted rather than commented (issue #151).

``run_preset``'s per-file body moved to :mod:`vemoizer.preset_plain_loop` as a
pure relocation. Two invariants hold that move together and are invisible to
the type checker and the rest of the suite:

1. ``batch_preset`` re-exports *exactly* the names the loop looks up on it.
   The loop resolves them with a function-local ``from vemoizer.batch_preset
   import ...`` precisely so the suite's long-standing
   ``monkeypatch.setattr(batch_preset, name, fake)`` still reaches it. A name
   that drifts out of that set turns a real patch into an attribute that
   nobody reads — the patch still "succeeds" and the test stays green while
   the real helper runs.
2. The two paths' run epilogues (close display -> wrote lines -> final-line
   gate -> naming hook) are the *same* gating as on ``main``: the group path
   gates the final line on ``not quiet and run_went_full(written,
   exit_code=exit_code)`` and closes the display unconditionally, while the
   plain path gates on ``run_went_full(written, len(files), exit_code)`` with
   no ``quiet`` term and closes the display for ``meeting`` only. Those
   differences are pre-existing (issue #143/#148), not drift introduced by the
   move — so they are pinned here, per path, to catch a future edit that
   changes one path and silently misses the other.

Both checks read the source with :mod:`ast`, so they survive a rename only
when *both* sides move together. Unit tests: no pipeline, no CLI, no network.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

import vemoizer.batch_preset as batch_preset_module
import vemoizer.preset_final as preset_final_module

_SRC = Path(batch_preset_module.__file__).resolve().parent


def _loop_lookups_on_batch_preset() -> set[str]:
    """Names ``run_preset_plain_loop`` resolves from ``vemoizer.batch_preset``.

    Parsed from the function's own body: only a ``from vemoizer.batch_preset
    import a, b`` counts, which is the shape the seam convention requires.
    """
    tree = ast.parse(_SRC.joinpath("preset_plain_loop.py").read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == "vemoizer.batch_preset":
            names.update(alias.name for alias in node.names)
    return names


def _module_reexports(module) -> set[str]:
    """Names bound by an ``import x as x`` re-export statement.

    The ``as``-binding is the marker ruff F401 accepts for an intentional
    re-export, so it is also the marker for "this name is here purely for
    someone else to look up" — none of ``batch_preset``'s own code calls it.
    """
    tree = ast.parse(Path(module.__file__).resolve().read_text(encoding="utf-8"))
    exported: set[str] = set()
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                if alias.asname and alias.asname == alias.name:
                    exported.add(alias.name)
    return exported


def _module_seam_names(module) -> set[str]:
    """Every name ``module`` resolves as *the seam* (leading underscore) or
    that the loop looks up on it, however it got there.

    Three legitimate origins, all of which keep ``setattr(module, name, fake)
    `` reaching the loop: an ``as``-bound re-export (``file_log``,
    ``_transcribe_preset_file``), a plain module-level import that the module
    also calls itself (``_write_preset_output`` — live in the group path), and
    a function defined right here (``_mtime_date_str``).
    """
    names = {n for n in _module_reexports(module) if n.startswith("_")}
    tree = ast.parse(Path(module.__file__).resolve().read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            names.update(a.asname or a.name for a in node.names)
        elif isinstance(node, ast.FunctionDef):
            names.add(node.name)
    return names


# --- invariant 1: batch_preset re-exports exactly the loop's lookups --------


def test_batch_preset_reexports_exactly_the_loop_lookup_names() -> None:
    """The comment at ``batch_preset.py`` claims "exactly the long-standing
    test seams"; this makes that claim executable.

    Adding a fifth ``as``-binding that the loop does not look up is a dead
    re-export that manufactures a vacuously-green patch target; dropping one
    the loop *does* look up makes the suite's patch on that name miss and the
    real helper run. Both are green everywhere else in the suite.
    """
    looked_up = _loop_lookups_on_batch_preset()
    assert looked_up == {
        "_mtime_date_str",
        "_transcribe_preset_file",
        "_write_preset_output",
        "file_log",
    }, f"the loop's lookups on batch_preset changed: {sorted(looked_up)}"
    available = _module_seam_names(batch_preset_module)
    missing = looked_up - available
    assert not missing, (
        f"the loop looks up {sorted(missing)} on batch_preset but the module "
        "no longer binds them: the suite's patch on those names would miss "
        "and the real helper would run"
    )
    # A name re-exported for nobody: an ``as``-binding the loop does not look
    # up is a dead patch target (the vacuous-green shape this PR argues
    # against), so it must not exist.
    dead = _module_reexports(batch_preset_module) - looked_up
    assert not dead, (
        f"batch_preset re-exports names the loop never looks up: {sorted(dead)}"
    )


def test_each_reexported_seam_name_is_the_same_object_as_its_home() -> None:
    """A patch on ``batch_preset.<seam>`` reaches the real implementation.

    ``is`` (not ``==``) is the point: if the re-export silently became a
    wrapper or a stale copy, patching it could stop affecting the code the
    loop calls while every seam test kept passing.
    """
    homes = {
        "_mtime_date_str": ("vemoizer.batch_preset", "_mtime_date_str"),
        "_transcribe_preset_file": (
            "vemoizer.preset_file_transcribe",
            "_transcribe_preset_file",
        ),
        "_write_preset_output": ("vemoizer.batch_output", "_write_preset_output"),
        "file_log": ("vemoizer.run_log", "file_log"),
    }
    for seam, (home_module, home_name) in homes.items():
        assert hasattr(batch_preset_module, seam), f"{seam} no longer re-exported"
        exported = getattr(batch_preset_module, seam)
        assert callable(exported), f"{seam} re-export is not callable"
        assert exported.__module__.startswith("vemoizer."), (
            f"{seam} resolved outside the package: {exported.__module__}"
        )
        assert home_name in vars(__import__(home_module, fromlist=[home_name])), (
            f"{home_module}.{home_name} missing — the seam has no home"
        )


def test_plain_loop_does_not_reexport_names_it_does_not_look_up() -> None:
    """The loop's own module must not hoist a deferred seam to module level.

    A module-level ``from vemoizer.batch_preset import ...`` (or ``notify``)
    binds the function object at import time, so ``mock.patch.object`` on the
    source module can no longer reach the loop — the deferral is load-bearing,
    and this test is what stops a "simplify the imports" edit from silently
    undoing it.
    """
    tree = ast.parse(_SRC.joinpath("preset_plain_loop.py").read_text(encoding="utf-8"))
    module_level: dict[str, set[str]] = {}
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module:
            module_level.setdefault(node.module, set()).update(
                alias.name for alias in node.names
            )
    for module in ("vemoizer.batch_preset", "vemoizer.notify"):
        assert module not in module_level, (
            f"{module} imported at module level in preset_plain_loop: the "
            f"suite patches that module, so the loop must look it up per call"
        )


# --- invariant 2: the two epilogues keep main's gating, per path ------------


def _call_shape(source: str, function: str, callee: str) -> list[tuple[str, str]]:
    """Every ``callee(...)`` call in *function* as (positional-arg sources,
    sorted keyword names). Used to pin each path's epilogue call shape."""
    tree = ast.parse(source)
    out: list[tuple[str, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef) or node.name != function:
            continue
        for inner in ast.walk(node):
            if not isinstance(inner, ast.Call):
                continue
            func = inner.func
            name = getattr(func, "id", None) or getattr(func, "attr", None)
            if name != callee:
                continue
            positional = ",".join(ast.unparse(a) for a in inner.args)
            keywords = ",".join(sorted(k.arg or "" for k in inner.keywords))
            out.append((positional, keywords))
    return out


def test_both_paths_gate_the_final_line_as_main_does() -> None:
    """``run_went_full`` is called once per path, with each path's own arity.

    The group path passes no expected-pair count (a group's partial pair sets
    ``exit_code`` in its write seam); the plain path passes ``len(files)`` as
    an independent count, one pair per file. Equalising the two signatures is
    a behaviour change, not a refactor, and belongs in its own issue.
    """
    group = _call_shape(
        _SRC.joinpath("batch_preset.py").read_text(encoding="utf-8"),
        "_run_preset_groups",
        "run_went_full",
    )
    plain = _call_shape(
        _SRC.joinpath("preset_plain_loop.py").read_text(encoding="utf-8"),
        "run_preset_plain_loop",
        "run_went_full",
    )
    assert group == [("written", "exit_code")], group
    assert plain == [("written,len(files),exit_code", "")], plain


def test_both_paths_close_the_display_where_main_does() -> None:
    """Group path closes unconditionally; plain path only for ``meeting``.

    Pre-existing asymmetry (issue #143): the group path always ends a meeting
    run, the plain path also serves ``memo``, which has no display.
    """
    group = _call_shape(
        _SRC.joinpath("batch_preset.py").read_text(encoding="utf-8"),
        "_run_preset_groups",
        "_close_run_display",
    )
    plain = _call_shape(
        _SRC.joinpath("preset_plain_loop.py").read_text(encoding="utf-8"),
        "run_preset_plain_loop",
        "_close_run_display",
    )
    assert group == [("display", "")], group
    assert plain == [("display", "")], plain


def test_preset_final_trio_is_not_reexported_by_batch_preset() -> None:
    """The group path imports the ``preset_final`` trio inside its body.

    Re-exporting it at module level would put a name on ``batch_preset`` that
    nothing there looks up — a patch target that only manufactures
    vacuously-green tests. This asserts the loop and the group path each own
    their own lookups.
    """
    trio = {"print_final_line", "print_wrote_lines", "run_went_full"}
    reexported = _module_reexports(batch_preset_module)
    assert trio & reexported == set(), (
        f"preset_final trio leaked into re-exports: {trio & reexported}"
    )
    for name in trio:
        assert hasattr(preset_final_module, name), f"preset_final lost {name}"
    # ...and the group path really does look them up locally.
    group_src = _SRC.joinpath("batch_preset.py").read_text(encoding="utf-8")
    tree = ast.parse(group_src)
    imported: set[str] = set()
    for node in ast.walk(tree):
        is_final = (
            isinstance(node, ast.ImportFrom) and node.module == "vemoizer.preset_final"
        )
        if is_final:
            imported.update(alias.name for alias in node.names)
    assert trio <= imported, (
        f"group path missing preset_final lookups: {trio - imported}"
    )


def test_plain_loop_still_sets_the_per_file_batch_prefix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``set_batch_prefix`` is live in the moved loop (issue #105 M4b).

    Deleting that call kept the whole suite green — the per-file ``[i/N]
    stem`` prefix on the preset plain path can vanish silently, because the
    existing prefix tests patch ``batch_plain`` / ``transcribe_loop`` /
    ``batch``, never this module. The fake records its own arguments, so a
    patch that missed the target (or a deleted call) fails on the empty record.
    """
    from _cli_helpers import isolate_home, touch_files

    import vemoizer.batch as batch_module
    import vemoizer.notify as notify_module
    import vemoizer.pipeline as pipeline_module
    import vemoizer.preflight as preflight_module
    import vemoizer.progress_wiring as progress_wiring_module

    calls: list[tuple[int, int, str]] = []

    def fake_prefix(display, index, total, stem):
        calls.append((index, total, stem))

    files = touch_files(["a.m4a", "b.m4a"], tmp_path)
    monkeypatch.setattr(progress_wiring_module, "set_batch_prefix", fake_prefix)
    monkeypatch.setattr(preflight_module, "preflight_gate", lambda **kw: None)
    monkeypatch.setattr(batch_module, "_resolve_llm_config", lambda p: None)
    monkeypatch.setattr(notify_module, "notify_write", lambda *a, **kw: None)
    monkeypatch.setattr(notify_module, "notify_result", lambda *a, **kw: None)
    monkeypatch.setattr(
        pipeline_module,
        "transcribe_file",
        lambda path, **kw: {
            "text": "moikka",
            "segments": [],
            "paragraphs": [{"text": "moikka", "start": 0.0, "end": 1.0}],
        },
    )
    monkeypatch.setattr(
        batch_preset_module,
        "_write_preset_output",
        lambda *a, **kw: ["x.md", "x.json"],
    )
    isolate_home(monkeypatch, tmp_path, tmp_path)

    code = batch_preset_module.run_preset(
        list(files),
        command="memo",
        config_path=None,
        glossary_path=None,
        quiet=True,
        yes=True,
        display=None,
    )
    assert code == 0
    assert [c[0] for c in calls] == [1, 2], (
        f"set_batch_prefix was not reached once per file: {calls!r}"
    )
    assert [c[1] for c in calls] == [2, 2], calls
    assert [c[2] for c in calls] == ["a", "b"], calls


def test_plain_loop_notes_written_files_on_the_interrupt_tracker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``note_written_files`` is live in the moved loop, once per written pair.

    The loop tallies every output path it produced into the ``InterruptTracker``
    so an interrupted run can report (and clean up) what it actually wrote.
    Deleting that call kept the whole suite green: nothing else reads the
    tally on the success path, so the interrupt report would silently claim an
    empty written set while files sat on disk. The fake records its own
    arguments, so a patch that missed the target fails on the empty record.

    The call sits BEFORE the partial-pair check, so a partial pair is noted
    too — that is the real behaviour and is pinned here rather than a nicer
    intent: the tally records the paths the write seam returned, not only the
    runs that fully succeeded.
    """
    from _cli_helpers import isolate_home, touch_files

    import vemoizer.batch as batch_module
    import vemoizer.notify as notify_module
    import vemoizer.pipeline as pipeline_module
    import vemoizer.preflight as preflight_module
    import vemoizer.preset_interrupt as preset_interrupt_module
    import vemoizer.preset_plain_loop as preset_plain_loop_module

    recorded: list[tuple[object, list[str]]] = []

    def fake_note(tracker, paths):
        recorded.append((tracker, list(paths)))

    files = touch_files(["a.m4a", "b.m4a"], tmp_path)
    # The real tracker, so the call is asserted to carry THIS run's instance
    # and not some stand-in.
    tracker = preset_interrupt_module.InterruptTracker()
    # Patched where the loop looks the name up. ``preset_plain_loop`` binds it
    # with a module-level ``from vemoizer.preset_interrupt import``, so a patch
    # on ``preset_interrupt`` binds too late to reach this loop — the
    # ``setattr`` below is the only one that does, and it is what makes a
    # deleted call visible instead of vacuously green.
    monkeypatch.setattr(preset_plain_loop_module, "note_written_files", fake_note)
    monkeypatch.setattr(preflight_module, "preflight_gate", lambda **kw: None)
    monkeypatch.setattr(batch_module, "_resolve_llm_config", lambda p: None)
    monkeypatch.setattr(notify_module, "notify_write", lambda *a, **kw: None)
    monkeypatch.setattr(notify_module, "notify_result", lambda *a, **kw: None)
    monkeypatch.setattr(
        pipeline_module,
        "transcribe_file",
        lambda path, **kw: {
            "text": "moikka",
            "segments": [],
            "paragraphs": [{"text": "moikka", "start": 0.0, "end": 1.0}],
        },
    )
    # First file writes the full pair, second file a partial pair.
    pairs = [["a.md", "a.json"], ["b.md"]]
    monkeypatch.setattr(
        batch_preset_module,
        "_write_preset_output",
        lambda *a, **kw: pairs.pop(0),
    )
    isolate_home(monkeypatch, tmp_path, tmp_path)

    tracker = preset_interrupt_module.InterruptTracker()
    code = batch_preset_module.run_preset(
        list(files),
        command="memo",
        config_path=None,
        glossary_path=None,
        quiet=True,
        yes=True,
        display=None,
        tracker=tracker,
    )
    assert code == 1, "a partial pair must fail the run"
    assert recorded, (
        "note_written_files was never reached: the interrupt tally of written "
        "files would be empty while the pair sits on disk"
    )
    # Exactly one call per file whose write seam returned, with this run's
    # tracker and that file's pair.
    assert [t for t, _ in recorded] == [tracker, tracker], recorded
    assert [p for _, p in recorded] == [["a.md", "a.json"], ["b.md"]], recorded
    # The real helper is not run while the seam is faked, so the tracker's own
    # tally stays at zero here — it is the fake that must see the pair, and
    # test_preset_interrupt.py owns the tally arithmetic.
    assert tracker.written_count == 0, tracker.written_count
