"""Byte-preservation write-back tests for the ``people`` key (issue #93).

``write_people_list`` must update the top-level ``people`` key without
touching the rest of the file:

- surgical path (valid TOML, no single-line top-level ``people`` to
  replace → insert; one → replace in place): comments, blank lines,
  array-of-tables, dotted keys, inline tables, and every unrelated key
  survive byte-for-byte;
- fallback path (multi-line people array, scalar people, stale
  table-nested people): re-emitted via the minimal TOML emitter only
  when the result re-parses to the intended data, otherwise the file is
  left byte-identical with one ``unsupported config layout`` warning;
- the written text always parses, names with quotes/backslash/unicode/
  control characters round-trip, and the write is atomic (tmp +
  replace, no tmp file left behind).
"""

from __future__ import annotations

import glob
import io
import sys
import tomllib
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from _cli_helpers import isolate_home

from vemoizer.llm_config import ConfigError, _strict_load
from vemoizer.people_config import write_people_list

# --- Helpers ---


def _config_path(tmp_path: Path) -> Path:
    d = tmp_path / ".vemoizer"
    d.mkdir(exist_ok=True)
    return d / "config.toml"


def _capture_stderr(
    fn: Callable[..., Any], *args: Any, **kwargs: Any
) -> tuple[Any, str]:
    old_stderr = sys.stderr
    sys.stderr = io.StringIO()
    try:
        result = fn(*args, **kwargs)
    finally:
        captured = sys.stderr
        sys.stderr = old_stderr
    return result, captured.getvalue()


def _layout_warnings(stderr_text: str) -> list[str]:
    return [
        line
        for line in stderr_text.strip().splitlines()
        if "unsupported config layout" in line
    ]


# --- Comment / table-nested lines are never the top-level people line ---


class TestPeopleLineScope:
    def test_commented_out_people_line_is_not_treated_as_top_level(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """A commented-out ``people = [...]`` line is never treated as
        the top-level people line: the surgical insert places the new
        people line at the start of the file, and the comment is
        preserved byte-for-byte."""
        isolate_home(monkeypatch, tmp_path, tmp_path)
        config = _config_path(tmp_path)
        original = '# people = ["commented out"]\nx = 1\n[llm]\nmodel = "m"\n'
        config.write_text(original, encoding="utf-8")

        write_people_list(config, ["Mikko"])

        text = config.read_text(encoding="utf-8")
        # The commented-out line is preserved; the new people line is
        # inserted at the very start, before the comment.
        assert text == (
            'people = ["Mikko"]\n'
            '# people = ["commented out"]\n'
            "x = 1\n"
            "[llm]\n"
            'model = "m"\n'
        )
        loaded = tomllib.loads(text)
        assert loaded["people"] == ["Mikko"]
        assert loaded["x"] == 1
        assert loaded["llm"] == {"model": "m"}


# --- Surgical path: insert when no top-level people ---


class TestSurgicalInsert:
    def test_comments_blank_lines_and_keys_preserved_byte_for_byte(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """First name added to a comment-heavy config: every other byte
        (comment lines, blank lines, unrelated keys) is untouched."""
        isolate_home(monkeypatch, tmp_path, tmp_path)
        config = _config_path(tmp_path)
        original = (
            "# top comment: user notes\n"
            "\n"
            "timeout = 42  # inline note\n"
            "\n"
            "[llm]\n"
            "# llm section comment\n"
            'model = "test-model"\n'
        )
        config.write_text(original, encoding="utf-8")

        write_people_list(config, ["Mikko"])

        text = config.read_text(encoding="utf-8")
        assert text == ('people = ["Mikko"]\n' + original)
        # The original content is a contiguous, unmodified suffix.
        assert text[len('people = ["Mikko"]\n') :] == original
        loaded = tomllib.loads(text)
        assert loaded == {
            "people": ["Mikko"],
            "timeout": 42,
            "llm": {"model": "test-model"},
        }

    def test_array_of_tables_dotted_key_inline_table_preserved(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """A config mixing [[array-of-tables]], a dotted key, and an inline
        table is preserved byte-for-byte when no top-level people exists
        (the surgical insert touches only a single added line)."""
        isolate_home(monkeypatch, tmp_path, tmp_path)
        config = _config_path(tmp_path)
        original = (
            "a.b.c = 1\n"
            'opts = {x = 1, y = "z"}\n'
            "[[parts]]\n"
            'name = "one"\n'
            "[[parts]]\n"
            'name = "two"\n'
        )
        config.write_text(original, encoding="utf-8")

        write_people_list(config, ["Aino"])

        text = config.read_text(encoding="utf-8")
        assert text == 'people = ["Aino"]\n' + original
        loaded = tomllib.loads(text)
        assert loaded["a"]["b"]["c"] == 1
        assert loaded["opts"] == {"x": 1, "y": "z"}
        assert loaded["parts"] == [{"name": "one"}, {"name": "two"}]
        assert loaded["people"] == ["Aino"]


# --- Surgical path: replace an existing single-line people ---


class TestSurgicalReplace:
    def test_second_read_failure_falls_back_to_refusal_not_traceback(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """When the config file disappears between the parse and the
        surgical re-read, no traceback escapes: the write is refused,
        one warning is emitted, and the (absent) file is never created."""
        isolate_home(monkeypatch, tmp_path, tmp_path)
        config = _config_path(tmp_path)
        # A config where BOTH the surgical path and the re-emit fail:
        # - table-nested people makes the surgical re-parse check fail
        #   (intended drops the nested key, but the inserted text keeps it)
        # - nested empty table [a.b] makes the re-emit fail (it emits
        #   `b = ""` which re-parses to {"a": {"b": ""}} not {"a": {"b": {}}})
        config.write_text('[llm]\npeople = ["stale"]\n[a]\n\n[a.b]\n', encoding="utf-8")

        original_read_text = Path.read_text
        first_call = [False]

        def _read_text_delete(self: Path, *args: Any, **kwargs: Any) -> str:
            """First read succeeds and deletes the file; second read raises."""
            if first_call[0]:
                raise OSError(2, "No such file")
            first_call[0] = True
            result = original_read_text(self, *args, **kwargs)
            self.unlink()
            return result

        monkeypatch.setattr(Path, "read_text", _read_text_delete)

        _, stderr = _capture_stderr(write_people_list, config, ["Mikko"])

        assert not config.exists()
        assert len(_layout_warnings(stderr)) == 1
        assert "unsupported config layout" in stderr

    def test_single_line_people_replaced_with_comment_kept(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """An existing single-line top-level people array is replaced in
        place; its trailing comment survives; everything else is
        byte-identical."""
        isolate_home(monkeypatch, tmp_path, tmp_path)
        config = _config_path(tmp_path)
        original = 'people = ["Old"]  # old list\nx = 1\n\n[llm]\nmodel = "m"\n'
        config.write_text(original, encoding="utf-8")

        write_people_list(config, ["Old", "New"])

        text = config.read_text(encoding="utf-8")
        assert text == (
            'people = ["Old", "New"] # old list\nx = 1\n\n[llm]\nmodel = "m"\n'
        )
        loaded = tomllib.loads(text)
        assert loaded == {
            "people": ["Old", "New"],
            "x": 1,
            "llm": {"model": "m"},
        }

    def test_top_level_replacement_beats_later_nested_people_line(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """A single-line top-level people + a table-nested stale people:
        the top-level array line is the match candidate, but the
        stale nested key makes the surgical result re-parse to a
        duplicate — so the re-emit fallback is used instead, dropping
        the stale key and keeping exactly one top-level people list."""
        isolate_home(monkeypatch, tmp_path, tmp_path)
        config = _config_path(tmp_path)
        config.write_text(
            'people = ["Top"]\n[llm]\npeople = ["Nested"]\n',
            encoding="utf-8",
        )

        write_people_list(config, ["Top", "New"])

        loaded = tomllib.loads(config.read_text(encoding="utf-8"))
        assert loaded["people"] == ["Top", "New"]
        assert "people" not in loaded["llm"]

    def test_stale_nested_people_with_top_level_list_pinned_by_strict_load(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """Regression test for the lens-review claim (issue #93): a config
        with a single-line top-level people AND a table-nested stale
        people (``[llm]`` → ``people``). Note the claim's literal config
        (top-level people lines AFTER the ``[llm]`` header) is not valid
        TOML: a bare key after a table header belongs to that table, so
        the only valid form is a top-level people line BEFORE the table
        plus the nested one under it — the surgical single-line replace
        path. The replace must not win: ``intended`` already has the
        nested key stripped, so the surgical result fails the re-parse
        check and the re-emit drops it. The result parses, carries both
        names at top level, keeps no nested people key, and passes the
        strict config load (which ``llm.people`` is not a known llm
        key).
        """
        isolate_home(monkeypatch, tmp_path, tmp_path)
        config = _config_path(tmp_path)
        config.write_text(
            'x = 1\npeople = ["Top"]\n[llm]\npeople = ["stale"]\n',
            encoding="utf-8",
        )

        write_people_list(config, ["Top", "New"])

        text = config.read_text(encoding="utf-8")
        loaded = tomllib.loads(text)
        assert loaded["people"] == ["Top", "New"]
        assert "people" not in loaded.get("llm", {})
        assert loaded["x"] == 1
        # The strict load must accept the file without ConfigError: a
        # surviving nested llm.people would fail it (llm.people is not a
        # known llm key).
        with pytest.raises(ConfigError, match="unknown top-level key or section 'x'"):
            _strict_load(config)


# --- Unsupported layout: surgical impossible, file refused unchanged ---


class TestUnsupportedLayoutRefusal:
    def test_unexpressible_layout_is_refused_with_one_warning(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """A config with a multi-line people array and a nested empty
        table cannot be expressed by either the surgical path (can't
        handle multi-line people) or the minimal TOML serializer (can't
        round-trip nested empty tables), so the write is refused: the
        file stays byte-identical and exactly one ``unsupported config
        layout`` warning line is emitted."""
        isolate_home(monkeypatch, tmp_path, tmp_path)
        config = _config_path(tmp_path)
        # The multi-line people array defeats the surgical replace (the
        # regex only matches single-line arrays), and the nested empty
        # table [a.b] defeats the re-emit (it would emit `b = ""` under
        # [a], which re-parses to {"a": {"b": ""}} not {"a": {"b": {}}}).
        original = 'people = [\n    "Old",\n]\n[a]\n\n[a.b]\n'
        config.write_text(original, encoding="utf-8")

        _, stderr = _capture_stderr(write_people_list, config, ["Mikko"])

        assert config.read_text(encoding="utf-8") == original
        assert _layout_warnings(stderr) == [
            "warning: could not update people in "
            f"{config}: unsupported config layout (config left unchanged)"
        ]


# --- Fallback path: re-emit when surgical does not apply ---


class TestFallbackReEmit:
    def test_multiline_people_array_falls_back_safely(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """A multi-line single-line people array the single-line regex
        cannot match is handled by the re-emit fallback; the file still
        parses and carries the new top-level people list."""
        isolate_home(monkeypatch, tmp_path, tmp_path)
        config = _config_path(tmp_path)
        original = (
            'people = [\n    "Old1",\n    "Old2",\n]\nx = 1\n[llm]\nmodel = "m"\n'
        )
        config.write_text(original, encoding="utf-8")

        write_people_list(config, ["New"])

        loaded = tomllib.loads(config.read_text(encoding="utf-8"))
        assert loaded["people"] == ["New"]
        assert loaded["x"] == 1
        assert loaded["llm"] == {"model": "m"}

    def test_stale_nested_people_dropped_via_fallback(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """No top-level people + a table-nested stale people: the surgical
        insert is impossible (it would leave the stale key and fail the
        re-parse check), so the re-emit drops the stale key."""
        isolate_home(monkeypatch, tmp_path, tmp_path)
        config = _config_path(tmp_path)
        original = '[llm]\nmodel = "m"\npeople = ["stale"]\n'
        config.write_text(original, encoding="utf-8")

        write_people_list(config, ["Fresh"])

        loaded = tomllib.loads(config.read_text(encoding="utf-8"))
        assert loaded == {"people": ["Fresh"], "llm": {"model": "m"}}

    def test_scalar_people_falls_back(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """A scalar top-level ``people`` is not a list: the re-emit
        fallback replaces it with the new list."""
        isolate_home(monkeypatch, tmp_path, tmp_path)
        config = _config_path(tmp_path)
        config.write_text('people = "Scalar"\nother = 1\n', encoding="utf-8")

        write_people_list(config, ["Listed"])

        loaded = tomllib.loads(config.read_text(encoding="utf-8"))
        assert loaded == {"people": ["Listed"], "other": 1}


# --- Names survive escaping round-trip ---


class TestEscapingRoundTrip:
    def test_quotes_backslash_unicode_and_control_chars(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """A name containing quotes, a backslash, unicode, and control
        characters survives the write and re-parses to the same string."""
        isolate_home(monkeypatch, tmp_path, tmp_path)
        config = _config_path(tmp_path)
        config.write_text("x = 1\n", encoding="utf-8")
        name = 'Qu"ote\\name \u00e4\u00f6\u00fc \x01\x7f\u0080\u2028'

        write_people_list(config, [name])

        loaded = tomllib.loads(config.read_text(encoding="utf-8"))
        assert loaded["people"] == [name]
        assert loaded["x"] == 1


# --- Atomicity ---


class TestAtomicity:
    def test_no_tmp_file_left_after_write(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """The write uses tmp + os.replace: no *.tmp-* file remains in the
        config directory after a successful write."""
        isolate_home(monkeypatch, tmp_path, tmp_path)
        config = _config_path(tmp_path)
        config.write_text('people = ["Old"]\n', encoding="utf-8")

        write_people_list(config, ["New"])

        leftovers = glob.glob(str(config.parent / "*.tmp-*"))
        assert leftovers == []
        assert tomllib.loads(config.read_text(encoding="utf-8"))["people"] == ["New"]

    def test_no_tmp_file_left_after_refusal(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ):
        """A config layout that is refused leaves neither a partial file
        nor a tmp file behind (byte-identical)."""
        isolate_home(monkeypatch, tmp_path, tmp_path)
        config = _config_path(tmp_path)
        original = "# broken by hand\n[llm\n   = oops\n"
        config.write_text(original, encoding="utf-8")

        write_people_list(config, ["X"])

        # Refused write: byte-identical, no tmp file left behind.
        assert glob.glob(str(config.parent / "*.tmp-*")) == []
        assert config.read_text(encoding="utf-8") == original
