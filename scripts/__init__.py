"""Developer-only helper scripts (corpus generation, FLEURS source).

This file only makes ``scripts`` an importable namespace so the sibling
helper modules (``fleurs_source``, ``wav_header``) can be reached from a
``from scripts.x import ...`` try-block in a sibling script when the script
directory is not on ``sys.path`` (e.g. the tests' importlib load path). It
defines no code of its own and is never imported by ``src/vemoizer``.
"""
