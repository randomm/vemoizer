"""The ``meeting`` and ``memo`` presets (issue #82, M2).

``resolve_options`` is the pure core of the new commands: it takes
*already-parsed* layer dicts and CLI overrides and returns a
:class:`RunOptions` ready for ``batch.run_batch`` (batch.py, sibling
workstream). No file reads, no printing, no config search here — that
I/O lives in ``glossary_layers`` (load_layers / merge) and ``llm``
(config search), and the batch runner calls :func:`resolve_options`
once and applies the result per file.

The composition the presets pin down (all on existing seams —
``pipeline.transcribe_file`` is unchanged):

- **meeting**: the ``meeting`` profile (whisper decode A, no consensus),
  diarization on (``2-6`` people), and the LLM repair pass. The
  glossary is the merged layers (prompt terms, ``@``-LLM-only terms,
  correction pairs) or the single ``--glossary`` file, which replaces
  both layers entirely.
- **memo**: the same whisper meeting decode, but for a solo memo — no
  diarization, no repair. The whisper ``initial_prompt`` stays empty
  (a 30-minute memo should not seed recognition with 500 prompt terms):
  the temp glossary file the batch runner writes contains ONLY the
  project layer's correction pairs, so ``glossary_prompt`` yields
  ``None`` (empty prompt) while ``apply_corrections`` still fires on
  the deterministic pairs (glossary.py: correction lines contribute
  nothing to the prompt).

Precedence (issue #82): built-in defaults < layers < CLI overrides.
The layers themselves merge with project layer first: on conflict the
project value wins.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from vemoizer.diarization import SpeakerCount

#: Preset names ``resolve_options`` understands; the CLI's ``meeting`` /
#: ``memo`` commands each pass their own name.
COMMANDS = ("meeting", "memo")

#: Default diarization bounds for meetings (people join and leave).
MEETING_SPEAKERS: tuple[int, int] = (2, 6)


@dataclass(frozen=True)
class RunOptions:
    """The resolved pipeline options for one file of a preset run.

    ``glossary_path`` is the path the batch runner hands to
    ``transcribe_file``: the explicit ``--glossary`` file when given,
    otherwise the temp file the batch runner writes with the merged
    content (meeting: terms + ``@`` lines + pairs; memo: project
    correction pairs only). ``None`` means no glossary at all.

    ``whisper_prompt`` and ``llm_terms`` split the merged terms the way
    glossary.py does: bare terms seed the whisper ``initial_prompt``
    (token-budgeted downstream by ``glossary.glossary_prompt``), while the
    ``@``-prefixed terms (``@`` stripped) go to the LLM stages
    (repair / notes) at no budget — they are spelling references, not
    recognition seeds.

    ``config_path`` is the explicit ``--config`` path when given, otherwise
    ``None``: the layered config search runs (``llm.load_default_config`` —
    nearest ``./.vemoizer/config.toml`` (walk-up, project wins) →
    ``~/.vemoizer/config.toml`` → legacy ``~/.config/vemoizer/config.toml``
    and other legacy paths). ``"os.devnull"`` remains a valid *explicit*
    value (e.g. for the eval harness); the presets themselves never emit
    it, because short-circuiting the search would bypass the LLM/notes/repair
    configuration in the user's ``.vemoizer`` config files.
    """

    profile: str
    diarize: bool
    repair: bool
    speakers: SpeakerCount | None
    glossary_path: str | None
    config_path: str | None
    whisper_prompt: list[str]
    llm_terms: list[str]
    corrections: dict[str, str]


def _split_terms(terms: list[str]) -> tuple[list[str], list[str]]:
    """Split merged terms into (whisper prompt terms, LLM-only terms).

    ``@``-prefixed terms are LLM-only (issue #82): never in the whisper
    prompt, passed to repair and notes with the ``@`` stripped. The
    whisper side keeps file order — it is load-bearing for
    ``glossary_prompt``'s priority inversion (last-listed term is
    highest priority).
    """
    prompt_terms: list[str] = []
    llm_terms: list[str] = []
    for term in terms:
        if term.startswith("@"):
            body = term[1:].strip()
            if body:
                llm_terms.append(body)
        else:
            prompt_terms.append(term)
    return prompt_terms, llm_terms


def _merge_layers(layers: dict[str, dict] | None) -> tuple[list[str], dict[str, str]]:
    """Merge the given layer dicts, project layer winning.

    Returns ``(terms, corrections)``: terms are the project layer's terms
    first (project priority for the whisper prompt budget), then the
    remaining home terms; corrections are the home layer's pairs, with
    the project layer's right side winning on the same wrong side. The
    dedupe of terms (case-insensitive, project spelling kept) and the
    token budget with drop-notices are ``glossary_layers.merge``'s job
    (sibling workstream) — it returns exactly this shape; this helper is
    the fallback for callers that skip the layered merge (e.g. unit
    tests passing a single layer dict).
    """
    layers = layers or {}
    home_terms = list(layers.get("home", {}).get("terms", []))
    project_terms = list(layers.get("project", {}).get("terms", []))
    seen = {t.lower() for t in project_terms}
    # Project terms first (highest priority), then home terms the project
    # did not already cover (case-insensitive, project spelling wins).
    terms = project_terms + [t for t in home_terms if t.lower() not in seen]

    corrections = dict(layers.get("home", {}).get("corrections", {}))
    corrections.update(layers.get("project", {}).get("corrections", {}))
    return terms, corrections


def resolve_options(
    command: str,
    layers: dict[str, dict] | None = None,
    cli_overrides: dict[str, object] | None = None,
) -> RunOptions:
    """Resolve the preset options for *command* (``"meeting"`` / ``"memo"``).

    Pure: *layers* and *cli_overrides* are already-parsed dicts — no file
    I/O, no printing, no config search. ``cli_overrides`` may carry
    ``glossary``, ``config``, ``profile``, ``diarize``, ``repair``,
    ``speakers`` (parsed ``SpeakerCount``) and ``glossary_terms`` /
    ``glossary_corrections`` (the merged result the batch runner already
    computed via ``glossary_layers.merge``); anything omitted falls back
    to the layer values, then to the preset defaults.
    """
    if command not in COMMANDS:
        known = ", ".join(COMMANDS)
        raise ValueError(f"unknown preset command {command!r} (known: {known})")
    overrides: dict[str, object] = dict(cli_overrides or {})
    if command == "meeting":
        base = RunOptions(
            profile="meeting",
            diarize=True,
            repair=True,
            speakers=MEETING_SPEAKERS,
            glossary_path=None,
            # None: the layered config search runs (see class docstring).
            config_path=None,
            whisper_prompt=[],
            llm_terms=[],
            corrections={},
        )
    else:
        # memo: whisper meeting decode, no diarization, no repair; the
        # whisper prompt stays empty (see module docstring).
        base = RunOptions(
            profile="meeting",
            diarize=False,
            repair=False,
            speakers=None,
            glossary_path=None,
            # None: the layered config search runs (see class docstring).
            config_path=None,
            whisper_prompt=[],
            llm_terms=[],
            corrections={},
        )
    # --glossary REPLACES both layers entirely (no merging): when it is
    # given, the single file is passed straight through as glossary_path
    # and the layer terms/corrections are ignored — the file's own
    # contents are what the pipeline reads.
    glossary_path = overrides.get("glossary")
    if glossary_path is not None:
        terms: list[str] = []
        corrections: dict[str, str] = {}
    else:
        terms, corrections = _merge_layers(layers)
        merged_terms = overrides.get("glossary_terms")
        if isinstance(merged_terms, list):
            terms = [str(t) for t in merged_terms]
        merged_corrections = overrides.get("glossary_corrections")
        if isinstance(merged_corrections, dict):
            corrections = {str(k): str(v) for k, v in merged_corrections.items()}
    prompt_terms, llm_terms = _split_terms(terms)
    if command == "memo":
        # Memo seam: the whisper prompt is always empty; the glossary file
        # carries ONLY the project layer's correction pairs (the batch
        # runner writes them; glossary_prompt then yields None).
        prompt_terms = []
    config_path = overrides.get("config")

    def _opt(key: str, default: object) -> object:
        # A None override means "the CLI didn't set this flag" — use the
        # preset default (None is not a valid CLI value for any of these).
        value = overrides.get(key)
        return value if value is not None else default

    return replace(
        base,
        profile=str(_opt("profile", base.profile)),
        diarize=bool(_opt("diarize", base.diarize)),
        repair=bool(_opt("repair", base.repair)),
        speakers=_opt("speakers", base.speakers),
        glossary_path=(
            str(glossary_path) if glossary_path is not None else base.glossary_path
        ),
        config_path=(str(config_path) if config_path is not None else base.config_path),
        whisper_prompt=prompt_terms,
        llm_terms=llm_terms,
        corrections=dict(corrections),
    )
