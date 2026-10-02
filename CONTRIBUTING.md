# Contributing to vemoizer

A short, human-oriented workflow for contributing to vemoizer. The quality
gates, project invariants, and agent-facing constraints live in
`AGENTS.md`; the canonical stage contract, model IDs, and CLI flag spec
live in `docs/pipeline-spec.md`. This file is the pointer, not the
restatement.

## Before you start

1. **Read `AGENTS.md`.** It covers the minimalist philosophy, the
   pre-creation challenge, file-size limits, quality gates, testing
   standards, code style, and the list of things we never commit.
2. **Read `docs/pipeline-spec.md`.** It is the single source of truth for
   the pipeline: audio contract, stage semantics, model revisions, CLI
   flags, config schema, and invariants.
3. **Read `SECURITY.md`.** It defines the local-only trust boundary —
   audio and transcripts never leave the machine for ASR, and the LLM
   key stays in the environment.

If a proposed change touches the ASR backends, the audio contract, the
consensus pipeline, or the LLM stage, update `AGENTS.md` and
`docs/pipeline-spec.md` in the same PR.

## Development setup

```bash
uv sync --group dev
```

Requires macOS on Apple Silicon, Python ≥ 3.11, and `ffmpeg` on PATH.

## Local quality gates

Run all four before pushing. CI verifies; it does not discover.

```bash
uv run pytest tests/
uv run ruff check src/ tests/
uv run ty check src/ tests/
uv run ruff format --check src/ tests/
```

If your change touches an ASR backend, the audio contract, alignment,
spans, re-decode, or the LLM stage, also run the WER regression gate and
paste its output into the PR body:

```bash
uv run vemoizer eval --backend all --check
```

Improvements are recorded with `--update-baseline` in a dedicated commit,
never mixed into a feature commit.

## Testing

- Unit tests must not download models or touch the network.
- Model-backed tests are opt-in: `uv run pytest -m models`.
- Audio fixtures stay small (a few seconds of speech per fixture, 16 kHz
  mono WAV). Never commit a real personal memo or a transcript derived
  from one.
- Accuracy claims in a PR come from `vemoizer eval` output, not from a
  model card.

## Git workflow

- Short-lived branches off `main`.
- Branch naming: `feature/issue-N-short-slug`, `fix/issue-N-short-slug`,
  or `chore/issue-N-short-slug`.
- Conventional commit types: `feat:`, `fix:`, `docs:`, `chore:`, `test:`,
  `refactor:`, `perf:`.
- The PR body must contain `Fixes #N` or `Closes #N` to auto-close the
  linked issue; a `(#N)` in a commit scope is not a close keyword.
- Squash-merge by default. Never commit directly to `main`; never
  force-push to `main`.

## Documentation

- The 200-PR test: if a fact will not still be true after 200 PRs, do
  not document it — put a code comment instead.
- Do not restate the canonical sources. Model IDs, revisions, and the
  stage contract belong in `docs/pipeline-spec.md`; ruff/ty config
  belongs in `pyproject.toml`.
- Do not create ALL_CAPS scratch files in the repo. If work must be
  deferred, file a GitHub issue — the issue is the TODO.

## Things we never commit

- Real personal voice memos, or transcripts derived from them.
- Model weights, GGUF/MLX conversions, or anything that belongs in the
  HuggingFace cache.
- API keys or tokens. The LLM key is read from the environment.
- Agent scratch files.
