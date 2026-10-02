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

All quality gates must pass locally before pushing; CI is for
verification, not discovery. The exact commands live in the "Pre-Push
Quality Gates" section of `AGENTS.md` — run those, not a local copy. If
your change touches an ASR backend, the audio contract, alignment,
spans, re-decode, or the LLM stage, the WER regression gate's output goes
in the PR body, and baseline updates land in a dedicated commit, never
mixed into a feature commit.

## Testing

- Unit tests must not download models or touch the network.
- Model-backed tests are opt-in (`pytest -m models`); audio fixtures stay
  small, and accuracy claims come from `vemoizer eval` output, not from a
  model card. The full testing standards live in `AGENTS.md`.

## Git workflow

The branch naming, conventional commit, and merge rules live in the
"Git Workflow" section of `AGENTS.md` — follow it there, including the
`Fixes #N` / `Closes #N` PR-body requirement and the no-direct-commits
to `main` rule.

## Documentation

- The 200-PR test: if a fact will not still be true after 200 PRs, do
  not document it — put a code comment instead.
- Do not restate the canonical sources, and do not create ALL_CAPS
  scratch files — deferred work becomes a GitHub issue. Full rules: the
  "Documentation Policy" section of `AGENTS.md`.

## Things we never commit

Real personal memos and their transcripts, model weights and cache
contents, API keys and tokens (the LLM key is read from the environment),
and agent scratch files. The authoritative list is the "Never Commit"
section of `AGENTS.md`.
