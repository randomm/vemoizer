# AGENTS.md

Instructions for AI coding agents (pi, Claude Code, etc.) working in this repo.
Human-oriented workflow guidance lives in `CONTRIBUTING.md`; this file defines
quality standards, workflows, and project constraints for agent work.

Last updated: 2026-09-02

**Maintenance note:** If you add or swap an ASR backend, change the audio
contract, or alter the consensus pipeline, update "Project Invariants" below
and `docs/pipeline-spec.md` in the same PR.

## What vemoizer Is

A local-first CLI that turns voice memos (iOS Voice Memos `.m4a`, typically
1-60 minutes) into text and Markdown. The hard problem is not English
accuracy; it is **Finnish with English seeping in** — acronyms, product
names, and technical terms embedded in Finnish prose.

The answer is a **consensus pipeline**: decode twice with different models,
find the spans where they disagree, re-decode only those spans with a third
model, and let a configured LLM adjudicate using surrounding context.

```
.m4a -> ffmpeg -> 16 kHz mono float32
     -> VAD (chunk long memos, drop silence)
     -> decode A: parakeet-tdt-0.6b-v3  (auto language ID, word timestamps)
     -> decode B: canary-1b-v2          (strongest off-the-shelf Finnish)
     -> align A and B on word timestamps + string similarity
     -> flag disputed spans (A != B, low token logprob, or LID flip)
     -> re-decode ONLY disputed slices with whisper-large-finnish-v3
     -> LLM adjudication over candidates + context
     -> diarization -> speaker labels
     -> LLM cleanup / summary -> text + Markdown
```

This is affordable because disputed spans are seconds long, not minutes, and
Parakeet runs at roughly 100x realtime, so the second decode is nearly free.

## Context7 Protocol

Before writing code that touches an external library API, check Context7 for
current documentation. Training data may be outdated; Context7 has the
authoritative current docs.

This repo's real external surface:

- `parakeet-mlx` — model loading, `generate()`, word-timestamp alignment API
- `mlx-audio` — the Canary-1b-v2 loading path on Apple Silicon
- `mlx-whisper` — `transcribe()` kwargs, `convert()` for HF checkpoints
- `silero-vad` — chunking API, ONNX vs JIT model selection
- `pyannote.audio` — 4.x pipeline API and its MPS behavior
- `huggingface_hub` — `snapshot_download`, revision pinning, offline mode
- `httpx` — request/timeout/error semantics for the LLM client
- `pytest`, `ruff` — invocation flags and config keys

Skip Context7 for pure-stdlib changes (alignment math, span bookkeeping,
output formatting).

## Minimalist Engineering Philosophy

**Every line of code is a liability.** Before creating anything:

- **LESS IS MORE**: Question necessity before creation
- **Challenge Everything**: Ask "Is this truly needed?" before implementing
- **Minimal Viable Solution**: Build the simplest thing that fully solves the problem
- **No Speculative Features**: Don't build for "future needs" - solve today's problem
- **Prefer Existing**: Reuse existing code/tools before creating new ones
- **One Purpose Per Component**: Each function/module should do one thing well

### Pre-Creation Challenge (MANDATORY)

Before creating ANY code, ask:
1. Is this explicitly required by the GitHub issue?
2. Can existing code/tools solve this instead?
3. What's the SIMPLEST way to meet the requirement?
4. Will removing this break core functionality?
5. Am I building for hypothetical future needs?

**If you cannot justify the necessity, DO NOT CREATE IT.**

## Software File Size

**Every file is a liability.** Long files delay mental context-switching,
discourage small PRs, and make blast-radius analysis harder.

| Category | Hard limit | Ideal |
|---|---|---|
| Source code (`.py`) | 500 lines | 300 lines |
| Test files (`.py`) | 800 lines | 400 lines |
| Config files (`.toml`, `.yml`, etc.) | 200 lines | 100 lines |

Audio fixtures and their reference transcripts under `tests/fixtures/` are
exempt — they are reference data, not application code.

**How to count:** Line count as reported by `wc -l` is the enforcement metric.

**Refactoring approach:** When a file grows toward the limit, extract a
**single responsibility** (e.g. "span alignment", "VAD chunking", "LLM
client") rather than a mechanical line-count chop.

## Pre-Push Quality Gates

**CI is for VERIFICATION, not DISCOVERY.** All checks must pass locally before
`git push`. Never push to "see if CI catches anything."

```bash
uv run pytest tests/
uv run ruff check src/ tests/
uv run ty check src/ tests/
uv run ruff format --check src/ tests/
```

If you changed an ASR backend, the audio contract, or the alignment logic,
additionally run the WER regression over the fixture corpus and paste the
before/after table into the PR body:

```bash
uv run vemoizer eval --corpus tests/fixtures/corpus
```

Do NOT:
- ❌ Use `--no-verify` to skip hooks
- ❌ Push with failing tests, lint, type-check, or format checks
- ❌ Commit formatting fixes as separate follow-up commits — format before the
  original commit
- ❌ Add a blanket `# noqa` or any `# type: ignore` to make a gate pass

## Testing Standards

- **TDD preferred**: write the failing test first when the behavior is
  well-specified. Alignment, span selection, and output formatting are all
  pure functions and are trivially testable this way.
- **Coverage threshold**: 80%+ for new code. Measure with:
  ```bash
  uv run pytest --cov=vemoizer --cov-report=term-missing tests/
  ```
- **Unit tests MUST NOT download models or touch the network.** Mock the
  `Transcriber` implementations. A test that needs real weights belongs
  behind a marker and is opt-in:
  ```bash
  uv run pytest -m models      # opt-in, requires a warm HF cache
  ```
- **Audio fixtures stay small.** A few seconds of speech per fixture, checked
  in as 16 kHz mono WAV. Never commit a real personal memo.
- **WER is a gate, not a vibe.** Accuracy claims in a PR must come from
  `vemoizer eval` output, not from a model card.
- CI runs on `macos-15` (pinned, not `macos-latest`) because the MLX stack is
  Apple Silicon only.

## Code Style & Conventions

- Python >= 3.11. Type hints required on public functions (`str | None`
  syntax, not `Optional`/`Union`).
- Ruff and ty rules are defined in `pyproject.toml`; do not paste the config
  here.
- Package layout: `src/vemoizer/`.
- **The `Transcriber` Protocol is the backend seam.** Every ASR model is
  reached through it. Never call a model library directly from pipeline or
  CLI code. This is what makes the consensus pipeline and the eval harness
  possible.
- No bare `except:`. Catch specific exception types.
- Model loading is slow; load lazily and log the wait. Never load a model at
  import time.
- Keep functions small; prefer explicit over clever.
- **No blanket suppressions**: no bare `# noqa` and no `# type: ignore`.
  A per-line `# noqa: <CODE> - <reason>` is acceptable only with a real
  justification; fix the underlying issue otherwise.

## Git Workflow

- Trunk-based: short-lived branches off `main`.
- **Branch naming**: `feature/issue-N-short-slug`, `fix/issue-N-short-slug`,
  `chore/issue-N-short-slug`.
- **Conventional commits**: `feat:`, `fix:`, `docs:`, `chore:`, `test:`,
  `refactor:`, `perf:`.
- **PR body must contain `Fixes #N`** (or `Closes #N`) to auto-close the
  linked issue. A `(#N)` in a commit scope is NOT a close keyword.
- Squash-merge by default. **Never commit to `main` directly.** No force-push
  to `main`.

**Agent merge policy:** Agents may squash-merge a PR once (a) CI is green on
all jobs, (b) all four quality gates passed locally, and (c) the PR has one
approving review. Pre-existing failures are not grounds for an exemption.
When any condition is unmet, stop at PR creation, apply the
`needs-human-attention` label, and notify the operator. **When in doubt, do
not merge.**

## Documentation Policy

### The 200-PR Test

Before creating documentation, ask: "Will this be true in 200 PRs?"

- **YES** (principle that endures) → document the principle (WHY)
- **NO** (implementation detail) → skip, or use a code comment (WHAT/HOW)

### Forbidden Documentation

- ❌ ALL_CAPS scratch files (`PLAN.md`, `RESEARCH.md`, `ANALYSIS.md`,
  `SUMMARY.md`, `IMPLEMENTATION_PLAN.md`, `TODO.md`, …) — agent work
  artifacts stay out of git. If work must be deferred, create a GitHub
  issue: **the issue IS the TODO.**
- ❌ Duplicating a canonical source. Model IDs, revisions, and the stage
  contract live in `docs/pipeline-spec.md`; ruff/ty config lives in
  `pyproject.toml` — point there, never restate.

### Documentation Locations

- Root: `README.md`, `AGENTS.md`, `CONTRIBUTING.md`, `SECURITY.md`
- Concept/spec docs: `docs/` with lowercase-hyphenated names
- Code comments: WHY, not WHAT

## Commands Reference

| Task | Command |
|------|---------|
| Install deps | `uv sync --group dev` |
| Run tests | `uv run pytest tests/` |
| Coverage report | `uv run pytest --cov=vemoizer --cov-report=term-missing tests/` |
| Model-backed tests (opt-in) | `uv run pytest -m models` |
| Lint | `uv run ruff check src/ tests/` |
| Lint + autofix | `uv run ruff check --fix src/ tests/` |
| Type check | `uv run ty check src/ tests/` |
| Format | `uv run ruff format src/ tests/` |
| Format check | `uv run ruff format --check src/ tests/` |
| Regenerate lockfile | `uv lock` |
| Transcribe a memo | `uv run vemoizer transcribe memo.m4a --out memo.md` |
| WER regression | `uv run vemoizer eval --corpus tests/fixtures/corpus` |
| Pre-download models | `uv run vemoizer models pull` |

Every `vemoizer` subcommand row above is the target CLI surface, not a
promise that it exists yet. The canonical flag spec is
`docs/pipeline-spec.md`.

## Project Invariants

1. **Transcription is local, full stop.** Audio and transcripts never leave
   the machine for ASR. The only network access in the ASR path is the
   one-time model download. There is no cloud-ASR fallback, and adding one
   is a product decision, not an implementation detail.

2. **The consensus pipeline IS the architecture.** Dual decode, timestamp
   alignment, disputed-span detection, targeted re-decode, LLM adjudication.
   Do not collapse it to a single decode as a "simplification" — it is the
   reason this project exists. Individual stages may be skipped by flag for
   speed; none may be deleted.

3. **Never force a single language on a memo.** Finnish with English
   code-switching is the normal case, not an edge case. Any code path that
   pins one language for a whole file is a bug. Language is a property of a
   span, not of a recording.

4. **Model weights are revision-pinned.** Download with
   `snapshot_download(repo_id, revision=<sha>)` and load from the returned
   local path, never from the bare repo ID. Upstream pushes must not be able
   to change the weights we run. This also keeps loading working under
   `HF_HUB_OFFLINE=1`.

5. **The LLM is optional, configured, and OpenAI-compatible.** No hardcoded
   provider, no hardcoded model ID, no hardcoded base URL in source. It is
   read from the config file, the API key from the environment. Every LLM
   call **fails open**: on timeout, error, or missing config, return the
   un-adjudicated transcript rather than failing the run.

6. **Audio contract: 16 kHz mono float32.** Decoding happens once, at ingest,
   via ffmpeg. Every internal boundary past that point speaks this format.
   No stage re-reads the source file.

7. **No model becomes a default without a WER run on our own corpus.**
   Published WERs come from model cards using different text-normalization
   pipelines and are not comparable. Treat them as a shortlist, never as a
   ranking.

## Never Commit

- Real personal voice memos, or transcripts derived from them
- Model weights, GGUF/MLX conversions, or anything else belonging in the
  HuggingFace cache
- API keys or tokens (the LLM key is read from the environment)
- Agent scratch files (see Forbidden Documentation)

## Environment

- **Apple Silicon only.** The MLX stack has no Intel path. Do NOT add x86
  compatibility code; fail fast with a clear message instead.
- Python >= 3.11 (authoritative source: `requires-python` in `pyproject.toml`).
- `ffmpeg` on PATH — the only non-Python system dependency.
- Models live in the HuggingFace cache (`~/.cache/huggingface/hub`). Budget
  roughly 5-6 GB for the three-model consensus set.
- LLM configuration lives in a user config file and selects any
  OpenAI-compatible endpoint. The API key is read from an environment
  variable named in that config; it is never stored in the repo.

## Cross-references

- `README.md` — what vemoizer is, for users
- `CONTRIBUTING.md` — human contributor workflow
- `SECURITY.md` — privacy and trust boundary
- `docs/pipeline-spec.md` — canonical stage contract, model IDs and pinned
  revisions, CLI flag spec, config-file schema
- `~/projects/kuiskaus` — sibling project; source of the `Transcriber`
  Protocol, the revision-pinning pattern, and the `macos-15` CI job
