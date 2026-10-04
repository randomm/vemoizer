# vemoizer

Local-first transcription for voice memos — built for **Finnish with English
seeping in**: acronyms, product names, and technical terms embedded in
Finnish prose. Runs entirely on your Mac (Apple Silicon, MLX); your audio
never leaves the machine for ASR.

```bash
uv run vemoizer transcribe memo.m4a
```

writes `memo.txt`, `memo.json`, `memo.srt`, `memo.vtt` and — when an LLM is
configured — `memo.md`: a Markdown note with a title, summary, action items,
and the paragraphed transcript.

## How it works

One decode is never enough for code-switched Finnish, so vemoizer runs a
**consensus pipeline**: decode twice with different model families, find
where they disagree, re-decode only those disputed slices with a third
(Finnish-fine-tuned) model, and let a configured LLM adjudicate using the
surrounding context.

```
.m4a → ffmpeg → 16 kHz mono float32
     → VAD (silero) → speech slices
     → decode A: Parakeet TDT 0.6B v3   (word timestamps)
     → decode B: Canary-1b-v2           (per-slice language auto-detection)
     → slice-level dispute detection    (normalized text similarity)
     → re-decode disputed slices: Whisper-large Finnish v3
     → LLM adjudication (optional, fails open)
     → optional speaker diarization (pyannote, CC-BY gated weights)
     → txt / json / srt / vtt / md
```

Every stage **fails open**: no LLM key, no re-decode model, no diarization
token — you still get a complete transcript.

Wall-clock time scales with memo length and Mac model; the consensus
stages make it slower than a single decode. Each transcribed file also
writes a full per-file log to `./.vemoizer/logs/<name>.log` (regardless of
`-v`), with HuggingFace tokens and LLM API keys redacted.

## Install as a tool

Install the `vemoizer` command on your PATH and pre-download the
revision-pinned models (per-model sizes in the
`docs/pipeline-spec.md` → "Model manifest" table):

```bash
uv tool install --editable ~/projects/vemoizer
vemoizer models pull
vemoizer doctor           # optional: verify the local setup
```

## Setup (developers)

```bash
uv sync --group dev
uv run vemoizer models pull     # pre-download the revision-pinned models
```

Requirements: macOS on Apple Silicon, Python ≥ 3.11, `ffmpeg` on PATH.

**LLM (optional).** Adjudication and the Markdown notes use any
OpenAI-compatible endpoint. The config file is searched in order: the
nearest `./.vemoizer/config.toml` (walked up from the current directory),
then `~/.vemoizer/config.toml`; the legacy `~/.config/vemoizer/config.toml`
still works with a deprecation notice:

```toml
[llm]
base_url = "https://api.example.com/v1"
model = "your-model"
api_key_env = "VEMOIZER_LLM_API_KEY"   # name of the env var holding the key
timeout_seconds = 30
```

**Fully offline LLM (Ollama / llama.cpp).** The same config works with a
local LLM server, making vemoizer fully offline — including adjudication
and the Markdown notes. When the named env var is unset or empty the LLM
stages are skipped entirely (no request is made); to get a fully offline
run, point `api_key_env` at a variable that is *set* — a placeholder
value is fine, since local servers such as Ollama do not enforce auth:

```toml
[llm]
base_url = "http://localhost:11434/v1"   # Ollama
model = "qwen2.5:14b"
api_key_env = "OLLAMA_API_KEY"            # set it (any value works)
timeout_seconds = 120
```

```bash
export OLLAMA_API_KEY="ollama"   # placeholder; Ollama ignores it
```

llama.cpp server works the same way: `base_url = "http://localhost:8080"`.
See `docs/pipeline-spec.md` → "Fully-offline LLM" for the full spec.

**Diarization (optional, `--diarize`).** Uses pyannote's gated CC-BY-4.0
weights: accept the license on HuggingFace and provide an access token
before first use. Attribution is printed whenever the stage runs.

## Commands

`meeting` and `memo` are the recommended entry points: they write dated
`YYYY-MM-DD <title>` outputs (`.md` + `.json`). `transcribe` is the expert
flag surface (stem-named outputs, per-file format selection).

To name speakers after a run: `vemoizer names <x.json>` (interactive) or
`vemoizer render --name LABEL=NAME <x.json>`; put default names in the
`people = [ ... ]` list of the config file.

## Accuracy is measured, not asserted

`vemoizer eval` scores each decode backend and the consensus over a
committed Finnish speech corpus and gates PRs against
`tests/fixtures/wer_baseline.json`. No model becomes a default without a
WER run on this corpus.

## Contributing

See `CONTRIBUTING.md` for the human workflow and `AGENTS.md` for the
quality gates and project invariants. The canonical stage contract, model
IDs and pinned revisions live in `docs/pipeline-spec.md`.
