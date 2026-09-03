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

A 64-minute memo processes in ~8 minutes on an M-series Mac, including the
consensus stages.

## Setup

```bash
uv sync --group dev
uv run vemoizer models pull     # pre-download the revision-pinned models (~6 GB)
```

Requirements: macOS on Apple Silicon, Python ≥ 3.11, `ffmpeg` on PATH.

**LLM (optional).** Adjudication and the Markdown notes use any
OpenAI-compatible endpoint, configured in `~/.config/vemoizer/config.toml`:

```toml
[llm]
base_url = "https://api.example.com/v1"
model = "your-model"
api_key_env = "VEMOIZER_LLM_API_KEY"   # name of the env var holding the key
timeout_seconds = 30
```

**Diarization (optional, `--diarize`).** Uses pyannote's gated CC-BY-4.0
weights: accept the license on HuggingFace and provide an access token
before first use. Attribution is printed whenever the stage runs.

## Commands

| Task | Command |
|---|---|
| Transcribe | `uv run vemoizer transcribe memo.m4a` |
| Pick formats | `uv run vemoizer transcribe memo.m4a --format txt,md` |
| Speaker labels | `uv run vemoizer transcribe memo.m4a --diarize` |
| WER regression gate | `uv run vemoizer eval --backend all --check` |
| Pre-download models | `uv run vemoizer models pull` |

## Accuracy is measured, not asserted

`vemoizer eval` scores each decode backend and the consensus over a
committed Finnish speech corpus and gates PRs against
`tests/fixtures/wer_baseline.json`. No model becomes a default without a
WER run on this corpus.

## Contributing

See `CONTRIBUTING.md` for the human workflow and `AGENTS.md` for the
quality gates and project invariants. The canonical stage contract, model
IDs and pinned revisions live in `docs/pipeline-spec.md`.
