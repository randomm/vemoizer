# vemoizer pipeline spec

Canonical stage contract, model IDs and pinned revisions, CLI flag spec, and
configuration schema. This document is the single source of truth for the
pipeline — `AGENTS.md` and `CONTRIBUTING.md` point here rather than restating
these details.

Last updated: 2026-09 (issue #15 — spec corrections: Parakeet repo IDs,
Canary load path, CC-BY diarization, runtime environment).

## Overview

vemoizer is a local-first CLI that turns voice memos (iOS Voice Memos `.m4a`,
typically 1–60 minutes) into text and Markdown. The hard problem is Finnish
with English code-switching — acronyms, product names, and technical terms
embedded in Finnish prose. The answer is a consensus pipeline: decode twice
with different models, find the spans where they disagree, re-decode only
those spans with a third model, and let a configured LLM adjudicate.

```
.m4a -> ffmpeg -> 16 kHz mono float32
     -> VAD (chunk long memos, drop silence)
     -> decode A: Parakeet TDT 0.6B v3  (auto language ID, word timestamps)
     -> decode B: Canary-1b-v2          (strongest off-the-shelf Finnish)
     -> align A and B on word onsets (DTW)
     -> flag disputed spans (low word similarity, or reported LID flip)
     -> re-decode ONLY disputed slices with Whisper-large Finnish v3
     -> LLM adjudication over candidates + context (optional, fails open)
     -> diarization (speaker labels, CC-BY gated weights)
     -> LLM cleanup / summary (optional, fails open) -> text + Markdown
```

This is affordable because disputed spans are seconds long, not minutes, and
Parakeet runs at roughly 100× realtime, so the second decode is nearly free.

The pipeline is the architecture (project invariant #2). Individual stages
may be skipped by flag for speed; none may be deleted.

## Audio contract

**16 kHz mono float32** (invariant #6). Decoding happens once, at ingest,
via ffmpeg. Every internal boundary past that point speaks this format; no
stage re-reads the source file.

Ingest argv (see `src/vemoizer/ingest.py`):

```
ffmpeg -nostdin -v error -ac 1 -ar 16000 -c:a pcm_f32le -f f32le - <input>
```

- `-nostdin` — never block on stdin
- `-v error` — only surface real errors
- `-ac 1 -ar 16000` — force mono, 16 kHz
- `-c:a pcm_f32le -f f32le -` — raw little-endian float32 PCM on stdout
- Duration is derived from the raw PCM byte count (`len(raw) // 4` samples),
  never from ffprobe: iOS Voice Memos carry edit lists that make container
  metadata lie.

## Stages

### 1. Ingest

`src/vemoizer/ingest.py`. Decodes any ffmpeg-readable container to the audio
contract. Pure subprocess + numpy: no model loading, no network.

### 2. VAD (silero-vad, ONNX)

`src/vemoizer/vad.py`. `silero-vad==6.2.1` in ONNX mode (via `onnxruntime`,
no torch): 512-sample (32 ms) windows at 16 kHz, per-window speech
probabilities, silero's reference state machine (threshold 0.5, min speech
250 ms, min silence 100 ms, pad 30 ms). Long memos are fed in 60-second
slices so memory stays bounded.

silero-vad natively supports 8/16 kHz only; other rates are rejected. The
VAD model weights ship inside the `silero-vad` pip package
(`silero_vad.onnx`) — no separate download.

### 3. Decode A — Parakeet TDT 0.6B v3

`src/vemoizer/parakeet_transcriber.py`.

- Base model: **`nvidia/parakeet-tdt-0.6b-v3`** (NVIDIA; NOT `microsoft/`).
- Load path: the MLX community port **`mlx-community/parakeet-tdt-0.6b-v3`**
  via the `parakeet-mlx` package (`from_pretrained` on the downloaded
  local path). The repo ID `nvidia/parakeet-tdt-0.6b-v3` names the upstream
  model the port is derived from; the load repo is the MLX port.
- Pinned revision: `ed2b7e8c15f9aaa0b5772e2efb986255eaef7e15`.
- 25 languages including Finnish, with internal auto language ID (the
  `parakeet-mlx` `AlignedResult` API does not surface the detected language,
  so per-span LID is not reported by this stage).
- Word timestamps are built in (CTC alignment): `AlignedResult.tokens`
  gives flat `{text, start, end}` per word; `sentences` gives
  `{text, start, end}` per sentence.
- Audio in: 16 kHz mono float32; mel features via
  `parakeet_mlx.audio.get_logmel(mx.array(audio), model.preprocessor_config)`.

### 4. Decode B — Canary-1b-v2

- Base model: **`nvidia/canary-1b-v2`** (~1.0 GB, 25 languages including
  Finnish; no published independent Finnish WER — the best evidence is a
  Finnish finetune, not this base model).
- **Load path correction:** the `mlx-audio` Canary module loads
  **MLX-formatted community ports** of Canary (e.g.
  `Mediform/canary-1b-v2-mlx-q8`, `base_model: nvidia/canary-1b-v2`),
  **not** the `nvidia/canary-1b-v2` F32 checkpoint directly. "mlx-audio
  loads canary-1b-v2" is an abbreviation that elides this — the correct
  statement is "the Canary path loads a community MLX port of
  `nvidia/canary-1b-v2`" (via mlx-audio's Canary module or an equivalent
  direct MLX port load).
- Word timestamps are not built in; the output normalizes to the
  `TranscriptionResult` contract before reaching alignment (`words` is
  optional in that contract).

### 5. Alignment (DTW on word onsets)

`src/vemoizer/alignment.py`. Pure, model-free DTW over the word onset lists
from decodes A and B produces a monotonic word pairing; each pair carries a
case/punctuation-insensitive similarity cost. One side may be a gap
(insertion/deletion).

### 6. Disputed-span flagging

`src/vemoizer/spans.py`. Decides which time ranges to re-decode:

- A pair is disputed when its character-level longest-common-subsequence
  similarity (casefold + punctuation stripped, normalized by the longer
  word) is **strictly below 0.75** (`DISPUTE_THRESHOLD`). The boundary
  value itself is not disputed.
- **LID flip:** two *reported* language tags that differ mark the pair
  disputed even when the texts match (invariant #3: language is a property
  of a span). A missing tag is not a reported language and never triggers a
  flip.
- Disputed slices that overlap or sit within 0.5 s (`SPAN_MERGE_GAP_S`) of
  each other are merged into one slice; slightly over-merging is cheaper
  than under-merging.
- Each slice runs from the start of its first disputed word to the end of
  its last, so re-decode always receives whole words.

### 7. Re-decode — Whisper-large Finnish v3

- Model: **`Finnish-NLP/whisper-large-finnish-v3`**, reached through
  `mlx-whisper` (`transcribe()` with `word_timestamps=True`; the flag is
  off by default).
- Pinned revision: `b23deb0b...` (full SHA recorded at download time).
- Native word timestamps + per-token logprobs.
- Only disputed slices (seconds, not minutes) are re-decoded — this is what
  keeps the third model affordable. Use float16 weights, not the ~6.5 GB
  float32 checkpoint.

### 8. LLM adjudication (optional, fails open)

For each disputed span, the configured LLM sees the candidate transcriptions
(Parakeet, Canary, Whisper) plus surrounding context and picks or composes
the final text.

- **Optional and configured** (invariant #5): the LLM is any
  OpenAI-compatible endpoint selected by the user config file; the API key
  is read from an environment variable named in that config. No hardcoded
  provider, model ID, or base URL in source.
- **Fails open:** on timeout, error, or missing config, the run returns the
  un-adjudicated transcript rather than failing. Every request sets an
  explicit timeout (an unset timeout means "hang").

### 9. Diarization (speaker labels)

Planned as issue #13 (`--diarize`).

- Library: `pyannote.audio==4.0.7` (code is MIT-licensed).
- **Weights: `pyannote/speaker-diarization-community-1` are
  CC-BY-4.0-licensed and gated on HuggingFace** — users must accept the
  license form and provide an access token before first use. The spec (and
  any UX copy) must state this explicitly; silent download is not an option
  under CC-BY-4.0.
- Known platform issue: the pipeline's MPS crash (linear interpolation on
  Metal) is fixed upstream in pyannote PR 1546 (linear → nearest); the
  version shipped must be ≥ that fix or the CPU fallback must be exercised.
- A CPU fallback exists for machines where the fixed MPS path is unavailable.
- Not yet a dependency of this package until issue #13 lands.

### 10. Output formatting

`src/vemoizer/output/`. Formats: `txt`, `json`, `srt`, `vtt` (default:
all four). Subtitle cue timestamps: SRT uses `HH:MM:SS,mmm -->` (comma,
1-based), VTT uses `HH:MM:SS.mmm -->` (dot) under a `WEBVTT` header.
Filenames are NFC-normalized (macOS APFS stores NFD).

## Model manifest

| Stage | Upstream model | Load repo (MLX) | Pinned revision | Notes |
|---|---|---|---|---|
| Decode A | `nvidia/parakeet-tdt-0.6b-v3` | `mlx-community/parakeet-tdt-0.6b-v3` | `ed2b7e8c15f9aaa0b5772e2efb986255eaef7e15` | parakeet-mlx; ~1.25 GB; word timestamps built in |
| Decode B | `nvidia/canary-1b-v2` | community MLX port, e.g. `Mediform/canary-1b-v2-mlx-q8` | `0b6b32ee...` (full SHA at implementation) | loads the MLX port, not the F32 checkpoint |
| Re-decode | `Finnish-NLP/whisper-large-finnish-v3` | itself (mlx-whisper) | recorded at download | f16; `word_timestamps=True` |
| Diarization | `pyannote/speaker-diarization-community-1` | n/a (pyannote.audio 4.0.7) | pinned per issue #13 | CC-BY-4.0, HF-gated (form + token) |
| VAD | silero-vad | bundled in `silero-vad==6.2.1` pip package | package version | ONNX mode, no separate download |

All downloads use `huggingface_hub.snapshot_download(repo_id,
revision=<full-SHA>)` and load from the returned local path, never from the
bare repo ID (invariant #4). Omitting `revision` caches a moving ref;
`HF_HUB_OFFLINE=1` is hard-off (raises if not cached).

## CLI spec

`vemoizer` (Typer; entry point in `pyproject.toml`):

### `vemoizer transcribe FILE... [options]`

Transcribe one or more audio files and write transcript files.

| Flag | Default | Meaning |
|---|---|---|
| `files` (positional, 1+) | — | audio file paths (`.m4a` etc.) |
| `--format` | `all` | `txt`, `json`, `srt`, `vtt`, or a comma-separated subset |
| `--quiet` / `-q` | off | suppress the summary output |
| `--verbose` / `-v` | off | per-stage progress logging to stderr |
| `--copy` | off | copy transcript text to the clipboard via pbcopy (macOS only) |
| `--low-memory` / `--no-low-memory` | auto | low-memory model-loading mode; auto-detected by total RAM when unset (on at ≤16 GiB, off if detection fails) |

Streams: progress bars and warnings go to **stderr**; transcripts and
summaries go to **stdout** (pipeable). On battery power a warning is
emitted to stderr before long transcription.

### Planned subcommands (target surface; not all wired yet)

- `vemoizer transcribe --diarize FILE...` — run speaker diarization
  (issue #13)
- `vemoizer eval --corpus <dir>` — WER regression over the fixture corpus
  (accuracy claims in PRs must come from this output, not model cards)
- `vemoizer models pull` — pre-download and revision-pin all models

## Configuration

The LLM (adjudication / cleanup / summary) is selected by a user config
file (TOML, parsed with stdlib `tomllib`): it names an OpenAI-compatible
base URL, model ID, and the **name of the environment variable** holding
the API key. The key itself is never stored in the repo or the config file
(invariant #5).

| Key | Meaning |
|---|---|
| `llm.base_url` | any OpenAI-compatible endpoint |
| `llm.model` | model ID to request |
| `llm.api_key_env` | environment variable name holding the API key |
| `llm.timeout_seconds` | request timeout; must be set (unset = hang) |

When no config exists or the endpoint fails, every LLM call fails open and
the un-adjudicated transcript is returned.

## Runtime environment

- **Platform: macOS on Apple Silicon only.** The MLX stack has no Intel
  path. Do not add x86 compatibility code; fail fast with a clear message
  (runtime check, not just dependency markers). CI runs on the `macos-15`
  runner for the same reason.
- **Python >= 3.11.**
- **`ffmpeg` on PATH** — the only non-Python system dependency (checked at
  ingest time; a missing binary produces an actionable install message).
- **`uv`** for dependency management (`uv sync --group dev`).
- **Models live in the HuggingFace cache** (`~/.cache/huggingface/hub`).
  Budget ~5–6 GB for the three-model consensus set (Parakeet ~1.25 GB,
  Canary port ~0.7–1 GB, Whisper-large-f16 ~3.3 GB); all three fit on a
  16 GB Mac when loaded lazily and sequentially (see `--low-memory`).
- **LLM**: optional, any OpenAI-compatible endpoint via config; API key
  from an environment variable named in the config.
- Transcription is local, full stop: audio and transcripts never leave the
  machine for ASR; the only network access in the ASR path is the one-time
  (revision-pinned) model download (invariant #1).

## Invariants (authoritative: AGENTS.md "Project Invariants")

1. Transcription is local, full stop. No cloud-ASR fallback.
2. The consensus pipeline is the architecture. Skip-by-flag yes, delete no.
3. Never force a single language on a memo — language is a span property.
4. Model weights are revision-pinned via `snapshot_download`.
5. The LLM is optional, configured, OpenAI-compatible, and fails open.
6. Audio contract: 16 kHz mono float32, decoded once at ingest.
7. No model becomes a default without a WER run on our own corpus.
