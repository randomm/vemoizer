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
     -> slice-level dispute detection (normalized text similarity per VAD slice)
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

### 5. Dispute detection (slice-level text similarity)

`src/vemoizer/slice_align.py`. Decode B emits no word timestamps, and
word-level comparison is wrong for Finnish anyway: measured on the
reference 64-minute memo, ~70 % of DTW word pairs "dispute" (morphology
and tokenizer drift between backends) while only ~15 % of speech *time*
has genuinely divergent slice text. The dispute unit is therefore the
**VAD slice** (median ~2 s, real bounds — no synthetic timestamps): a
slice is disputed when the char-level similarity of its normalized A/B
texts falls below `SLICE_DISPUTE_THRESHOLD` (0.55, calibrated on the
reference memo). Each disputed span carries the slice's detected language
(invariant #3). When the span-count cap trims the set, the most severe
disputes survive, not the earliest.

Guardrails (`src/vemoizer/spans.py`): spans clip to `MAX_SPAN_S` (15 s),
cap at `MAX_SPANS` (300), and a disputed fraction above
`MAX_DISPUTED_FRACTION` (25 %) aborts consensus and ships decode A —
applied only above 60 s of speech (a short clip disputing wholly is
normal and cheap). `VEMOIZER_DISABLE_CONSENSUS=1` is the kill-switch.

`src/vemoizer/alignment.py` (word-onset DTW) remains the documented
mechanism for when decode B gains real word timestamps (e.g. Canary
cross-attention alignment); the slice-level detector is the seam it will
replace.

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

- Model: **`Finnish-NLP/whisper-large-finnish-v3`**, loaded as the
  community MLX conversion `FredrikKarlssonSpeech/whisper-large-finnish-v3-mlx`
  through `mlx-whisper` (`transcribe()` with `word_timestamps=True`,
  `temperature=0.0`, `condition_on_previous_text=False`). mlx-whisper cannot
  consume the raw HF transformers checkpoint and ships no converter, so the
  MLX port is the load repo — the same pattern as decode B's Canary port.
- Pinned revision: `f51f0310c1b2a3e5acb16905c1a7245bb9476846`.
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

`--diarize` flag (opt-in, off by default); wired into the pipeline since issue #37.

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
- Dependency of this package: `pyannote.audio==4.0.7` (CC-BY-4.0 gated weights, see above).

### 10. Assembly

`src/vemoizer/readability.py`. Adjudicated verdicts are spliced INTO
decode A's sentence segments (full coverage — with zero disputes the
output text is byte-identical to decode A's), and consecutive segments
group into paragraphs at silence gaps ≥ 1.5 s or speaker changes.

### 10a. Fail-loud contract (issue #73 / #78)

A total decode failure (``decode_all`` returns ``None`` when ``len(slices) > 0``
and no slice succeeded) must not look like a successful empty transcript.
The pipeline sets ``result["error"]`` to
``"decode A produced no output for any of {N} slices (model may have failed to load)"``
and the CLI exits non-zero via the existing ``"error"`` branch.

Warnings (``result["warnings"]``, printed to stderr by the CLI):
- **Meeting fallback:** when ``decode_meeting`` returns ``None`` and the
dictation decode succeeds, the warning is exactly
``"meeting decode failed; fell back to dictation path"``. If the dictation
decode also totally fails, the result is the ``#73`` error (no warning).
- **Diarization failure:** when ``diarize=True`` and the diarization stage
raises, the warning is exactly
``"diarization failed; continuing without speaker labels"``. The transcript
still ships (fail-open); no ``"error"`` key.

The CLI exit rules run after the ``"error"`` branch and before the
file-writing loop (issue #78):

- **Empty-transcript:** when ``not result.get("text") and not
  result.get("segments") and "error" not in result``, the CLI exits non-zero
  and writes no output files. Legitimately silent audio (empty decode, no
  error) is treated as a failure under this simple rule; there is no
  pipeline marker that distinguishes silence from failure.
- **Diarize no labels:** when ``diarize`` and ``result.get("segments")``
  and no segment carries a ``"speaker"`` key, the CLI exits non-zero. The
  two rules are mutually exclusive by construction (one requires empty
  output, the other requires non-empty segments), so a run never
  double-reports; a result with an ``"error"`` key exits via the error
  branch and neither rule runs.

### 11. LLM notes (optional, fails open)

`src/vemoizer/notes.py`. The configured LLM turns the assembled
transcript into `{title, summary, key_points, action_items}`; transcripts
over 24 K chars are map-reduced in ~12 K-char chunks. Any failure returns
no notes and lands one line in the run's warnings — the transcript is
never affected.

### 12. Output formatting

`src/vemoizer/output/`. Formats: `txt`, `json`, `srt`, `vtt`, `md`
(default: all five). `md` renders the notes (sections omitted when
absent) plus the paragraphed, speaker-labelled transcript. Subtitle cue timestamps: SRT uses `HH:MM:SS,mmm -->` (comma,
1-based), VTT uses `HH:MM:SS.mmm -->` (dot) under a `WEBVTT` header.
Filenames are NFC-normalized (macOS APFS stores NFD).

### 13. Dated output naming (issue #82)

`src/vemoizer/output/naming.py` adds three exports on top of the
existing NFC helpers:

- `sanitize_title(raw: str) -> str` — NFC-normalizes the string, drops
  path separators, control characters (including zero-width joiners and
  BOM), collapses internal whitespace to single spaces, removes leading
  and trailing dots/spaces, and caps the result at 80 characters. An
  empty result signals the caller to fall back.
- `dated_basename(title, *, date_str=None, fallback_stem=None) -> str`
  — builds `YYYY-MM-DD <title>` (date defaults to today, ISO format).
  The title is sanitized first; if sanitising leaves an empty string,
  *fallback_stem* (the first source file's stem) is sanitized and used.
  Raises `ValueError` if both are empty.
- `collision_free_path(directory, base, suffix) -> Path` — probes the
  real filesystem (via `Path.exists`) on the NFC-normalized name and
  appends ` (2)`, ` (3)`, … before the suffix until a free name is
  found. Never overwrites an existing file.

## Model manifest

| Stage | Upstream model | Load repo (MLX) | Pinned revision | Notes |
|---|---|---|---|---|
| Decode A | `nvidia/parakeet-tdt-0.6b-v3` | `mlx-community/parakeet-tdt-0.6b-v3` | `ed2b7e8c15f9aaa0b5772e2efb986255eaef7e15` | parakeet-mlx; ~1.25 GB; word timestamps built in |
| Decode B | `nvidia/canary-1b-v2` | community MLX port, e.g. `Mediform/canary-1b-v2-mlx-q8` | `0b6b32ee...` (full SHA at implementation) | loads the MLX port, not the F32 checkpoint |
| Re-decode | `Finnish-NLP/whisper-large-finnish-v3` | `FredrikKarlssonSpeech/whisper-large-finnish-v3-mlx` | `f51f0310c1b2a3e5acb16905c1a7245bb9476846` | community MLX conversion (mlx-whisper cannot read the raw HF checkpoint); `word_timestamps=True` |
| Diarization | `pyannote/speaker-diarization-community-1` | n/a (pyannote.audio 4.0.7) | `3533c8cf8e369892e6b79ff1bf80f7b0286a54ee` | CC-BY-4.0, HF-gated (form + token) |
| VAD | silero-vad | bundled in `silero-vad==6.2.1` pip package | package version | ONNX mode, no separate download |

All downloads use `huggingface_hub.snapshot_download(repo_id,
revision=<full-SHA>)` and load from the returned local path, never from the
bare repo ID (invariant #4). Omitting `revision` caches a moving ref;
`HF_HUB_OFFLINE=1` is hard-off (raises if not cached).

## CLI spec

`vemoizer` (Typer; entry point in `pyproject.toml`). Three commands are
wired: `transcribe` (expert, unchanged), `meeting`, and `memo` (preset
commands added in issue #82). `eval` is registered with `hidden=True`
and does not appear in the main `--help`.

### `vemoizer meeting FILES... [options]` (issue #82)

Transcribe one or more meeting recordings: whisper decode (profile
`meeting`, no consensus), diarization on by default (2–6 speakers),
LLM repair pass on by default. Output is `.md` + `.json` to the CWD
with a dated, sanitized title and NFC collision suffix; one `wrote
<relative path>` line per file is printed at the end.

The glossary is the merged result of `~/.vemoizer/glossary.txt` and
the nearest `./.vemoizer/glossary.txt` (project layer winning, M0
token budget applied, `@`-prefixed terms LLM-only — see Glossary
layers below). `--glossary` replaces both layers entirely (no merge).
`--config` replaces the layered config search entirely.

| Flag | Default | Meaning |
|---|---|---|
| `files` (positional, 1+) | — | audio file paths |
| `--quiet` / `-q` | off | suppress the `wrote <path>` summary lines |
| `--verbose` / `-v` | off | per-stage progress logging to stderr |
| `--config` | layered search | explicit LLM config path (replaces the search) |
| `--glossary` | layered merge | explicit glossary file (replaces both `.vemoizer` layers) |
| `--repair` / `--no-repair` | on | LLM repair pass over the final paragraphs |
| `--speakers` | 2-6 | diarization bounds (`N` or `MIN-MAX`) |
| `--no-diarize` | off (diarize on) | skip speaker diarization |
| `--low-memory` / `--no-low-memory` | auto | low-memory model-loading mode |

### `vemoizer memo FILES... [options]` (issue #82)

Transcribe one or more solo memos: whisper meeting decode (profile
`meeting`, no consensus), **no** diarization, LLM repair pass on by
default. Output naming is identical to `meeting` (`.md` + `.json` to
CWD with dated title and NFC collision suffix).

The memo seam (issue #82, DESIGN DECISION): the whisper
`initial_prompt` stays empty for a memo (a 30-minute memo should not
seed recognition with hundreds of prompt terms). The batch runner
therefore writes a temporary glossary file containing ONLY the project
layer's correction pairs and passes it through the existing
glossary_path argument — so `glossary_prompt` yields `None` (empty
prompt) while `apply_corrections` still fires on the deterministic
pairs. `--glossary` replaces the layers entirely; in that case only
that file's correction pairs are used for memo (prompt terms ignored).

| Flag | Default | Meaning |
|---|---|---|
| `files` (positional, 1+) | — | audio file paths |
| `--quiet` / `-q` | off | suppress the `wrote <path>` summary lines |
| `--verbose` / `-v` | off | per-stage progress logging to stderr |
| `--config` | layered search | explicit LLM config path (replaces the search) |
| `--glossary` | layered merge | explicit glossary file (correction pairs only for memo) |
| `--repair` / `--no-repair` | on | LLM repair pass over the final paragraphs |
| `--low-memory` / `--no-low-memory` | auto | low-memory model-loading mode |

### `vemoizer transcribe FILE... [options]`

Transcribe one or more audio files and write transcript files.
Unchanged from before issue #82 — the expert command that exposes
every pipeline flag explicitly. The per-file loop that used to live
here moved to `src/vemoizer/batch.py` (new module) so the `meeting`
and `memo` presets can reuse it.

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

### Config search order (issue #82)

`llm.load_default_config(path=None)` searches in this precedence order
(lowest → highest, later layers override earlier ones at the whole-file
level — there is no per-key merging):

1. **`--config` flag** (explicit path): short-circuits the search
   entirely. The special value `"os.devnull"` (the `os.devnull` string
   as documented in `presets.RunOptions.config_path`) loads nothing and
   returns `None`.
2. **`~/.vemoizer/config.toml`** (home layer): strict validation — an
   unknown key under `[llm]` or an unknown top-level key/section raises
   `ConfigError` naming the offending key. Missing file: silently
   skipped (fail-open) to the next layer.
3. **Nearest `./.vemoizer/config.toml`** walking up from CWD to the
   filesystem root (project layer): same strict rules. Nearest wins —
   the walk stops at the first directory that has a `.vemoizer` folder.
   Symlink loops are prevented by tracking the resolved real path of
   each directory visited.
4. **Legacy paths** (fail-open, pre-M2 semantics, unchanged):
   `~/.config/vemoizer/config.toml` then `~/.vemoizer.toml`. When the
   `~/.config/…` file is the one actually used, a one-line deprecation
   notice is printed to stderr. No notice is printed when a newer
   layer won or when the `~/.vemoizer.toml` legacy path is used.

Strict validation applies only to layers 2 and 3 (the new `.vemoizer`
files). Legacy layers keep the old fail-open contract so existing users
are not broken.

### Glossary layers (issue #82)

`src/vemoizer/glossary_layers.py` (new module) provides the two-layer
glossary for the `meeting` and `memo` presets:

- `load_layers(home_path=None, project_path=None)` — file I/O only.
  Reads `~/.vemoizer/glossary.txt` (home layer) and the nearest
  `./.vemoizer/glossary.txt` (project layer, same walk-up as config
  search). A missing or unreadable file contributes `([], {})`.
  Returns `(home_terms, home_corrections, project_terms,
  project_corrections)`, where `terms` preserves `@`-prefixed LLM-only
  entries (no stripping here — `glossary_prompt` and the pipeline
  handle that) and `corrections` maps wrong-side key → right-side value.
- `merge(project_terms, project_corrections, home_terms, home_corrections,
  tokenizer=None, budget=GLOSSARY_PROMPT_TOKEN_BUDGET)` — pure
  (no I/O, no printing). Returns `(merged_terms, merged_corrections,
  notices)`:  
  - Prompt terms: project first, case-insensitive dedupe keeping the
    project spelling. The M0 token budget is applied after merge and
    the lowest-priority (earliest-listed) terms are dropped first; each
    dropped term is named in `notices`. `@`-prefixed LLM-only terms
    are excluded from the budget entirely (they never enter the whisper
    prompt).
  - Correction pairs: union of both layers; for the same wrong-side key
    the project's right side wins. Correction lines never contribute
    prompt terms.
  - `notices`: one string per dropped term. `batch.py` prints these to
    stderr; `merge` itself never prints.

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
