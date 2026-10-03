# vemoizer pipeline spec

Canonical stage contract, model IDs and pinned revisions, CLI flag spec, and
configuration schema. This document is the single source of truth for the
pipeline — `AGENTS.md` and `CONTRIBUTING.md` point here rather than restating
these details.

Last updated: 2026-10 (issue #110 — `sanitize_title` now maps `:` to an
en-dash separator and drops the other filesystem-invalid characters
`* ? " < > |`; spec corrections earlier: Parakeet repo IDs, Canary load
path, CC-BY diarization, runtime environment).

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
- **Telemetry is disabled (invariant #1, issue #103):** `pyannote.audio` 4.x
  ships OpenTelemetry usage metrics enabled by default, sending spans to
  `otel.pyannote.ai` carrying version, session id, and audio duration /
  speaker-count hints — which would break invariant #1. Before `pyannote`
  is lazily imported, the loader sets `PYANNOTE_METRICS_ENABLED=false` and
  `OTEL_SDK_DISABLED=true` (both via `setdefault`, as belt-and-braces), so
  a user can opt in by exporting `PYANNOTE_METRICS_ENABLED=true`.

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

#### 12b. M6 reader-ready Markdown and quality report (issue #75)

The `md` format is the reader-facing deliverable. M6 adds, all via the
run dict (the seam the CLI/batch layer controls — `format_md` stays a
pure function of the dict):

- **Header block above `# {title}`** — date, duration
  (`[hh:mm:ss]`, from `transcript["duration_s"]`), parts count (from
  `part_markers`), a speaker legend with talk share (derived from
  `transcript["paragraphs"]` start/end/speaker), and glossary
  provenance (`transcript["glossary_source"]`). Each line is omitted
  when its input is absent.
- **`[hh:mm:ss]` per-paragraph prefix** — from `para["start"]`; omitted
  when the key is missing (no placeholder).
- **Labelled suspect prefix** — `suspect="garble"` → `⚠ epäselvä`,
  `suspect="number"` → `⚠ luku` (replaces the old bare `⚠ ` prefix).
- **Section language** — `format_md(transcript, language=...)` selects
  Finnish (default) or English headings; the value rides on the run
  dict as `transcript["language"]` (loaded by
  `llm_config.load_language` from the config layer's top-level `language`
  key, `"fi"` default).
- **`<details>` quality-report block** — the string stored on
  `transcript["quality_report"]` (computed by the CLI/batch layer via
  `report.build_quality_report` BEFORE the destructive
  `result.pop("warnings")`) renders after the transcript section;
  absent/empty → no block (fail-open).

**Run-dict keys added by the pipeline** (`transcribe_file`, stage 10):
`duration_s` (decoded-audio seconds, `len(audio) / SAMPLE_RATE` — never
ffprobe) and `language` (`"fi"` / `"en"`). The gate resolution for the
original "pipeline.py is not touched" constraint is superseded: the
decoded audio exists only inside `transcribe_file`, so the duration rides
on the dict rather than being passed as a parameter.

**Run-dict keys added by the batch layer** (`batch_output` / 
`batch_preset`): `glossary_source` (the real layer file path(s) the run
read, `"<paths> (N terms)"` where N counts the whisper-prompt terms only
— `@`-prefixed LLM-only terms never reached the whisper prompt and are
excluded from the count — report/header provenance only, never stored by
the pipeline) and `glossary_terms` (the whisper-prompt terms, popped by
`batch_output._process_result` before output writing so it is the
report's matching input, never mirrored into an output file).

**The quality report** (`src/vemoizer/report.py`, `render_report` /
`build_quality_report`) is a pure function of the run dict plus
`diarize_requested` and the glossary provenance — it is NOT an output
format (not in `OUTPUT_FORMATS`). The batch layer computes it per file
BEFORE `_check_result` pops `result["warnings"]` (a report computed
after the pop would see an empty warnings list), stores it on
`transcript["quality_report"]` (the `md` embeds it as the `<details>`
block), and prints it to stdout (suppressed by `--quiet`; still printed
when `--format` excludes `md`). A report failure never breaks the md
write (fail-open, invariant #5).

**Warnings pre-pop contract** (issue #75): `batch_output._check_result`
owns the `result.pop("warnings", [])` + stderr print for every batch
path (expert `transcribe`, `run_batch` groups, the preset plain loop).
The quality report is computed from the warnings list BEFORE that pop
(`_render_quality_report` in `batch_output`), so the report's warnings
section is populated in every real run.

### 12a. M5a JSON sidecar keys (issue #89)

The dated `.json` written next to the `.md` by the `meeting` and `memo`
presets carries four extra keys, assembled by
`src/vemoizer/sidecar.py` (`build_sidecar`) before `format_json` mirrors
them (present-only, so old JSON without the keys and the expert
`transcribe` JSON are unaffected):

- `notes` — the LLM notes verbatim (`{title, summary, key_points,
  action_items}`); the key is omitted when there were no notes (absent
  or `None`, i.e. LLM fail-open). Never fabricated.
- `source` — one entry per recorded part: `{path, part_offset_s,
  duration_s?}`. Single-file runs: one entry with the file path,
  `part_offset_s: 0.0`. Grouped runs: one entry per `part_markers` part,
  `path` is the real part file path and `part_offset_s` the marker's
  cumulative decoded-PCM start offset. `duration_s` is the part's
  measured decoded-PCM duration, omitted when the measurement failed
  (fail-open).
- `options` — `{command, glossary_files, glossary_sha256}`.
  `glossary_files` lists the exact glossary files the run used (project
  layer first, then home; or the single explicit `--glossary` file); `[]`
  when the run had none. `glossary_sha256` is the sha256 over the run's
  **prompt-term set**, hashed by `prompt_term_set_hash` (see below), or
  `null` when `glossary_files` is empty.
- `speaker_names` — `{label: name}` speaker-name map, `{}` by default
  (the render command's `--name` persists into it atomically — temp
  file in the same directory + `os.replace`). A `clips` key is never
  written.
- `duration_s` — the run's measured total PCM duration in seconds
  (float), mirrored into the JSON sidecar when present (e.g.
  `9.15`). Absent when the ffmpeg duration measurement failed
  (fail-open) or the key was not set. Render reproduces the
  `Kesto: [HH:MM:SS]` header line from it.
- `glossary_source` — a human-readable string (e.g.
  `/path/to/.vemoizer/glossary.txt (1 terms)`) describing the glossary
  the run used. Set by the preset seam when a glossary was present;
  absent otherwise. Render reproduces the `Sanasto: …` header line
  from it.

**`glossary_sha256` — prompt-term-set hash (issue #107)**: the
hash covers only the glossary *prompt terms* — the non-correction
(`=>`-free), non-`@`-prefixed lines — after layer merge, deduped
case-insensitively with first-seen spelling winning (project over home),
newline-joined, then sha256. This is the canonical deduplicated
prompt-term set, the input to `glossary_prompt` before its token-budget
truncation; correction pairs and `@`-names are render-safe and
ever excluded. `render` recomputes the same hash over the current
glossary files via the shared `prompt_term_set_hash`, so a mismatch
means the prompt-term set genuinely changed.

A sidecar written before this change stores a raw-bytes hash (sha256
over the concatenated raw bytes of the glossary files, no term-level
filtering). Because the stored raw-bytes value will not match the new
prompt-term-set hash recomputed by render, the first render of such an
old sidecar prints the drift warning (the two hashes differ); the
stored hash is never updated by render, so the warning repeats on
every subsequent render until the sidecar is regenerated by a fresh
run. This is a one-time migration cost, not a functional regression.

These keys let `vemoizer render` re-apply glossary corrections and
speaker names without a re-transcribe; old `.json` files without them
render unchanged (the `render` command is fail-open over missing keys).

`format_json` mirrors each key only when present and non-empty (the
`speaker_names` `{}` default is omitted), so the expert `transcribe`
JSON and pre-M5 sidecars are byte-identical to before.

The round-trip guarantee: `render` of its own JSON with the same
glossary and no names produces a Markdown byte-identical to the
Markdown the run wrote **iff** the current effective glossary's sha256
equals the stored `options.glossary_sha256`. When they differ,
corrections and names are still applied but byte-identity is not
guaranteed (the hash-mismatch warning fires).

### 13. Dated output naming (issue #82)

`src/vemoizer/output/naming.py` adds three exports on top of the
existing NFC helpers:

- `sanitize_title(raw: str) -> str` — NFC-normalizes the string, drops
  path separators, control characters (including zero-width joiners and
  BOM), and the other filesystem-invalid characters; maps `:` to an
  en-dash separator (`" – "`, so an LLM title "Planning: NG Nordic"
  becomes "Planning – NG Nordic" — a bare colon survives into a filename
  that Finder renders as a path separator and that sync tools and
  Windows/SMB shares reject); drops `* ? " < > |` outright; collapses
  internal whitespace to single spaces, removes leading and trailing
  dots/spaces, and caps the result at 80 characters. An empty result
  signals the caller to fall back.
- `dated_basename(title, *, date_str=None, fallback_stem=None) -> str`
  — builds `YYYY-MM-DD <title>` (date defaults to today, ISO format).
  The title is sanitized first; if sanitising leaves an empty string,
  *fallback_stem* (the first source file's stem) is sanitized and used.
  Raises `ValueError` if both are empty.
- `collision_free_path(directory, base, suffix) -> Path` — probes the
  real filesystem (via `Path.exists`) on the NFC-normalized name and
  appends ` (2)`, ` (3)`, … before the suffix until a free name is
  found. Never overwrites an existing file.

### 13a. M5a JSON sidecar keys (issue #89)

See section 12a — the single canonical sidecar-key contract lives there.

### 14. Speaker clips (issue #90)

`src/vemoizer/speaker_clips.py` selects short audio windows from a
rendered sidecar's `paragraphs` and decodes them from the `source` parts
for preview playback. It is a pure selector plus audio helpers; nothing
it does changes the pipeline output.

**Selection rules** (`select_clips(paragraphs, segments, *, per_speaker=3,
min_s=3.0, max_s=5.0, total_duration=None) -> dict[label, list[ClipWindow]]`
— `total_duration` overrides the audio-end used to clamp windows; it
defaults to the end of the last paragraph):
`ClipWindow` is a hashable 3-tuple subclass of `tuple` with named fields
`start_s`, `end_s`, `quote`; the same value is the key of the
dict returned by `extract_clips`:

- A *turn* is one labelled, non-suspect paragraph — labelled iff its
  `speaker` is a non-empty string (missing key, `None`, or `""` are all
  unlabelled and excluded).
- A turn of duration D is a candidate iff D ≥ 2 s and its
  whitespace-collapsed, casefolded text is **not** a member of
  `BACKCHANNELS ∪ BACKCHANNEL_PHRASES` (no word-count condition — a
  backchannel word inside a longer sentence is a normal turn).
- The window is the middle `min(max_s, D)` seconds of the turn, centered
  on the turn midpoint and clamped to the audio duration. `min_s` is a
  display annotation, not a disqualification threshold.
- Picks for one speaker spread across the beginning, middle and end of
  the meeting: the longest candidate from each third in order
  (beginning → middle → end); when a third is empty, the longest remaining
  candidate regardless of third. Fewer than `per_speaker` clean turns
  yields fewer clips, without error. The result is sorted by `start_s`.

Extraction (`extract_clips(source, windows, tmp_dir)`) maps each window
onto the sidecar `source` part whose `[part_offset_s, part_offset_s +
duration_s)` contains it (a part without a measured `duration_s` extends
unbounded), and decodes only the needed span with ffmpeg
`-ss {start} -t {dur}` placed **before** `-i` (input seeking, O(window))
to 16 kHz mono WAV. `duration_s` is the part's measured decoded-PCM
duration from the sidecar's `source` entries (via `pcm_duration_seconds`),
never ffprobe. A window that straddles two parts, a window whose end
exceeds the mapped part's `duration_s`, or a window whose source file is
missing/unreadable yields `None` — never an exception.
`play(path) -> bool` runs `afplay` (argv list, 30 s timeout) and fails
open (returns `False`) on non-darwin, missing afplay, missing file,
non-zero exit, or timeout.

**No clips on disk**: `clip_session()` is a context manager yielding a
fresh 0o700 temp dir; it removes the directory (and every clip in it) on
normal exit, exception, and KeyboardInterrupt. Clips are preview-only —
nothing clip-related is ever written to a persistent location, and no
`clips` key is ever written to the sidecar.

## Model manifest

| Stage | Upstream model | Load repo (MLX) | Pinned revision | Expected size | Notes |
|---|---|---|---|---|---|
| Decode A | `nvidia/parakeet-tdt-0.6b-v3` | `mlx-community/parakeet-tdt-0.6b-v3` | `ed2b7e8c15f9aaa0b5772e2efb986255eaef7e15` | ~2.3 GiB | parakeet-mlx; word timestamps built in |
| Decode B | `nvidia/canary-1b-v2` | community MLX port, e.g. `Mediform/canary-1b-v2-mlx-q8` | `0b6b32ee...` (full SHA at implementation) | ~1.1 GiB | loads the MLX port, not the F32 checkpoint |
| Re-decode | `Finnish-NLP/whisper-large-finnish-v3` | `FredrikKarlssonSpeech/whisper-large-finnish-v3-mlx` | `f51f0310c1b2a3e5acb16905c1a7245bb9476846` | ~2.9 GiB | community MLX conversion (mlx-whisper cannot read the raw HF checkpoint); `word_timestamps=True` |
| Meeting decode | `openai/whisper-large-v3-turbo` | `mlx-community/whisper-large-v3-turbo` | `a4aaeec0636e6fef84abdcbe3544cb2bf7e9f6fb` | ~1.5 GiB | MLX community conversion; decoded in 30 s windows so the glossary `initial_prompt` re-seeds every window (issue #76) |
| Diarization | `pyannote/speaker-diarization-community-1` | n/a (pyannote.audio 4.0.7) | `3533c8cf8e369892e6b79ff1bf80f7b0286a54ee` | ~32 MiB | CC-BY-4.0, HF-gated (form + token) |
| VAD | silero-vad | bundled in `silero-vad==6.2.1` pip package | package version | bundled | ONNX mode, no separate download |

Expected sizes are the per-model constants in `_EXPECTED_SIZE_GIB` in
`src/vemoizer/models.py` (issue #67), measured against the local HuggingFace
cache and shown per-model in the `models pull` report. The five pipeline
models total roughly 7.8 GiB on disk; loaded lazily and sequentially, all
fit on a 16 GB Mac.

The first five rows are the `MODELS` registry in `src/vemoizer/models.py`
in pipeline order (parakeet, canary, whisper-finnish, whisper-turbo,
pyannote); `vemoizer models pull` downloads all five at their pinned
revisions. The transcriber, re-decode, and diarization modules read their
repo ID and revision from this registry — no repo/SHA pair is defined in
two places (issue #79; drift tests guard the constants).

All downloads use `huggingface_hub.snapshot_download(repo_id,
revision=<full-SHA>)` and load from the returned local path, never from the
bare repo ID (invariant #4). Omitting `revision` caches a moving ref;
`HF_HUB_OFFLINE=1` is hard-off (raises if not cached).

### Glossary prompt and echo filter (meeting decode)

The glossary prompt passed to whisper as `initial_prompt` is a plain
comma-separated term list ending in a period (e.g. `"Pia, NG-TOPI, IBC.")`
— no label word (the former `Sanasto:` prefix was removed in issue #109).
After each window's decode, the segments pass a post-decode echo filter
(`echo_filter.filter_echo_segments`) that drops a segment only when
every token is a glossary term (or the former label `Sanasto`) **and** it
is a run of ≥ 2 terms or carries the label (a single bare term is kept).
The filter is fail-open: on any error the segments are returned unfiltered,
with one warning logged, or — if the words-extraction fallback also
fails — two warnings (one for the error, one for the words degradation).

## CLI spec

`vemoizer` (Typer; entry point in `pyproject.toml`). Four commands are
wired: `transcribe` (expert, unchanged), `meeting`, `memo` (preset
commands added in issue #82), and `render` (M5a, issue #89 — model-free
re-render of a stored sidecar). `eval` is registered with `hidden=True`
and does not appear in the main `--help`.

### `vemoizer meeting FILES... [options]` (issue #82)

Transcribe one or more meeting recordings: whisper decode (profile
`meeting`, no consensus), diarization on by default (2–6 speakers),
LLM repair pass on by default. Output is `.md` + `.json` to the CWD
with a dated, sanitized title and NFC collision suffix; one `wrote
<relative path>` line per file is printed at the end.

With 2+ files, the M3 split-recording grouping flow runs (issue #87):
natural sort, boundary decodes, continuation/break proposals, confirmation
(`--yes` / `--no-group` / interactive), ffmpeg concat, one decode per
group, part markers. A non-TTY invocation without `--yes`/`--no-group`
fails immediately (exit 2) before any decode. The dated output name uses
the first source file's **modification date** (not `creation_time`, which
on iOS exports is the copy/export time).

The glossary is the merged result of `~/.vemoizer/glossary.txt` and
the nearest `./.vemoizer/glossary.txt` (project layer winning, M0
token budget applied when the prompt is built — see Glossary layers
below), with `@`-prefixed terms LLM-only. `--glossary` replaces both
layers entirely (no merge). `--config` replaces the layered config
search entirely.

End-of-meeting naming prompt (issue #95): after an interactive `meeting`
run finishes (single file, the `--no-group` loop, or the grouped run —
after the `wrote <path>` lines), the run asks `Name the speakers now?
[y/N]` once; on yes it runs the existing `vemoizer names` flow on each
written sidecar (`.json` resolved against the CWD) that has 2+ labelled
speakers. The prompt is skipped entirely — no output, no prompt — for
`memo` (never prompts), `--yes`, and non-interactive (non-TTY stdin or
stdout) runs, and when no written sidecar is eligible. The hook never
changes the run's exit code, and a per-sidecar naming failure only warns
(`warning: naming failed for <name>: <ExcClassName>`) before continuing.

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
| `--yes` | off | group mode for 2+ files: run the boundary decodes and accept every continuation proposal without a prompt (mutually exclusive with `--no-group`) |
| `--no-group` | off | skip split-recording grouping entirely — each file is transcribed standalone (no boundary decode, no concat, no part markers); mutually exclusive with `--yes` |
| `--language` | `auto` | recognition language for the whisper decode: `auto` (detect per window), `fi`, or `en` (case-insensitive); a `[meeting] language` key in the config file pins the same choice for meeting (and memo) runs (issue #108) |

### `vemoizer memo FILES... [options]` (issue #82)

Transcribe one or more solo memos: whisper meeting decode (profile
`meeting`, no consensus), **no** diarization, LLM repair pass on by
default. Output naming is identical to `meeting` (`.md` + `.json` to
CWD with dated title and NFC collision suffix).

The memo seam (issue #82, DESIGN DECISION): the whisper
`initial_prompt` stays empty for a memo (a 30-minute memo should not
seed recognition with hundreds of prompt terms). The batch runner
therefore writes a temporary glossary file containing ONLY the merged
correction pairs (home + project layers, project right-side winning on
the same wrong-side key) and passes it through the existing
glossary_path argument — so `glossary_prompt` yields `None` (empty
prompt) while `apply_corrections` still fires on the deterministic
pairs. `--glossary` replaces the layers entirely; in that case the file
is filtered to its own correction pairs via a second temp file, keeping
the same empty-prompt invariant (prompt terms ignored).

| Flag | Default | Meaning |
|---|---|---|
| `files` (positional, 1+) | — | audio file paths |
| `--quiet` / `-q` | off | suppress the `wrote <path>` summary lines |
| `--verbose` / `-v` | off | per-stage progress logging to stderr |
| `--config` | layered search | explicit LLM config path (replaces the search) |
| `--glossary` | layered merge | explicit glossary file (correction pairs only for memo) |
| `--repair` / `--no-repair` | on | LLM repair pass over the final paragraphs |

### `vemoizer render X.json [options]` (issue #89, M5a)

Re-apply the CURRENT glossary correction pairs and any `--name` values
to a stored meeting/memo sidecar (the `.json` written next to the `.md`
by `meeting` / `memo`) and re-emit the Markdown. No model, no LLM —
works on a machine without the MLX stack. A speaker renaming or
adding a correction pair never requires a re-transcribe.

Glossary resolution mirrors `meeting` / `memo`: the layered glossary
(project + home layers, project right-side winning) is used unless
`--glossary` replaces both layers entirely. The stored
`options.glossary_sha256` (the run's prompt-term-set hash, see §12a)
is compared against the current prompt-term-set hash recomputed from
the current glossary files; when they differ, exactly one stderr line
warns that new PROMPT terms need a re-transcribe (corrections and names
are applied regardless). A missing glossary file warns to stderr and
proceeds (fail-open).

`--name LABEL=NAME` values are persisted into the sidecar's
`speaker_names` key by rewriting the JSON in place atomically (temp
file in the same directory + `os.replace`). The output `.md` is written
atomically (temp file in the same directory + `os.replace`) to
`<stem>.md` next to the sidecar, **overwriting** any existing file
(the rendered output is fully derived from the sidecar, so the
previous file is replaced — no ` (2)` suffix), unless `--out` is
given. `os.replace` over a symlink whose pointee is a regular file replaces
the symlink itself (not the pointed-to file), making the write symlink-safe;
a symlink to a non-regular file is written through in place. The existing
file's mode is preserved (default path and `--out` alike).

Exit codes: `0` on success, `1` on an unreadable or malformed sidecar,
`2` on a malformed `--name` value.

| Flag | Default | Meaning |
|---|---|---|
| `X.json` (positional) | — | the `.json` sidecar written by a meeting or memo run |
| `--glossary` | layered merge | explicit glossary file (replaces both `.vemoizer` layers) |
| `--name` | — | set speaker name: `LABEL=NAME` (repeatable); persisted into the sidecar |
| `--out` | next to the sidecar | write the `.md` to this path instead |

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
| `--copy` | off | copy transcript text to the clipboard via pbcopy (macOS only; single-file runs — a warning is printed when 2+ files are passed) |
| `--yes` | off | group mode for 2+ files: run the 20 s boundary decodes and accept every continuation proposal without a prompt (mutually exclusive with `--no-group`) |
| `--no-group` | off | skip split-recording grouping entirely — each file is transcribed standalone (no boundary decode, no concat, no part markers) |

Split-recording grouping (issue #77, 2+ files only): the inputs are
naturally sorted (NFC stem, trailing integer as the numeric key), the
last 20 s of each file and the first 20 s of its successor are decoded
with the Whisper boundary model (only those 20 s windows — ffmpeg
`-ss`/`-t`; the full file is never decoded for the probe), and each
boundary is proposed as *continue* or *break* from closing-cue matching
over the normalised edge text (silent/failed edges degrade to *break*,
never a false continuation). The proposal is confirmed — `--yes`
(accept all; boundary decodes still run), `--no-group` (no grouping at
all), or interactively (Enter accept, `e` a full partition edit, `q`
quit). A non-TTY invocation without `--yes`/`--no-group` fails fast
before any decode. Accepted multi-part groups are joined with the ffmpeg
concat demuxer (`-c copy`, same audio-stream check per part) into a temp
file and decoded ONCE; the part start offsets (decoded PCM, never
ffprobe) are written to the result as `part_markers` (`{"offset",
"label"}`) so the JSON sidecar and Markdown carry a
`— osa N (äänitys X) —` marker per part. Single-part groups carry no
`part_markers` key at all. A single file skips grouping entirely.

Part offsets are measured by decoding EACH part separately (one extra
streaming decode pass per part, no PCM materialised — only the byte
count) rather than deriving them from the merged decode: the contract
is that offsets come from decoded PCM byte counts, never ffprobe or
container metadata (iOS Voice Memos edit lists make container duration
lie), and `-c copy` concatenation of AAC does not guarantee that the
merged decode's length equals the sum of the parts' lengths (frame
padding and edit lists), so the merged decode is not a safe basis for
the offsets. The measured cost of that extra decode is a streaming pass
only (no PCM materialised): on this machine (Apple Silicon, synthetic
20-minute 16 kHz pink-noise .m4a, 3 runs) each per-part decode took
0.530-0.649 s (median 0.575 s) — about 1.6 s per hour of audio, roughly
2265x real time — versus ~12 min/h for the whisper-per-window decode
itself (PR #84 spike), i.e. well under 1 % of a run's wall time; a 3 x
7-minute group measured 0.837 s end to end. That figure is a measurement
on a synthetic file on one machine, not a guarantee.

`--out` with 2+ files is honored only when the run ends up as a single
group (one combined transcript to that path); `--out -` (stdout) is
always fine (groups stream in order). An explicit `--out` file path with
2+ files that would produce more than one group fails up front (exit 2,
before any decode) instead of later groups overwriting the earlier ones.

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

### Fully-offline LLM (Ollama / llama.cpp)

The `[llm]` config works with **any** OpenAI-compatible endpoint, including
local servers. Pointing `base_url` at a local LLM runtime makes vemoizer
fully offline — audio, transcripts, adjudication, and the Markdown notes
never leave the machine. Note: if `HTTP_PROXY`/`HTTPS_PROXY` are set
(e.g. by a VPN or corporate MDM), the request to the local server may be
routed through the proxy; unset them or set `NO_PROXY=localhost` for a
genuinely offline run.

**Key behavior.** Whether the LLM stages run at all depends on the named
env var, not on the server: when the variable named by `api_key_env` is
**unset or empty, the LLM stages are skipped entirely** (no HTTP request is
made; adjudication and the Markdown notes are silently skipped). To get a
fully offline run you must point `api_key_env` at a variable that is
**set** — a placeholder value is enough, because local servers such as
Ollama do not enforce authentication and ignore the `Authorization` header.

**Ollama** (tested, `http://localhost:11434/v1`):

```toml
[llm]
base_url = "http://localhost:11434/v1"
model = "qwen2.5:14b"       # any model installed via `ollama pull`
api_key_env = "OLLAMA_API_KEY"  # name of the env var — it must be SET
timeout_seconds = 120
```

```bash
ollama pull qwen2.5:14b
# Point api_key_env at a var that is set; Ollama ignores the value.
export OLLAMA_API_KEY="ollama"   # placeholder; no real key is needed
```

**llama.cpp server** (tested pattern, `http://localhost:8080`):

```toml
[llm]
base_url = "http://localhost:8080"
model = "qwen2.5-14b"        # model name as exposed by the llama.cpp server
api_key_env = "LLAMA_CPP_API_KEY"  # must be set for the LLM stages to run
timeout_seconds = 120
```

```bash
llama-server -m qwen2.5-14b-q8_0.gguf --port 8080
export LLAMA_CPP_API_KEY="local"   # placeholder when auth is not enforced
```

The client POSTs to `{base_url}/chat/completions` with the standard OpenAI
chat-completions body. When the key env var is **set**, an
`Authorization: Bearer <value>` header is sent; local servers that do not
enforce auth simply ignore it. When the key env var is **unset or empty**,
no request is sent at all and the LLM stages fail open (the un-adjudicated
transcript is returned and the run continues). Any other failure (timeout,
connection refused, HTTP error) also fails open per invariant #5.

| Key | Meaning |
|---|---|
| `llm.base_url` | any OpenAI-compatible endpoint |
| `llm.model` | model ID to request |
| `llm.api_key_env` | environment variable name holding the API key |
| `llm.timeout_seconds` | request timeout; must be set (unset = hang) |
| `language` | section language for the Markdown header and quality report: `"fi"` (default) or `"en"` (top-level key, issue #75) |
| `meeting.language` | recognition-language override for the whisper meeting decode: `"auto"` (default, per-window detection), `"fi"`, or `"en"` pins every window (issue #108) |

When no config exists or the endpoint fails, every LLM call fails open and
the un-adjudicated transcript is returned.

### Config search order (issue #82)

`llm.load_default_config(path=None)` searches in this precedence order
(lowest → highest, later layers override earlier ones at the whole-file
level — there is no per-key merging):

1. **`--config` flag** (explicit path): short-circuits the search
   entirely; the path is passed straight through to
   `load_default_config(path)` (missing/unreadable files fail open to
   no LLM). The special value `"os.devnull"` loads nothing and returns
   `None`. The presets (`meeting` / `memo`) pass `None` when no
   `--config` is given, so the layered search runs — they never emit the
   sentinel themselves (only explicit callers such as the eval harness
   do).
2. **Nearest `./.vemoizer/config.toml`** (project layer): strict
   validation — an unknown key under `[llm]` or an unknown top-level
   key/section raises `ConfigError` naming the offending key. The
   walk-up starts at the CWD and stops at the filesystem root; nearest
   wins. Symlink loops are prevented by tracking the resolved real path
   of each directory visited.
3. **`~/.vemoizer/config.toml`** (home layer): same strict rules. Used
   only when the project walk-up finds nothing.
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
    project spelling. The M0 token budget is applied when the prompt is
    built (by `glossary_prompt`, after the merge) and the
    lowest-priority (earliest-listed) terms are dropped first; each
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
  Budget ~7.8 GiB for the five pipeline models (Parakeet ~2.3 GiB, Canary
  ~1.1 GiB, Whisper-Finnish ~2.9 GiB, Whisper-turbo ~1.5 GiB, pyannote
  ~32 MiB — per-model sizes in the Model manifest); all fit on a 16 GB
  Mac when loaded lazily and sequentially.
- **LLM**: optional, any OpenAI-compatible endpoint via config; API key
  from an environment variable named in the config.
- Transcription is local, full stop: audio and transcripts never leave the
  machine for ASR; the only network access in the ASR path is the one-time
  (revision-pinned) model download (invariant #1). Third-party telemetry is
  disabled: pyannote's OpenTelemetry metrics are turned off before import
  (issue #103).

## Per-file run log (issue #111, M4c)

Every transcribed file (or group) writes a full log to
`./.vemoizer/logs/<NFC-stem>.log` regardless of `-v`. The span per seam
(see the four seams below) opens the file with `'w'` at block start, so a
re-run truncates the same file and a file that fails immediately still
leaves a (possibly near-empty) log. The directory is `0700`, the file
`0600`.

Terminal noise: in non-verbose runs a `level < WARNING` filter is added to
the **existing** terminal stderr handlers only (the root `basicConfig`
handler under `-v`, and `huggingface_hub`'s own `StreamHandler`) — nothing
is added to the root logger, so `vemoizer.*` and third-party `WARNING+`
reach stderr exactly as today via the last-resort handler. In verbose mode
no filter is added (INFO flows to both the terminal and the file).

`huggingface_hub` is the special case: the same file handler is attached
directly to that logger **only when `propagate` is `False` at block entry**
(runtime check, not an assumption); when `propagate` is `True` (the
default) nothing is attached there and HF records reach the file via root
propagation. All handler/level/filter changes are restored in a `finally`
on every exit path (normal, exception, `KeyboardInterrupt`).

Redaction: the file handler's `Formatter` subclass rewrites the formatted
message *and* exception text, replacing `hf_[A-Za-z0-9]{8,}` →
`hf_<redacted>`, case-insensitive `Bearer\s+\S+` → `Bearer <redacted>`, and
the value of the env var named by the LLM config's `api_key_env` (when set
and ≥ 8 chars). Transcript text is never logged by any stage.

Fail-open: if the log directory/file cannot be created or opened (read-only
CWD, permissions, ENOSPC, hostile stem), the run behaves identically to no
file logging — at most one short stderr notice per CLI invocation
(`--quiet` suppresses it), no exception leaks. A mid-run write failure is
swallowed silently.

The four seams (`with file_log(stem)` per output):

- **seam (a)** `transcribe_batch` (expert single file): the span starts
  *after* the `_resolve_llm_config` check so a `ConfigError` abort creates
  no log file.
- **seam (b)** the `run_preset` plain per-file loop (single / memo /
  `--no-group`): the span wraps the entire per-file iteration (transcribe
  through the write/notification).
- **seam (c)** the `run_batch` group loop (grouped meeting): the span
  wraps the per-group transcribe + result handling; the log is named after
  the group's **first part** NFC stem.
- **seam (d)** the `run_batch` plain loop (`batch_plain._run_plain`):
  the span wraps each file's transcribe + result handling in the
  single-file / `--no-group` short-circuit path (expert multi-file
  `--no-group`, and the expert single-file path when routed through
  `run_batch`); the log is named after the file's NFC stem.

## Invariants (authoritative: AGENTS.md "Project Invariants")

1. Transcription is local, full stop. No cloud-ASR fallback.
2. The consensus pipeline is the architecture. Skip-by-flag yes, delete no.
3. Never force a single language on a memo — language is a span property.
4. Model weights are revision-pinned via `snapshot_download`.
5. The LLM is optional, configured, OpenAI-compatible, and fails open.
6. Audio contract: 16 kHz mono float32, decoded once at ingest.
7. No model becomes a default without a WER run on our own corpus.
