# Corpus attribution

This corpus combines two distinct audio sources. Each source has its own
licence and attribution requirements.

## Piper TTS (synthetic Finnish)

The stems `fi_*`, `en_*`, and `meeting_sample` under this directory are
synthesized from the Piper voice `fi_FI-harri-low` (dataset licensed CC0;
the voice's Ryan lineage is unencumbered). See `scripts/gen_fixtures.py`
for the revision-pinned model ID and regeneration instructions.

These clips are **not** real speech — they exist to exercise the audio
contract and the consensus pipeline on clean, noise-free audio. They are
the regression gate, not the challenge set (see issue #62).

## FLEURS Finnish (real human speech)

The stems `fleurs_fi_*` under this directory are genuine human speech,
selected from the **FLEURS** dataset:

- Repository: [`google/fleurs`](https://huggingface.co/datasets/google/fleurs)
- Config: `fi_fi` (Finnish, Finland)
- Split: `train` (the `parquet-data/fi_fi/train-00000-of-00001.parquet`
  file)
- Licence: **CC-BY-4.0** (Creative Commons Attribution 4.0 International)
  — <https://creativecommons.org/licenses/by/4.0/>

CC-BY-4.0 requires that appropriate credit be given to the creator and
the licensor, a link to the licence, and an indication of whether changes
were made. No endorsement is implied.

**Credit (per CC-BY-4.0):**

> Audio clips from the [FLEURS dataset](https://huggingface.co/datasets/google/fleurs),
> config `fi_fi`, split `train`, by Google (FLEURS team). Licensed under
> [CC-BY-4.0](https://creativecommons.org/licenses/by/4.0/).

**Selection parameters** (reproducible; see `scripts/gen_real_speech_corpus.py`):

- Seed: `20261003`
- Number of clips: `28`
- Duration window: `[3.5, 5.5]` seconds
- Word-count window: `[8, 18]` words
- Selection method: the committed 28 clips are selected by the script's
  default `--ids` list (the authoritative clip ids are recorded below);
  the seeded window draw (`random.Random(seed).sample` over candidates
  sorted by `(id, row_index)`) is the documented original draw that
  produced this set. Re-running the script with the default args
  regenerates exactly the committed 28 clips, byte-identically.

**Committed clip ids** (the authoritative selection, in `(id, take)`
order; a take `b` is the second row of the same FLEURS `id` in
`(id, row_index)` order):

```
24, 25, 34, 36, 36b, 204, 235, 238, 246, 252, 533, 596, 604, 630,
656, 692, 711, 732, 748, 748b, 800, 991, 1039, 1046, 1096, 1290,
1324, 1343
```

The same input parquet + the same id list always produces the same set
of clips (byte-identical WAVs), so the corpus is stable across re-runs.
The parquet file is a one-time dev-time download (never a runtime
dependency of `vemoizer`); the script does not fetch on its own.

**Note on the `fleurs_fi_<id>`, `fleurs_fi_<id>b`, ... stems:** the FLEURS
parquet keys rows by utterance `id`, not by clip — several rows share an
`id` (different takes of the same reference transcript). The stem is the
4-digit `id`; on a collision (two selected takes of the same utterance),
the second+ take in `(id, row_index)` order gets a `b`, `c`, ... suffix,
so no two clips collide on a stem. The `.wav`/`.txt` pair contract is
per-stem, not per-utterance.

**Code-switching gap:** no public dataset contains Finnish with English
technical terms (the project's core use case, see AGENTS.md invariant #3),
so this corpus improves the gate's realism (real human speech, real
channel noise, real speaker variability) but does not measure
code-switching. That gap is documented in `docs/pipeline-spec.md`.
