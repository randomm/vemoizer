# SECURITY.md

Privacy and trust boundary for vemoizer.

## Trust boundary

**Transcription is local, full stop.** Audio and transcripts never leave
the machine for ASR. The only network access in the ASR path is the
one-time model download. There is no cloud-ASR fallback, and adding one
is a product decision, not an implementation detail.

Third-party telemetry is disabled (issue #103). `pyannote.audio` 4.x ships
OpenTelemetry usage metrics enabled by default, sending spans to
`otel.pyannote.ai` with duration and speaker-count metadata — see
`docs/pipeline-spec.md` §9 for the policy. `OTEL_SDK_DISABLED=true` is set
as belt-and-braces; both are `setdefault`, so a user can opt in.

The optional LLM stage (adjudication, Markdown notes, speaker naming)
sends transcript text — not audio — to the OpenAI-compatible endpoint
configured by the user. No LLM is configured by default; every LLM call
fails open when the config, the key, or the endpoint is unavailable.

## Secrets

- The config file names the environment variable that holds the LLM API
  key (via `api_key_env`); a literal `api_key` entry in the config is
  rejected by the strict config loader as an unknown key. The key itself
  is never stored in the config file, the repository, or written to disk
  by vemoizer.
- Never commit API keys or tokens.
- Never commit real personal voice memos, or transcripts derived from
  them.
- Model weights are revision-pinned and live in the HuggingFace cache,
  never in the repository.

## Reporting a vulnerability

Report privately by opening a GitHub Security Advisory on this
repository (Settings → Security → "New draft advisory"), or email the
maintainer. Do not open a public issue with security details.
