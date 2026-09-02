"""OpenAI-compatible LLM client for adjudication and cleanup.

The LLM is optional, configured, and OpenAI-compatible (AGENTS.md
invariant #5). This module implements the client half: config loading
from a user TOML file, an ``httpx``-based request builder, and a
fail-open contract that returns the un-adjudicated transcript on ANY
failure (timeout, HTTP error, missing config, missing API key, malformed
response, empty response, ...).

The client never raises. The caller's un-adjudicated text is returned
whenever the endpoint cannot deliver a usable answer.

Design notes:

- **No hardcoded provider, model, or URL.** All three are read from the
  user's config file. The base URL is used verbatim as a prefix; the
  request is sent to ``{base_url}/chat/completions``.
- **Timeout is always set.** ``httpx`` defaults to ``timeout=None`` (hang
  until the OS gives up); every call in this module passes an explicit
  ``timeout=`` value from config.
- **API key is env-only.** The config names an environment variable; the
  key itself is never stored in the file. If the variable is unset or
  empty the client fails open.
- **Response parsing is defensive.** The OpenAI-compatible contract is
  ``choices[0].message.content`` as a string; some providers return a
  list, some omit the key entirely. Anything that is not a non-empty
  string falls through to the fail-open path.
- **No network in tests.** Unit tests patch ``httpx.Client`` (per
  AGENTS.md: no respx dependency in this ticket).
"""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

#: The JSON key under which the LLM configuration lives in the user TOML
#: file. Missing section → no LLM configured (fail-open, no error).
LLM_CONFIG_SECTION: str = "llm"

#: Default system prompt for adjudication. Intentionally short: the
#: model's task is to pick or compose the final text for a disputed
#: span, not to do editorial rewriting.
_ADJUDICATION_SYSTEM_PROMPT: str = (
    "You are a transcription adjudicator. Given the surrounding context "
    "and candidate transcriptions for a disputed span, return ONLY the "
    "correct transcription for that span — no tags, no commentary, no "
    "prefix. Keep every non-filler word. If no candidate is clearly "
    "correct, compose the most plausible text from them."
)


@dataclass(frozen=True)
class LLMConfig:
    """Parsed ``[llm]`` section of the user config file.

    Fields:
      base_url: any OpenAI-compatible endpoint. Used verbatim; the
        request goes to ``{base_url}/chat/completions``. Trailing slash
        is normalized.
      model: model ID to request. Never hardcoded in source.
      api_key_env: name of the environment variable holding the API key.
        The key itself is never stored in the config file.
      timeout_seconds: explicit request timeout. ``httpx`` defaults to
        ``timeout=None`` (hang), so an unset value is a config error —
        the client fails open on it rather than hanging.
    """

    base_url: str
    model: str
    api_key_env: str
    timeout_seconds: float


def load_config(path: Path | str) -> LLMConfig | None:
    """Parse the user config file and return the ``[llm]`` section.

    Returns ``None`` (fail-open signal) when:
      - the file does not exist,
      - the file is not valid TOML,
      - the ``[llm]`` section is absent,
      - the section is present but malformed (missing keys, wrong types,
        non-positive timeout).

    Raises nothing. Callers treat ``None`` as "no LLM configured" and
    return the un-adjudicated transcript.
    """
    try:
        p = Path(path)
        if not p.is_file():
            return None
        with p.open("rb") as f:
            raw = tomllib.load(f)
    except (OSError, tomllib.TOMLDecodeError, ValueError):
        return None

    section = raw.get(LLM_CONFIG_SECTION)
    if not isinstance(section, dict):
        return None

    base_url = section.get("base_url")
    model = section.get("model")
    api_key_env = section.get("api_key_env")
    timeout = section.get("timeout_seconds")

    if not isinstance(base_url, str) or not base_url.strip():
        return None
    if not isinstance(model, str) or not model.strip():
        return None
    if not isinstance(api_key_env, str) or not api_key_env.strip():
        return None
    if not isinstance(timeout, (int, float)) or isinstance(timeout, bool):
        return None
    if timeout <= 0:
        return None

    return LLMConfig(
        base_url=base_url.rstrip("/"),
        model=model.strip(),
        api_key_env=api_key_env,
        timeout_seconds=float(timeout),
    )


def _build_user_prompt(
    span_text: str,
    candidates: list[dict[str, str]],
    context: str,
) -> str:
    """Build the user message for an adjudication request.

    The prompt carries:
      - the surrounding context (what came just before / after the span)
      - the disputed span's raw text as decoded (useful when candidates
        are all garbled but the context disambiguates)
      - every candidate transcription, labelled by source
    """
    parts: list[str] = []
    if context:
        parts.append(f"Context: {context}")
    if span_text:
        parts.append(f"Disputed span: {span_text}")
    parts.append("Candidates:")
    for cand in candidates:
        parts.append(f"  - {cand.get('source', '?')}: {cand.get('text', '')}")
    parts.append("Return ONLY the corrected transcription for the disputed span.")
    return "\n".join(parts)


class LLMClient:
    """OpenAI-compatible LLM client with a fail-open contract.

    The client never raises. Every public method returns the input text
    (or ``""``) when the endpoint cannot deliver a usable answer. This
    is the "fail-open" contract required by AGENTS.md invariant #5:
    on timeout, error, or missing config, the run returns the
    un-adjudicated transcript rather than failing.
    """

    def __init__(self, config: LLMConfig) -> None:
        self._config = config

    @property
    def config(self) -> LLMConfig:
        return self._config

    def _api_key(self) -> str | None:
        """Read the API key from the environment variable named in config.

        Returns ``None`` when the variable is unset or empty. Callers
        treat ``None`` as fail-open (no request, return the input).
        """
        value = os.environ.get(self._config.api_key_env)
        if value is None or not value.strip():
            return None
        return value

    def _build_request(
        self,
        system_prompt: str,
        user_prompt: str,
    ) -> tuple[str, dict[str, Any], dict[str, str]]:
        """Build the (url, body, headers) tuple for an OpenAI-compatible POST.

        The URL is ``{base_url}/chat/completions`` — the base URL is used
        verbatim (no provider detection, no path surgery). The body is
        the standard OpenAI chat-completions shape.
        """
        url = f"{self._config.base_url}/chat/completions"
        body: dict[str, Any] = {
            "model": self._config.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": 0,
        }
        api_key = self._api_key()
        headers: dict[str, str] = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        return url, body, headers

    def _post(
        self, url: str, body: dict[str, Any], headers: dict[str, str]
    ) -> str | None:
        """Send the request and parse the response.

        Returns the parsed ``choices[0].message.content`` string, or
        ``None`` on any failure (timeout, HTTP error, malformed JSON,
        missing content, non-string content). Never raises.
        """
        try:
            with httpx.Client(timeout=self._config.timeout_seconds) as client:
                resp = client.post(url, json=body, headers=headers)
                if resp.status_code >= 400:
                    resp.raise_for_status()
                data = resp.json()
        except (httpx.HTTPError, ValueError, OSError):
            # httpx.HTTPError covers RequestError, TimeoutException,
            # HTTPStatusError. ValueError covers json.JSONDecodeError
            # (which is a ValueError) and any other JSON parse failure.
            # OSError covers network-level failures that httpx does not
            # wrap into its own hierarchy (e.g. DNS, socket, file-descriptor
            # exhaustion on the Client constructor itself). The fail-open
            # contract is "never raises" — the caller's un-adjudicated text
            # is returned on ANY failure, not just the expected ones.
            return None

        choices = data.get("choices") if isinstance(data, dict) else None
        if not isinstance(choices, list) or not choices:
            return None
        first = choices[0]
        if not isinstance(first, dict):
            return None
        message = first.get("message")
        if not isinstance(message, dict):
            return None
        content = message.get("content")
        if not isinstance(content, str):
            return None
        content = content.strip()
        if not content:
            return None
        return content

    def adjudicate(
        self,
        span_text: str,
        candidates: list[dict[str, str]],
        context: str = "",
    ) -> str:
        """Adjudicate one disputed span. Fails open to ``span_text``.

        Args:
            span_text: the raw text of the disputed span as decoded.
              Returned unchanged when the endpoint cannot deliver a
              usable answer. Empty string is allowed (the span itself
              might be empty; the fail-open path just returns ``""``).
            candidates: list of ``{"source": str, "text": str}`` dicts —
              the candidate transcriptions from decode A, decode B, and
              re-decode. Order is the pipeline's order.
            context: surrounding text (what came just before / after the
              span). Optional; included in the prompt when non-empty.

        Returns:
            The adjudicated span text, or ``span_text`` on any failure.
            The function never raises.
        """
        if not candidates:
            return span_text

        if not span_text.strip() and all(
            not (c.get("text") or "").strip() for c in candidates
        ):
            # Nothing to adjudicate: every candidate is empty.
            return span_text

        api_key = self._api_key()
        if api_key is None:
            return span_text

        user_prompt = _build_user_prompt(span_text, candidates, context)
        url, body, headers = self._build_request(
            _ADJUDICATION_SYSTEM_PROMPT, user_prompt
        )
        result = self._post(url, body, headers)
        if result is None:
            return span_text
        return result


def adjudicate_span(
    config: LLMConfig,
    span_text: str,
    candidates: list[dict[str, str]],
    context: str = "",
) -> str:
    """Module-level convenience for adjudicating one span with an ad-hoc client.

    Constructs a fresh ``LLMClient`` per call. For long runs with many
    spans, construct one ``LLMClient`` and call ``.adjudicate()`` on it
    instead (avoids repeated client-construction overhead; the httpx
    client itself is still created per request).

    Fails open to ``span_text`` on any failure.
    """
    return LLMClient(config).adjudicate(span_text, candidates, context)
