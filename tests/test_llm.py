"""Tests for the OpenAI-compatible LLM client (issue #9).

Covers:
- ``load_config``: TOML parsing, missing/malformed sections, timeout
  validation, trailing-slash normalization.
- ``LLMClient.adjudicate``: fail-open matrix (timeout, HTTP 4xx/5xx,
  malformed JSON, empty/missing/non-string content, missing env key,
  empty candidates), success path, per-span semantics.
- No-hardcoding regression: base URL + model come from config, key comes
  from the environment.
- Request shape: URL is ``{base_url}/chat/completions``; body carries
  ``model``, ``messages``, ``temperature=0``; headers carry the Bearer
  token from the env-named variable.

No network in unit tests: ``httpx.Client`` is patched (per AGENTS.md,
no respx dependency in this ticket).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import httpx
import pytest

from vemoizer.llm import (
    LLMClient,
    LLMConfig,
    adjudicate_span,
    load_config,
)

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "llm"
CONFIG_DIR = FIXTURES_DIR / "config"
RESPONSES_DIR = FIXTURES_DIR / "openai_responses"
ADJUDICATION_DIR = FIXTURES_DIR / "adjudication"

#: A well-formed config used by most LLMClient tests.
DEFAULT_CONFIG = LLMConfig(
    base_url="https://api.example.com/v1",
    model="gpt-4o-mini",
    api_key_env="VEMOIZER_LLM_API_KEY",
    timeout_seconds=10.0,
)


def _load_response(name: str) -> dict[str, Any]:
    with (RESPONSES_DIR / name).open() as f:
        return json.load(f)


def _load_adjudication(name: str) -> dict[str, Any]:
    with (ADJUDICATION_DIR / name).open() as f:
        return json.load(f)


def _mock_response(body: Any, status_code: int = 200) -> MagicMock:
    """Build a MagicMock that behaves like an httpx.Response.

    ``body`` is whatever ``resp.json()`` returns — typically a dict, but
    some malformed-provider cases produce a list or other JSON value. The
    client's defensive parsing must handle all of them, so the mock helper
    accepts any JSON-serializable value.
    """
    resp = MagicMock(spec=httpx.Response)
    resp.status_code = status_code
    resp.json = MagicMock(return_value=body)
    # raise_for_status() should be a no-op on 2xx, raise HTTPStatusError on 4xx/5xx.
    if status_code >= 400:
        err = httpx.HTTPStatusError("error", request=MagicMock(), response=resp)
        resp.raise_for_status = MagicMock(side_effect=err)
    else:
        resp.raise_for_status = MagicMock()
    return resp


def _mock_client_factory(mock_client: MagicMock) -> MagicMock:
    """Wrap a mock client so ``with httpx.Client(...) as c`` yields it.

    ``httpx.Client`` is a context manager; ``with`` calls ``__enter__`` on
    the instance. By default a fresh MagicMock returns a *different*
    MagicMock from ``__enter__`` than the instance itself, so ``client``
    inside the ``with`` block would not be the mock we configured. This
    helper makes ``__enter__`` return the configured mock.
    """
    mock_client.__enter__ = MagicMock(return_value=mock_client)
    mock_client.__exit__ = MagicMock(return_value=False)
    return mock_client


class TestLoadConfig:
    """TOML parsing + validation of the ``[llm]`` section."""

    def test_full_config(self) -> None:
        cfg = load_config(CONFIG_DIR / "full.toml")
        assert cfg is not None
        assert cfg.base_url == "https://api.example.com/v1"
        assert cfg.model == "gpt-4o-mini"
        assert cfg.api_key_env == "VEMOIZER_LLM_API_KEY"
        assert cfg.timeout_seconds == 10.0

    def test_missing_file_returns_none(self, tmp_path: Path) -> None:
        assert load_config(tmp_path / "does-not-exist.toml") is None

    def test_no_section_returns_none(self) -> None:
        assert load_config(CONFIG_DIR / "no_section.toml") is None

    def test_missing_base_url_returns_none(self) -> None:
        assert load_config(CONFIG_DIR / "missing_base_url.toml") is None

    def test_missing_model_returns_none(self) -> None:
        assert load_config(CONFIG_DIR / "missing_model.toml") is None

    def test_missing_api_key_env_returns_none(self) -> None:
        assert load_config(CONFIG_DIR / "missing_api_key_env.toml") is None

    def test_missing_timeout_returns_none(self) -> None:
        # invariant: timeout must always be set (httpx default None = hang)
        assert load_config(CONFIG_DIR / "missing_timeout.toml") is None

    def test_zero_timeout_returns_none(self) -> None:
        assert load_config(CONFIG_DIR / "zero_timeout.toml") is None

    def test_negative_timeout_returns_none(self) -> None:
        assert load_config(CONFIG_DIR / "negative_timeout.toml") is None

    def test_integer_timeout_accepted(self) -> None:
        cfg = load_config(CONFIG_DIR / "integer_timeout.toml")
        assert cfg is not None
        assert cfg.timeout_seconds == 5.0

    def test_trailing_slash_normalized(self) -> None:
        cfg = load_config(CONFIG_DIR / "trailing_slash.toml")
        assert cfg is not None
        assert cfg.base_url == "https://api.example.com/v1"  # no trailing slash

    def test_bad_toml_returns_none(self) -> None:
        assert load_config(CONFIG_DIR / "bad_toml.toml") is None

    def test_accepts_str_path(self) -> None:
        cfg = load_config(str(CONFIG_DIR / "full.toml"))
        assert cfg is not None

    def test_directory_path_returns_none(self) -> None:
        # Passing a directory (not a file) must fail open, not raise.
        assert load_config(CONFIG_DIR) is None


class TestBuildRequest:
    """Request shape: URL, body, headers. No hardcoding regression."""

    def test_url_is_base_url_plus_chat_completions(self) -> None:
        client = LLMClient(DEFAULT_CONFIG)
        url, _, _ = client._build_request("sys", "user")
        assert url == "https://api.example.com/v1/chat/completions"

    def test_body_carry_config_model(self) -> None:
        client = LLMClient(DEFAULT_CONFIG)
        _, body, _ = client._build_request("sys", "user")
        assert body["model"] == "gpt-4o-mini"

    def test_body_has_system_and_user_messages(self) -> None:
        client = LLMClient(DEFAULT_CONFIG)
        _, body, _ = client._build_request("system prompt", "user prompt")
        assert len(body["messages"]) == 2
        assert body["messages"][0] == {"role": "system", "content": "system prompt"}
        assert body["messages"][1] == {"role": "user", "content": "user prompt"}

    def test_body_temperature_zero(self) -> None:
        client = LLMClient(DEFAULT_CONFIG)
        _, body, _ = client._build_request("sys", "user")
        assert body["temperature"] == 0

    def test_headers_carry_bearer_token_from_env(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("VEMOIZER_LLM_API_KEY", "sk-test-123")
        client = LLMClient(DEFAULT_CONFIG)
        _, _, headers = client._build_request("sys", "user")
        assert headers["Authorization"] == "Bearer sk-test-123"

    def test_headers_empty_when_env_unset(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("VEMOIZER_LLM_API_KEY", raising=False)
        client = LLMClient(DEFAULT_CONFIG)
        _, _, headers = client._build_request("sys", "user")
        assert headers == {}

    def test_headers_empty_when_env_empty(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("VEMOIZER_LLM_API_KEY", "")
        client = LLMClient(DEFAULT_CONFIG)
        _, _, headers = client._build_request("sys", "user")
        assert headers == {}

    def test_no_hardcoded_provider_or_url(self) -> None:
        # The base URL is entirely config-driven. Change the config,
        # change the URL.
        cfg = LLMConfig(
            base_url="https://custom.internal.example/api",
            model="local-llama-70b",
            api_key_env="UNUSED",
            timeout_seconds=5.0,
        )
        client = LLMClient(cfg)
        url, body, _ = client._build_request("sys", "user")
        assert url == "https://custom.internal.example/api/chat/completions"
        assert body["model"] == "local-llama-70b"


class TestAdjudicateFailOpen:
    """The fail-open contract: on ANY failure, return span_text unchanged."""

    def test_missing_env_key_returns_span_text(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("VEMOIZER_LLM_API_KEY", raising=False)
        client = LLMClient(DEFAULT_CONFIG)
        result = client.adjudicate(
            "some span", [{"source": "parakeet", "text": "some span"}]
        )
        assert result == "some span"

    def test_empty_env_key_returns_span_text(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("VEMOIZER_LLM_API_KEY", "")
        client = LLMClient(DEFAULT_CONFIG)
        result = client.adjudicate(
            "some span", [{"source": "parakeet", "text": "some span"}]
        )
        assert result == "some span"

    def test_empty_candidates_returns_span_text(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("VEMOIZER_LLM_API_KEY", "sk-test")
        client = LLMClient(DEFAULT_CONFIG)
        assert client.adjudicate("span", []) == "span"

    def test_all_candidates_empty_returns_span_text(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("VEMOIZER_LLM_API_KEY", "sk-test")
        client = LLMClient(DEFAULT_CONFIG)
        fixture = _load_adjudication("span_all_candidates_empty.json")
        result = client.adjudicate(
            fixture["span_text"], fixture["candidates"], fixture["context"]
        )
        assert result == fixture["span_text"]

    def test_timeout_returns_span_text(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("VEMOIZER_LLM_API_KEY", "sk-test")
        client = LLMClient(DEFAULT_CONFIG)

        mock_client = _mock_client_factory(MagicMock(spec=httpx.Client))
        mock_client.post = MagicMock(side_effect=httpx.TimeoutException("timed out"))
        with patch("vemoizer.llm.httpx.Client", return_value=mock_client):
            result = client.adjudicate("span", [{"source": "parakeet", "text": "span"}])
        assert result == "span"

    def test_connection_error_returns_span_text(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("VEMOIZER_LLM_API_KEY", "sk-test")
        client = LLMClient(DEFAULT_CONFIG)

        mock_client = _mock_client_factory(MagicMock(spec=httpx.Client))
        mock_client.post = MagicMock(side_effect=httpx.ConnectError("no route"))
        with patch("vemoizer.llm.httpx.Client", return_value=mock_client):
            result = client.adjudicate("span", [{"source": "parakeet", "text": "span"}])
        assert result == "span"

    def test_http_4xx_returns_span_text(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("VEMOIZER_LLM_API_KEY", "sk-test")
        client = LLMClient(DEFAULT_CONFIG)

        body = _load_response("error_body.json")
        mock_resp = _mock_response(body, status_code=429)
        mock_client = _mock_client_factory(MagicMock(spec=httpx.Client))
        mock_client.post = MagicMock(return_value=mock_resp)
        with patch("vemoizer.llm.httpx.Client", return_value=mock_client):
            result = client.adjudicate("span", [{"source": "parakeet", "text": "span"}])
        assert result == "span"

    def test_http_5xx_returns_span_text(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("VEMOIZER_LLM_API_KEY", "sk-test")
        client = LLMClient(DEFAULT_CONFIG)

        body = _load_response("error_body.json")
        mock_resp = _mock_response(body, status_code=503)
        mock_client = _mock_client_factory(MagicMock(spec=httpx.Client))
        mock_client.post = MagicMock(return_value=mock_resp)
        with patch("vemoizer.llm.httpx.Client", return_value=mock_client):
            result = client.adjudicate("span", [{"source": "parakeet", "text": "span"}])
        assert result == "span"

    def test_invalid_json_returns_span_text(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("VEMOIZER_LLM_API_KEY", "sk-test")
        client = LLMClient(DEFAULT_CONFIG)

        mock_resp = _mock_response({})
        mock_resp.json = MagicMock(side_effect=ValueError("not json"))
        mock_client = _mock_client_factory(MagicMock(spec=httpx.Client))
        mock_client.post = MagicMock(return_value=mock_resp)
        with patch("vemoizer.llm.httpx.Client", return_value=mock_client):
            result = client.adjudicate("span", [{"source": "parakeet", "text": "span"}])
        assert result == "span"

    def test_empty_choices_returns_span_text(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("VEMOIZER_LLM_API_KEY", "sk-test")
        client = LLMClient(DEFAULT_CONFIG)

        body = _load_response("empty_choices.json")
        mock_resp = _mock_response(body)
        mock_client = _mock_client_factory(MagicMock(spec=httpx.Client))
        mock_client.post = MagicMock(return_value=mock_resp)
        with patch("vemoizer.llm.httpx.Client", return_value=mock_client):
            result = client.adjudicate("span", [{"source": "parakeet", "text": "span"}])
        assert result == "span"

    def test_missing_message_returns_span_text(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("VEMOIZER_LLM_API_KEY", "sk-test")
        client = LLMClient(DEFAULT_CONFIG)

        body = _load_response("missing_message.json")
        mock_resp = _mock_response(body)
        mock_client = _mock_client_factory(MagicMock(spec=httpx.Client))
        mock_client.post = MagicMock(return_value=mock_resp)
        with patch("vemoizer.llm.httpx.Client", return_value=mock_client):
            result = client.adjudicate("span", [{"source": "parakeet", "text": "span"}])
        assert result == "span"

    def test_missing_content_returns_span_text(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("VEMOIZER_LLM_API_KEY", "sk-test")
        client = LLMClient(DEFAULT_CONFIG)

        body = _load_response("missing_content.json")
        mock_resp = _mock_response(body)
        mock_client = _mock_client_factory(MagicMock(spec=httpx.Client))
        mock_client.post = MagicMock(return_value=mock_resp)
        with patch("vemoizer.llm.httpx.Client", return_value=mock_client):
            result = client.adjudicate("span", [{"source": "parakeet", "text": "span"}])
        assert result == "span"

    def test_empty_content_returns_span_text(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("VEMOIZER_LLM_API_KEY", "sk-test")
        client = LLMClient(DEFAULT_CONFIG)

        body = _load_response("empty_content.json")
        mock_resp = _mock_response(body)
        mock_client = _mock_client_factory(MagicMock(spec=httpx.Client))
        mock_client.post = MagicMock(return_value=mock_resp)
        with patch("vemoizer.llm.httpx.Client", return_value=mock_client):
            result = client.adjudicate("span", [{"source": "parakeet", "text": "span"}])
        assert result == "span"

    def test_non_string_content_returns_span_text(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("VEMOIZER_LLM_API_KEY", "sk-test")
        client = LLMClient(DEFAULT_CONFIG)

        body = _load_response("non_string_content.json")
        mock_resp = _mock_response(body)
        mock_client = _mock_client_factory(MagicMock(spec=httpx.Client))
        mock_client.post = MagicMock(return_value=mock_resp)
        with patch("vemoizer.llm.httpx.Client", return_value=mock_client):
            result = client.adjudicate("span", [{"source": "parakeet", "text": "span"}])
        assert result == "span"

    def test_response_not_dict_returns_span_text(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("VEMOIZER_LLM_API_KEY", "sk-test")
        client = LLMClient(DEFAULT_CONFIG)

        # A valid JSON response that is not a dict (e.g., a list) must fail open.
        mock_resp = _mock_response([])
        mock_client = _mock_client_factory(MagicMock(spec=httpx.Client))
        mock_client.post = MagicMock(return_value=mock_resp)
        with patch("vemoizer.llm.httpx.Client", return_value=mock_client):
            result = client.adjudicate("span", [{"source": "parakeet", "text": "span"}])
        assert result == "span"


class TestAdjudicateSuccess:
    """The happy path: a 200 response with a usable content string."""

    def test_success_returns_adjudicated_text(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("VEMOIZER_LLM_API_KEY", "sk-test")
        client = LLMClient(DEFAULT_CONFIG)

        body = _load_response("success_finnish_codeswitched.json")
        mock_resp = _mock_response(body)
        mock_client = _mock_client_factory(MagicMock(spec=httpx.Client))
        mock_client.post = MagicMock(return_value=mock_resp)
        with patch("vemoizer.llm.httpx.Client", return_value=mock_client):
            fixture = _load_adjudication("span_finnish_codeswitched.json")
            result = client.adjudicate(
                fixture["span_text"], fixture["candidates"], fixture["context"]
            )
        assert result == fixture["expected"]

    def test_success_multi_token(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("VEMOIZER_LLM_API_KEY", "sk-test")
        client = LLMClient(DEFAULT_CONFIG)

        body = _load_response("success_multi_token.json")
        mock_resp = _mock_response(body)
        mock_client = _mock_client_factory(MagicMock(spec=httpx.Client))
        mock_client.post = MagicMock(return_value=mock_resp)
        with patch("vemoizer.llm.httpx.Client", return_value=mock_client):
            result = client.adjudicate("span", [{"source": "parakeet", "text": "span"}])
        assert result == "MLX-pohjainen sovellus käynnistyy"

    def test_timeout_is_always_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """httpx default timeout=None means hang; the client must always set one."""
        monkeypatch.setenv("VEMOIZER_LLM_API_KEY", "sk-test")
        client = LLMClient(DEFAULT_CONFIG)

        body = _load_response("success_finnish_codeswitched.json")
        mock_resp = _mock_response(body)
        mock_client = _mock_client_factory(MagicMock(spec=httpx.Client))
        mock_client.post = MagicMock(return_value=mock_resp)
        with patch("vemoizer.llm.httpx.Client", return_value=mock_client) as mock_cls:
            client.adjudicate("span", [{"source": "parakeet", "text": "span"}])

        # The Client was constructed with an explicit timeout.
        assert mock_cls.call_args.kwargs.get("timeout") == 10.0
        assert mock_cls.call_args.kwargs.get("timeout") is not None

    def test_request_payload_contains_context_and_candidates(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("VEMOIZER_LLM_API_KEY", "sk-test")
        client = LLMClient(DEFAULT_CONFIG)

        body = _load_response("success_finnish_codeswitched.json")
        mock_resp = _mock_response(body)
        mock_client = _mock_client_factory(MagicMock(spec=httpx.Client))
        mock_client.post = MagicMock(return_value=mock_resp)
        with patch("vemoizer.llm.httpx.Client", return_value=mock_client):
            fixture = _load_adjudication("span_finnish_codeswitched.json")
            client.adjudicate(
                fixture["span_text"], fixture["candidates"], fixture["context"]
            )

        # Inspect the request body that was sent.
        post_call = mock_client.post.call_args
        sent_body = post_call.kwargs["json"]
        user_message = sent_body["messages"][1]["content"]
        # Context must be in the prompt.
        assert fixture["context"] in user_message
        # Every candidate source must be in the prompt.
        for cand in fixture["candidates"]:
            assert cand["source"] in user_message
            assert cand["text"] in user_message

    def test_url_in_request_matches_config(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("VEMOIZER_LLM_API_KEY", "sk-test")
        client = LLMClient(DEFAULT_CONFIG)

        body = _load_response("success_finnish_codeswitched.json")
        mock_resp = _mock_response(body)
        mock_client = _mock_client_factory(MagicMock(spec=httpx.Client))
        mock_client.post = MagicMock(return_value=mock_resp)
        with patch("vemoizer.llm.httpx.Client", return_value=mock_client):
            client.adjudicate("span", [{"source": "parakeet", "text": "span"}])

        post_call = mock_client.post.call_args
        sent_url = post_call.args[0] if post_call.args else post_call.kwargs["url"]
        assert sent_url == "https://api.example.com/v1/chat/completions"


class TestPerSpanSemantics:
    """One span failing must not affect other spans; the whole run never raises."""

    def test_one_span_fails_others_still_adjudicated(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("VEMOIZER_LLM_API_KEY", "sk-test")
        client = LLMClient(DEFAULT_CONFIG)

        # Span 1: timeout → fail open to span_text.
        mock_client_timeout = _mock_client_factory(MagicMock(spec=httpx.Client))
        mock_client_timeout.post = MagicMock(side_effect=httpx.TimeoutException("t/o"))

        # Span 2: success.
        body = _load_response("success_finnish_codeswitched.json")
        mock_resp = _mock_response(body)
        mock_client_success = _mock_client_factory(MagicMock(spec=httpx.Client))
        mock_client_success.post = MagicMock(return_value=mock_resp)

        # Build a fake "client factory" that alternates: first call times out,
        # second succeeds.
        with patch(
            "vemoizer.llm.httpx.Client",
            side_effect=[
                mock_client_timeout,
                mock_client_success,
            ],
        ):
            span1 = "span one text"
            candidates1 = [{"source": "parakeet", "text": "span one text"}]
            r1 = client.adjudicate(span1, candidates1)
            span2 = "MLX poijainen sovellus"
            candidates2 = [
                {"source": "parakeet", "text": "MLX-pohjainen sovellus"},
                {"source": "canary", "text": "MLX poijainen sovollus"},
            ]
            r2 = client.adjudicate(span2, candidates2)

        # Span 1 failed open; span 2 succeeded.
        assert r1 == span1
        assert r2 == "MLX-pohjainen sovellus"

    def test_never_raises_on_any_failure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The whole run never raises, even on catastrophic client errors."""
        monkeypatch.setenv("VEMOIZER_LLM_API_KEY", "sk-test")
        client = LLMClient(DEFAULT_CONFIG)

        # httpx.Client constructor itself raises (e.g., bad config).
        with patch("vemoizer.llm.httpx.Client", side_effect=OSError("no fd")):
            result = client.adjudicate("span", [{"source": "parakeet", "text": "span"}])
        assert result == "span"


class TestModuleLevelConvenience:
    """adjudicate_span() is a thin wrapper over LLMClient.adjudicate()."""

    def test_delegates_to_client(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("VEMOIZER_LLM_API_KEY", "sk-test")

        body = _load_response("success_finnish_codeswitched.json")
        mock_resp = _mock_response(body)
        mock_client = _mock_client_factory(MagicMock(spec=httpx.Client))
        mock_client.post = MagicMock(return_value=mock_resp)
        with patch("vemoizer.llm.httpx.Client", return_value=mock_client):
            result = adjudicate_span(
                DEFAULT_CONFIG,
                "span",
                [{"source": "parakeet", "text": "span"}],
                "context here",
            )
        assert result == "MLX-pohjainen sovellus"

    def test_fails_open_on_missing_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("VEMOIZER_LLM_API_KEY", raising=False)
        result = adjudicate_span(
            DEFAULT_CONFIG,
            "span",
            [{"source": "parakeet", "text": "span"}],
        )
        assert result == "span"
