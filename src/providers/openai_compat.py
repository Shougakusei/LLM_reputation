from __future__ import annotations

import asyncio
import json
import os
import random

import httpx

from src.core.config import ProviderCfg
from src.providers.base import (
    Completion,
    HttpAttempt,
    LLMProvider,
    Message,
    ProviderHTTPError,
    ProviderParseError,
    ProviderUnavailable,
)

_RETRY_BASE_S = 5.0
_RETRY_CAP_S = 30.0
_MAX_ATTEMPTS = 2


class OpenAICompatibleProvider:
    """Talks to any OpenAI-compatible /chat/completions endpoint (Ollama, OpenAI,
    Cerebras, Gemini, ...). Retries 429/5xx/network with exponential backoff."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        *,
        timeout_s: float = 120.0,
        reasoning: bool = True,
        reasoning_effort: str = "",
        extra_body: dict | None = None,
        stream: bool = False,
        client: httpx.AsyncClient | None = None,
    ):
        self._url = base_url.rstrip("/") + "/chat/completions"
        self._headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        self._model = model
        self._reasoning = reasoning
        self._reasoning_effort = reasoning_effort
        self._extra_body = extra_body or {}
        self._stream = stream
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(
            timeout=httpx.Timeout(connect=10.0, read=timeout_s, write=30.0, pool=10.0)
        )

    async def complete(
        self,
        *,
        system: str,
        messages: list[Message],
        temperature: float,
        max_tokens: int,
    ) -> Completion:
        payload = {
            "model": self._model,
            "messages": [
                {"role": "system", "content": system},
                *({"role": m.role, "content": m.content} for m in messages),
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        # Controlling reasoning for reasoning models: turn it off — an explicit {"enabled": false}
        # flag (Non-think); we don't send it enabled by default (the provider decides on its own,
        # non-reasoning models ignore the field). reasoning_effort is sent only if set (groundwork
        # for the future: high/max).
        if not self._reasoning:
            payload["reasoning"] = {"enabled": False}
        if self._reasoning_effort:
            payload["reasoning_effort"] = self._reasoning_effort
        if self._stream:
            payload["stream"] = True
            payload["stream_options"] = {"include_usage": True}   # usage arrives in the last chunk
        # extra_body — provider-specific fields (e.g. vLLM chat_template_kwargs); merged
        # last so the config can override the base fields when needed.
        payload.update(self._extra_body)
        # _post_with_retries guarantees a resp with an extractable response (a broken envelope
        # and a malformed shape are retried the same as 5xx). Here we parse again — this time
        # to extract the content.
        resp, attempts = await self._post_with_retries(payload)   # raises already carry request+attempts
        raw_text = resp.text
        data = self._decode(resp)
        content = data["choices"][0]["message"].get("content")    # extractability guaranteed by the gate
        usage = data.get("usage") or {}
        pt = int(usage.get("prompt_tokens", 0))
        ct = int(usage.get("completion_tokens", 0))
        final = HttpAttempt(
            status="ok", status_code=resp.status_code, request=payload,
            response=content or "", response_raw=raw_text, error=None,
            prompt_tokens=pt, completion_tokens=ct,
        )
        return Completion(
            text=content or "", prompt_tokens=pt, completion_tokens=ct,
            raw=data, request=payload, attempts=attempts + (final,),
        )

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    def _decode(self, resp: httpx.Response) -> dict:
        """The response as one OpenAI-shaped dict: parsed JSON, or the folded SSE stream."""
        return _fold_sse(resp.text) if self._stream else resp.json()

    async def _post_with_retries(self, payload: dict) -> tuple[httpx.Response, tuple[HttpAttempt, ...]]:
        """Make the request with retries. Return (resp with a valid JSON body, retry attempts).

        Accumulates EVERY HTTP attempt (for the L2 log). Retries as transient: network,
        429/5xx, and a **broken JSON envelope** (`resp.json()` failed to parse — usually a
        proxy/connection drop). Terminal (raise with `request`+`attempts`): 4xx and retry
        exhaustion. Returns the raw `resp` (body already verified as JSON); content
        extraction happens in `complete`.
        """
        attempts: list[HttpAttempt] = []
        last_exc: Exception | None = None
        for attempt in range(_MAX_ATTEMPTS):
            retry_after: float | None = None
            try:
                resp = await self._client.post(
                    self._url, json=payload, headers=self._headers
                )
            except (httpx.TransportError, httpx.TimeoutException) as e:
                last_exc = e
                attempts.append(HttpAttempt(
                    status="network", status_code=None, request=payload,
                    response=None, response_raw=None, error=str(e)))
            else:
                code = resp.status_code
                if code < 400:
                    try:
                        data = self._decode(resp)                # gate: is the body extractable?
                        data["choices"][0]["message"].get("content")
                    except ValueError:
                        last_exc = ProviderParseError("response was not valid JSON")
                        attempts.append(HttpAttempt(   # broken envelope — retry it like a 5xx
                            status="bad_json", status_code=code, request=payload,
                            response=None, response_raw=resp.text, error="not valid JSON"))
                    except (KeyError, IndexError, TypeError, AttributeError):
                        last_exc = ProviderParseError(f"unexpected response shape: {data!r}")
                        attempts.append(HttpAttempt(   # valid JSON but not extractable — also retry it
                            status="bad_shape", status_code=code, request=payload,
                            response=None, response_raw=resp.text, error="unexpected shape"))
                    else:
                        return resp, tuple(attempts)
                elif code == 429 or 500 <= code < 600:
                    last_exc = ProviderUnavailable(f"HTTP {code} from {self._url}")
                    attempts.append(HttpAttempt(           # also record the 5xx/429 body
                        status="server_error", status_code=code, request=payload,
                        response=None, response_raw=resp.text, error=f"HTTP {code}"))
                    retry_after = _parse_retry_after(resp.headers.get("Retry-After"))
                else:
                    err = ProviderHTTPError(code, resp.text)
                    err.request = payload
                    err.attempts = tuple(attempts) + (HttpAttempt(
                        status="http_error", status_code=code, request=payload,
                        response=None, response_raw=resp.text, error=f"HTTP {code}"),)
                    raise err

            if attempt == _MAX_ATTEMPTS - 1:
                break
            delay = retry_after if retry_after is not None else _backoff_delay(attempt)
            await asyncio.sleep(delay)

        # retries exhausted: each attempt is already in attempts (with the 5xx / network body)
        err = ProviderUnavailable(f"exhausted {_MAX_ATTEMPTS} attempts to {self._url}")
        err.request = payload
        err.attempts = tuple(attempts)
        raise err from last_exc


def make_provider(
    cfg: ProviderCfg, *, client: httpx.AsyncClient | None = None
) -> LLMProvider:
    api_key = (os.environ.get(cfg.api_key_env) if cfg.api_key_env else None) or "sk-noauth"
    return OpenAICompatibleProvider(
        cfg.base_url, api_key, cfg.model, timeout_s=cfg.timeout_s,
        reasoning=cfg.reasoning, reasoning_effort=cfg.reasoning_effort,
        extra_body=cfg.extra_body, stream=cfg.stream, client=client
    )


def _fold_sse(text: str) -> dict:
    """Fold an SSE chat stream into one OpenAI-shaped response dict.

    Raises ValueError (like a broken JSON body) when the text holds no `data:` chunks.
    """
    content: list[str] = []
    reasoning: list[str] = []
    usage: dict = {}
    seen = False
    for line in text.splitlines():
        if not line.startswith("data:"):
            continue
        chunk = line[5:].strip()
        if chunk == "[DONE]":
            continue
        d = json.loads(chunk)
        seen = True
        if d.get("usage"):
            usage = d["usage"]
        for choice in d.get("choices") or []:
            delta = choice.get("delta") or {}
            if delta.get("content"):
                content.append(delta["content"])
            r = delta.get("reasoning") or delta.get("reasoning_content")
            if r:
                reasoning.append(r)
    if not seen:
        raise ValueError("no SSE data chunks in the response")
    message: dict = {"role": "assistant", "content": "".join(content)}
    if reasoning:
        message["reasoning"] = "".join(reasoning)
    return {"choices": [{"message": message}], "usage": usage}


def _backoff_delay(attempt: int) -> float:
    capped = min(_RETRY_CAP_S, _RETRY_BASE_S * (2**attempt))
    return capped + random.random() * _RETRY_BASE_S


def _parse_retry_after(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        # HTTP-date form is not handled; caller falls back to exponential backoff.
        return None
