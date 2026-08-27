from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from src.core.config import ProviderCfg
from src.providers.openai_compat import _MAX_ATTEMPTS
from src.providers import (
    Message,
    OpenAICompatibleProvider,
    ProviderHTTPError,
    ProviderUnavailable,
    make_provider,
)


async def _no_sleep(*_a, **_k):
    return None


def _ok_response(content="hi", usage=None):
    body = {"choices": [{"message": {"content": content}, "finish_reason": "stop"}]}
    if usage is not None:
        body["usage"] = usage
    return httpx.Response(200, json=body)


def _provider_with(handler, **kw):
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return OpenAICompatibleProvider("http://x/v1", "k", "m", client=client, **kw)


async def _call(provider, **kw):
    defaults = dict(
        system="s", messages=[Message("user", "q")], temperature=0.0, max_tokens=10
    )
    defaults.update(kw)
    return await provider.complete(**defaults)


async def test_request_shape():
    captured = {}

    def handler(req):
        captured["req"] = req
        captured["body"] = json.loads(req.content)
        return _ok_response(usage={"prompt_tokens": 3, "completion_tokens": 5})

    p = _provider_with(handler)
    await _call(
        p,
        system="SYS",
        messages=[
            Message("user", "hello"),
            Message("assistant", "hi"),
            Message("user", "again"),
        ],
        temperature=0.3,
        max_tokens=64,
    )
    body = captured["body"]
    assert body["model"] == "m"
    assert body["messages"][0] == {"role": "system", "content": "SYS"}
    assert body["messages"][1] == {"role": "user", "content": "hello"}
    assert body["messages"][2] == {"role": "assistant", "content": "hi"}
    assert body["temperature"] == 0.3
    assert body["max_tokens"] == 64
    assert captured["req"].headers["Authorization"] == "Bearer k"
    assert captured["req"].url.path == "/v1/chat/completions"


async def test_reasoning_disabled_sends_enabled_false():
    # reasoning=False -> Non-think: the payload carries {"reasoning": {"enabled": false}}.
    captured = {}

    def handler(req):
        captured["body"] = json.loads(req.content)
        return _ok_response()

    p = _provider_with(handler, reasoning=False)
    await _call(p)
    assert captured["body"]["reasoning"] == {"enabled": False}
    assert "reasoning_effort" not in captured["body"]


async def test_reasoning_enabled_by_default_sends_nothing():
    # Default reasoning=True sends nothing — non-reasoning models shouldn't receive the field.
    captured = {}

    def handler(req):
        captured["body"] = json.loads(req.content)
        return _ok_response()

    p = _provider_with(handler)            # reasoning defaults to True
    await _call(p)
    assert "reasoning" not in captured["body"]
    assert "reasoning_effort" not in captured["body"]


async def test_reasoning_effort_sent_when_set():
    captured = {}

    def handler(req):
        captured["body"] = json.loads(req.content)
        return _ok_response()

    p = _provider_with(handler, reasoning_effort="high")
    await _call(p)
    assert captured["body"]["reasoning_effort"] == "high"


async def test_make_provider_threads_reasoning_from_cfg():
    # make_provider must thread reasoning from ProviderCfg into the payload.
    captured = {}

    def handler(req):
        captured["body"] = json.loads(req.content)
        return _ok_response()

    cfg = ProviderCfg(base_url="http://x/v1", model="m", reasoning=False)
    p = make_provider(cfg, client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    await _call(p)
    assert captured["body"]["reasoning"] == {"enabled": False}


async def test_extra_body_merged_into_payload():
    # extra_body goes into the payload as-is (vLLM: chat_template_kwargs for Qwen3 no-think).
    captured = {}

    def handler(req):
        captured["body"] = json.loads(req.content)
        return _ok_response()

    p = _provider_with(handler, extra_body={"chat_template_kwargs": {"enable_thinking": False}})
    await _call(p)
    assert captured["body"]["chat_template_kwargs"] == {"enable_thinking": False}


async def test_make_provider_threads_extra_body_from_cfg():
    captured = {}

    def handler(req):
        captured["body"] = json.loads(req.content)
        return _ok_response()

    cfg = ProviderCfg(base_url="http://x/v1", model="m",
                      extra_body={"chat_template_kwargs": {"enable_thinking": False}})
    p = make_provider(cfg, client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    await _call(p)
    assert captured["body"]["chat_template_kwargs"] == {"enable_thinking": False}


async def test_completion_carries_sent_request():
    captured = {}

    def handler(req):
        captured["body"] = json.loads(req.content)
        return _ok_response("answer")

    p = _provider_with(handler)
    c = await _call(p, system="SYS", temperature=0.3, max_tokens=64)
    # the Completion's request is EXACTLY the payload that was sent
    assert c.request == captured["body"]
    assert c.request["model"] == "m"
    assert c.request["messages"][0] == {"role": "system", "content": "SYS"}


async def test_non_json_body_retried_then_exhausts(monkeypatch):
    monkeypatch.setattr(asyncio, "sleep", _no_sleep)
    raw = "<html>502 Bad Gateway</html>"
    calls = {"n": 0}

    def handler(req):
        calls["n"] += 1
        return httpx.Response(200, text=raw)

    p = _provider_with(handler)
    with pytest.raises(ProviderUnavailable) as ei:        # a broken JSON envelope is retried, not failed immediately
        await _call(p, system="SYS")
    assert calls["n"] == _MAX_ATTEMPTS
    e = ei.value
    assert len(e.attempts) == _MAX_ATTEMPTS and all(a.status == "bad_json" for a in e.attempts)
    assert e.attempts[-1].response_raw == raw             # the raw body is preserved, even non-JSON
    assert e.attempts[-1].status_code == 200
    assert e.request["messages"][0]["content"] == "SYS"   # what was sent is also on hand


async def test_non_json_then_valid_recovers(monkeypatch):
    monkeypatch.setattr(asyncio, "sleep", _no_sleep)
    calls = {"n": 0}

    def handler(req):
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(200, text="garbage")
        return _ok_response("ok")

    p = _provider_with(handler)
    c = await _call(p)
    assert c.text == "ok"
    assert [a.status for a in c.attempts] == ["bad_json", "ok"]   # one bad reply survived


async def test_http_error_enriches_error():
    p = _provider_with(lambda req: httpx.Response(404, text="nope"))
    with pytest.raises(ProviderHTTPError) as ei:
        await _call(p)
    last = ei.value.attempts[-1]
    assert last.status == "http_error"
    assert last.status_code == 404
    assert last.response_raw == "nope"
    assert ei.value.request is not None


async def test_bad_shape_retried_then_exhausts(monkeypatch):
    monkeypatch.setattr(asyncio, "sleep", _no_sleep)
    calls = {"n": 0}

    def handler(req):
        calls["n"] += 1
        return httpx.Response(200, json={"usage": {}})    # valid JSON, but no choices

    p = _provider_with(handler)
    with pytest.raises(ProviderUnavailable) as ei:        # a bad shape is also retried
        await _call(p)
    assert calls["n"] == _MAX_ATTEMPTS
    e = ei.value
    assert len(e.attempts) == _MAX_ATTEMPTS and all(a.status == "bad_shape" for a in e.attempts)
    assert e.request is not None
    assert "usage" in e.attempts[-1].response_raw         # verbatim body (resp.text)


async def test_every_retry_is_an_attempt(monkeypatch):
    monkeypatch.setattr(asyncio, "sleep", _no_sleep)
    calls = {"n": 0}

    def handler(req):
        calls["n"] += 1
        if calls["n"] < _MAX_ATTEMPTS:        # fail until the last allowed attempt
            return httpx.Response(503, text=f"busy{calls['n']}")
        return _ok_response("ok", usage={"prompt_tokens": 1, "completion_tokens": 1})

    p = _provider_with(handler)
    c = await _call(p)
    # each HTTP attempt is its own row; final ok, the retries carry the 5xx body
    assert [a.status for a in c.attempts] == ["server_error"] * (_MAX_ATTEMPTS - 1) + ["ok"]
    assert c.attempts[0].status_code == 503
    assert c.attempts[0].response_raw == "busy1"
    assert c.attempts[-1].response == "ok"
    assert c.attempts[-1].prompt_tokens == 1


async def test_exhausted_network_records_all_attempts(monkeypatch):
    monkeypatch.setattr(asyncio, "sleep", _no_sleep)

    def handler(req):
        raise httpx.ConnectError("boom", request=req)

    p = _provider_with(handler)
    with pytest.raises(ProviderUnavailable) as ei:
        await _call(p)
    e = ei.value
    assert len(e.attempts) == _MAX_ATTEMPTS           # all attempts recorded
    assert all(a.status == "network" for a in e.attempts)
    assert e.attempts[0].status_code is None          # there was no response
    assert e.request is not None


async def test_parses_text_and_usage():
    p = _provider_with(
        lambda req: _ok_response(
            "answer", usage={"prompt_tokens": 7, "completion_tokens": 11}
        )
    )
    c = await _call(p)
    assert c.text == "answer"
    assert c.prompt_tokens == 7
    assert c.completion_tokens == 11
    assert c.raw["choices"][0]["message"]["content"] == "answer"


async def test_usage_missing_defaults_zero():
    p = _provider_with(lambda req: _ok_response("x"))
    c = await _call(p)
    assert (c.prompt_tokens, c.completion_tokens) == (0, 0)


async def test_null_content_becomes_empty():
    p = _provider_with(lambda req: _ok_response(content=None))
    c = await _call(p)
    assert c.text == ""


async def test_retries_5xx_then_succeeds(monkeypatch):
    monkeypatch.setattr(asyncio, "sleep", _no_sleep)
    calls = {"n": 0}

    def handler(req):
        calls["n"] += 1
        if calls["n"] < _MAX_ATTEMPTS:        # fail until the last allowed attempt
            return httpx.Response(503, text="busy")
        return _ok_response("ok", usage={"prompt_tokens": 1, "completion_tokens": 1})

    p = _provider_with(handler)
    c = await _call(p)
    assert c.text == "ok"
    assert calls["n"] == _MAX_ATTEMPTS


async def test_retries_transport_error(monkeypatch):
    monkeypatch.setattr(asyncio, "sleep", _no_sleep)
    calls = {"n": 0}

    def handler(req):
        calls["n"] += 1
        if calls["n"] < 2:
            raise httpx.ConnectError("boom", request=req)
        return _ok_response("ok")

    p = _provider_with(handler)
    c = await _call(p)
    assert c.text == "ok"
    assert calls["n"] == 2


async def test_honors_retry_after(monkeypatch):
    slept = []

    async def fake_sleep(s):
        slept.append(s)

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)
    calls = {"n": 0}

    def handler(req):
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(429, headers={"Retry-After": "2"}, text="slow down")
        return _ok_response("ok")

    p = _provider_with(handler)
    await _call(p)
    assert slept == [2.0]


@pytest.mark.parametrize("code", [400, 401, 404])
async def test_4xx_fail_fast(code):
    calls = {"n": 0}

    def handler(req):
        calls["n"] += 1
        return httpx.Response(code, text="nope")

    p = _provider_with(handler)
    with pytest.raises(ProviderHTTPError) as ei:
        await _call(p)
    assert ei.value.status_code == code
    assert calls["n"] == 1


async def test_exhausts_retries(monkeypatch):
    monkeypatch.setattr(asyncio, "sleep", _no_sleep)
    calls = {"n": 0}

    def handler(req):
        calls["n"] += 1
        return httpx.Response(503, text="busy")

    p = _provider_with(handler)
    with pytest.raises(ProviderUnavailable):
        await _call(p)
    assert calls["n"] == _MAX_ATTEMPTS


async def test_missing_choices_retried_then_exhausts(monkeypatch):
    monkeypatch.setattr(asyncio, "sleep", _no_sleep)
    p = _provider_with(lambda req: httpx.Response(200, json={"usage": {}}))
    with pytest.raises(ProviderUnavailable):    # bad_shape is retried, then exhausted
        await _call(p)


async def test_aclose_injected_not_closed():
    client = httpx.AsyncClient(transport=httpx.MockTransport(lambda req: _ok_response()))
    p = OpenAICompatibleProvider("http://x/v1", "k", "m", client=client)
    await p.aclose()
    assert not client.is_closed
    await client.aclose()


async def test_aclose_owned_closed():
    p = OpenAICompatibleProvider("http://x/v1", "k", "m")
    await p.aclose()
    assert p._client.is_closed


async def test_make_provider_uses_env_key(monkeypatch):
    monkeypatch.setenv("MY_KEY", "secret123")
    cfg = ProviderCfg(base_url="http://x/v1", model="m", api_key_env="MY_KEY")
    p = make_provider(cfg)
    try:
        assert p._headers["Authorization"] == "Bearer secret123"
    finally:
        await p.aclose()


async def test_make_provider_stub_key_when_unset(monkeypatch):
    monkeypatch.delenv("MISSING_KEY", raising=False)
    cfg = ProviderCfg(base_url="http://x/v1", model="m", api_key_env="MISSING_KEY")
    p = make_provider(cfg)
    try:
        assert p._headers["Authorization"] == "Bearer sk-noauth"
    finally:
        await p.aclose()


# ---- streaming mode (stream-only models: the SSE stream is folded into one Completion) ----

def _sse_response(chunks, usage=None):
    lines = ["data: " + json.dumps({"choices": [{"delta": c, "finish_reason": None}]}) for c in chunks]
    if usage is not None:
        lines.append("data: " + json.dumps({"choices": [], "usage": usage}))
    lines.append("data: [DONE]")
    return httpx.Response(200, text="\n\n".join(lines) + "\n",
                          headers={"Content-Type": "text/event-stream"})


async def test_stream_mode_sends_stream_flags_and_folds_chunks():
    captured = {}

    def handler(req):
        captured["body"] = json.loads(req.content)
        return _sse_response([{"role": "assistant", "content": "hel"}, {"content": "lo"}],
                             usage={"prompt_tokens": 7, "completion_tokens": 2})

    c = await _call(_provider_with(handler, stream=True))
    assert captured["body"]["stream"] is True
    assert captured["body"]["stream_options"] == {"include_usage": True}
    assert c.text == "hello" and (c.prompt_tokens, c.completion_tokens) == (7, 2)
    assert c.raw["choices"][0]["message"]["content"] == "hello"
    assert c.attempts[-1].response == "hello" and "data:" in c.attempts[-1].response_raw


async def test_stream_mode_keeps_reasoning_out_of_content():
    def handler(req):
        return _sse_response([{"reasoning": "think"}, {"reasoning": "ing"}, {"content": "4"}])

    c = await _call(_provider_with(handler, stream=True))
    assert c.text == "4"
    assert c.raw["choices"][0]["message"]["reasoning"] == "thinking"


async def test_stream_mode_non_sse_body_is_retried(monkeypatch):
    monkeypatch.setattr("asyncio.sleep", _no_sleep)
    calls = {"n": 0}

    def handler(req):
        calls["n"] += 1
        return httpx.Response(200, text="garbage")

    with pytest.raises(ProviderUnavailable) as ei:
        await _call(_provider_with(handler, stream=True))
    assert calls["n"] == _MAX_ATTEMPTS
    assert all(a.status == "bad_json" for a in ei.value.attempts)


async def test_non_stream_mode_sends_no_stream_flags():
    captured = {}

    def handler(req):
        captured["body"] = json.loads(req.content)
        return _ok_response()

    await _call(_provider_with(handler))
    assert "stream" not in captured["body"] and "stream_options" not in captured["body"]


async def test_make_provider_threads_stream_from_cfg():
    captured = {}

    def handler(req):
        captured["body"] = json.loads(req.content)
        return _sse_response([{"content": "x"}])

    cfg = ProviderCfg(base_url="http://x/v1", model="m", stream=True)
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    c = await _call(make_provider(cfg, client=client))
    assert captured["body"]["stream"] is True and c.text == "x"
