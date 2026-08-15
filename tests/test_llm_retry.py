"""
LLM Client 重试测试：瞬时错误（网络断连 / 5xx）自动重试，4xx 不重试。
"""
import asyncio

import httpx
import pytest

from app.config import Settings
from app.errors import LLMError
from app.llm.client import OpenAICompatClient, _parse_args


def _ok_response(content="ok"):
    return httpx.Response(200, json={
        "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        "model": "test",
    })


class _FlakyTransport(httpx.AsyncBaseTransport):
    """前 N 次请求失败（模拟网络断连），之后成功。"""

    def __init__(self, fail_count=1, fail_with=None):
        self.fail_count = fail_count
        self.calls = 0
        self.fail_with = fail_with or httpx.ConnectError("connection refused")

    async def handle_async_request(self, request):
        self.calls += 1
        if self.calls <= self.fail_count:
            raise self.fail_with
        return _ok_response()


def _make_client(transport, settings_kwargs=None):
    settings = Settings(
        llm_base_url="https://test.example.com/v1",
        llm_api_key="test-key",
        llm_model="test-model",
        **(settings_kwargs or {}),
    )
    client = OpenAICompatClient(settings)
    client._client = httpx.AsyncClient(transport=transport)
    return client


@pytest.mark.asyncio
async def test_retry_on_network_error():
    """网络断连 1 次后成功：重试生效。"""
    transport = _FlakyTransport(fail_count=1)
    client = _make_client(transport, {"llm_max_retries": 2})
    resp = await client.chat([{"role": "user", "content": "hi"}])
    assert resp.content == "ok"
    assert transport.calls == 2  # 1 次失败 + 1 次成功


@pytest.mark.asyncio
async def test_retry_on_5xx():
    """5xx 重试。"""
    class _FiveHundred(httpx.AsyncBaseTransport):
        calls = 0

        async def handle_async_request(self, request):
            self.calls += 1
            if self.calls == 1:
                return httpx.Response(500, text="upstream error")
            return _ok_response("recovered")

    transport = _FiveHundred()
    client = _make_client(transport, {"llm_max_retries": 2})
    resp = await client.chat([{"role": "user", "content": "hi"}])
    assert resp.content == "recovered"
    assert transport.calls == 2


@pytest.mark.asyncio
async def test_no_retry_on_400():
    """4xx 不重试（参数/鉴权问题重试无意义）。"""
    class _FourHundred(httpx.AsyncBaseTransport):
        calls = 0

        async def handle_async_request(self, request):
            self.calls += 1
            return httpx.Response(400, text="bad request")

    transport = _FourHundred()
    client = _make_client(transport, {"llm_max_retries": 3})
    with pytest.raises(LLMError) as exc:
        await client.chat([{"role": "user", "content": "hi"}])
    assert "400" in str(exc.value)
    assert transport.calls == 1  # 只试一次


@pytest.mark.asyncio
async def test_retry_exhausted_raises():
    """重试耗尽后抛 LLMError。"""
    transport = _FlakyTransport(fail_count=5)  # 一直失败
    client = _make_client(transport, {"llm_max_retries": 2})
    with pytest.raises(LLMError):
        await client.chat([{"role": "user", "content": "hi"}])
    assert transport.calls == 3  # 1 + 2 次重试


def test_parse_args():
    assert _parse_args({"function": {"arguments": '{"a": 1}'}}) == {"a": 1}
    assert _parse_args({"function": {"arguments": "not-json"}}) == {}
    assert _parse_args({"function": {}}) == {}
    assert _parse_args({}) == {}
