"""
统一 LLM Client（核心设计）。

目标：
1. 核心 Agent Runtime 不绑定任何具体厂商 SDK，只面向一个极薄的抽象：
       async def chat(messages, tools) -> LLMResponse
2. 默认优先支持 OpenAI-compatible HTTP API（通过 httpx 直接调用 /chat/completions）。
3. 提供确定性 Stub 实现：没有 API Key 时也能离线跑通全部 Demo / 测试 / 评测，
   便于教学与 CI，绝不伪造真实模型的行为。

LLMResponse 约定（对 Agent 循环而言只有两种结果）：
    - tool_calls 非空  -> 模型要求调用工具
    - tool_calls 为空  -> 模型给出最终回答（final answer）
"""
import asyncio
import json
import re
from abc import ABC, abstractmethod
from typing import Any

import httpx
from pydantic import BaseModel, Field

from app.config import Settings
from app.errors import LLMError


# ---------------------------------------------------------------------------
# 数据结构：模型响应的统一表示
# ---------------------------------------------------------------------------
class ToolCallRequest(BaseModel):
    """模型发起的一次工具调用请求。"""

    id: str = Field(description="工具调用唯一 ID（OpenAI 格式的 tool_call_id）")
    name: str = Field(description="工具名")
    arguments: dict[str, Any] = Field(default_factory=dict, description="已解析为 dict 的调用参数")


class LLMResponse(BaseModel):
    """统一 LLM 响应。"""

    content: str | None = None  # 模型生成的文本；如果是工具调用则通常为 None
    tool_calls: list[ToolCallRequest] = Field(default_factory=list)  # 本次决策发起的工具调用
    finish_reason: str = "stop"  # stop | tool_calls | length | ...
    # usage 用宽松 dict：部分网关返回嵌套对象（如 prompt_tokens_details），
    # 严格 dict[str, int] 会导致校验失败。读取处用 .get("prompt_tokens", 0) 等兜底。
    usage: dict = Field(default_factory=dict)
    model: str = ""

    @property
    def is_final_answer(self) -> bool:
        """是否最终回答（没有工具调用即为最终回答）。"""
        return not self.tool_calls


class BaseLLMClient(ABC):
    """所有 LLM 实现必须满足的最小接口。"""

    @abstractmethod
    async def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        """
        发送一轮对话。

        :param messages: OpenAI 风格消息列表 [{"role": "user|assistant|tool|system", "content": ...}]
        :param tools:    工具 JSON Schema 列表（OpenAI function calling 格式）
        :return:         LLMResponse
        """
        raise NotImplementedError


# ---------------------------------------------------------------------------
# 真实实现：OpenAI-compatible HTTP API
# ---------------------------------------------------------------------------
class OpenAICompatClient(BaseLLMClient):
    """通过 httpx 直连 OpenAI-compatible /chat/completions 接口。"""

    def __init__(self, settings: Settings):
        if not settings.llm_base_url:
            raise LLMError("LLM_BASE_URL 未配置，无法创建 OpenAICompatClient")
        self.settings = settings
        # 统一把 base url 规范为不含尾部斜杠的形式
        self.base_url = settings.llm_base_url.rstrip("/")
        self.api_key = settings.llm_api_key
        self.model = settings.llm_model
        self.timeout = settings.llm_timeout_seconds
        # 复用同一个异步连接池
        self._client = httpx.AsyncClient(timeout=self.timeout)

    async def aclose(self) -> None:
        """关闭底层连接池。"""
        await self._client.aclose()

    def _build_url(self) -> str:
        # 兼容两种 base url：http://host/v1 或 http://host（自动补 /v1 与 /chat/completions）
        if self.base_url.endswith("/chat/completions"):
            return self.base_url
        if self.base_url.endswith("/v1"):
            return f"{self.base_url}/chat/completions"
        return f"{self.base_url}/v1/chat/completions"

    async def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        payload: dict[str, Any] = {"model": self.model, "messages": messages}
        if tools:
            payload["tools"] = tools
            # OpenAI 要求 tool_choice 缺省为 auto，显式声明更清晰
            payload["tool_choice"] = kwargs.get("tool_choice", "auto")

        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        url = self._build_url()
        max_retries = self.settings.llm_max_retries
        backoff = self.settings.llm_retry_backoff

        last_exc: Exception | None = None
        for attempt in range(max_retries + 1):
            try:
                resp = await self._client.post(url, json=payload, headers=headers)
            except httpx.HTTPError as exc:
                # 网络层错误（断连/超时）：瞬时，重试
                last_exc = exc
                if attempt < max_retries:
                    await asyncio.sleep(backoff * (2**attempt))
                    continue
                raise LLMError(f"LLM 网络请求失败: {exc}") from exc

            if resp.status_code != 200:
                # 5xx / 429：瞬时，重试；4xx：不重试（参数/鉴权问题）
                if resp.status_code >= 500 or resp.status_code == 429:
                    last_exc = LLMError(f"LLM 接口返回 {resp.status_code}: {resp.text[:200]}")
                    if attempt < max_retries:
                        await asyncio.sleep(backoff * (2**attempt))
                        continue
                raise LLMError(f"LLM 接口返回 {resp.status_code}: {resp.text[:300]}")

            try:
                data = resp.json()
                choice = data["choices"][0]
                message = choice.get("message", {})
                usage = data.get("usage", {}) or {}
            except (KeyError, IndexError, ValueError) as exc:
                raise LLMError(f"LLM 响应格式非法: {exc}") from exc

            return LLMResponse(
                content=message.get("content"),
                tool_calls=[
                    ToolCallRequest(
                        id=raw.get("id", ""),
                        name=(raw.get("function") or {}).get("name", ""),
                        arguments=_parse_args(raw),
                    )
                    for raw in message.get("tool_calls") or []
                ],
                finish_reason=choice.get("finish_reason", "stop"),
                usage=usage,
                model=data.get("model", self.model),
            )

        raise LLMError(f"LLM 调用重试 {max_retries} 次后仍失败: {last_exc}") from last_exc


# ---------------------------------------------------------------------------
# 确定性 Stub：无 Key 也能跑的教学实现
# ---------------------------------------------------------------------------
class StubLLMClient(BaseLLMClient):
    """
    确定性规则假模型。

    设计意图：教学与 CI 中不依赖外部 API。它模拟"一个会正确理解本 Demo
    场景的模型"，规则全部硬编码，输出完全可预期：

    - 最后一条消息是 tool 结果  -> 依据工具结果生成最终回答（第二/三轮）
    - 用户消息包含 "计算 <表达式>" -> 调用 calculator
    - 用户消息包含 "<城市>天气"   -> 调用 get_weather（支持多城市）
    - 其他（如 "你好"）          -> 直接最终回答，不调用任何工具
    """

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or Settings()
        self.model = self.settings.llm_model

    # ---- 工具调用规则 ----
    _CALC_RE = re.compile(r"计算\s*([0-9+\-*/().%^ \t]+)")
    # 城市解析：去掉常见前缀/分隔词后取尾部城市名，支持"同时查询北京和上海天气"
    _CITY_PREFIXES = ("帮我查询", "查询一下", "查一下", "同时查询", "查询", "帮我查", "看看", "查查", "查")
    _CITY_SEPARATORS = re.compile(r"[和、与，,及\s]+")

    def _parse_cities(self, user_text: str) -> list[str]:
        """从用户文本中解析城市列表（最后一个"天气"之前的部分按分隔词拆分）。"""
        idx = user_text.rfind("天气")
        if idx == -1:
            return []
        head = user_text[:idx].rstrip("的")
        for prefix in self._CITY_PREFIXES:
            if head.startswith(prefix):
                head = head[len(prefix):]
                break
        cities: list[str] = []
        for part in self._CITY_SEPARATORS.split(head):
            m = re.search(r"[\u4e00-\u9fa5]+$", part)  # 取每段尾部的连续汉字
            if m:
                token = m.group(0)
                cities.append(token if len(token) <= 4 else token[-4:])
        return cities

    def _parse_last_user_message(self, messages: list[dict]) -> str:
        """取最后一条 user 消息的文本（跳过中间的 tool / assistant 消息）。"""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                content = msg.get("content") or ""
                return content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)
        return ""

    def _answer_from_tool_results(self, messages: list[dict]) -> str | None:
        """
        如果消息末尾连续若干条都是 tool 结果（其前一条是 assistant 工具调用），
        则依据这些结果生成最终回答。这是 ReAct 循环"工具结果重新进入消息"后的收尾。
        """
        tool_msgs = []
        for msg in reversed(messages):
            if msg.get("role") == "tool":
                tool_msgs.append(msg)
            else:
                break
        if not tool_msgs:
            return None
        tool_msgs.reverse()

        lines = []
        for tm in tool_msgs:
            name = tm.get("name", "")
            try:
                envelope = json.loads(tm.get("content") or "{}")
            except json.JSONDecodeError:
                envelope = {}
            if envelope.get("success"):
                data = envelope.get("data")
                if name == "calculator":
                    lines.append(f"计算结果：{data}")
                elif name == "get_weather":
                    # 城市从信封 metadata.args 中取（args 由执行器记录）
                    city = ((envelope.get("metadata") or {}).get("args") or {}).get("city", "")
                    lines.append(f"{city}天气：{data}")
                else:
                    lines.append(f"{name} 返回：{data}")
            else:
                err = envelope.get("error") or {}
                lines.append(f"工具 {name} 执行失败：{err.get('message', '未知错误')}")
        return "；".join(lines) + "。"

    def _estimate_usage(self, messages: list[dict], output_len: int) -> dict[str, int]:
        """按 4 字符≈1 token 的粗略估计，保证评测的 token 指标有数据可看。"""
        prompt_chars = sum(len(str(m.get("content") or "")) for m in messages)
        return {
            "prompt_tokens": max(1, prompt_chars // 4),
            "completion_tokens": max(1, output_len // 4),
            "total_tokens": max(2, (prompt_chars + output_len) // 4),
        }

    async def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        # 规则 1：末尾有工具结果 -> 生成最终回答（模拟模型看到 Tool Result 后收尾）
        answer = self._answer_from_tool_results(messages)
        if answer is not None:
            return LLMResponse(
                content=answer,
                finish_reason="stop",
                usage=self._estimate_usage(messages, len(answer)),
                model=self.model,
            )

        user_text = self._parse_last_user_message(messages)

        # 规则 2：计算器
        m = self._CALC_RE.search(user_text)
        if m:
            expr = m.group(1).strip()
            return LLMResponse(
                tool_calls=[ToolCallRequest(id="call_stub_calc", name="calculator", arguments={"expression": expr})],
                finish_reason="tool_calls",
                usage=self._estimate_usage(messages, len(expr)),
                model=self.model,
            )

        # 规则 3：天气（支持"同时查询北京和上海天气"这类多城市）
        cities = self._parse_cities(user_text)
        if cities:
            calls = [
                ToolCallRequest(id=f"call_stub_weather_{i}", name="get_weather", arguments={"city": c})
                for i, c in enumerate(cities)
            ]
            return LLMResponse(
                tool_calls=calls,
                finish_reason="tool_calls",
                usage=self._estimate_usage(messages, sum(len(c) for c in cities)),
                model=self.model,
            )

        # 规则 4：兜底最终回答（如 "你好"）
        fallback = f"你好！我收到了你的消息：「{user_text}」。这是一个离线 Stub 模型，配置真实 LLM_BASE_URL / LLM_API_KEY 后我会接入真实模型。"
        return LLMResponse(
            content=fallback,
            finish_reason="stop",
            usage=self._estimate_usage(messages, len(fallback)),
            model=self.model,
        )


# ---------------------------------------------------------------------------
# 工具参数解析（OpenAI 格式：arguments 是 JSON 字符串）
# ---------------------------------------------------------------------------
def _parse_args(raw: dict) -> dict:
    """解析 tool_call 的 arguments（JSON 字符串 -> dict）。

    参数解析失败不应让整个调用崩溃，记为空参数并交由 Gateway 校验。
    """
    fn = raw.get("function") or {}
    try:
        args = json.loads(fn.get("arguments") or "{}")
        return args if isinstance(args, dict) else {}
    except (json.JSONDecodeError, TypeError):
        return {}


# ---------------------------------------------------------------------------
# 工厂：根据配置创建客户端
# ---------------------------------------------------------------------------
def create_llm_client(settings: Settings | None = None) -> BaseLLMClient:
    """按配置创建 LLM 客户端。"""
    settings = settings or Settings()
    provider = settings.llm_provider_resolved
    if provider == "openai":
        return OpenAICompatClient(settings)
    return StubLLMClient(settings)
