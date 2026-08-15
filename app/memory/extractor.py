"""
回合结束后的记忆提炼（Fact Extractor）。

为什么不能直接存原始消息？
    - 原始消息噪音大（"你好"、"谢谢"这类寒暄）；
    - token 浪费：把整段对话塞进向量库，检索时也容易命中无关内容。
    正确做法：提炼成"事实句"（如"用户上次查询了北京天气"），
    只存信息密度高的内容。

策略（与 context_builder 的 summary 策略同构）：
    stub —— 确定性规则提炼（不调用模型，测试 / 离线默认）
    llm  —— 调用 LLM 提炼（需要真实模型）
    off  —— 关闭自动提炼（build 前由上层拦截）
"""
import json

from app.config import Settings
from app.llm.client import BaseLLMClient


class MemoryExtractor:
    """把一轮 Agent 回合的 messages 提炼为事实句列表。"""

    def __init__(
        self,
        settings: Settings | None = None,
        llm: BaseLLMClient | None = None,
    ) -> None:
        self.settings = settings or Settings()
        self.llm = llm
        self.strategy = self.settings.memory_extract_strategy

    # ------------------------------------------------------------------
    # 主入口（async：llm 策略需要 await）
    # ------------------------------------------------------------------
    async def extract(self, messages: list[dict]) -> list[str]:
        """从一轮回合的完整 messages 提炼事实句。"""
        if self.strategy == "off":
            return []
        if self.strategy == "llm" and self.llm is not None:
            return await self._llm_extract(messages)
        return self._stub_extract(messages)

    # ------------------------------------------------------------------
    # 确定性规则提炼（默认，离线可用）
    # ------------------------------------------------------------------
    def _stub_extract(self, messages: list[dict]) -> list[str]:
        """规则提炼：抽取 用户提问 + 工具结果 中的关键信息。"""
        facts: list[str] = []

        # 1) 用户最后一条有效提问（去掉寒暄）
        for msg in reversed(messages):
            if msg.get("role") == "user":
                content = msg.get("content") or ""
                if isinstance(content, str) and len(content.strip()) > 4:
                    if not _is_chitchat(content):
                        facts.append(f"用户询问：{content.strip()[:80]}")
                break

        # 2) 工具执行结果（成功的结果值得记住）
        for msg in messages:
            if msg.get("role") == "tool":
                try:
                    envelope = json.loads(msg.get("content") or "{}")
                except (TypeError, json.JSONDecodeError):
                    continue
                if envelope.get("success"):
                    name = msg.get("name", "")
                    data = envelope.get("data")
                    if isinstance(data, str):
                        facts.append(f"工具 {name} 返回：{data[:80]}")

        # 去重（同一文本只保留一次）
        seen: set[str] = set()
        out: list[str] = []
        for f in facts:
            if f not in seen:
                seen.add(f)
                out.append(f)
        return out

    # ------------------------------------------------------------------
    # LLM 提炼（真实模型）
    # ------------------------------------------------------------------
    async def _llm_extract(self, messages: list[dict]) -> list[str]:
        """调用 LLM 提炼事实（教学演示；生产可用更专业的提炼 Prompt）。"""
        prompt = (
            "请从下面这段 Agent 对话中提炼出值得长期记住的事实，"
            "每条一行，最多 5 条。只输出事实本身，不要编号。\n"
            + json.dumps(messages, ensure_ascii=False)
        )
        response = await self.llm.chat([{"role": "user", "content": prompt}], tools=None)
        lines = [ln.strip() for ln in (response.content or "").splitlines() if ln.strip()]
        return lines[:5]


_CHITCHAT = {
    "你好", "你是谁", "再见", "谢谢", "介绍一下你自己", "在吗", "hello", "hi", "你好呀",
}


def _is_chitchat(text: str) -> bool:
    t = text.strip().lower()
    return t in _CHITCHAT or len(t) <= 2
