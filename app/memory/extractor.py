"""
回合结束后的记忆提炼（Fact Extractor）。

为什么不能直接存原始消息？
    - 原始消息噪音大（"你好"、"谢谢"这类寒暄）；
    - token 浪费：把整段对话塞进向量库，检索时也容易命中无关内容。
    正确做法：提炼成"事实句"（如"用户上次查询了北京天气"），
    只存信息密度高的内容。

提炼分类（memory_type）：
    fact        具体事实（"用户查询了北京天气"）→ 默认写入会话级记忆
    preference  用户偏好（"用户关注 RAG 应用开发"）→ 写入全局记忆（跨会话复用）
    conclusion  任务结论（"调研表明动态图通信适合 MARL"）→ 写入全局记忆

策略（与 context_builder 的 summary 策略同构）：
    stub —— 确定性规则提炼（不调用模型，测试 / 离线默认）
    llm  —— 调用 LLM 结构化提炼（需要真实模型）：输出 JSON
            {"facts": [...], "preferences": [...], "conclusions": [...]}
    off  —— 关闭自动提炼（build 前由上层拦截）

LLM 输出防御：
    1. 要求严格 JSON，容忍 ```json 代码块包裹 / 前后废话；
    2. 解析失败 → 降级 stub 规则提炼，绝不抛异常；
    3. 每类最多 5 条，截断超长。
"""
import json
import re

from app.config import Settings
from app.llm.client import BaseLLMClient

# 每类提炼条数上限
MAX_PER_TYPE = 5
# 单条事实长度上限
MAX_FACT_LEN = 160

_LLM_EXTRACT_PROMPT = """请从下面的 Agent 对话中提炼值得长期记住的信息，按三类输出严格 JSON（不要 Markdown 代码块、不要多余文字）：

{{"facts": ["具体事实，如「用户查询了北京天气」"],
 "preferences": ["用户偏好/习惯/身份信息，如「用户关注 RAG 应用开发」"],
 "conclusions": ["本次对话得出的结论，如「调研表明动态图通信适合 MARL」"]}}

要求：
1. 每类最多 {max_per_type} 条，每条一句话、不超过 {max_fact_len} 字；
2. 只输出 JSON 本身；没有某类内容则输出空数组；
3. 寒暄（你好/谢谢）不要提炼。

对话内容：
{messages}"""


class MemoryExtractor:
    """把一轮 Agent 回合的 messages 提炼为分类事实。"""

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
    async def extract(self, messages: list[dict]) -> list[dict]:
        """从一轮回合的完整 messages 提炼分类事实。

        返回：[{"text": str, "memory_type": "fact|preference|conclusion", "scope": "session|global"}]
        """
        if self.strategy == "off":
            return []
        if self.strategy == "llm" and self.llm is not None:
            return await self._llm_extract(messages)
        return self._stub_extract(messages)

    # ------------------------------------------------------------------
    # 确定性规则提炼（默认，离线可用）
    # ------------------------------------------------------------------
    def _stub_extract(self, messages: list[dict]) -> list[dict]:
        """规则提炼：抽取 用户提问（fact） + 工具结果（fact/conclusion）。"""
        items: list[dict] = []

        # 1) 用户最后一条有效提问（去掉寒暄）→ fact / 会话级
        for msg in reversed(messages):
            if msg.get("role") == "user":
                content = msg.get("content") or ""
                if isinstance(content, str) and len(content.strip()) > 4:
                    if not _is_chitchat(content):
                        items.append(
                            {
                                "text": f"用户询问：{content.strip()[:MAX_FACT_LEN]}",
                                "memory_type": "fact",
                                "scope": "session",
                            }
                        )
                break

        # 2) 工具执行结果（成功的结果值得记住）→ conclusion / 全局
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
                        items.append(
                            {
                                "text": f"工具 {name} 返回：{data[:MAX_FACT_LEN]}",
                                "memory_type": "conclusion",
                                "scope": "global",
                            }
                        )

        return _dedupe(items)

    # ------------------------------------------------------------------
    # LLM 结构化提炼（真实模型，JSON 分类输出）
    # ------------------------------------------------------------------
    async def _llm_extract(self, messages: list[dict]) -> list[dict]:
        """调用 LLM 提炼为 facts/preferences/conclusions 三类；失败降级 stub。"""
        prompt = _LLM_EXTRACT_PROMPT.format(
            max_per_type=MAX_PER_TYPE,
            max_fact_len=MAX_FACT_LEN,
            messages=json.dumps(messages, ensure_ascii=False),
        )
        try:
            response = await self.llm.chat([{"role": "user", "content": prompt}], tools=None)
        except Exception:
            return self._stub_extract(messages)

        parsed = _parse_classified_json(response.content or "")
        if parsed is None:
            return self._stub_extract(messages)

        items: list[dict] = []
        # 分类 → 类型 + 作用域（偏好/结论跨会话复用，事实默认会话级）
        for text in parsed.get("facts", [])[:MAX_PER_TYPE]:
            items.append({"text": _clean_fact(text), "memory_type": "fact", "scope": "session"})
        for text in parsed.get("preferences", [])[:MAX_PER_TYPE]:
            items.append({"text": _clean_fact(text), "memory_type": "preference", "scope": "global"})
        for text in parsed.get("conclusions", [])[:MAX_PER_TYPE]:
            items.append({"text": _clean_fact(text), "memory_type": "conclusion", "scope": "global"})
        return _dedupe([i for i in items if i["text"]])


def _parse_classified_json(raw: str) -> dict | None:
    """解析 LLM 返回的分类 JSON（容忍代码块 / 前后文字）。"""
    if not raw:
        return None
    text = raw.strip()
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if fence:
        text = fence.group(1).strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        obj = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    if not isinstance(obj, dict):
        return None
    # 容错：只保留三类数组字段
    return {k: v for k, v in obj.items() if k in ("facts", "preferences", "conclusions") and isinstance(v, list)}


def _clean_fact(text: str) -> str:
    """清洗单条事实：去空白、截断、跳过空/寒暄。"""
    t = text.strip().strip('"').strip("'")
    if len(t) <= 2 or _is_chitchat(t):
        return ""
    return t[:MAX_FACT_LEN]


def _dedupe(items: list[dict]) -> list[dict]:
    """按文本去重（同文本只保留一次）。"""
    seen: set[str] = set()
    out: list[dict] = []
    for item in items:
        key = item["text"]
        if key not in seen:
            seen.add(key)
            out.append(item)
    return out


_CHITCHAT = {
    "你好", "你是谁", "再见", "谢谢", "介绍一下你自己", "在吗", "hello", "hi", "你好呀",
}


def _is_chitchat(text: str) -> bool:
    t = text.strip().lower()
    return t in _CHITCHAT or len(t) <= 2
