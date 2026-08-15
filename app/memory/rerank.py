"""
记忆重排序器（Reranker，Stage 8 Level 2）。

为什么需要重排？—— 朴素 Top-K 只信"向量相似度"，但向量相似度有盲区：
    - 语义相似但字面差异大（"京城的天气" vs "北京天气"）可能漏召；
    - 检索出来的候选里，靠前的未必是最有用的。
    重排用**更细的评分**把候选重新排一遍，让"精排"决定最终注入 Top-K。

策略（与项目三策略同构）：
    stub —— 规则重排：关键词重叠 + 语义分数 加权（确定性，离线默认）
    llm  —— LLM 重排：让模型对候选打分排序（真实重排，需真实 LLM）
    off  —— 跳过重排（等价朴素 Top-K，测试/低配场景）

流程：
    粗召回（Top-K × candidate_multiplier）→ Reranker.rerank() → 取最终 Top-K
"""
import json
import re
from typing import Any

from app.config import Settings
from app.llm.client import BaseLLMClient
from app.memory.models import MemoryRecord


# ---------------------------------------------------------------------------
# 基础：中文/英文 token 化（用于关键词重叠）
# ---------------------------------------------------------------------------
def _tokenize(text: str) -> set[str]:
    """简单 token 化：中文按单字/双字，英文按单词，兼顾关键词重叠计算。"""
    text = text.lower()
    tokens: set[str] = set()
    # 英文单词
    for w in re.findall(r"[a-z0-9_]+", text):
        if len(w) >= 2:
            tokens.add(w)
    # 中文：连续汉字切 2-gram（"北京天气" -> {"北京","京天","天气"}）
    cjk = re.findall(r"[\u4e00-\u9fff]+", text)
    for chunk in cjk:
        if len(chunk) == 1:
            tokens.add(chunk)
        for i in range(len(chunk) - 1):
            tokens.add(chunk[i : i + 2])
    return tokens


class BaseReranker:
    """重排器抽象。"""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or Settings()

    async def rerank(self, query: str, candidates: list[MemoryRecord]) -> list[MemoryRecord]:
        """对候选重排（原地修改 score），返回按新分数降序的列表。"""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# stub：规则重排（默认，离线可用）
# ---------------------------------------------------------------------------
class RuleReranker(BaseReranker):
    """
    规则重排：final = semantic_score × (1 - kw_w) + keyword_overlap × kw_w

    semantic_score  —— 粗召回的相似度 × 时间衰减（repository.search 已算好）
    keyword_overlap —— 查询与候选文本的关键词重叠率 [0,1]
                      （纠正向量盲区：字面强相关但语义向量分不高的记忆）
    """

    def __init__(self, settings: Settings | None = None) -> None:
        super().__init__(settings)
        self.keyword_weight = self.settings.memory_rerank_keyword_weight

    @staticmethod
    def _overlap(query_tokens: set[str], text_tokens: set[str]) -> float:
        if not query_tokens:
            return 0.0
        # 重叠率 = 交集大小 / 查询 token 数（0~1）
        return len(query_tokens & text_tokens) / len(query_tokens)

    async def rerank(self, query: str, candidates: list[MemoryRecord]) -> list[MemoryRecord]:
        if not candidates:
            return []
        q_tokens = _tokenize(query)
        kw_w = self.keyword_weight
        for c in candidates:
            overlap = self._overlap(q_tokens, _tokenize(c.text))
            final = c.score * (1 - kw_w) + overlap * kw_w
            c.score = round(final, 4)
        candidates.sort(key=lambda r: r.score, reverse=True)
        return candidates


# ---------------------------------------------------------------------------
# llm：LLM 重排（真实重排，需真实 LLM）
# ---------------------------------------------------------------------------
class LLMReranker(BaseReranker):
    """
    LLM 重排：让模型从候选中挑出与查询最相关的 N 条并排序。

    教学演示实现：把候选编号后交给模型，解析模型输出的编号顺序。
    生产可用更稳的"逐条打分"或交叉编码器（如 bge-reranker）。
    """

    def __init__(self, settings: Settings | None = None, llm: BaseLLMClient | None = None) -> None:
        super().__init__(settings)
        self.llm = llm

    async def rerank(self, query: str, candidates: list[MemoryRecord]) -> list[MemoryRecord]:
        if not candidates:
            return []
        if self.llm is None:
            return candidates  # 无模型时降级为原顺序

        lines = "\n".join(f"[{i}] {c.text}" for i, c in enumerate(candidates))
        prompt = (
            "以下是按相似度预检索到的记忆片段。请根据与查询的相关程度重新排序，"
            "从最相关到最不相关，只输出编号序列（如 3,0,2,1），不要其他内容。\n\n"
            f"查询：{query}\n\n{lines}"
        )
        try:
            response = await self.llm.chat([{"role": "user", "content": prompt}], tools=None)
        except Exception:
            return candidates  # 重排失败降级原顺序，不阻塞主流程

        # 解析模型输出的编号序列
        raw = response.content or ""
        nums = re.findall(r"\d+", raw)
        order: list[int] = []
        for n in nums:
            i = int(n)
            if 0 <= i < len(candidates) and i not in order:
                order.append(i)

        if not order:
            return candidates
        # 未提及的候选排最后（保持原相对顺序）
        mentioned = set(order)
        rest = [i for i in range(len(candidates)) if i not in mentioned]
        ranked = [candidates[i] for i in order + rest]
        # 重排后分数仅作展示：降序编号（最相关分最高）
        for idx, c in enumerate(ranked):
            c.score = round(max(0.0, 1.0 - idx * 0.01), 4)
        return ranked


# ---------------------------------------------------------------------------
# 工厂
# ---------------------------------------------------------------------------
def create_reranker(
    settings: Settings | None = None, llm: BaseLLMClient | None = None
) -> BaseReranker | None:
    """按配置创建重排器；off 或未知策略返回 None（跳过重排）。"""
    settings = settings or Settings()
    strategy = settings.memory_rerank_strategy
    if strategy == "llm":
        return LLMReranker(settings, llm=llm)
    if strategy == "stub":
        return RuleReranker(settings)
    return None
