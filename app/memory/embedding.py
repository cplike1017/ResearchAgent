"""
统一 Embedding 客户端（核心设计，与 llm/client.py 完全同构）。

目标：
1. 记忆层不绑定任何厂商 SDK，只面向一个极薄的抽象：
       async def embed(texts: list[str]) -> list[list[float]]
2. 默认支持 OpenAI-compatible /embeddings HTTP API（httpx 直连）。
3. 提供确定性 Stub 实现：没有 Key 时离线可跑（hash 伪向量），便于教学与 CI。
"""
import hashlib
import math
from abc import ABC, abstractmethod
from typing import Any

import httpx

from app.config import Settings
from app.errors import LLMError  # 复用统一错误模型


class BaseEmbeddingClient(ABC):
    """所有 Embedding 实现必须满足的最小接口。"""

    @abstractmethod
    async def embed(self, texts: list[str]) -> list[list[float]]:
        """把一批文本转为向量。返回 list[list[float]]，与 texts 一一对应。"""
        raise NotImplementedError

    @property
    @abstractmethod
    def dim(self) -> int:
        """向量维度（存储建表需要知道）。"""
        raise NotImplementedError


# ---------------------------------------------------------------------------
# 真实实现：OpenAI-compatible /embeddings
# ---------------------------------------------------------------------------
class OpenAICompatEmbeddingClient(BaseEmbeddingClient):
    """通过 httpx 直连 OpenAI-compatible /embeddings 接口。"""

    def __init__(self, settings: Settings):
        if not settings.embedding_base_url:
            raise LLMError("EMBEDDING_BASE_URL 未配置，无法创建 OpenAICompatEmbeddingClient")
        self.settings = settings
        self.base_url = settings.embedding_base_url.rstrip("/")
        self.api_key = settings.embedding_api_key
        self.model = settings.embedding_model
        self.timeout = settings.embedding_timeout_seconds
        self._dim = settings.embedding_dim
        self._client = httpx.AsyncClient(timeout=self.timeout)

    @property
    def dim(self) -> int:
        return self._dim

    async def aclose(self) -> None:
        await self._client.aclose()

    def _build_url(self) -> str:
        # 兼容：http://host/v1/embeddings 或 http://host/v1（自动补 /embeddings）
        if self.base_url.endswith("/embeddings"):
            return self.base_url
        if self.base_url.endswith("/v1"):
            return f"{self.base_url}/embeddings"
        return f"{self.base_url}/v1/embeddings"

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        payload: dict[str, Any] = {"model": self.model, "input": texts}
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        try:
            resp = await self._client.post(self._build_url(), json=payload, headers=headers)
        except httpx.HTTPError as exc:
            raise LLMError(f"Embedding 网络请求失败: {exc}") from exc

        if resp.status_code != 200:
            raise LLMError(f"Embedding 接口返回 {resp.status_code}: {resp.text[:300]}")

        try:
            data = resp.json()
            items = sorted(data["data"], key=lambda x: x.get("index", 0))
            vectors = [item["embedding"] for item in items]
        except (KeyError, IndexError, ValueError) as exc:
            raise LLMError(f"Embedding 响应格式非法: {exc}") from exc

        if vectors:
            self._dim = len(vectors[0])  # 以实际返回为准
        return vectors


# ---------------------------------------------------------------------------
# 确定性 Stub：无 Key 也能跑的教学实现
# ---------------------------------------------------------------------------
class StubEmbeddingClient(BaseEmbeddingClient):
    """
    确定性伪向量模型。

    用文本 hash 生成稳定向量：相同文本永远得到相同向量（可复现、可检索），
    相似文本在 hash 空间的相似度无语义意义 —— 教学与测试足够，
    真实语义检索必须配置 OpenAI-compatible embedding。
    """

    def __init__(self, settings: Settings | None = None):
        self.settings = settings or Settings()
        self._dim = self.settings.embedding_dim

    @property
    def dim(self) -> int:
        return self._dim

    def _hash_vec(self, text: str) -> list[float]:
        """把文本 hash 到单位向量（dim 维），确定性。"""
        vec = []
        for i in range(self._dim):
            h = hashlib.sha256(f"{text}:{i}".encode("utf-8")).digest()
            # 取 8 字节映射到 [-1, 1]
            val = int.from_bytes(h[:8], "big") / (2**64) * 2 - 1
            vec.append(val)
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    async def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._hash_vec(t) for t in texts]


# ---------------------------------------------------------------------------
# 工厂
# ---------------------------------------------------------------------------
def create_embedding_client(settings: Settings | None = None) -> BaseEmbeddingClient:
    """按配置创建 Embedding 客户端。"""
    settings = settings or Settings()
    provider = settings.embedding_provider_resolved
    if provider == "openai":
        return OpenAICompatEmbeddingClient(settings)
    return StubEmbeddingClient(settings)
