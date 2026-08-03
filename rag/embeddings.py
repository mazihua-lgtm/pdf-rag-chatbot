"""rag.embeddings · 嵌入模型封装.

策略:
- 默认: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
  中文友好、零 API 成本(首次下载 ~120MB)
- 备选 A: Voyage AI (Anthropic 官方推荐, 需要 VOYAGE_API_KEY)
- 备选 B: OpenAI text-embedding-3-small (需要 OPENAI_API_KEY)

切换方式: 设置环境变量 EMBEDDING_BACKEND=local|voyage|openai
"""

from __future__ import annotations

import os
from typing import Any, List, Optional

# LangChain embeddings — 选择性导入以避免未安装时整体炸掉
try:
    from langchain.embeddings.base import Embeddings  # type: ignore
except ImportError:  # pragma: no cover
    Embeddings = Any  # type: ignore[misc,assignment]

# 本地 sentence-transformers 适配器 — 默认安装
try:
    from langchain.embeddings import HuggingFaceEmbeddings  # type: ignore

    _HAS_HF = True
except ImportError:  # pragma: no cover
    _HAS_HF = False

# Voyage / OpenAI 适配器 — 可选依赖
try:
    from langchain.embeddings.voyage import VoyageEmbeddings  # type: ignore

    _HAS_VOYAGE = True
except ImportError:  # pragma: no cover
    _HAS_VOYAGE = False

try:
    from langchain.embeddings.openai import OpenAIEmbeddings  # type: ignore

    _HAS_OPENAI = True
except ImportError:  # pragma: no cover
    _HAS_OPENAI = False


DEFAULT_LOCAL_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
DEFAULT_VOYAGE_MODEL = "voyage-3"
DEFAULT_OPENAI_MODEL = "text-embedding-3-small"


def get_embedding_backend() -> str:
    """读取 EMBEDDING_BACKEND 环境变量, 默认 local.

    可选: local | voyage | openai
    """
    backend = os.getenv("EMBEDDING_BACKEND", "local").strip().lower()
    if backend not in {"local", "voyage", "openai"}:
        raise ValueError(f"未知 EMBEDDING_BACKEND: {backend}, 允许: local/voyage/openai")
    return backend


def get_embeddings(
    backend: Optional[str] = None,
    model_name: Optional[str] = None,
    cache_folder: Optional[str] = None,
) -> "Embeddings":
    """根据配置返回对应的 Embeddings 实例.

    Args:
        backend: "local" | "voyage" | "openai", 不传则读 env
        model_name: 模型名, 不传则用各后端的默认模型
        cache_folder: 本地模型缓存目录(默认 ~/.cache/torch/sentence_transformers)

    Returns:
        LangChain Embeddings 实例, 可直接用于 Chroma / FAISS / ...

    Raises:
        RuntimeError: 指定 backend 的依赖未安装或 key 缺失
    """
    chosen = (backend or get_embedding_backend()).lower()

    if chosen == "local":
        if not _HAS_HF:
            raise RuntimeError(
                "本地嵌入需要 sentence-transformers: `pip install sentence-transformers`"
            )
        return HuggingFaceEmbeddings(  # type: ignore[call-arg]
            model_name=model_name or DEFAULT_LOCAL_MODEL,
            cache_folder=cache_folder,
            model_kwargs={"device": os.getenv("EMBEDDING_DEVICE", "cpu")},
            encode_kwargs={"normalize_embeddings": True, "batch_size": 32},
        )

    if chosen == "voyage":
        if not _HAS_VOYAGE:
            raise RuntimeError("Voyage 嵌入需要: `pip install voyageai`")
        api_key = os.getenv("VOYAGE_API_KEY")
        if not api_key:
            raise RuntimeError("VOYAGE_API_KEY 未设置, 请在 .env 中填写")
        return VoyageEmbeddings(  # type: ignore[call-arg]
            voyage_api_key=api_key,
            model=model_name or DEFAULT_VOYAGE_MODEL,
        )

    if chosen == "openai":
        if not _HAS_OPENAI:
            raise RuntimeError("OpenAI 嵌入需要: `pip install openai tiktoken`")
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY 未设置, 请在 .env 中填写")
        return OpenAIEmbeddings(  # type: ignore[call-arg]
            openai_api_key=api_key,
            model=model_name or DEFAULT_OPENAI_MODEL,
        )

    raise ValueError(f"未知 backend: {chosen}")


def embed_query(text: str, backend: Optional[str] = None) -> List[float]:
    """便捷函数: 单文本嵌入."""
    emb = get_embeddings(backend=backend)
    return emb.embed_query(text)


def embed_documents(texts: List[str], backend: Optional[str] = None) -> List[List[float]]:
    """便捷函数: 批量文档嵌入."""
    emb = get_embeddings(backend=backend)
    return emb.embed_documents(texts)


__all__ = [
    "get_embeddings",
    "get_embedding_backend",
    "embed_query",
    "embed_documents",
    "DEFAULT_LOCAL_MODEL",
    "DEFAULT_VOYAGE_MODEL",
    "DEFAULT_OPENAI_MODEL",
]


# === 自检 ===
# 运行 `python rag/embeddings.py` 触发快速健康检查(不联网、不调 API)
if __name__ == "__main__":  # pragma: no cover
    print("Embeddings module self-check")
    print(f"  backend env:  {os.getenv('EMBEDDING_BACKEND', 'local (default)')}")
    print(f"  HF available: {_HAS_HF}")
    print(f"  Voyage available: {_HAS_VOYAGE}")
    print(f"  OpenAI available: {_HAS_OPENAI}")
    print(f"  default local model: {DEFAULT_LOCAL_MODEL}")
