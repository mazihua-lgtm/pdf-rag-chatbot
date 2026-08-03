"""rag.vector_store · Chroma 向量库封装.

特性:
- 本地持久化到 ./chroma_db/ (可改 CHROMA_PERSIST_DIR)
- 单一 collection 管理 (默认 pdf_rag_chinese)
- add_documents() 自动生成稳定 ID (避免重复入库)
- similarity_search 支持返回 source_documents 配合 citation 后处理
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any, List, Optional, Tuple

from langchain.schema import Document
from langchain.vectorstores import Chroma

from rag.embeddings import get_embeddings

DEFAULT_COLLECTION = "pdf_rag_chinese"


def _stable_id(doc: Document) -> str:
    """生成稳定 ID — 基于 source + page + chunk_index + content 前 200 字符.

    同一文件重复入库不会重复写入.
    """
    md = doc.metadata or {}
    key = f"{md.get('source','')}|{md.get('page','')}|{md.get('chunk_index','')}|{doc.page_content[:200]}"
    return hashlib.md5(key.encode("utf-8")).hexdigest()


def _get_persist_dir(persist_dir: Optional[str] = None) -> str:
    return persist_dir or os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")


def get_vector_store(
    persist_dir: Optional[str] = None,
    collection_name: Optional[str] = None,
    embeddings: Optional[Any] = None,
) -> Chroma:
    """拿到 / 创建一个持久化 Chroma.

    Args:
        persist_dir: 持久化目录, 默认 ./chroma_db 或 env CHROMA_PERSIST_DIR
        collection_name: collection 名, 默认 pdf_rag_chinese 或 env CHROMA_COLLECTION
        embeddings: 自定义 Embeddings 实例, 默认 get_embeddings()

    Returns:
        Chroma 客户端, 可直接 add_texts / similarity_search
    """
    if embeddings is None:
        embeddings = get_embeddings()

    pd = _get_persist_dir(persist_dir)
    # Chroma 在新版本里会自动创建目录
    Path(pd).mkdir(parents=True, exist_ok=True)

    cn = collection_name or os.getenv("CHROMA_COLLECTION", DEFAULT_COLLECTION)
    return Chroma(
        collection_name=cn,
        embedding_function=embeddings,
        persist_directory=pd,
    )


def add_documents(
    store: Chroma,
    docs: List[Document],
    ids: Optional[List[str]] = None,
) -> List[str]:
    """批量添加文档, 返回写入的 ID 列表.

    用 _stable_id 去重 — 同一文件重复 ingest 只入库一次.
    """
    if not docs:
        return []
    if ids is None:
        ids = [_stable_id(d) for d in docs]
    store.add_documents(documents=docs, ids=ids)
    # Chroma 0.4.x persist 已自动; 新版也保留兼容
    try:
        store.persist()  # type: ignore[attr-defined]
    except AttributeError:
        pass
    return ids


def similarity_search_with_citations(
    store: Chroma,
    query: str,
    k: int = 4,
    score_threshold: Optional[float] = None,
) -> List[Tuple[Document, float]]:
    """检索 + 返回 (Document, distance).

    Args:
        store: Chroma 实例
        query: 用户问题
        k: 召回段落数
        score_threshold: 距离阈值, 低于该值(更相似)的会被保留

    Returns:
        List[(Document, score)] — score 是 L2 距离, 越小越相关
    """
    kwargs: dict = {"k": k}
    # Chroma 的 relevance_score_fn 不一致, 这里走直接的距离返回
    raw = store.similarity_search_with_score(query, **kwargs)
    if score_threshold is None:
        return raw
    return [(d, s) for d, s in raw if s <= score_threshold]


def format_citation(doc: Document) -> str:
    """把 Document metadata 格式化为中文 citation 字符串.

    期望输出形如:
        招股书_2023.pdf, P.42, 段落 3
        未找到相关段落
    """
    md = doc.metadata or {}
    source = md.get("source") or "未知文档"
    page = md.get("page")
    chunk_idx = md.get("chunk_index")
    parts = [source]
    if page is not None:
        parts.append(f"P.{page}")
    if chunk_idx is not None:
        parts.append(f"段落 {chunk_idx}")
    return ", ".join(parts)


def count(store: Chroma) -> int:
    """返回 collection 内的文档数."""
    try:
        return store._collection.count()  # type: ignore[attr-defined]
    except Exception:  # pragma: no cover
        return -1


def reset(persist_dir: Optional[str] = None, collection_name: Optional[str] = None) -> None:
    """清空 collection(用于重新构建索引)."""
    store = get_vector_store(persist_dir=persist_dir, collection_name=collection_name)
    try:
        store.delete_collection()  # type: ignore[attr-defined]
    except Exception:  # pragma: no cover
        # 老版本没 delete_collection — 直接落库没用, 提示用户手动删 chroma_db/
        pass


__all__ = [
    "get_vector_store",
    "add_documents",
    "similarity_search_with_citations",
    "format_citation",
    "count",
    "reset",
    "DEFAULT_COLLECTION",
]
