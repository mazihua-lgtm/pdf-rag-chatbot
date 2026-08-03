"""rag.qa · RetrievalQA 链 + citation 后处理.

设计:
- 用 LangChain RetrievalQA(基于 StuffDocumentsChain)
- LLM: Anthropic Claude (ChatAnthropic)
- 强制返回 source_documents 配合 rag.vector_store.format_citation
- 当模型没输出合法 [来源: ...] 标注时, 后处理兜底补一个
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import Any, List, Optional

from langchain.chains import RetrievalQA
from langchain.chat_models import ChatAnthropic
from langchain.prompts import ChatPromptTemplate
from langchain.schema import Document
from langchain.vectorstores import Chroma

from rag.prompts import QA_PROMPT_MESSAGES
from rag.vector_store import format_citation, similarity_search_with_citations

DEFAULT_CLAUDE_MODEL = os.getenv("CLAUDE_MODEL", "claude-3-5-sonnet-latest")
DEFAULT_RETRIEVAL_K = int(os.getenv("RETRIEVAL_K", "4"))

# 兜底用的 citation 标记正则
_CITATION_RE = re.compile(r"\[来源:\s*[^\]]+\]")


@dataclass
class QAResult:
    """单次问答的结构化返回."""

    question: str
    answer: str
    citations: List[str] = field(default_factory=list)
    source_documents: List[Document] = field(default_factory=list)

    def __str__(self) -> str:
        s = self.answer.rstrip()
        if self.citations:
            s += "\n\n" + "\n".join(f"[来源: {c}]" for c in self.citations)
        return s


def _get_anthropic_key() -> str:
    key = os.getenv("ANTHROPIC_API_KEY")
    if not key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY 未设置, 请在 .env 中填写或运行:\n"
            "  export ANTHROPIC_API_KEY=sk-ant-xxxxx"
        )
    return key


def build_llm(
    model: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: int = 2048,
    timeout: int = 60,
) -> ChatAnthropic:
    """构造 ChatAnthropic 客户端.

    注意: 即使 model 不传, 也会校验 ANTHROPIC_API_KEY.
    """
    return ChatAnthropic(
        anthropic_api_key=_get_anthropic_key(),
        model=model or DEFAULT_CLAUDE_MODEL,
        temperature=temperature,
        max_tokens_to_sample=max_tokens,
        timeout=timeout,
    )


def build_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(QA_PROMPT_MESSAGES)


def build_qa_chain(
    store: Chroma,
    k: int = DEFAULT_RETRIEVAL_K,
    model: Optional[str] = None,
    temperature: float = 0.0,
) -> RetrievalQA:
    """构造一个 RetrievalQA.

    return_source_documents=True 保证返回 source_documents 以便我们做 citation 兜底.
    """
    llm = build_llm(model=model, temperature=temperature)
    prompt = build_prompt()
    retriever = store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k},
    )
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
    )


def _extract_citations(answer: str) -> List[str]:
    """从答案末尾解析出所有 [来源: xxx] 标注."""
    found = _CITATION_RE.findall(answer)
    cleaned: List[str] = []
    for line in found:
        # 去掉 "[来源:" 和末尾 "]"
        body = line.split(":", 1)[1].rstrip("]").strip()
        # 多来源用 "; " 分隔
        for piece in body.split(";"):
            p = piece.strip()
            if p and p not in cleaned:
                cleaned.append(p)
    return cleaned


def _append_fallback_citation(answer: str, sources: List[Document]) -> str:
    """如果模型没给 [来源: ...] 标注, 用我们自己的 format_citation 兜底."""
    answer = answer.rstrip()
    if _CITATION_RE.search(answer):
        return answer
    if not sources:
        return answer + "\n\n[来源: 未找到相关段落]"
    cites: List[str] = []
    for d in sources[:3]:  # 最多列 3 个来源
        c = format_citation(d)
        if c not in cites:
            cites.append(c)
    return answer + "\n\n" + "\n".join(f"[来源: {c}]" for c in cites)


def ask(
    store: Chroma,
    question: str,
    k: int = DEFAULT_RETRIEVAL_K,
    model: Optional[str] = None,
    temperature: float = 0.0,
) -> QAResult:
    """对向量库问一条问题, 返回结构化结果.

    Args:
        store: Chroma 实例(已 ingest 过)
        question: 用户中文问题
        k: 召回段落数
        model: Claude 模型名, 默认从环境读
        temperature: 0 表示最稳

    Returns:
        QAResult(answer + citations + source_documents)
    """
    if not question or not question.strip():
        raise ValueError("question 不能为空")

    chain = build_qa_chain(store, k=k, model=model, temperature=temperature)
    out: dict = chain.invoke({"query": question})
    raw_answer = out.get("result", "") or ""
    source_docs: List[Document] = out.get("source_documents", []) or []

    # 后处理: 兜底补 citation
    final_answer = _append_fallback_citation(raw_answer, source_docs)
    citations = _extract_citations(final_answer)

    return QAResult(
        question=question,
        answer=raw_answer.strip(),
        citations=citations,
        source_documents=source_docs,
    )


def ask_with_scores(
    store: Chroma,
    question: str,
    k: int = DEFAULT_RETRIEVAL_K,
    score_threshold: Optional[float] = None,
) -> List[tuple]:
    """不做 LLM 生成, 只检索并返回 (Document, score) 列表. 用于 debug."""
    return similarity_search_with_citations(
        store=store,
        query=question,
        k=k,
        score_threshold=score_threshold,
    )


__all__ = [
    "QAResult",
    "build_qa_chain",
    "build_llm",
    "build_prompt",
    "ask",
    "ask_with_scores",
    "DEFAULT_CLAUDE_MODEL",
    "DEFAULT_RETRIEVAL_K",
]
