"""tests/test_basic.py · 离线基础测试.

不调用任何 API, 覆盖:
- prompt 模板完整性
- citation 正则解析
- loader 分块元数据
- vector_store citation 格式化
"""

from __future__ import annotations

import sys
from pathlib import Path

# 让 pytest 在仓库根目录运行也能 import rag.*
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from langchain.schema import Document  # noqa: E402

from rag.loader import load_pdf, split_documents  # noqa: E402
from rag.prompts import QA_PROMPT_MESSAGES, QA_SYSTEM_PROMPT, QA_USER_TEMPLATE  # noqa: E402
from rag.qa import _extract_citations  # noqa: E402
from rag.vector_store import format_citation  # noqa: E402


# === prompts ===

def test_prompts_have_required_keywords():
    assert "来源" in QA_SYSTEM_PROMPT
    assert "{context}" in QA_USER_TEMPLATE
    assert "{question}" in QA_USER_TEMPLATE
    assert len(QA_PROMPT_MESSAGES) == 2
    roles = [m[0] for m in QA_PROMPT_MESSAGES]
    assert roles == ["system", "human"]


# === citation 解析 ===

def test_extract_citations_single():
    ans = "答案正文。\n\n[来源: a.pdf, P.3, 段落 0]"
    out = _extract_citations(ans)
    assert out == ["a.pdf, P.3, 段落 0"]


def test_extract_citations_multiple():
    ans = "正文\n\n[来源: a.pdf, P.3, 段落 0; b.pdf, P.5, 段落 1]"
    out = _extract_citations(ans)
    assert out == ["a.pdf, P.3, 段落 0", "b.pdf, P.5, 段落 1"]


def test_extract_citations_not_found():
    out = _extract_citations("纯答案, 没有来源标注。")
    assert out == []


# === citation 格式化 ===

def test_format_citation_full():
    doc = Document(
        page_content="x",
        metadata={"source": "demo.pdf", "page": 7, "chunk_index": 2},
    )
    assert format_citation(doc) == "demo.pdf, P.7, 段落 2"


def test_format_citation_missing_fields():
    doc = Document(page_content="x", metadata={"source": "only.pdf"})
    assert format_citation(doc) == "only.pdf"


# === loader 分块 ===

def test_split_documents_metadata_preserved():
    page = Document(
        page_content="第一段。" + ("中文测试。" * 100) + "\n\n第二段。" + ("继续测试。" * 100),
        metadata={"source": "demo.pdf", "page": 1, "total_pages": 1, "backend": "pypdf"},
    )
    chunks = split_documents([page], chunk_size=400, chunk_overlap=50)
    assert len(chunks) >= 2
    for c in chunks:
        assert c.metadata.get("source") == "demo.pdf"
        assert c.metadata.get("page") == 1
        assert "chunk_index" in c.metadata
        assert len(c.page_content) >= 20


def test_split_documents_empty_input():
    assert split_documents([]) == []
