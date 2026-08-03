"""rag.loader · PDF 加载与分块.

策略:
- 优先 pdfplumber（对中文表格更友好，能保留 layout）
- 后备 pypdf（无原生依赖，安装简单）
- 每页作为一个 page-level Document
- 再用 RecursiveCharacterTextSplitter 按段落切分
- metadata 保留 source / page / chunk_index，citation 后处理靠它
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

try:
    import pdfplumber  # type: ignore

    _HAS_PDFPLUMBER = True
except ImportError:  # pragma: no cover
    _HAS_PDFPLUMBER = False

try:
    from pypdf import PdfReader  # type: ignore

    _HAS_PYPDF = True
except ImportError:  # pragma: no cover
    _HAS_PYPDF = False


# 默认分块参数 — 按中文段落切，800 字一段、相邻 100 字重叠
DEFAULT_CHUNK_SIZE = 800
DEFAULT_CHUNK_OVERLAP = 100

# 中文段落分隔符优先级（Recursive splitter 用）
_CN_SEPARATORS = ["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""]


@dataclass
class LoaderConfig:
    """加载器配置.

    Attributes:
        chunk_size: 单块最大字符数
        chunk_overlap: 相邻块重叠字符数
        min_chars: 短于该长度的段落会被丢弃（避免空白碎片）
        backend: 强制指定后端, "pdfplumber" / "pypdf" / "auto"
    """

    chunk_size: int = DEFAULT_CHUNK_SIZE
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP
    min_chars: int = 20
    backend: str = "auto"


def _clean_text(text: str) -> str:
    """去掉控制字符 + 合并多余空白(中文常见全角空格、NBSP)."""
    if not text:
        return ""
    # 去掉控制字符但保留中文标点
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text)
    # 全角空格 / NBSP → 普通空格
    text = text.replace("\u3000", " ").replace("\xa0", " ")
    # 合并连续空白
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _load_with_pdfplumber(path: str) -> List[Document]:
    """使用 pdfplumber 按页加载, 返回 page-level Document 列表."""
    docs: List[Document] = []
    with pdfplumber.open(path) as pdf:  # type: ignore[name-defined]
        for page_idx, page in enumerate(pdf.pages, start=1):
            # extract_text 失败(扫描页)→ 跳过该页
            text = page.extract_text() or ""
            text = _clean_text(text)
            if not text:
                continue
            docs.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": os.path.basename(path),
                        "path": path,
                        "page": page_idx,
                        "total_pages": len(pdf.pages),
                        "backend": "pdfplumber",
                    },
                )
            )
    return docs


def _load_with_pypdf(path: str) -> List[Document]:
    """使用 pypdf 后备加载."""
    docs: List[Document] = []
    reader = PdfReader(path)
    total = len(reader.pages)
    for idx, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception:  # pragma: no cover
            text = ""
        text = _clean_text(text)
        if not text:
            continue
        docs.append(
            Document(
                page_content=text,
                metadata={
                    "source": os.path.basename(path),
                    "path": path,
                    "page": idx,
                    "total_pages": total,
                    "backend": "pypdf",
                },
            )
        )
    return docs


def _select_backend(backend: str) -> str:
    if backend == "auto":
        if _HAS_PDFPLUMBER:
            return "pdfplumber"
        if _HAS_PYPDF:
            return "pypdf"
        raise RuntimeError(
            "pdfplumber / pypdf 均未安装, 请运行 `pip install pdfplumber` 或 `pip install pypdf`."
        )
    if backend == "pdfplumber" and not _HAS_PDFPLUMBER:
        raise RuntimeError("指定使用 pdfplumber 但未安装: `pip install pdfplumber`")
    if backend == "pypdf" and not _HAS_PYPDF:
        raise RuntimeError("指定使用 pypdf 但未安装: `pip install pypdf`")
    return backend


def _load_pdf_pages(path: str, backend: str = "auto") -> List[Document]:
    """加载 PDF 为 page-level Document 列表."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"PDF 文件不存在: {path}")
    if p.suffix.lower() != ".pdf":
        raise ValueError(f"仅支持 .pdf 文件, 收到: {p.suffix}")

    chosen = _select_backend(backend)
    if chosen == "pdfplumber":
        return _load_with_pdfplumber(path)
    return _load_with_pypdf(path)


def split_documents(
    docs: List[Document],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> List[Document]:
    """把 page-level Document 切成 chunk-level Document.

    保留并补充 metadata:
    - chunk_index: 当前页内的块序号
    """
    if not docs:
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=_CN_SEPARATORS,
        length_function=len,
        is_separator_regex=False,
    )

    out: List[Document] = []
    for doc in docs:
        chunks = splitter.split_text(doc.page_content)
        for c_idx, chunk in enumerate(chunks):
            chunk = _clean_text(chunk)
            if len(chunk) < 20:  # 过滤太碎的块
                continue
            md = dict(doc.metadata)
            md["chunk_index"] = c_idx
            out.append(Document(page_content=chunk, metadata=md))
    return out


def load_pdf(
    path: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    backend: str = "auto",
) -> List[Document]:
    """一站式加载 + 分块.

    Args:
        path: PDF 文件路径
        chunk_size: 单块最大字符数(默认 800)
        chunk_overlap: 相邻块重叠字符数(默认 100)
        backend: "auto" | "pdfplumber" | "pypdf"

    Returns:
        List[Document]: 已分块的 Document 列表, 每条带 source / page / chunk_index 元数据

    Raises:
        FileNotFoundError: 文件不存在
        ValueError: 非 PDF 文件
        RuntimeError: 后端缺失
    """
    pages = _load_pdf_pages(path, backend=backend)
    if not pages:
        return []
    return split_documents(pages, chunk_size=chunk_size, chunk_overlap=chunk_overlap)


def batch_load_pdfs(
    paths: Iterable[str],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    backend: str = "auto",
    on_error: str = "skip",
) -> List[Document]:
    """批量入库多个 PDF.

    Args:
        paths: PDF 路径迭代器
        on_error: "skip"(跳过坏文件) | "raise"(出错即中止)
    """
    all_docs: List[Document] = []
    for p in paths:
        try:
            all_docs.extend(
                load_pdf(p, chunk_size=chunk_size, chunk_overlap=chunk_overlap, backend=backend)
            )
        except Exception as e:
            if on_error == "raise":
                raise
            print(f"[loader] 跳过 {p}: {e}")
    return all_docs


__all__ = [
    "LoaderConfig",
    "load_pdf",
    "batch_load_pdfs",
    "split_documents",
    "DEFAULT_CHUNK_SIZE",
    "DEFAULT_CHUNK_OVERLAP",
]
