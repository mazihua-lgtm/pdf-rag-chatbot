"""scripts/ingest.py · CLI 批量入库工具.

Usage:
  python scripts/ingest.py file1.pdf file2.pdf ...
  python scripts/ingest.py ./pdfs/*.pdf
  python scripts/ingest.py --reset  # 先清空向量库
  python scripts/ingest.py --reset path/to/a.pdf path/to/b.pdf
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
from pathlib import Path
from typing import List

# 让 `python scripts/ingest.py` 也能 import rag.*
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rag.loader import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE, load_pdf  # noqa: E402
from rag.vector_store import add_documents, count, get_vector_store, reset  # noqa: E402


def _expand_paths(items: List[str]) -> List[str]:
    """展开 glob, 过滤非 .pdf, 返回已存在文件的绝对路径列表."""
    out: List[str] = []
    for it in items:
        # 直接是文件
        if os.path.isfile(it):
            out.append(os.path.abspath(it))
            continue
        # 是 glob 模式
        matches = sorted(glob.glob(it))
        if matches:
            for m in matches:
                if os.path.isfile(m) and m.lower().endswith(".pdf"):
                    out.append(os.path.abspath(m))
            continue
        print(f"[ingest] 跳过 (不存在或无匹配): {it}")
    # 去重
    seen = set()
    dedup: List[str] = []
    for p in out:
        if p not in seen:
            seen.add(p)
            dedup.append(p)
    return dedup


def main() -> int:
    p = argparse.ArgumentParser(
        description="pdf-rag-chatbot · 批量入库 PDF 到本地 Chroma 向量库",
    )
    p.add_argument("paths", nargs="*", help="PDF 文件或 glob 模式, 如 ./pdfs/*.pdf")
    p.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE,
                   help=f"单块最大字符数 (默认 {DEFAULT_CHUNK_SIZE})")
    p.add_argument("--chunk-overlap", type=int, default=DEFAULT_CHUNK_OVERLAP,
                   help=f"相邻块重叠字符数 (默认 {DEFAULT_CHUNK_OVERLAP})")
    p.add_argument("--reset", action="store_true",
                   help="入库前先清空向量库 (慎用!)")
    p.add_argument("--quiet", action="store_true", help="减少日志输出")
    args = p.parse_args()

    if args.reset:
        print("[ingest] --reset 已指定, 即将清空向量库 ...")
        reset()
        print("[ingest] 已清空。")

    store = get_vector_store()
    before = count(store)
    print(f"[ingest] 当前向量库已有 {before} 段落")

    paths = _expand_paths(args.paths)
    if not paths:
        print("[ingest] 没有可处理的 PDF, 退出。")
        return 1

    total_chunks = 0
    for path in paths:
        if not args.quiet:
            print(f"[ingest] 处理 {os.path.basename(path)} ...")
        try:
            docs = load_pdf(path, chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
        except Exception as e:  # noqa: BLE001
            print(f"[ingest] ❌ {path}: {e}")
            continue
        if not docs:
            print(f"[ingest] ⚠️ {path}: 未提取到文本 (可能为扫描件)")
            continue
        ids = add_documents(store, docs)
        total_chunks += len(ids)
        if not args.quiet:
            print(f"[ingest] ✅ {os.path.basename(path)}: 入库 {len(ids)} 段落")

    after = count(store)
    print(f"\n[ingest] 完成. 本次新增 {total_chunks} 段落, 库内总数 {after} 段落.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
