"""pdf-rag-chatbot · Gradio Web UI.

3 Tab:
- 上传文档: 文件上传 + 入库按钮
- 智能问答: 问题输入 + 答案输出 (含 citation)
- 关于: 项目说明 + GitHub 链接

启动: python app.py
默认地址: http://127.0.0.1:7860
"""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

import gradio as gr

from rag.loader import DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE, load_pdf
from rag.qa import DEFAULT_RETRIEVAL_K, ask
from rag.vector_store import add_documents, count, get_vector_store

REPO_URL = "https://github.com/mazihua-lgtm/pdf-rag-chatbot"
ABOUT_MD = f"""# PDF RAG Chatbot · 中文财报问答

> 一个**针对中文 PDF（财报 / 招股书 / 年报）的 RAG 问答 demo**。

## ✨ 核心功能

- 📄 **PDF 解析** — pdfplumber (主) + pypdf (备)，自动按段落切分
- 🧠 **本地嵌入** — sentence-transformers 多语言 MiniLM，零 API 成本
- 🗃️ **本地向量库** — Chroma 持久化到 `./chroma_db/`
- 🤖 **Claude 问答** — 强制要求答案末尾标注 `[来源: ...]`
- 🌐 **Gradio UI** — 当前页面

## 🚀 快速开始

1. 复制 `.env.example` 为 `.env`，填入 `ANTHROPIC_API_KEY`
2. 在「上传文档」Tab 上传 PDF
3. 切到「智能问答」Tab 开始提问

## 🔗 链接

- 仓库: <{REPO_URL}>
- Issues: <{REPO_URL}/issues>
- 雇佣作者: <https://github.com/mazihua-lgtm/hire-mazihua>
- 邮件: mz@mazihua.dev
- 赞助: 见 [SPONSOR.md]({REPO_URL}/blob/main/SPONSOR.md)

---

> 本地访问: http://127.0.0.1:7860
"""


def _save_uploads(files: List) -> List[str]:
    """把上传的临时文件落地到一个稳定路径, 返回路径列表."""
    out: List[str] = []
    for f in files or []:
        # Gradio 4.x 给的是 NamedString / tempfile 对象, .name 即本地路径
        src = getattr(f, "name", None) or (f if isinstance(f, str) else None)
        if not src or not os.path.exists(src):
            continue
        # 复制到持久目录以防 tempfile 被清理
        dst_dir = Path("./uploads")
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / os.path.basename(src)
        if not dst.exists():
            shutil.copy2(src, dst)
        out.append(str(dst))
    return out


def ingest_files(
    files,
    chunk_size: int,
    chunk_overlap: int,
) -> str:
    """上传 Tab 的处理函数: 解析 + 入库."""
    if not files:
        return "⚠️ 请先选择至少一个 PDF 文件。"
    paths = _save_uploads(files)
    if not paths:
        return "⚠️ 文件读取失败, 请重新上传。"

    store = get_vector_store()
    total_chunks = 0
    summary: List[str] = []
    for p in paths:
        try:
            docs = load_pdf(p, chunk_size=int(chunk_size), chunk_overlap=int(chunk_overlap))
        except Exception as e:  # noqa: BLE001
            summary.append(f"❌ {os.path.basename(p)}: {e}")
            continue
        if not docs:
            summary.append(f"⚠️ {os.path.basename(p)}: 未提取到文本 (可能为扫描件)")
            continue
        ids = add_documents(store, docs)
        total_chunks += len(ids)
        summary.append(f"✅ {os.path.basename(p)}: 入库 {len(ids)} 个段落")

    total = count(store)
    return (
        "## 入库结果\n\n"
        + "\n".join(summary)
        + f"\n\n**本批新增**: {total_chunks} 段落\n"
        + f"**向量库总数**: {total} 段落"
    )


def answer_question(
    question: str,
    k: int,
    history: Optional[List[Tuple[str, str]]] = None,
) -> Tuple[str, List[Tuple[str, str]]]:
    """问答 Tab 的处理函数."""
    if not question or not question.strip():
        return "请输入问题。", history or []
    store = get_vector_store()
    if count(store) <= 0:
        return "⚠️ 向量库为空, 请先在「上传文档」Tab 导入 PDF。", history or []
    try:
        result = ask(store, question=question.strip(), k=int(k))
    except Exception as e:  # noqa: BLE001
        return f"❌ 调用失败: {e}", history or []

    text = str(result)
    history = list(history or [])
    history.append((question.strip(), text))
    return text, history


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="PDF RAG Chatbot", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            "# 📚 PDF RAG Chatbot\n"
            "**中文财报 / 招股书 / 年报 问答** · "
            "[GitHub]({}) · [SPONSOR]({}/blob/main/SPONSOR.md)".format(REPO_URL, REPO_URL)
        )

        with gr.Tabs():
            # ===== Tab 1: 上传文档 =====
            with gr.Tab("📤 上传文档"):
                gr.Markdown(
                    "上传 PDF, 自动按段落切分并入库到本地 Chroma。\n"
                    "默认分块: 800 字 / 重叠 100 字 (可在下方调整)。"
                )
                with gr.Row():
                    file_input = gr.File(
                        label="选择 PDF 文件 (可多选)",
                        file_count="multiple",
                        file_types=[".pdf"],
                    )
                with gr.Row():
                    chunk_size = gr.Slider(
                        minimum=200, maximum=2000, step=100,
                        value=DEFAULT_CHUNK_SIZE, label="chunk_size (单块最大字符)",
                    )
                    chunk_overlap = gr.Slider(
                        minimum=0, maximum=400, step=20,
                        value=DEFAULT_CHUNK_OVERLAP, label="chunk_overlap (相邻重叠)",
                    )
                ingest_btn = gr.Button("🚀 开始入库", variant="primary")
                ingest_out = gr.Markdown(label="入库结果")

                ingest_btn.click(
                    fn=ingest_files,
                    inputs=[file_input, chunk_size, chunk_overlap],
                    outputs=[ingest_out],
                )

            # ===== Tab 2: 智能问答 =====
            with gr.Tab("💬 智能问答"):
                gr.Markdown("基于已入库的 PDF 回答问题, 答案末尾会自动标注 `[来源: ...]`。")
                chatbot = gr.Chatbot(label="对话", height=400)
                with gr.Row():
                    question_box = gr.Textbox(
                        label="你的问题",
                        placeholder="例: 该公司 2023 年营业收入是多少?",
                        scale=4,
                    )
                    k_slider = gr.Slider(
                        minimum=1, maximum=10, step=1,
                        value=DEFAULT_RETRIEVAL_K, label="k (召回段落数)", scale=1,
                    )
                with gr.Row():
                    ask_btn = gr.Button("🤖 提问", variant="primary")
                    clear_btn = gr.Button("🧹 清空对话")

                def _submit(q, k_, hist):
                    ans, new_hist = answer_question(q, k_, hist)
                    return "", new_hist

                ask_btn.click(
                    fn=_submit,
                    inputs=[question_box, k_slider, chatbot],
                    outputs=[question_box, chatbot],
                )
                question_box.submit(
                    fn=_submit,
                    inputs=[question_box, k_slider, chatbot],
                    outputs=[question_box, chatbot],
                )
                clear_btn.click(fn=lambda: [], outputs=[chatbot])

            # ===== Tab 3: 关于 =====
            with gr.Tab("ℹ️ 关于"):
                gr.Markdown(ABOUT_MD)

    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.queue().launch(
        server_name=os.getenv("GRADIO_SERVER_NAME", "127.0.0.1"),
        server_port=int(os.getenv("GRADIO_SERVER_PORT", "7860")),
        show_error=True,
    )
