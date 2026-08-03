# PDF RAG Chatbot · 中文财报问答

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green)](LICENSE)
[![LangChain](https://img.shields.io/badge/LangChain-0.1%2B-orange)](https://python.langchain.com/)
[![Chroma](https://img.shields.io/badge/Chroma-vector--store-purple)](https://www.trychroma.com/)

> 一个**针对中文 PDF（财报 / 招股书 / 年报）的 RAG 问答 demo**：上传 PDF → 解析分块 → 嵌入 → 存入本地 Chroma → 用 Claude 回答问题，并强制要求答案末尾标注 `[来源: 文档, 段落]`。

这是 [mazihua-lgtm](https://github.com/mazihua-lgtm) 接单服务中 **RAG / 知识库问答** 类目的官方 demo。生产化落地（多租户、权限控制、审计日志、可观测性）随时可谈。

---

## 🎬 Demo

> 截图占位 — 上传后将以 GIF 形式展示：上传招股书 → 等待分块完成 → 问「该公司 2023 年营业收入是多少？」→ 答案末尾出现 `[来源: 招股书_2023, P.42]`。

（待录屏回填）

---

## ⚡ 快速开始

```bash
# 1. 克隆
git clone https://github.com/mazihua-lgtm/pdf-rag-chatbot.git
cd pdf-rag-chatbot

# 2. 安装依赖（建议 Python 3.10+）
pip install -r requirements.txt

# 3. 配置 API Key
cp .env.example .env
# 编辑 .env，填入 ANTHROPIC_API_KEY（也支持 VOYAGE_API_KEY / OPENAI_API_KEY 作为嵌入备选）

# 4. 启动 Web UI
python app.py
# 浏览器打开 http://127.0.0.1:7860

# 或者批量入库（CLI 模式）
python scripts/ingest.py path/to/你的财报.pdf
```

零 API 嵌入方案：`embeddings.py` 默认使用本地 `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`，无需任何 API Key 即可完成 embedding，只在生成答案时调用 Claude。

---

## ✨ 功能列表

- 📄 **PDF 解析** — 支持中文财报/招股书，优先用 `pdfplumber`（表格友好），自动按段落切分
- ✂️ **智能分块** — 500–1000 字一段、相邻重叠 100 字，保留「页码 + 段落号」元数据
- 🧠 **本地嵌入** — 默认使用多语言 MiniLM 模型，零 API 成本、支持中文
- 🗃️ **本地向量库** — Chroma 持久化到 `./chroma_db/`，二次启动秒级加载
- 🤖 **Claude 问答** — LangChain `RetrievalQA` + Anthropic Claude，中文 prompt，强制要求 **citation 标注**
- 🌐 **Gradio UI** — 三 Tab：上传 / 问答 / 关于，本地访问 `http://127.0.0.1:7860`
- 🛠️ **CLI 工具** — `python scripts/ingest.py file.pdf` 一行入库

---

## 📝 示例问题 + 答案格式

文件：`examples/sample_questions.md` 提供 20+ 中文财报常用问题模板。

**示例问答：**

> **Q：** 该公司 2023 年度的营业收入和归母净利润分别是多少？
>
> **A：** 根据《XX 公司 2023 年度报告》合并利润表数据：
> - 2023 年营业收入为 **358.62 亿元**（同比 +12.4%）
> - 归属于母公司股东的净利润为 **47.13 亿元**（同比 +18.7%）
>
> 两项指标均创历史新高，主要受益于核心产品销量增长及海外业务扩张。  
> `[来源: XX_2023年报, P.42, 段落3]`

> **Q：** 这份招股书的募集资金用途是什么？
>
> **A：** 招股书披露，本次发行募集资金扣除发行费用后，将投资于以下项目：
> 1. **研发中心建设项目** — 拟投入 4.5 亿元
> 2. **生产基地扩建项目** — 拟投入 6.0 亿元
> 3. **补充流动资金** — 拟投入 1.5 亿元
>
> 项目实施期为 24 个月。  
> `[来源: XX_招股说明书, P.218, 段落1]`

> **Q：** 2022 年的研发费用占营收比例？
>
> **A：** 提供该上下文的文档未明确披露 2022 年研发费用占比，您可以提供 2022 年年报或问询函回复以获取准确数据。  
> `[来源: 未找到相关段落]`

---

## 🧱 技术栈

| 层 | 选型 | 备注 |
|---|---|---|
| **LLM 框架** | LangChain 0.1+ | `RetrievalQA` + 自定义 prompt |
| **LLM 模型** | Anthropic Claude 3.5 Sonnet | `claude-3-5-sonnet-20241022` |
| **嵌入（默认）** | sentence-transformers | `paraphrase-multilingual-MiniLM-L12-v2`，零 API |
| **嵌入（备选）** | voyage-3 / OpenAI text-embedding-3-small | 需对应 API Key |
| **向量库** | Chroma 0.4+ | 本地持久化 |
| **PDF 解析** | pdfplumber（主） + pypdf（备） | pdfplumber 对中文表格更友好 |
| **Web UI** | Gradio 4.x | 三 Tab 布局 |
| **Python** | 3.9+ | 推荐 3.10 / 3.11 |

---

## 🎯 适用场景

1. **券商 / 投研** — 把 100 份上市公司年报/招股书扔进去，团队成员问「X 公司毛利率趋势？」秒回带页码
2. **律所尽调** — 上传并购标的法律意见书 + 财报，自动回答尽调清单问题，citation 可直接贴进报告
3. **企业知识库** — 把内部制度、产品手册、技术文档做 RAG，新员工自助问答，比 Confluence 搜索精准得多

---

## 🛒 接单服务定位

这是一个 **RAG 服务的 demo**，完整生产化版本包含：

- ✅ 多租户隔离 + 用户权限
- ✅ 审计日志（每一次问答留痕）
- ✅ 流式输出（SSE / WebSocket）
- ✅ 文档版本管理（diff）
- ✅ 多模态（图片/表格/公式）
- ✅ 可观测性（LangSmith / Phoenix）
- ✅ 私有化部署（K8s / 离线模型）

如需定制，联系 [mz@mazihua.dev](mailto:mz@mazihua.dev) 或查看 [hire-mazihua](https://github.com/mazihua-lgtm/hire-mazihua)。

更多支持与赞助：见 [SPONSOR.md](SPONSOR.md)。

---

## 📁 项目结构

```
pdf-rag-chatbot/
├── app.py                ← Gradio Web UI
├── rag/
│   ├── loader.py         ← PDF 解析 + 分块
│   ├── embeddings.py     ← 嵌入模型封装（默认本地，可切 voyage / OpenAI）
│   ├── vector_store.py   ← Chroma 封装
│   ├── qa.py             ← RetrievalQA + citation
│   └── prompts.py        ← 中文 prompt 模板
├── scripts/
│   └── ingest.py         ← CLI 批量入库
├── tests/
│   └── test_basic.py     ← 单元测试
├── examples/
│   └── sample_questions.md
├── README.md
├── LICENSE               ← MIT
├── requirements.txt
├── .env.example
└── .gitignore
```

---

## 🧪 测试

```bash
pytest tests/ -v
```

不需要 API Key 即可运行的离线测试覆盖：分块逻辑、prompt 模板、Citation 后处理。

需要 API 的端到端测试默认 `skip`，要打开请设置环境变量 `RUN_LIVE_TEST=1`。

---

## 📜 License

MIT — 见 [LICENSE](LICENSE)。

## 🙏 致谢

- [LangChain](https://python.langchain.com/)
- [Chroma](https://www.trychroma.com/)
- [Anthropic Claude](https://www.anthropic.com/)
- [Gradio](https://gradio.app/)
- [sentence-transformers](https://www.sbert.net/)
