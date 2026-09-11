# RAG Studio：个人多模态知识库 Demo

一个可以在本机运行的 RAG 问答 Demo，支持 PDF、Markdown、TXT 和图片资料。项目使用 Vue 3 可视化界面、FastAPI 后端、DeepSeek 远程生成答案、Ollama 本地运行 Qwen3 Embedding、PostgreSQL + pgvector 保存向量，并通过 BM25、RRF 与 Cross-Encoder Rerank 完成混合检索。

本项目不使用 Docker、Redis、消息队列或持久化缓存。文档在 Streamlit 页面中同步解析和索引，适合个人演示与效果实验。

## 功能

- PDF、Markdown、TXT 和图片导入；
- 扫描页和图片使用 PaddleOCR / RapidOCR，可选 Ollama VLM 补充图表、表格和版面描述；
- 文档 Hash 检测、版本化 Chunk 和同步增量更新；
- Qwen3 Embedding + pgvector 语义召回；
- 中文 BM25 + RRF 混合召回；
- DeepSeek Query 改写、HyDE 和多角度扩写；
- `BAAI/bge-reranker-v2-m3` Cross-Encoder 精排；
- 受约束 Prompt 与 DeepSeek 答案生成；
- 文档名、页码、章节和原文片段引用；
- Hit@K、MRR、nDCG 检索评测脚本。
- Vue 页面实时展示 Query 改写、召回路径、RRF/Rerank 分数和有效引用。

## 环境要求

- Python 3.11～3.14（当前 Python 3.14 环境使用 RapidOCR ONNXRuntime；PaddleOCR 作为可选增强）；
- 本机 PostgreSQL，并已安装 `vector` 扩展；
- 本机 Ollama；
- 可访问 DeepSeek API。

macOS 使用 Homebrew 时可以安装 PostgreSQL 和 pgvector：

```bash
brew install postgresql@17 pgvector
brew services start postgresql@17
createdb rag
```

具体安装路径会随 Homebrew 和 PostgreSQL 版本变化；如果 `CREATE EXTENSION vector` 失败，需要先确认 pgvector 已安装到当前 PostgreSQL 实例。

## 安装

建议创建 Python 3.12 虚拟环境：

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

需要处理图片或扫描 PDF 时安装 OCR（当前 Python 3.14 推荐 RapidOCR）：

```bash
pip install -e ".[ocr]"
```

如果使用支持 PaddlePaddle 的 Python 版本，可额外安装 PaddleOCR：

```bash
pip install -e ".[ocr-paddle]"
```

Cross-Encoder 已包含在基础依赖中；首次问答会从 Hugging Face 下载重排模型。需要 RAGAS 时安装：

```bash
pip install -e ".[eval]"
```

## 模型准备

启动 Ollama 并拉取 Qwen3 Embedding：

```bash
ollama serve
ollama pull qwen3-embedding:0.6b
ollama list
```

本机已确认可用标签为 `qwen3-embedding:0.6b`，并实测返回 1024 维向量；其他机器仍应以 `ollama list` 的结果为准。

如需 VLM 补充复杂图片语义，可额外拉取模型并在 `.env` 中开启：

```bash
ollama pull moondream
```

企业文档模式会对独立图片、含嵌入图像的 PDF 页面，以及 OCR 文字量少或置信度低的扫描页面追加 VLM 视觉描述。本地测试默认使用体积较小的 `moondream:latest`；如需更强的中文表格理解，可切换为 `qwen2.5vl:3b` 或更大模型。

## 配置

```bash
cp .env.example .env
```

至少修改以下内容：

```dotenv
DATABASE_URL=postgresql+psycopg://localhost:5432/rag
DEEPSEEK_API_KEY=你的_API_Key
OLLAMA_EMBED_MODEL=qwen3-embedding:0.6b
EMBEDDING_DIMENSION=1024
RERANK_ENABLED=true
RERANK_MODEL=BAAI/bge-reranker-v2-m3
```

`EMBEDDING_DIMENSION` 必须和 Ollama 实际返回的向量维度一致。更换维度后需要删除并重新创建本 Demo 的数据表。

## 初始化数据库

```bash
python -m scripts.init_db
```

这个命令会启用 `vector` 扩展、创建表，并建立 pgvector HNSW 索引。

## 构建 Vue 页面

首次从源码启动时执行：

```bash
cd frontend
npm install
npm run build
cd ..
```

`frontend/dist` 构建产物会由 FastAPI 直接托管，运行时不需要再启动 Node.js 服务。

## 启动 Vue Demo

```bash
python -m scripts.run_app
```

浏览器打开 `http://127.0.0.1:8000`。macOS 也可直接双击项目根目录的 `启动RAG.command`，脚本会补齐本地环境、初始化数据库并打开页面。

打开页面后：

1. 创建知识库；
2. 上传并索引文件；
3. 切换到“知识问答”并提问；
4. 在“检索信息”中检查查询改写和引用。

页面也提供“一键加载测试样本”，可以不准备文件直接验证完整问答链路。

原 Streamlit 页面仍保留为轻量备用入口：

```bash
streamlit run web/streamlit_app.py
```

## HTTP API

FastAPI 文档地址为 `http://127.0.0.1:8000/docs`。主要接口包括：

- `GET /api/health`：数据库、Ollama、DeepSeek 和 Reranker 状态；
- `POST /api/knowledge-bases`：创建知识库；
- `POST /api/knowledge-bases/{id}/documents`：上传并同步索引文件；
- `POST /api/knowledge-bases/{id}/ask`：执行六步 RAG 问答并返回引用与检索轨迹。

## 六步 RAG 检索流程

1. **Query 预处理**：DeepSeek 生成改写查询、2～3 个互补角度以及 HyDE 假设文档；原问题始终保留。
2. **Query Embedding**：普通查询使用 Qwen3 Query 指令模板，HyDE 按文档文本向量化；数据库仅检索由同一 Ollama 模型生成的向量。
3. **多路召回**：原问题、改写和扩写分别执行 pgvector 与 BM25 检索，HyDE 执行向量检索，所有排名使用 RRF 融合。
4. **Rerank 精排**：使用 `BAAI/bge-reranker-v2-m3` Cross-Encoder 对 Query 和候选 Chunk 联合打分。
5. **Prompt 拼装**：限制 DeepSeek 只能使用提供的证据，忽略资料中的指令，证据不足时明确拒答。
6. **生成与溯源**：关键事实使用 `[S1]` 格式引用，并映射到真实文档、页码、章节和原文片段。

## 检索评测

评测集采用 JSONL，每行包含问题和相关文档名：

```json
{"question":"这个项目使用什么向量数据库？","relevant_documents":["项目说明.md"]}
```

运行：

```bash
python -m scripts.evaluate_retrieval tests/fixtures/eval.example.jsonl \
  --knowledge-base 你的知识库UUID --k 5
```

完整生成效果可使用可选的 RAGAS 脚本。样本字段为 `user_input`、`retrieved_contexts`、`response` 和 `reference`：

```bash
python -m scripts.evaluate_ragas tests/fixtures/ragas.example.jsonl
```

RAGAS 会调用 DeepSeek 作为评委，并使用 Ollama Embedding 计算相关性，因此会产生远程 API 消耗。

## 测试

```bash
pytest
```

算法单元测试不需要连接数据库和模型。完整联调需要 PostgreSQL、Ollama 和 DeepSeek API 均可用。

## 当前 Demo 边界

- 单用户、单进程、本机运行；
- 文件导入期间页面会等待索引完成；
- BM25 在每次查询时根据当前有效 Chunk 构建，适合个人规模资料；
- Cross-Encoder 默认开启，首次运行需要下载 `bge-reranker-v2-m3`；如机器资源不足可在 `.env` 中临时关闭；
- OCR 默认开启，PaddleOCR 不可用时自动回退 RapidOCR；VLM 默认按企业文档模式开启；
- 不包含用户权限、并发任务、缓存、生产监控和分布式部署。

完整技术说明见 [项目技术方案.md](./项目技术方案.md)。
