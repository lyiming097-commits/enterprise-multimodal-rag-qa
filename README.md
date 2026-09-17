# 企业知识库 RAG 问答系统

这是一个面向企业知识库场景的多模态 RAG 问答系统，支持 PDF、Markdown、TXT 和图片资料。项目使用 Vue 3 可视化界面、FastAPI 后端、DeepSeek 远程生成答案、Ollama 本地运行 Qwen3 Embedding、PostgreSQL + pgvector 保存向量，并通过 BM25、RRF 与 Cross-Encoder Rerank 完成混合检索。

项目支持本地开发与 Docker 部署。文档在上传后同步解析和索引，Streamlit 页面作为轻量备用入口；当前版本面向企业知识库的单实例演示，不包含用户权限、并发任务和生产监控。

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

## 仓库结构

```text
app/                 FastAPI 接口、文档解析、检索与生成源码
frontend/            Vue 前端源码及依赖清单
scripts/             数据库初始化、启动及评测脚本
tests/               单元测试与评测示例
web/                 Streamlit 备用界面
data/uploads/        上传目录（Git 仅保留 .gitkeep）
docker/              容器入口脚本
Dockerfile           应用镜像构建
docker-compose.yml   数据库、Ollama 与应用服务编排
.env.example         环境变量示例
pyproject.toml       Python 依赖与构建配置
启动RAG.command      macOS 本地启动入口
项目技术方案.md       架构与技术说明
```

源码按上述目录维护；根目录不再保留同名源码、前端配置和测试文件的重复副本。

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

`EMBEDDING_DIMENSION` 必须和 Ollama 实际返回的向量维度一致。更换维度后需要删除并重新创建本项目的数据表。

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

## 启动 Vue 服务

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

Streamlit 页面保留为轻量备用入口：

```bash
streamlit run web/streamlit_app.py
```

## Docker Compose 部署

需要 Docker Engine / Docker Desktop、Compose v2，以及用于预先构建前端的 Node.js 和 npm。

1. 从 `.env.example` 复制 `.env`，填写 `DEEPSEEK_API_KEY` 和 `POSTGRES_PASSWORD`。数据库密码若含 URL 保留字符，需要同步调整 Compose 中的数据库连接 URL。
2. 检查 `docker-compose.yml` 中 Ollama 的模型挂载。当前文件保留了开发机的 `C:/Users/Lenovo/.ollama/models:/root/.ollama/models:ro`：其他机器应改成自己的模型目录，或删除这一条绑定挂载，保留 `ollama:/root/.ollama` 命名卷并在容器内下载模型。
3. 在宿主机预先构建前端（Dockerfile 会复制 `frontend/dist`）：

```bash
cd frontend
npm ci
npm run build
cd ..
```

4. 启动数据库和 Ollama。若已通过只读目录挂载准备好的模型，可跳过 `pull`；否则执行以下命令下载：

```bash
docker compose up -d db ollama
docker compose exec ollama ollama pull qwen3-embedding:0.6b
docker compose exec ollama ollama list
docker compose up -d --build app
docker compose logs -f app
```

模型必须与 Compose 中的 `OLLAMA_EMBED_MODEL` 一致。入口脚本会初始化数据库并启动 FastAPI，服务地址为 `http://127.0.0.1:8000`。

Compose 使用命名卷保存数据库、上传文件和模型缓存。当前容器配置开启 OCR、关闭 VLM；需要 VLM 时，应先准备视觉模型并调整容器环境变量。不要通过 `docker compose down -v` 停止日常服务，该命令会删除命名卷数据。

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

## 当前版本边界

- 当前默认单用户、单进程，可通过 Docker Compose 部署为独立服务；
- 文件导入期间页面会等待索引完成；
- BM25 在每次查询时根据当前有效 Chunk 构建，适合单实例知识库；
- Cross-Encoder 默认开启，首次运行需要下载 `bge-reranker-v2-m3`；如机器资源不足可在 `.env` 中临时关闭；
- OCR 默认开启，PaddleOCR 不可用时自动回退 RapidOCR；VLM 默认按企业文档模式开启；
- 不包含用户权限、并发任务、生产监控和分布式高可用。

完整技术说明见 [项目技术方案.md](./项目技术方案.md)。
