from __future__ import annotations

import sys
from pathlib import Path
from uuid import UUID

import streamlit as st
from sqlalchemy import select, text

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.core.config import get_settings  # noqa: E402
from app.core.models import OllamaEmbeddingClient  # noqa: E402
from app.db.models import Document, KnowledgeBase  # noqa: E402
from app.db.session import SessionLocal, engine  # noqa: E402
from app.generation.rag import RAGService  # noqa: E402
from app.ingestion.indexer import IndexingService  # noqa: E402
from app.ingestion.parsers import ParserFactory  # noqa: E402
from app.ingestion.storage import save_upload  # noqa: E402

settings = get_settings()
st.set_page_config(page_title="企业知识库 RAG", page_icon="📚", layout="wide")
st.title("📚 企业知识库 RAG")


def database_ready() -> tuple[bool, str]:
    try:
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        return True, "PostgreSQL 已连接"
    except Exception as exc:
        return False, f"PostgreSQL 不可用：{exc}"


def ollama_ready() -> tuple[bool, str]:
    try:
        result = OllamaEmbeddingClient(settings).check()
        if result["ok"]:
            return True, f"Ollama 已连接：{settings.ollama_embed_model}"
        return False, "Ollama 已连接，但未找到配置的 Embedding 模型"
    except Exception as exc:
        return False, f"Ollama 不可用：{exc}"


db_ok, db_message = database_ready()
if not db_ok:
    st.error(db_message)
    st.code("python -m scripts.init_db", language="bash")
    st.stop()

with st.sidebar:
    st.header("运行状态")
    st.success(db_message)
    ollama_ok, ollama_message = ollama_ready()
    (st.success if ollama_ok else st.warning)(ollama_message)
    if settings.rerank_enabled:
        st.caption(f"Cross-Encoder：{settings.rerank_model}")
    else:
        st.warning("Cross-Encoder Rerank 当前已关闭")

    st.header("知识库")
    with SessionLocal() as session:
        knowledge_bases = list(session.scalars(select(KnowledgeBase).order_by(KnowledgeBase.name)))

    new_name = st.text_input("新建知识库", placeholder="例如：产品知识库")
    if st.button("创建", use_container_width=True, disabled=not new_name.strip()):
        try:
            with SessionLocal() as session:
                session.add(KnowledgeBase(name=new_name.strip()))
                session.commit()
            st.rerun()
        except Exception as exc:
            st.error(f"创建失败：{exc}")

    if not knowledge_bases:
        st.info("请先创建一个知识库。")
        st.stop()

    labels = {str(item.id): item.name for item in knowledge_bases}
    selected_id = st.selectbox(
        "当前知识库",
        options=list(labels),
        format_func=lambda value: labels[value],
    )
    knowledge_base_id = UUID(selected_id)

if st.session_state.get("active_knowledge_base") != selected_id:
    st.session_state.active_knowledge_base = selected_id
    st.session_state.messages = []
    st.session_state.pop("last_debug", None)


documents_tab, chat_tab, debug_tab = st.tabs(["文档管理", "知识问答", "检索信息"])

with documents_tab:
    st.subheader("上传并建立索引")
    supported = sorted(suffix.lstrip(".") for suffix in ParserFactory.supported_suffixes())
    uploads = st.file_uploader(
        "支持 PDF、Markdown、TXT 和图片",
        type=supported,
        accept_multiple_files=True,
    )
    if st.button("开始索引", type="primary", disabled=not uploads or not ollama_ok):
        progress = st.progress(0)
        for index, upload in enumerate(uploads, start=1):
            try:
                content = upload.getvalue()
                path, digest = save_upload(content, upload.name, settings.storage_dir)
                with st.spinner(f"正在处理 {upload.name}..."), SessionLocal() as session:
                    document, chunk_count, changed = IndexingService(session).index_file(
                        knowledge_base_id=knowledge_base_id,
                        path=path,
                        filename=upload.name,
                        sha256=digest,
                        mime_type=upload.type,
                    )
                action = "完成索引" if changed else "内容未变化，跳过"
                st.success(f"{upload.name}：{action}，当前 {chunk_count} 个 Chunk")
            except Exception as exc:
                st.error(f"{upload.name} 处理失败：{exc}")
            progress.progress(index / len(uploads))

    st.subheader("已导入文档")
    with SessionLocal() as session:
        documents = list(
            session.scalars(
                select(Document)
                .where(Document.knowledge_base_id == knowledge_base_id)
                .order_by(Document.updated_at.desc())
            )
        )
    if not documents:
        st.caption("还没有文档。")
    for document in documents:
        cols = st.columns([4, 1, 1, 1])
        cols[0].write(document.filename)
        cols[1].write(document.status)
        cols[2].write(f"v{document.current_version}")
        if cols[3].button("删除", key=f"delete-{document.id}"):
            with SessionLocal() as session:
                IndexingService(session).delete_document(document.id)
            st.rerun()
        if document.error:
            st.caption(document.error)

with chat_tab:
    if "messages" not in st.session_state:
        st.session_state.messages = []
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            for citation in message.get("citations", []):
                location = f"第 {citation['page']} 页" if citation.get("page") else ""
                st.caption(f"[{citation['citation_id']}] {citation['source']} {location}")

    question = st.chat_input("输入关于当前知识库的问题")
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            try:
                with st.spinner("正在检索并生成答案..."), SessionLocal() as session:
                    result = RAGService(session).ask(
                        knowledge_base_id,
                        question,
                        history=st.session_state.messages[:-1],
                    )
                st.markdown(result.answer)
                citation_dicts = []
                for citation in result.citations:
                    item = {
                        "citation_id": citation.citation_id,
                        "source": citation.source,
                        "page": citation.page,
                        "section": citation.section,
                    }
                    citation_dicts.append(item)
                    location = f"第 {citation.page} 页" if citation.page else ""
                    st.caption(f"[{citation.citation_id}] {citation.source} {location}")
                st.session_state.messages.append(
                    {"role": "assistant", "content": result.answer, "citations": citation_dicts}
                )
                st.session_state.last_debug = result.as_dict()
            except Exception as exc:
                st.error(f"问答失败：{exc}")

with debug_tab:
    st.caption("展示最近一次问答的六步 RAG 链路，不保存缓存。")
    if debug := st.session_state.get("last_debug"):
        st.markdown("#### 1. Query 预处理")
        st.json(debug["query_plan"])

        st.markdown("#### 2. Query Embedding")
        st.write(
            f"模型：`{settings.ollama_embed_model}`；维度："
            f"`{settings.embedding_dimension}`；与建库模型一致。"
        )

        st.markdown("#### 3. 向量检索 + BM25 多路召回 + RRF")
        st.dataframe(debug["retrieval_trace"], use_container_width=True)

        st.markdown("#### 4. Cross-Encoder Rerank")
        rerank_applied = any(
            item.get("rerank_score") is not None for item in debug["retrieval_trace"]
        )
        if rerank_applied:
            st.success(f"已使用 {settings.rerank_model} 精排")
        else:
            st.warning("本次未产生 Rerank 分数，请检查模型是否启用和成功加载。")

        st.markdown("#### 5. 受约束 Prompt 拼装")
        st.write(f"问题路由：`{debug['route']}`；只允许依据检索资料回答，证据不足时拒答。")

        st.markdown("#### 6. 生成与溯源")
        st.json(debug["citations"])
    else:
        st.info("完成一次问答后即可查看。")
