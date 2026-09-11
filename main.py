from __future__ import annotations

import json
from collections.abc import Generator
from pathlib import Path
from typing import Annotated, Any
from uuid import UUID

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sqlalchemy import func, select, text
from sqlalchemy.orm import Session

from app.core.config import get_settings
from app.core.models import OllamaEmbeddingClient, OllamaVisionClient
from app.db.models import Chunk, Document, KnowledgeBase
from app.db.session import SessionLocal, engine
from app.generation.rag import RAGService
from app.ingestion.indexer import IndexingService
from app.ingestion.parsers import ParserFactory
from app.ingestion.storage import save_upload

ROOT = Path(__file__).resolve().parents[2]
FRONTEND_DIST = ROOT / "frontend" / "dist"
MAX_UPLOAD_BYTES = 30 * 1024 * 1024
SAMPLE_FILENAME = "示例-项目配置.md"
SAMPLE_CONTENT = """# 项目配置

这个知识库问答项目使用 PostgreSQL 和 pgvector 保存文档向量。

向量化模型是通过 Ollama 本地运行的 Qwen3-Embedding-0.6B。

在线问答使用 DeepSeek 生成答案，检索使用 Query 改写、HyDE、多路召回、RRF 融合和 Cross-Encoder 精排。
"""


class KnowledgeBaseCreate(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    description: str | None = Field(default=None, max_length=1000)


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=4000)
    history: list[dict[str, str]] = Field(default_factory=list)


class ApiMessage(BaseModel):
    message: str


def get_db() -> Generator[Session, None, None]:
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


DbSession = Annotated[Session, Depends(get_db)]


def create_app() -> FastAPI:
    application = FastAPI(
        title="Personal RAG Studio API",
        description="Local API for the six-step RAG demo",
        version="0.2.0",
    )
    application.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    register_api_routes(application)
    register_frontend(application)
    return application


def register_api_routes(application: FastAPI) -> None:
    @application.get("/api/health")
    def health() -> dict[str, Any]:
        settings = get_settings()
        database = {"ok": False, "message": "PostgreSQL 未连接"}
        try:
            with engine.connect() as connection:
                extension = connection.scalar(
                    text("SELECT extversion FROM pg_extension WHERE extname='vector'")
                )
            database = {
                "ok": bool(extension),
                "message": f"PostgreSQL + pgvector {extension}" if extension else "pgvector 未安装",
            }
        except Exception as exc:
            database["message"] = _friendly_error(exc)

        ollama: dict[str, Any] = {"ok": False, "message": "Ollama 未连接"}
        try:
            result = OllamaEmbeddingClient(settings).check()
            ollama = {
                "ok": result["ok"],
                "message": (
                    f"Ollama · {settings.ollama_embed_model}"
                    if result["ok"]
                    else "Ollama 已连接，Embedding 模型未安装"
                ),
            }
        except Exception as exc:
            ollama["message"] = _friendly_error(exc)

        ocr_status = _ocr_status(settings)
        vlm: dict[str, Any] = {
            "enabled": settings.vlm_enabled,
            "ok": False,
            "message": f"VLM 已关闭（{settings.ollama_vlm_model}）",
        }
        if settings.vlm_enabled:
            try:
                result = OllamaVisionClient(settings).check()
                vlm = {
                    "enabled": True,
                    "ok": result["ok"],
                    "message": (
                        f"Ollama VLM · {settings.ollama_vlm_model}"
                        if result["ok"]
                        else f"VLM 模型未安装：{settings.ollama_vlm_model}"
                    ),
                }
            except Exception as exc:
                vlm["message"] = _friendly_error(exc)

        deepseek_ok = bool(settings.deepseek_api_key.get_secret_value())
        vlm_required_ok = not settings.vlm_enabled or vlm["ok"]
        return {
            "ok": (
                database["ok"]
                and ollama["ok"]
                and ocr_status["ok"]
                and deepseek_ok
                and vlm_required_ok
            ),
            "database": database,
            "ollama": ollama,
            "ocr": ocr_status,
            "vlm": vlm,
            "deepseek": {
                "ok": deepseek_ok,
                "message": (
                    f"DeepSeek · {settings.deepseek_model}"
                    if settings.deepseek_api_key.get_secret_value()
                    else "DeepSeek API Key 未配置"
                ),
            },
            "reranker": {
                "ok": settings.rerank_enabled,
                "message": settings.rerank_model if settings.rerank_enabled else "已关闭",
                "cached": _model_cached(settings.rerank_model),
            },
            "embedding": {
                "model": settings.ollama_embed_model,
                "dimension": settings.embedding_dimension,
            },
            "supported_extensions": sorted(ParserFactory.supported_suffixes()),
        }

    @application.get("/api/knowledge-bases")
    def list_knowledge_bases(session: DbSession) -> list[dict[str, Any]]:
        rows = session.execute(
            select(
                KnowledgeBase,
                func.count(func.distinct(Document.id)).label("document_count"),
                func.count(func.distinct(Chunk.id))
                .filter(Chunk.is_active.is_(True))
                .label("chunk_count"),
            )
            .outerjoin(Document, Document.knowledge_base_id == KnowledgeBase.id)
            .outerjoin(Chunk, Chunk.knowledge_base_id == KnowledgeBase.id)
            .group_by(KnowledgeBase.id)
            .order_by(KnowledgeBase.updated_at.desc())
        ).all()
        return [
            {
                "id": str(item.id),
                "name": item.name,
                "description": item.description,
                "document_count": document_count,
                "chunk_count": chunk_count,
                "index_version": item.index_version,
                "updated_at": item.updated_at.isoformat(),
            }
            for item, document_count, chunk_count in rows
        ]

    @application.post("/api/knowledge-bases", status_code=status.HTTP_201_CREATED)
    def create_knowledge_base(payload: KnowledgeBaseCreate, session: DbSession) -> dict[str, Any]:
        name = payload.name.strip()
        if session.scalar(select(KnowledgeBase.id).where(KnowledgeBase.name == name)):
            raise HTTPException(status_code=409, detail="知识库名称已存在")
        item = KnowledgeBase(name=name, description=(payload.description or "").strip() or None)
        session.add(item)
        session.commit()
        return {"id": str(item.id), "name": item.name, "description": item.description}

    @application.get("/api/knowledge-bases/{knowledge_base_id}/documents")
    def list_documents(knowledge_base_id: UUID, session: DbSession) -> list[dict[str, Any]]:
        _require_knowledge_base(session, knowledge_base_id)
        rows = session.execute(
            select(
                Document,
                func.count(Chunk.id).label("chunk_count"),
                func.count(Chunk.id)
                .filter(Chunk.source_metadata["element_type"].astext == "ocr_text")
                .label("ocr_chunk_count"),
                func.count(Chunk.id)
                .filter(Chunk.source_metadata["element_type"].astext == "vlm_description")
                .label("vlm_chunk_count"),
            )
            .outerjoin(Chunk, (Chunk.document_id == Document.id) & Chunk.is_active.is_(True))
            .where(Document.knowledge_base_id == knowledge_base_id, Document.status != "deleted")
            .group_by(Document.id)
            .order_by(Document.updated_at.desc())
        ).all()
        return [
            {
                "id": str(item.id),
                "filename": item.filename,
                "mime_type": item.mime_type,
                "status": item.status,
                "version": item.current_version,
                "chunk_count": chunk_count,
                "error": item.error,
                "updated_at": item.updated_at.isoformat(),
                "extraction": {
                    "ocr_chunks": ocr_chunk_count,
                    "vlm_chunks": vlm_chunk_count,
                    "used_ocr": ocr_chunk_count > 0,
                    "used_vlm": vlm_chunk_count > 0,
                },
            }
            for item, chunk_count, ocr_chunk_count, vlm_chunk_count in rows
        ]

    @application.post("/api/knowledge-bases/{knowledge_base_id}/documents")
    async def upload_documents(
        knowledge_base_id: UUID,
        session: DbSession,
        files: Annotated[list[UploadFile], File(description="待索引的知识库文件")],
    ) -> dict[str, Any]:
        _require_knowledge_base(session, knowledge_base_id)
        if not files:
            raise HTTPException(status_code=400, detail="请选择文件")
        results: list[dict[str, Any]] = []
        for upload in files:
            filename = upload.filename or "document"
            suffix = Path(filename).suffix.lower()
            if suffix not in ParserFactory.supported_suffixes():
                results.append({"filename": filename, "ok": False, "message": "不支持的文件格式"})
                continue
            content = await upload.read(MAX_UPLOAD_BYTES + 1)
            if len(content) > MAX_UPLOAD_BYTES:
                results.append({"filename": filename, "ok": False, "message": "文件超过 30 MB"})
                continue
            try:
                results.append(
                    _index_content(
                        session,
                        knowledge_base_id,
                        content,
                        filename,
                        upload.content_type,
                    )
                )
            except Exception as exc:
                results.append({"filename": filename, "ok": False, "message": _friendly_error(exc)})
        return {"results": results}

    @application.post("/api/knowledge-bases/{knowledge_base_id}/sample")
    def load_sample(knowledge_base_id: UUID, session: DbSession) -> dict[str, Any]:
        _require_knowledge_base(session, knowledge_base_id)
        try:
            return _index_content(
                session,
                knowledge_base_id,
                SAMPLE_CONTENT.encode("utf-8"),
                SAMPLE_FILENAME,
                "text/markdown",
            )
        except Exception as exc:
            raise HTTPException(status_code=500, detail=_friendly_error(exc)) from exc

    @application.delete("/api/documents/{document_id}", response_model=ApiMessage)
    def delete_document(document_id: UUID, session: DbSession) -> ApiMessage:
        document = session.get(Document, document_id)
        if not document or document.status == "deleted":
            raise HTTPException(status_code=404, detail="文档不存在")
        IndexingService(session).delete_document(document_id)
        return ApiMessage(message="文档已从知识库移除")

    @application.get("/api/knowledge-bases/{knowledge_base_id}/chunks")
    def list_chunks(
        knowledge_base_id: UUID,
        session: DbSession,
        document_id: UUID | None = None,
        limit: int = 100,
        offset: int = 0,
        include_inactive: bool = False,
    ) -> dict[str, Any]:
        _require_knowledge_base(session, knowledge_base_id)
        limit = max(1, min(limit, 500))
        offset = max(0, offset)
        filters = [Chunk.knowledge_base_id == knowledge_base_id]
        if document_id:
            filters.append(Chunk.document_id == document_id)
        if not include_inactive:
            filters.append(Chunk.is_active.is_(True))
        total = session.scalar(select(func.count(Chunk.id)).where(*filters)) or 0
        rows = session.execute(
            select(Chunk, Document.filename)
            .join(Document, Document.id == Chunk.document_id)
            .where(*filters)
            .order_by(Chunk.document_id, Chunk.document_version, Chunk.ordinal)
            .offset(offset)
            .limit(limit)
        ).all()
        return {
            "total": int(total),
            "limit": limit,
            "offset": offset,
            "items": [
                {
                    "id": str(chunk.id),
                    "document_id": str(chunk.document_id),
                    "filename": filename,
                    "version": chunk.document_version,
                    "ordinal": chunk.ordinal,
                    "content": chunk.content,
                    "page": chunk.page,
                    "section": chunk.section_path,
                    "token_count": chunk.token_count,
                    "is_active": chunk.is_active,
                    "embedding_model": chunk.embedding_model,
                    "metadata": chunk.source_metadata or {},
                }
                for chunk, filename in rows
            ],
        }

    @application.delete("/api/chunks/{chunk_id}", response_model=ApiMessage)
    def delete_chunk(chunk_id: UUID, session: DbSession) -> ApiMessage:
        chunk = session.get(Chunk, chunk_id)
        if not chunk:
            raise HTTPException(status_code=404, detail="分片不存在")
        if chunk.is_active:
            chunk.is_active = False
            knowledge_base = session.get(KnowledgeBase, chunk.knowledge_base_id)
            if knowledge_base:
                knowledge_base.index_version += 1
            session.commit()
        return ApiMessage(message="分片已删除（保留历史记录）")

    @application.post("/api/knowledge-bases/{knowledge_base_id}/ask")
    def ask(knowledge_base_id: UUID, payload: AskRequest, session: DbSession) -> dict[str, Any]:
        _require_knowledge_base(session, knowledge_base_id)
        active_chunks = session.scalar(
            select(func.count(Chunk.id)).where(
                Chunk.knowledge_base_id == knowledge_base_id, Chunk.is_active.is_(True)
            )
        )
        if not active_chunks:
            raise HTTPException(status_code=400, detail="当前知识库还没有可检索内容")
        history = [
            item
            for item in payload.history[-10:]
            if item.get("role") in {"user", "assistant"} and item.get("content")
        ]
        try:
            return (
                RAGService(session)
                .ask(knowledge_base_id, payload.question, history=history)
                .as_dict()
            )
        except Exception as exc:
            raise HTTPException(status_code=500, detail=_friendly_error(exc)) from exc

    @application.post("/api/knowledge-bases/{knowledge_base_id}/ask/stream")
    def ask_stream(
        knowledge_base_id: UUID, payload: AskRequest, session: DbSession
    ) -> StreamingResponse:
        _require_knowledge_base(session, knowledge_base_id)
        active_chunks = session.scalar(
            select(func.count(Chunk.id)).where(
                Chunk.knowledge_base_id == knowledge_base_id, Chunk.is_active.is_(True)
            )
        )
        if not active_chunks:
            raise HTTPException(status_code=400, detail="当前知识库还没有可检索内容")
        history = [
            item
            for item in payload.history[-10:]
            if item.get("role") in {"user", "assistant"} and item.get("content")
        ]

        def event_generator() -> Generator[str, None, None]:
            try:
                for event in RAGService(session).ask_stream(
                    knowledge_base_id, payload.question, history=history
                ):
                    yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
            except Exception as exc:
                error_event = {"type": "error", "message": _friendly_error(exc)}
                yield f"data: {json.dumps(error_event, ensure_ascii=False)}\n\n"

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )


def register_frontend(application: FastAPI) -> None:
    assets = FRONTEND_DIST / "assets"
    if assets.exists():
        application.mount("/assets", StaticFiles(directory=assets), name="assets")

    @application.get("/{full_path:path}", include_in_schema=False)
    def serve_vue(full_path: str) -> FileResponse:
        index = FRONTEND_DIST / "index.html"
        if not index.exists():
            raise HTTPException(
                status_code=503,
                detail="Vue 页面尚未构建，请在 frontend 目录执行 npm install && npm run build",
            )
        requested = (FRONTEND_DIST / full_path).resolve()
        if full_path and requested.is_relative_to(FRONTEND_DIST.resolve()) and requested.is_file():
            return FileResponse(requested)
        return FileResponse(index)


def _require_knowledge_base(session: Session, knowledge_base_id: UUID) -> KnowledgeBase:
    item = session.get(KnowledgeBase, knowledge_base_id)
    if not item:
        raise HTTPException(status_code=404, detail="知识库不存在")
    return item


def _index_content(
    session: Session,
    knowledge_base_id: UUID,
    content: bytes,
    filename: str,
    mime_type: str | None,
) -> dict[str, Any]:
    settings = get_settings()
    path, digest = save_upload(content, filename, settings.storage_dir)
    document, chunk_count, changed = IndexingService(session, settings).index_file(
        knowledge_base_id=knowledge_base_id,
        path=path,
        filename=filename,
        sha256=digest,
        mime_type=mime_type,
    )
    return {
        "id": str(document.id),
        "filename": filename,
        "ok": True,
        "changed": changed,
        "chunk_count": chunk_count,
        "message": "索引完成" if changed else "内容未变化，已跳过",
    }


def _model_cached(model_name: str) -> bool:
    cache_name = "models--" + model_name.replace("/", "--")
    return (Path.home() / ".cache" / "huggingface" / "hub" / cache_name).exists()


def _ocr_status(settings: Any) -> dict[str, Any]:
    if not settings.ocr_enabled:
        return {
            "enabled": False,
            "ok": False,
            "backend": settings.ocr_backend,
            "message": "OCR 已关闭",
        }
    backends: list[str] = []
    try:
        import paddleocr  # type: ignore[import-not-found,import-untyped]  # noqa: F401

        backends.append("paddleocr")
    except ImportError:
        pass
    try:
        import rapidocr_onnxruntime  # type: ignore[import-not-found,import-untyped]  # noqa: F401

        backends.append("rapidocr_onnxruntime")
    except ImportError:
        pass
    return {
        "enabled": True,
        "ok": bool(backends),
        "backend": settings.ocr_backend,
        "available_backends": backends,
        "message": f"OCR · {', '.join(backends)}" if backends else "OCR 依赖未安装",
    }


def _friendly_error(exc: Exception) -> str:
    message = str(exc).strip()
    if not message:
        return exc.__class__.__name__
    return message[:600]


app = create_app()
