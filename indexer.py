from __future__ import annotations

from pathlib import Path
from uuid import UUID

from sqlalchemy import func, select, update
from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.core.models import OllamaEmbeddingClient
from app.db.models import Chunk, Document, DocumentVersion, KnowledgeBase
from app.ingestion.chunker import TextChunker
from app.ingestion.parsers import ParserFactory


class IndexingService:
    def __init__(
        self,
        session: Session,
        settings: Settings | None = None,
        embedder: OllamaEmbeddingClient | None = None,
    ) -> None:
        self.session = session
        self.settings = settings or get_settings()
        self.embedder = embedder or OllamaEmbeddingClient(self.settings)
        self.chunker = TextChunker(self.settings.chunk_size, self.settings.chunk_overlap)

    def index_file(
        self,
        knowledge_base_id: UUID,
        path: Path,
        filename: str,
        sha256: str,
        mime_type: str | None = None,
    ) -> tuple[Document, int, bool]:
        knowledge_base = self.session.get(KnowledgeBase, knowledge_base_id)
        if not knowledge_base:
            raise ValueError("知识库不存在")

        document = self.session.scalar(
            select(Document).where(
                Document.knowledge_base_id == knowledge_base_id,
                Document.filename == filename,
            )
        )
        if document and document.sha256 == sha256 and document.status == "ready":
            return document, self._active_chunk_count(document.id), False

        if document is None:
            document = Document(
                knowledge_base_id=knowledge_base_id,
                filename=filename,
                source_path=str(path),
                mime_type=mime_type,
                sha256=sha256,
                status="indexing",
            )
            self.session.add(document)
            self.session.flush()
        else:
            document.status = "indexing"
            document.error = None
            document.source_path = str(path)
            document.mime_type = mime_type

        latest_version = self.session.scalar(
            select(func.max(DocumentVersion.version)).where(
                DocumentVersion.document_id == document.id
            )
        )
        next_version = (latest_version or 0) + 1
        version = DocumentVersion(
            document_id=document.id,
            version=next_version,
            sha256=sha256,
            source_path=str(path),
            status="indexing",
            chunk_config_version=f"chars-{self.settings.chunk_size}-{self.settings.chunk_overlap}",
        )
        self.session.add(version)
        self.session.commit()

        try:
            parser = ParserFactory.for_path(path)
            elements = parser.parse(path)
            drafts = self.chunker.split(elements)
            if not drafts:
                raise RuntimeError("文档解析后没有可索引内容")

            vectors: list[list[float]] = []
            batch_size = self.settings.embedding_batch_size
            for start in range(0, len(drafts), batch_size):
                vectors.extend(
                    self.embedder.embed_documents(
                        [item.content for item in drafts[start : start + batch_size]]
                    )
                )

            self.session.execute(
                update(Chunk).where(Chunk.document_id == document.id).values(is_active=False)
            )
            for draft, vector in zip(drafts, vectors, strict=True):
                self.session.add(
                    Chunk(
                        knowledge_base_id=knowledge_base_id,
                        document_id=document.id,
                        document_version=next_version,
                        ordinal=draft.ordinal,
                        content=draft.content,
                        content_hash=draft.content_hash,
                        token_count=draft.token_count,
                        page=draft.page,
                        section_path=draft.section_path,
                        source_metadata=draft.metadata,
                        embedding=vector,
                        embedding_model=self.settings.ollama_embed_model,
                        is_active=True,
                    )
                )
            document.sha256 = sha256
            document.current_version = next_version
            document.status = "ready"
            document.error = None
            version.status = "ready"
            knowledge_base.index_version += 1
            self.session.commit()
            return document, len(drafts), True
        except Exception as exc:
            self.session.rollback()
            failed_document = self.session.get(Document, document.id)
            failed_version = self.session.scalar(
                select(DocumentVersion).where(
                    DocumentVersion.document_id == document.id,
                    DocumentVersion.version == next_version,
                )
            )
            if failed_document:
                failed_document.status = "failed"
                failed_document.error = str(exc)[:2000]
            if failed_version:
                failed_version.status = "failed"
                failed_version.error = str(exc)[:2000]
            self.session.commit()
            raise

    def delete_document(self, document_id: UUID) -> None:
        document = self.session.get(Document, document_id)
        if not document:
            raise ValueError("文档不存在")
        knowledge_base = self.session.get(KnowledgeBase, document.knowledge_base_id)
        self.session.execute(
            update(Chunk).where(Chunk.document_id == document.id).values(is_active=False)
        )
        document.status = "deleted"
        if knowledge_base:
            knowledge_base.index_version += 1
        self.session.commit()

    def _active_chunk_count(self, document_id: UUID) -> int:
        return len(
            self.session.scalars(
                select(Chunk.id).where(Chunk.document_id == document_id, Chunk.is_active.is_(True))
            ).all()
        )
