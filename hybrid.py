from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from uuid import UUID

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.core.models import OllamaEmbeddingClient
from app.db.models import Chunk, Document
from app.retrieval.bm25 import BM25Document, BM25Index
from app.retrieval.fusion import reciprocal_rank_fusion
from app.retrieval.query_plan import QueryPlan
from app.retrieval.reranker import CrossEncoderReranker


@dataclass(slots=True)
class RetrievedChunk:
    chunk_id: UUID
    document_id: UUID
    content: str
    source: str
    page: int | None
    section: str | None
    metadata: dict[str, Any] = field(default_factory=dict)
    dense_score: float | None = None
    sparse_score: float | None = None
    fusion_score: float = 0.0
    rerank_score: float | None = None
    retrieval_paths: list[str] = field(default_factory=list)


class HybridRetriever:
    def __init__(
        self,
        session: Session,
        settings: Settings | None = None,
        embedder: OllamaEmbeddingClient | None = None,
    ) -> None:
        self.session = session
        self.settings = settings or get_settings()
        self.embedder = embedder or OllamaEmbeddingClient(self.settings)
        self.reranker = CrossEncoderReranker(
            self.settings.rerank_model,
            enabled=self.settings.rerank_enabled,
            batch_size=self.settings.rerank_batch_size,
        )

    def retrieve(
        self,
        knowledge_base_id: UUID,
        question: str,
        queries: list[str] | None = None,
        query_plan: QueryPlan | None = None,
    ) -> list[RetrievedChunk]:
        plan = query_plan or self._legacy_plan(question, queries)
        candidates: dict[UUID, RetrievedChunk] = {}
        rankings: list[list[tuple[UUID, float]]] = []
        all_chunks = self._load_active_chunks(knowledge_base_id)
        sparse_index = BM25Index(
            [BM25Document(doc_id=str(item.chunk_id), text=item.content) for item in all_chunks]
        )
        by_id = {item.chunk_id: item for item in all_chunks}

        query_vectors = self.embedder.embed_queries(plan.dense_queries)
        for index, (_query, vector) in enumerate(
            zip(plan.dense_queries, query_vectors, strict=True), start=1
        ):
            path = f"dense_query_{index}"
            dense = self._dense_by_vector(knowledge_base_id, vector)
            self._record_dense(candidates, dense, path)
            rankings.append([(item.chunk_id, item.dense_score or 0.0) for item in dense])

        if plan.hypothetical_document:
            hyde_vector = self.embedder.embed_documents([plan.hypothetical_document])[0]
            hyde = self._dense_by_vector(knowledge_base_id, hyde_vector)
            self._record_dense(candidates, hyde, "dense_hyde")
            rankings.append([(item.chunk_id, item.dense_score or 0.0) for item in hyde])

        for index, query in enumerate(plan.lexical_queries, start=1):
            path = f"bm25_query_{index}"
            sparse_scores = sparse_index.search(query, self.settings.sparse_top_k)
            sparse_ranking: list[tuple[UUID, float]] = []
            for raw_id, score in sparse_scores:
                chunk_id = UUID(raw_id)
                item = candidates.get(chunk_id) or by_id[chunk_id]
                item.sparse_score = max(item.sparse_score or 0.0, score)
                if path not in item.retrieval_paths:
                    item.retrieval_paths.append(path)
                candidates[chunk_id] = item
                sparse_ranking.append((chunk_id, score))
            rankings.append(sparse_ranking)

        fused = reciprocal_rank_fusion(rankings)
        ordered: list[RetrievedChunk] = []
        for chunk_id, score in fused[: self.settings.fusion_top_k]:
            item = candidates[chunk_id]
            item.fusion_score = score
            ordered.append(item)
        return self.reranker.rerank(question, ordered, self.settings.rerank_top_k)

    def _dense_by_vector(
        self, knowledge_base_id: UUID, vector: list[float]
    ) -> list[RetrievedChunk]:
        distance = Chunk.embedding.cosine_distance(vector).label("distance")
        rows = self.session.execute(
            select(Chunk, Document.filename, distance)
            .join(Document, Document.id == Chunk.document_id)
            .where(
                Chunk.knowledge_base_id == knowledge_base_id,
                Chunk.is_active.is_(True),
                Chunk.embedding.is_not(None),
                Chunk.embedding_model == self.settings.ollama_embed_model,
            )
            .order_by(distance)
            .limit(self.settings.dense_top_k)
        ).all()
        return [
            RetrievedChunk(
                chunk_id=chunk.id,
                document_id=chunk.document_id,
                content=chunk.content,
                source=filename,
                page=chunk.page,
                section=chunk.section_path,
                metadata=chunk.source_metadata,
                dense_score=1.0 - float(row_distance),
            )
            for chunk, filename, row_distance in rows
        ]

    def _load_active_chunks(self, knowledge_base_id: UUID) -> list[RetrievedChunk]:
        rows = self.session.execute(
            select(Chunk, Document.filename)
            .join(Document, Document.id == Chunk.document_id)
            .where(Chunk.knowledge_base_id == knowledge_base_id, Chunk.is_active.is_(True))
        ).all()
        return [
            RetrievedChunk(
                chunk_id=chunk.id,
                document_id=chunk.document_id,
                content=chunk.content,
                source=filename,
                page=chunk.page,
                section=chunk.section_path,
                metadata=chunk.source_metadata,
            )
            for chunk, filename in rows
        ]

    @staticmethod
    def _record_dense(
        candidates: dict[UUID, RetrievedChunk],
        ranking: list[RetrievedChunk],
        path: str,
    ) -> None:
        for item in ranking:
            existing = candidates.get(item.chunk_id)
            if existing is None:
                item.retrieval_paths.append(path)
                candidates[item.chunk_id] = item
                continue
            existing.dense_score = max(existing.dense_score or -1.0, item.dense_score or -1.0)
            if path not in existing.retrieval_paths:
                existing.retrieval_paths.append(path)

    @staticmethod
    def _legacy_plan(question: str, queries: list[str] | None) -> QueryPlan:
        if not queries:
            return QueryPlan.fallback(question)
        return QueryPlan(
            original_query=question,
            rewritten_query=queries[0],
            expanded_queries=queries[1:],
        )
