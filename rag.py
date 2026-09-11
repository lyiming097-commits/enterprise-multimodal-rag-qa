from __future__ import annotations

import re
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from typing import Any
from uuid import UUID

from sqlalchemy.orm import Session

from app.core.config import Settings, get_settings
from app.core.models import DeepSeekClient, OllamaEmbeddingClient
from app.retrieval.hybrid import HybridRetriever, RetrievedChunk
from app.retrieval.query_plan import QueryPlan


@dataclass(slots=True)
class Citation:
    citation_id: str
    chunk_id: str
    source: str
    page: int | None
    section: str | None
    excerpt: str


@dataclass(slots=True)
class AnswerResult:
    answer: str
    citations: list[Citation]
    rewritten_queries: list[str]
    query_plan: QueryPlan
    retrieved_count: int
    route: str
    second_retrieval: bool
    retrieval_trace: list[dict[str, Any]]

    def as_dict(self) -> dict[str, Any]:
        return {
            "answer": self.answer,
            "citations": [asdict(item) for item in self.citations],
            "rewritten_queries": self.rewritten_queries,
            "query_plan": self.query_plan.as_dict(),
            "retrieved_count": self.retrieved_count,
            "route": self.route,
            "second_retrieval": self.second_retrieval,
            "retrieval_trace": self.retrieval_trace,
        }


class RAGService:
    def __init__(self, session: Session, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        embedder = OllamaEmbeddingClient(self.settings)
        self.retriever = HybridRetriever(session, self.settings, embedder)
        self.llm = DeepSeekClient(self.settings)

    def ask(
        self,
        knowledge_base_id: UUID,
        question: str,
        history: list[dict[str, str]] | None = None,
    ) -> AnswerResult:
        question = question.strip()
        if not question:
            raise ValueError("问题不能为空")
        route = classify_route(question)
        query_plan = self.llm.preprocess_query(question)
        chunks = self.retriever.retrieve(
            knowledge_base_id,
            question,
            query_plan=query_plan,
        )
        second_retrieval = False
        if self._low_confidence(chunks):
            expanded = self.llm.expand_query(question, query_plan.lexical_queries)
            if expanded and expanded not in query_plan.lexical_queries:
                query_plan.expanded_queries.append(expanded)
                fallback_plan = QueryPlan(
                    original_query=question,
                    rewritten_query=expanded,
                )
                second = self.retriever.retrieve(
                    knowledge_base_id,
                    question,
                    query_plan=fallback_plan,
                )
                chunks = merge_results(chunks, second)
                second_retrieval = True
        context_limit = self.settings.max_context_chunks + (2 if route != "lookup" else 0)
        selected = chunks[:context_limit]
        contexts = [self._runtime_context()]
        contexts.extend(
            self._context(item, index)
            for index, item in enumerate(selected, start=1)
        )
        answer = self.llm.answer(question, contexts, history, route=route)
        citations = self._validated_citations(answer, contexts)
        return AnswerResult(
            answer=answer,
            citations=citations,
            rewritten_queries=query_plan.lexical_queries,
            query_plan=query_plan,
            retrieved_count=len(chunks),
            route=route,
            second_retrieval=second_retrieval,
            retrieval_trace=[self._trace(item) for item in chunks],
        )

    def ask_stream(
        self,
        knowledge_base_id: UUID,
        question: str,
        history: list[dict[str, str]] | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Run retrieval synchronously while yielding progress and answer tokens."""
        question = question.strip()
        if not question:
            raise ValueError("问题不能为空")
        route = classify_route(question)
        yield {"type": "stage", "stage": "query_preprocess", "message": "正在进行查询改写"}
        query_plan = self.llm.preprocess_query(question)
        yield {"type": "query_plan", "query_plan": query_plan.as_dict()}

        yield {"type": "stage", "stage": "retrieval", "message": "正在执行向量和 BM25 检索"}
        chunks = self.retriever.retrieve(knowledge_base_id, question, query_plan=query_plan)
        second_retrieval = False
        if self._low_confidence(chunks):
            yield {
                "type": "stage",
                "stage": "fallback_retrieval",
                "message": "首轮结果较弱，正在扩展查询",
            }
            expanded = self.llm.expand_query(question, query_plan.lexical_queries)
            if expanded and expanded not in query_plan.lexical_queries:
                query_plan.expanded_queries.append(expanded)
                fallback_plan = QueryPlan(original_query=question, rewritten_query=expanded)
                second = self.retriever.retrieve(
                    knowledge_base_id, question, query_plan=fallback_plan
                )
                chunks = merge_results(chunks, second)
                second_retrieval = True
        yield {"type": "stage", "stage": "rerank", "message": "Cross-Encoder 精排完成"}
        trace = [self._trace(item) for item in chunks]
        yield {"type": "retrieval", "retrieval_trace": trace}

        context_limit = self.settings.max_context_chunks + (2 if route != "lookup" else 0)
        selected = chunks[:context_limit]
        contexts = [self._runtime_context()]
        contexts.extend(
            self._context(item, index)
            for index, item in enumerate(selected, start=1)
        )
        yield {"type": "stage", "stage": "generation", "message": "DeepSeek 正在生成回答"}
        answer_parts: list[str] = []
        for token in self.llm.stream_answer(question, contexts, history, route=route):
            answer_parts.append(token)
            yield {"type": "token", "content": token}
        answer = "".join(answer_parts).strip()
        citations = self._validated_citations(answer, contexts)
        result = AnswerResult(
            answer=answer,
            citations=citations,
            rewritten_queries=query_plan.lexical_queries,
            query_plan=query_plan,
            retrieved_count=len(chunks),
            route=route,
            second_retrieval=second_retrieval,
            retrieval_trace=trace,
        )
        yield {"type": "done", "result": result.as_dict()}

    @staticmethod
    def _low_confidence(chunks: list[RetrievedChunk]) -> bool:
        return len(chunks) < 2 or (bool(chunks) and chunks[0].fusion_score < 0.02)

    @staticmethod
    def _context(chunk: RetrievedChunk, index: int) -> dict[str, Any]:
        return {
            "citation_id": f"S{index}",
            "chunk_id": str(chunk.chunk_id),
            "source": chunk.source,
            "page": chunk.page,
            "section": chunk.section,
            "content": chunk.content,
        }

    def _runtime_context(self) -> dict[str, Any]:
        """Expose the running Demo configuration as authoritative evidence."""
        return {
            "citation_id": "S0",
            "chunk_id": "runtime-config",
            "source": "当前 Demo 运行配置",
            "page": None,
            "section": "系统配置",
            "content": (
                f"当前 Demo 实际使用 PostgreSQL + pgvector 作为向量数据库；"
                f"向量化模型为 Ollama {self.settings.ollama_embed_model}，"
                f"向量维度为 {self.settings.embedding_dimension}；"
                f"文本生成模型为远程 DeepSeek {self.settings.deepseek_model}。"
                "这些是当前运行配置的真实值，优先于知识库资料中‘建议使用’或‘例如’的方案描述。"
            ),
        }

    @staticmethod
    def _trace(chunk: RetrievedChunk) -> dict[str, Any]:
        return {
            "chunk_id": str(chunk.chunk_id),
            "source": chunk.source,
            "page": chunk.page,
            "section": chunk.section,
            "dense_score": chunk.dense_score,
            "sparse_score": chunk.sparse_score,
            "fusion_score": chunk.fusion_score,
            "rerank_score": chunk.rerank_score,
            "retrieval_paths": chunk.retrieval_paths,
            "excerpt": chunk.content[:160],
        }

    @staticmethod
    def _validated_citations(answer: str, contexts: list[dict[str, Any]]) -> list[Citation]:
        referenced = set(re.findall(r"\[(S\d+)\]", answer, flags=re.IGNORECASE))
        citations: list[Citation] = []
        for item in contexts:
            if item["citation_id"].upper() not in {value.upper() for value in referenced}:
                continue
            citations.append(
                Citation(
                    citation_id=item["citation_id"],
                    chunk_id=item["chunk_id"],
                    source=item["source"],
                    page=item["page"],
                    section=item["section"],
                    excerpt=item["content"][:300],
                )
            )
        return citations


def classify_route(question: str) -> str:
    if any(word in question for word in ("比较", "区别", "差异", "对比", "分别")):
        return "comparison"
    if any(word in question for word in ("总结", "概括", "摘要", "主要内容", "全文")):
        return "summary"
    return "lookup"


def merge_results(
    first: list[RetrievedChunk], second: list[RetrievedChunk]
) -> list[RetrievedChunk]:
    merged = {item.chunk_id: item for item in first}
    for item in second:
        existing = merged.get(item.chunk_id)
        if existing is None or item.fusion_score > existing.fusion_score:
            merged[item.chunk_id] = item
    return sorted(
        merged.values(),
        key=lambda item: item.rerank_score if item.rerank_score is not None else item.fusion_score,
        reverse=True,
    )
