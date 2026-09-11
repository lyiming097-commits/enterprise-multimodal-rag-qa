from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from typing import Any

from app.ingestion.parsers import DocumentElement, normalize_text


@dataclass(slots=True)
class ChunkDraft:
    ordinal: int
    content: str
    content_hash: str
    token_count: int
    page: int | None
    section_path: str | None
    metadata: dict[str, Any] = field(default_factory=dict)


class TextChunker:
    """Structure-aware character chunker with deterministic overlap."""

    def __init__(self, chunk_size: int = 700, overlap: int = 100) -> None:
        if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
            raise ValueError("切块参数不合法")
        self.chunk_size = chunk_size
        self.overlap = overlap

    def split(self, elements: list[DocumentElement]) -> list[ChunkDraft]:
        chunks: list[ChunkDraft] = []
        ordinal = 0
        for element in elements:
            content = normalize_text(element.text)
            if not content:
                continue
            for part in self._split_text(content):
                metadata = dict(element.metadata)
                if element.bbox:
                    metadata["bbox"] = list(element.bbox)
                metadata["element_type"] = element.element_type
                if element.confidence is not None:
                    metadata["confidence"] = element.confidence
                chunks.append(
                    ChunkDraft(
                        ordinal=ordinal,
                        content=part,
                        content_hash=hashlib.sha256(part.encode("utf-8")).hexdigest(),
                        token_count=estimate_tokens(part),
                        page=element.page,
                        section_path=element.section_path,
                        metadata=metadata,
                    )
                )
                ordinal += 1
        return chunks

    def _split_text(self, text: str) -> list[str]:
        if len(text) <= self.chunk_size:
            return [text]
        sentences = [item for item in re.split(r"(?<=[。！？!?；;\n])", text) if item]
        parts: list[str] = []
        current = ""
        for sentence in sentences:
            if len(sentence) > self.chunk_size:
                if current:
                    parts.append(current.strip())
                    current = ""
                parts.extend(self._window(sentence))
                continue
            if current and len(current) + len(sentence) > self.chunk_size:
                parts.append(current.strip())
                prefix = current[-self.overlap :] if self.overlap else ""
                current = prefix + sentence
            else:
                current += sentence
        if current.strip():
            parts.append(current.strip())
        return parts

    def _window(self, text: str) -> list[str]:
        step = self.chunk_size - self.overlap
        return [
            text[start : start + self.chunk_size].strip() for start in range(0, len(text), step)
        ]


def estimate_tokens(text: str) -> int:
    chinese = len(re.findall(r"[\u4e00-\u9fff]", text))
    non_chinese_words = len(re.findall(r"[A-Za-z0-9_]+", text))
    punctuation = len(re.findall(r"[^\w\s\u4e00-\u9fff]", text))
    return chinese + non_chinese_words + max(1, punctuation // 2)
