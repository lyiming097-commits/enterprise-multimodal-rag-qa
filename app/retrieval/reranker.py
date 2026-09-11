from __future__ import annotations

import logging
from typing import Any, Protocol, TypeVar

logger = logging.getLogger(__name__)


class RerankCandidate(Protocol):
    content: str
    rerank_score: float | None


T = TypeVar("T", bound=RerankCandidate)


class CrossEncoderReranker:
    """Lazy-loaded Cross-Encoder reranker for Chinese and multilingual passages."""

    def __init__(
        self,
        model_name: str,
        enabled: bool = True,
        batch_size: int = 8,
    ) -> None:
        self.model_name = model_name
        self.enabled = enabled
        self.batch_size = batch_size
        self._model: Any = None

    def rerank(self, query: str, candidates: list[T], top_k: int) -> list[T]:
        if not self.enabled or not candidates:
            return candidates[:top_k]
        model = self._load()
        pairs = [(query, item.content) for item in candidates]
        scores = model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        for item, score in zip(candidates, scores, strict=True):
            item.rerank_score = float(score)
        return sorted(
            candidates,
            key=lambda item: item.rerank_score if item.rerank_score is not None else float("-inf"),
            reverse=True,
        )[:top_k]

    def _load(self) -> Any:
        if self._model is None:
            try:
                from sentence_transformers import (  # type: ignore[import-not-found,import-untyped]
                    CrossEncoder,
                )
            except ImportError as exc:
                raise RuntimeError(
                    "Rerank 已启用但 sentence-transformers 未安装，请重新执行 pip install -e ."
                ) from exc
            logger.info("正在加载 Cross-Encoder：%s", self.model_name)
            self._model = CrossEncoder(
                self.model_name,
                max_length=512,
                trust_remote_code=True,
            )
        return self._model


# 保留旧名称，避免外部导入中断。
LocalReranker = CrossEncoderReranker
