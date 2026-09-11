from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass


def tokenize(text: str) -> list[str]:
    tokens = [item.lower() for item in re.findall(r"[A-Za-z0-9_\-]+", text)]
    for sequence in re.findall(r"[\u4e00-\u9fff]+", text):
        tokens.extend(sequence)
        tokens.extend(sequence[index : index + 2] for index in range(len(sequence) - 1))
    return tokens


@dataclass(slots=True)
class BM25Document:
    doc_id: str
    text: str


class BM25Index:
    def __init__(self, documents: list[BM25Document], k1: float = 1.5, b: float = 0.75) -> None:
        self.documents = documents
        self.k1 = k1
        self.b = b
        self.tokens = [tokenize(item.text) for item in documents]
        self.term_frequencies = [Counter(items) for items in self.tokens]
        self.doc_lengths = [len(items) for items in self.tokens]
        self.avg_doc_length = sum(self.doc_lengths) / len(self.doc_lengths) if documents else 0
        self.document_frequency: Counter[str] = Counter()
        for items in self.tokens:
            self.document_frequency.update(set(items))

    def search(self, query: str, top_k: int = 10) -> list[tuple[str, float]]:
        if not self.documents:
            return []
        query_terms = tokenize(query)
        scores: list[tuple[str, float]] = []
        total = len(self.documents)
        for index, document in enumerate(self.documents):
            score = 0.0
            length = self.doc_lengths[index]
            frequencies = self.term_frequencies[index]
            for term in query_terms:
                frequency = frequencies.get(term, 0)
                if frequency == 0:
                    continue
                doc_frequency = self.document_frequency[term]
                idf = math.log(1 + (total - doc_frequency + 0.5) / (doc_frequency + 0.5))
                normalizer = frequency + self.k1 * (
                    1 - self.b + self.b * length / max(self.avg_doc_length, 1)
                )
                score += idf * frequency * (self.k1 + 1) / normalizer
            if score > 0:
                scores.append((document.doc_id, score))
        return sorted(scores, key=lambda item: item[1], reverse=True)[:top_k]
