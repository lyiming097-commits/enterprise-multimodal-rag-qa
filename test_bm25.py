from app.retrieval.bm25 import BM25Document, BM25Index, tokenize


def test_chinese_tokenizer_contains_bigrams() -> None:
    tokens = tokenize("向量数据库 pgvector")
    assert "向量" in tokens
    assert "pgvector" in tokens


def test_bm25_ranks_relevant_document_first() -> None:
    index = BM25Index(
        [
            BM25Document("a", "项目使用 PostgreSQL 和 pgvector 向量数据库"),
            BM25Document("b", "今天适合去公园散步"),
        ]
    )
    results = index.search("项目的向量数据库")
    assert results[0][0] == "a"
