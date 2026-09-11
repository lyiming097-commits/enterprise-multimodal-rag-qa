from app.ingestion.chunker import TextChunker
from app.ingestion.parsers import DocumentElement


def test_chunker_preserves_source_metadata() -> None:
    chunks = TextChunker(chunk_size=12, overlap=2).split(
        [
            DocumentElement(
                text="第一句话。第二句话很长。第三句话。",
                page=3,
                section_path="第一章 > 示例",
                metadata={"kind": "demo"},
            )
        ]
    )
    assert len(chunks) >= 2
    assert all(item.page == 3 for item in chunks)
    assert all(item.section_path == "第一章 > 示例" for item in chunks)
    assert all(item.metadata["kind"] == "demo" for item in chunks)
    assert [item.ordinal for item in chunks] == list(range(len(chunks)))


def test_chunk_hash_is_deterministic() -> None:
    element = DocumentElement(text="固定内容")
    first = TextChunker().split([element])[0]
    second = TextChunker().split([element])[0]
    assert first.content_hash == second.content_hash
