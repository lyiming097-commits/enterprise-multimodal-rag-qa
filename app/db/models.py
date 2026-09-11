from uuid import UUID

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    BigInteger,
    Boolean,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.config import get_settings
from app.db.base import Base, TimestampMixin, UUIDPrimaryKeyMixin


class KnowledgeBase(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "knowledge_bases"

    name: Mapped[str] = mapped_column(String(200), unique=True, nullable=False)
    description: Mapped[str | None] = mapped_column(Text)
    index_version: Mapped[int] = mapped_column(BigInteger, default=0, nullable=False)

    documents: Mapped[list["Document"]] = relationship(back_populates="knowledge_base")


class Document(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "documents"

    knowledge_base_id: Mapped[UUID] = mapped_column(
        ForeignKey("knowledge_bases.id", ondelete="CASCADE"), index=True, nullable=False
    )
    filename: Mapped[str] = mapped_column(String(500), nullable=False)
    source_path: Mapped[str] = mapped_column(Text, nullable=False)
    mime_type: Mapped[str | None] = mapped_column(String(200))
    sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(String(30), default="pending", index=True)
    current_version: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    error: Mapped[str | None] = mapped_column(Text)

    knowledge_base: Mapped[KnowledgeBase] = relationship(back_populates="documents")
    versions: Mapped[list["DocumentVersion"]] = relationship(
        back_populates="document", cascade="all, delete-orphan"
    )

    __table_args__ = (
        UniqueConstraint("knowledge_base_id", "filename", name="uq_document_kb_filename"),
    )


class DocumentVersion(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "document_versions"

    document_id: Mapped[UUID] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), index=True, nullable=False
    )
    version: Mapped[int] = mapped_column(Integer, nullable=False)
    sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    source_path: Mapped[str] = mapped_column(Text, nullable=False)
    parser_version: Mapped[str] = mapped_column(String(100), default="v1", nullable=False)
    chunk_config_version: Mapped[str] = mapped_column(String(100), default="v1", nullable=False)
    status: Mapped[str] = mapped_column(String(30), default="pending", nullable=False)
    error: Mapped[str | None] = mapped_column(Text)

    document: Mapped[Document] = relationship(back_populates="versions")

    __table_args__ = (UniqueConstraint("document_id", "version", name="uq_document_version"),)


class Chunk(UUIDPrimaryKeyMixin, TimestampMixin, Base):
    __tablename__ = "chunks"

    knowledge_base_id: Mapped[UUID] = mapped_column(
        ForeignKey("knowledge_bases.id", ondelete="CASCADE"), index=True, nullable=False
    )
    document_id: Mapped[UUID] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), index=True, nullable=False
    )
    document_version: Mapped[int] = mapped_column(Integer, nullable=False)
    ordinal: Mapped[int] = mapped_column(Integer, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    content_hash: Mapped[str] = mapped_column(String(64), index=True, nullable=False)
    token_count: Mapped[int] = mapped_column(Integer, nullable=False)
    page: Mapped[int | None] = mapped_column(Integer)
    section_path: Mapped[str | None] = mapped_column(Text)
    source_metadata: Mapped[dict] = mapped_column(JSONB, default=dict, nullable=False)
    embedding: Mapped[list[float] | None] = mapped_column(
        Vector(get_settings().embedding_dimension)
    )
    embedding_model: Mapped[str | None] = mapped_column(String(200))
    is_active: Mapped[bool] = mapped_column(Boolean, default=False, index=True, nullable=False)

    __table_args__ = (
        UniqueConstraint(
            "document_id",
            "document_version",
            "ordinal",
            name="uq_chunk_document_version_ordinal",
        ),
        Index("ix_chunks_kb_active", "knowledge_base_id", "is_active"),
    )
