from sqlalchemy import text

from app.core.config import get_settings
from app.db import models  # noqa: F401
from app.db.base import Base
from app.db.session import engine


def main() -> None:
    settings = get_settings()
    settings.prepare_directories()
    with engine.begin() as connection:
        connection.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    Base.metadata.create_all(engine)
    with engine.begin() as connection:
        connection.execute(
            text(
                "CREATE INDEX IF NOT EXISTS ix_chunks_embedding_hnsw "
                "ON chunks USING hnsw (embedding vector_cosine_ops) "
                "WHERE is_active = true"
            )
        )
    print("数据库初始化完成。")


if __name__ == "__main__":
    main()

