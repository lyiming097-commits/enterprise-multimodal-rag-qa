#!/bin/sh
set -e

echo "==> 初始化数据库（pgvector 扩展 + 建表 + HNSW 索引）"
python -m scripts.init_db

echo "==> 启动 FastAPI 应用"
exec uvicorn app.api.main:app --host 0.0.0.0 --port 8000
