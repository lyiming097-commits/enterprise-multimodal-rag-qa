from __future__ import annotations

import argparse

import uvicorn


def main() -> None:
    parser = argparse.ArgumentParser(description="启动企业知识库 Vue + FastAPI RAG 服务")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", default=8000, type=int)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()
    uvicorn.run("app.api.main:app", host=args.host, port=args.port, reload=args.reload)


if __name__ == "__main__":
    main()
