import argparse
import json
from dataclasses import asdict
from pathlib import Path
from uuid import UUID

from app.db.session import SessionLocal
from app.evaluation.retrieval import evaluate_rankings
from app.retrieval.hybrid import HybridRetriever


def main() -> None:
    parser = argparse.ArgumentParser(description="评测知识库检索效果")
    parser.add_argument("dataset", type=Path, help="JSONL 评测集")
    parser.add_argument("--knowledge-base", required=True, type=UUID)
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()

    cases = [
        json.loads(line)
        for line in args.dataset.read_text(encoding="utf-8").splitlines()
        if line
    ]
    rankings: list[list[str]] = []
    relevant: list[set[str]] = []
    with SessionLocal() as session:
        retriever = HybridRetriever(session)
        for case in cases:
            results = retriever.retrieve(args.knowledge_base, case["question"])
            rankings.append([item.source for item in results])
            relevant.append(set(case["relevant_documents"]))
    metrics = evaluate_rankings(rankings, relevant, args.k)
    print(json.dumps(asdict(metrics), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
