import argparse
import json
from pathlib import Path

from app.core.models import DeepSeekClient, OllamaLangChainEmbeddings


def main() -> None:
    try:
        from ragas import EvaluationDataset, evaluate  # type: ignore[import-untyped]
        from ragas.embeddings import LangchainEmbeddingsWrapper  # type: ignore[import-untyped]
        from ragas.llms import LangchainLLMWrapper  # type: ignore[import-untyped]
        from ragas.metrics import (  # type: ignore[import-untyped]
            Faithfulness,
            LLMContextPrecisionWithReference,
            LLMContextRecall,
            ResponseRelevancy,
        )
    except ImportError as exc:
        raise SystemExit("请先执行 pip install -e '.[eval]'") from exc

    parser = argparse.ArgumentParser(description="使用 RAGAS 评测已生成的问答样本")
    parser.add_argument("dataset", type=Path, help="RAGAS JSONL 数据集")
    args = parser.parse_args()
    samples = [
        json.loads(line)
        for line in args.dataset.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    dataset = EvaluationDataset.from_list(samples)
    evaluator_llm = LangchainLLMWrapper(DeepSeekClient().llm)
    evaluator_embeddings = LangchainEmbeddingsWrapper(OllamaLangChainEmbeddings())
    result = evaluate(
        dataset=dataset,
        metrics=[
            Faithfulness(llm=evaluator_llm),
            LLMContextPrecisionWithReference(llm=evaluator_llm),
            LLMContextRecall(llm=evaluator_llm),
            ResponseRelevancy(llm=evaluator_llm, embeddings=evaluator_embeddings),
        ],
    )
    print(result)


if __name__ == "__main__":
    main()
