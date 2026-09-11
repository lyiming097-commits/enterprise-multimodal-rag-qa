from app.evaluation.retrieval import evaluate_rankings


def test_retrieval_metrics() -> None:
    metrics = evaluate_rankings(
        rankings=[["a", "b"], ["x", "c"]],
        relevant_items=[{"a"}, {"c"}],
        k=2,
    )
    assert metrics.hit_at_k == 1.0
    assert metrics.mrr == 0.75
    assert 0 < metrics.ndcg_at_k <= 1
