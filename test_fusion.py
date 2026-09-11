from app.retrieval.fusion import reciprocal_rank_fusion


def test_rrf_rewards_results_from_multiple_retrievers() -> None:
    fused = reciprocal_rank_fusion(
        [
            [("dense-only", 0.99), ("both", 0.8)],
            [("both", 9.0), ("sparse-only", 8.0)],
        ]
    )
    assert fused[0][0] == "both"
