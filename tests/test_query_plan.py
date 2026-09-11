from app.retrieval.query_plan import QueryPlan


def test_query_plan_keeps_original_and_deduplicates_queries() -> None:
    plan = QueryPlan.from_payload(
        "这个项目用什么数据库？",
        {
            "rewritten_query": "项目采用的数据库技术",
            "expanded_queries": [
                "项目采用的数据库技术",
                "向量数据库选型",
                "数据存储组件",
                "第四个角度会被截断",
            ],
            "hypothetical_document": "该项目的数据存储方案包括关系数据库和向量扩展。",
        },
    )
    assert plan.original_query == "这个项目用什么数据库？"
    assert plan.lexical_queries == [
        "这个项目用什么数据库？",
        "项目采用的数据库技术",
        "向量数据库选型",
        "数据存储组件",
        "第四个角度会被截断",
    ]
    assert plan.hypothetical_document is not None


def test_query_plan_falls_back_to_original_question() -> None:
    plan = QueryPlan.from_payload(" 原始问题 ", {})
    assert plan.lexical_queries == ["原始问题"]
    assert plan.hypothetical_document is None
