from fastapi.routing import APIRoute

from app.api.main import create_app


def test_vue_api_routes_are_registered() -> None:
    paths = {route.path for route in create_app().routes if isinstance(route, APIRoute)}

    assert "/api/health" in paths
    assert "/api/knowledge-bases" in paths
    assert "/api/knowledge-bases/{knowledge_base_id}/documents" in paths
    assert "/api/knowledge-bases/{knowledge_base_id}/ask" in paths
    assert "/api/knowledge-bases/{knowledge_base_id}/ask/stream" in paths
    assert "/api/knowledge-bases/{knowledge_base_id}/chunks" in paths
    assert "/api/chunks/{chunk_id}" in paths
    assert "/{full_path:path}" in paths
