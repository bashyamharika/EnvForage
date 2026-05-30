import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import select
from sqlalchemy.orm import selectinload

from app.database import get_db
from app.main import app
from app.models.profile import EnvironmentProfile

pytestmark = pytest.mark.asyncio

@pytest.fixture
async def client(db_session_factory):
    """Provide an AsyncClient for testing FastAPI routes, overriding the DB dependency."""
    async def _get_db_override():
        async with db_session_factory() as session:
            yield session

    app.dependency_overrides[get_db] = _get_db_override
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as c:
        yield c
    del app.dependency_overrides[get_db]

async def test_profile_crud_lifecycle(client, db_session):
    profile_slug = "test-ml-cuda"
    profile_data = {
        "slug": profile_slug,
        "name": "Test ML CUDA Profile",
        "description": "An environment profile for PyTorch ML testing",
        "tags": ["ml", "cuda", "pytorch"],
        "os_support": ["LINUX", "WSL"],
        "cuda_required": True,
        "python_versions": ["3.10", "3.11"],
        "cuda_versions": ["12.1"],
        "packages": [
            {"package_name": "torch", "version_spec": "==2.3.0", "cuda_variant": "cu121", "is_optional": False, "install_order": 1},
            {"package_name": "torchvision", "version_spec": "==0.18.0", "cuda_variant": "cu121", "is_optional": True, "install_order": 2}
        ]
    }

    create_response = await client.post("/api/v1/profiles", json=profile_data)
    assert create_response.status_code == 201

    get_response = await client.get(f"/api/v1/profiles/{profile_slug}")
    assert get_response.status_code == 200

    delete_response = await client.delete(f"/api/v1/profiles/{profile_slug}")
    assert delete_response.status_code == 204

async def test_create_duplicate_slug_conflict(client):
    profile_data = {"slug": "duplicate-slug", "name": "Original Profile", "os_support": ["LINUX"], "python_versions": ["3.10"]}
    await client.post("/api/v1/profiles", json=profile_data)
    res2 = await client.post("/api/v1/profiles", json=profile_data)
    assert res2.status_code == 409
    assert "already exists" in str(res2.json().get("detail", "")).lower()

def _assert_error_code(response_data, expected_code):
    """Helper to handle both flat string and nested dict error responses."""
    detail = response_data.get("detail")
    if isinstance(detail, dict) and "error" in detail:
        error = detail["error"]
        code = error["code"] if isinstance(error, dict) else error
        assert code == expected_code
    else:
        assert detail == expected_code

async def test_get_nonexistent_profile_returns_404(client):
    response = await client.get("/api/v1/profiles/nonexistent-profile-slug")
    assert response.status_code == 404
    _assert_error_code(response.json(), "PROFILE_NOT_FOUND")

async def test_delete_nonexistent_profile_returns_404(client):
    response = await client.delete("/api/v1/profiles/nonexistent-profile-slug")
    assert response.status_code == 404
    _assert_error_code(response.json(), "PROFILE_NOT_FOUND")

async def test_create_profile_with_wrong_admin_key_returns_401(client):
    response = await client.post("/api/v1/profiles", json={}, headers={"X-Admin-API-Key": "wrong"})
    assert response.status_code == 401
    _assert_error_code(response.json(), "INVALID_ADMIN_KEY") == "PROFILE_NOT_FOUND"
