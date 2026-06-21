import pytest
import pytest_asyncio
from httpx import AsyncClient, ASGITransport
from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
from sqlalchemy.orm import DeclarativeBase

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from app.main import app
from app.db.session import get_db, Base


# ── In-memory SQLite DB for testing ───────────────────────────────────────────

TEST_DB_URL = "sqlite+aiosqlite:///:memory:"

test_engine = create_async_engine(TEST_DB_URL, echo=False)
TestSession = async_sessionmaker(bind=test_engine, class_=AsyncSession, expire_on_commit=False)


async def override_get_db():
    async with TestSession() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


@pytest_asyncio.fixture(autouse=True)
async def setup_db():
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    app.dependency_overrides[get_db] = override_get_db
    yield
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    app.dependency_overrides.clear()


@pytest_asyncio.fixture
async def client():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as c:
        yield c


# ── Tests ──────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_health(client):
    r = await client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "ok"
    assert "users" in data
    assert "items" in data


@pytest.mark.asyncio
async def test_create_user(client):
    r = await client.post("/users/", json={"external_id": "user_test_001"})
    assert r.status_code == 201
    data = r.json()
    assert data["external_id"] == "user_test_001"
    assert "id" in data


@pytest.mark.asyncio
async def test_create_user_duplicate(client):
    await client.post("/users/", json={"external_id": "user_dup"})
    r = await client.post("/users/", json={"external_id": "user_dup"})
    assert r.status_code == 409


@pytest.mark.asyncio
async def test_create_item(client):
    r = await client.post("/items/", json={
        "external_id": "movie_001",
        "title": "The Dark Knight",
        "genres": ["Action", "Thriller"],
        "tags": ["blockbuster"],
        "content_vector": [0.1] * 32,
    })
    assert r.status_code == 201
    assert r.json()["title"] == "The Dark Knight"


@pytest.mark.asyncio
async def test_log_interaction(client):
    await client.post("/users/", json={"external_id": "u1"})
    await client.post("/items/", json={
        "external_id": "i1", "title": "Inception",
        "genres": ["Sci-Fi"], "tags": [], "content_vector": [0.0] * 32,
    })
    r = await client.post("/interactions/", json={
        "user_external_id": "u1",
        "item_external_id": "i1",
        "rating": 4.5,
    })
    assert r.status_code == 201
    assert r.json()["rating"] == 4.5


@pytest.mark.asyncio
async def test_recommend_no_model(client):
    await client.post("/users/", json={"external_id": "u2"})
    r = await client.get("/recommend/u2")
    # Without a trained model, we expect a 500 or graceful error
    assert r.status_code in (200, 500)


@pytest.mark.asyncio
async def test_user_not_found(client):
    r = await client.get("/recommend/nonexistent_user")
    assert r.status_code == 404
