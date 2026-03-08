"""
Tests for RedisBackend. Skipped automatically when Redis is unavailable.
Run with: docker run -p 6379:6379 redis/redis-stack-server:latest
"""
import pytest

from tests.conftest import dummy_embed, near_embed

REDIS_URL = "redis://localhost:6379"


@pytest.fixture
def b():
    redis = pytest.importorskip("redis", reason="redis package not installed")
    from semantic_cache.backends.redis_backend import RedisBackend

    backend = RedisBackend(url=REDIS_URL, namespace="test", dims=16)
    try:
        backend._sync.ping()
    except Exception:
        pytest.skip("Redis not reachable — start redis-stack-server on port 6379")

    backend.clear_sync()
    yield backend
    backend.clear_sync()


# ------------------------------------------------------------------ index

def test_index_created_idempotently(b):
    """Calling ensure_index twice must not raise."""
    b._index_ready = False
    b._ensure_index_sync()
    b._ensure_index_sync()   # second call — should be a no-op


# ------------------------------------------------------------------ sync search / store

def test_miss_on_empty(b):
    assert b.search_sync(dummy_embed("hello"), threshold=0.85) is None


def test_hit_on_exact_vector(b):
    vec = dummy_embed("What is Redis?")
    b.store_sync(vec, "Redis is an in-memory data store.")

    result = b.search_sync(vec, threshold=0.85)
    assert result == "Redis is an in-memory data store."


def test_miss_on_dissimilar_vector(b):
    b.store_sync(dummy_embed("What is Redis?"), "Redis is an in-memory data store.")
    result = b.search_sync(dummy_embed("Who wrote Hamlet?"), threshold=0.85)
    assert result is None


# ------------------------------------------------------------------ stats

def test_stats_hit_and_miss(b):
    vec = dummy_embed("stats test redis")
    b.store_sync(vec, "answer")

    b.search_sync(vec, threshold=0.85)                            # hit
    b.search_sync(dummy_embed("unrelated query"), threshold=0.85) # miss

    s = b.stats_sync()
    assert s["hits"] == 1
    assert s["misses"] == 1


# ------------------------------------------------------------------ clear

def test_clear_removes_all_entries(b):
    b.store_sync(dummy_embed("test clear"), "value")
    b.clear_sync()
    assert b.search_sync(dummy_embed("test clear"), threshold=0.85) is None
    assert b.stats_sync()["total"] == 0


# ------------------------------------------------------------------ async

@pytest.mark.asyncio
async def test_async_store_and_search(b):
    vec = dummy_embed("async redis test")
    await b.store(vec, "async answer")
    result = await b.search(vec, threshold=0.85)
    assert result == "async answer"


@pytest.mark.asyncio
async def test_async_stats(b):
    vec = dummy_embed("async stats redis")
    await b.store(vec, "value")
    await b.search(vec, threshold=0.85)   # hit
    s = await b.stats()
    assert s["hits"] == 1


@pytest.mark.asyncio
async def test_async_clear(b):
    vec = dummy_embed("async clear test")
    await b.store(vec, "value")
    await b.clear()
    result = await b.search(vec, threshold=0.85)
    assert result is None
