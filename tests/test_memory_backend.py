"""
Tests for MemoryBackend in isolation — no SemanticCache involved.
"""
import time

import pytest

from semantic_cache.backends.memory import MemoryBackend
from tests.conftest import dummy_embed, near_embed


@pytest.fixture
def b():
    backend = MemoryBackend()
    yield backend
    backend.clear_sync()


# ------------------------------------------------------------------ search

def test_miss_on_empty_backend(b):
    result = b.search_sync(dummy_embed("hello"), threshold=0.85)
    assert result is None


def test_hit_on_identical_vector(b):
    vec = dummy_embed("What is the speed of light?")
    b.store_sync(vec, "299,792,458 m/s")

    result = b.search_sync(vec, threshold=0.85)
    assert result == "299,792,458 m/s"


def test_hit_on_similar_vector(b):
    """A near-paraphrase embedding should still hit above the default threshold."""
    original = dummy_embed("What is machine learning?")
    similar  = near_embed("What is machine learning?", noise=0.03)

    b.store_sync(original, "ML is a subset of AI.")
    result = b.search_sync(similar, threshold=0.85)
    assert result == "ML is a subset of AI."


def test_miss_on_dissimilar_vector(b):
    b.store_sync(dummy_embed("What is the speed of light?"), "299,792,458 m/s")

    result = b.search_sync(dummy_embed("Who wrote Hamlet?"), threshold=0.85)
    assert result is None


def test_returns_best_match_among_multiple_entries(b):
    b.store_sync(dummy_embed("capital of France"), "Paris")
    b.store_sync(dummy_embed("capital of Germany"), "Berlin")

    # Querying with the exact France vector should return Paris
    result = b.search_sync(dummy_embed("capital of France"), threshold=0.85)
    assert result == "Paris"


# ------------------------------------------------------------------ TTL

def test_expired_entry_is_not_returned(b):
    vec = dummy_embed("expiry test")
    b.store_sync(vec, "should expire", ttl=1)

    time.sleep(1.1)
    result = b.search_sync(vec, threshold=0.85)
    assert result is None


def test_non_expired_entry_is_returned(b):
    vec = dummy_embed("not expired")
    b.store_sync(vec, "still valid", ttl=60)

    result = b.search_sync(vec, threshold=0.85)
    assert result == "still valid"


# ------------------------------------------------------------------ stats

def test_stats_empty(b):
    s = b.stats_sync()
    assert s == {"hits": 0, "misses": 0, "hit_rate": 0.0, "total": 0}


def test_stats_after_hit_and_miss(b):
    vec = dummy_embed("stats test")
    b.store_sync(vec, "cached answer")

    b.search_sync(vec, threshold=0.85)                       # hit
    b.search_sync(dummy_embed("unrelated"), threshold=0.85)  # miss

    s = b.stats_sync()
    assert s["hits"] == 1
    assert s["misses"] == 1
    assert s["total"] == 2
    assert s["hit_rate"] == 0.5


# ------------------------------------------------------------------ clear

def test_clear_removes_entries_and_resets_stats(b):
    vec = dummy_embed("clear test")
    b.store_sync(vec, "answer")
    b.search_sync(vec, threshold=0.85)  # hit

    b.clear_sync()

    # Check stats reset BEFORE doing another search (which would add a miss)
    assert b.stats_sync()["total"] == 0
    # Then verify the entry itself is gone
    assert b.search_sync(vec, threshold=0.85) is None


# ------------------------------------------------------------------ async

@pytest.mark.asyncio
async def test_async_hit(b):
    vec = dummy_embed("async hit test")
    await b.store(vec, "async answer")
    result = await b.search(vec, threshold=0.85)
    assert result == "async answer"


@pytest.mark.asyncio
async def test_async_stats(b):
    vec = dummy_embed("async stats")
    await b.store(vec, "value")
    await b.search(vec, threshold=0.85)   # hit
    s = await b.stats()
    assert s["hits"] == 1
