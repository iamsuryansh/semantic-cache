"""
Tests for @cache.cached on `async def` functions.
"""
import asyncio

import pytest

from semantic_cache import SemanticCache
from semantic_cache.backends.memory import MemoryBackend
from tests.conftest import dummy_embed


@pytest.fixture
def cache():
    return SemanticCache(
        backend=MemoryBackend(),
        embed_fn=dummy_embed,
        threshold=0.85,
    )


# ------------------------------------------------------------------ basic hit / miss

@pytest.mark.asyncio
async def test_miss_calls_async_function(cache):
    call_count = 0

    @cache.cached
    async def ask(prompt: str) -> str:
        nonlocal call_count
        call_count += 1
        return "async answer"

    await ask("What is asyncio?")
    assert call_count == 1


@pytest.mark.asyncio
async def test_second_identical_call_is_a_hit(cache):
    call_count = 0

    @cache.cached
    async def ask(prompt: str) -> str:
        nonlocal call_count
        call_count += 1
        return "async answer"

    await ask("What is asyncio?")
    # The cache write is a background task — yield control so it completes
    await asyncio.sleep(0)
    await ask("What is asyncio?")

    assert call_count == 1


@pytest.mark.asyncio
async def test_dissimilar_prompt_is_a_miss(cache):
    call_count = 0

    @cache.cached
    async def ask(prompt: str) -> str:
        nonlocal call_count
        call_count += 1
        return "answer"

    await ask("What is Python?")
    await asyncio.sleep(0)
    await ask("Who wrote Hamlet?")

    assert call_count == 2


# ------------------------------------------------------------------ key arg

@pytest.mark.asyncio
async def test_explicit_key_arg_async(cache):
    call_count = 0

    @cache.cached(key_arg="query")
    async def search(query: str, top_k: int = 5) -> str:
        nonlocal call_count
        call_count += 1
        return "results"

    await search("python tutorials", top_k=3)
    await asyncio.sleep(0)
    await search("python tutorials", top_k=10)

    assert call_count == 1


# ------------------------------------------------------------------ stats / flush

@pytest.mark.asyncio
async def test_astats(cache):
    @cache.cached
    async def ask(prompt: str) -> str:
        return "answer"

    await ask("q1")               # miss
    await asyncio.sleep(0)
    await ask("q1")               # hit
    await ask("q2")               # miss

    s = await cache.astats()
    assert s["misses"] == 2
    assert s["hits"] == 1


@pytest.mark.asyncio
async def test_aflush_clears_cache(cache):
    call_count = 0

    @cache.cached
    async def ask(prompt: str) -> str:
        nonlocal call_count
        call_count += 1
        return "answer"

    await ask("hello")
    # sleep(0.05): the background store uses run_in_executor; sleep(0) (one
    # event-loop tick) isn't always enough for the thread callback to land.
    # 50ms gives the pool time to complete before we flush.
    await asyncio.sleep(0.05)
    assert call_count == 1

    await cache.aflush()

    await ask("hello")
    assert call_count == 2


# ------------------------------------------------------------------ direct aget/aset

@pytest.mark.asyncio
async def test_direct_aset_and_aget(cache):
    await cache.aset("What is 2+2?", "4")
    result = await cache.aget("What is 2+2?")
    assert result == "4"


@pytest.mark.asyncio
async def test_aget_returns_none_on_miss(cache):
    result = await cache.aget("not in cache")
    assert result is None


# ------------------------------------------------------------------ fail-open

@pytest.mark.asyncio
async def test_fail_open_on_broken_async_backend():
    from unittest.mock import AsyncMock, MagicMock

    broken = MagicMock(spec=MemoryBackend)
    broken.search = AsyncMock(side_effect=RuntimeError("backend down"))
    broken.store  = AsyncMock(side_effect=RuntimeError("backend down"))
    broken.stats  = AsyncMock(return_value={"hits": 0, "misses": 0, "hit_rate": 0.0, "total": 0})

    cache = SemanticCache(backend=broken, embed_fn=dummy_embed)
    call_count = 0

    @cache.cached
    async def ask(prompt: str) -> str:
        nonlocal call_count
        call_count += 1
        return "answer"

    result = await ask("test")
    assert result == "answer"
    assert call_count == 1
