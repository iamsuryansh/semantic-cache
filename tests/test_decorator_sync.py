"""
Tests for @cache.cached on plain `def` functions.
"""
import pytest

from semantic_cache import SemanticCache
from semantic_cache.backends.memory import MemoryBackend
from tests.conftest import dummy_embed, near_embed


@pytest.fixture
def cache():
    return SemanticCache(
        backend=MemoryBackend(),
        embed_fn=dummy_embed,
        threshold=0.85,
    )


# ------------------------------------------------------------------ basic hit / miss

def test_miss_calls_function(cache):
    call_count = 0

    @cache.cached
    def ask(prompt: str) -> str:
        nonlocal call_count
        call_count += 1
        return "answer"

    ask("What is Python?")
    assert call_count == 1


def test_second_identical_call_is_a_hit(cache):
    call_count = 0

    @cache.cached
    def ask(prompt: str) -> str:
        nonlocal call_count
        call_count += 1
        return "answer"

    ask("What is Python?")
    ask("What is Python?")
    assert call_count == 1   # second call served from cache


def test_similar_prompt_is_a_hit(cache):
    """near_embed simulates a paraphrase — should be served from cache."""
    responses = []

    @cache.cached
    def ask(prompt: str) -> str:
        return f"response for: {prompt}"

    r1 = ask("What is machine learning?")

    # Patch embed_fn to return a similar (but not identical) vector for the next call
    original_embed = cache._embed_fn
    cache._embed_fn = lambda t: near_embed("What is machine learning?", noise=0.03)

    r2 = ask("Explain machine learning to me")

    cache._embed_fn = original_embed  # restore

    assert r1 == r2


def test_dissimilar_prompt_is_a_miss(cache):
    call_count = 0

    @cache.cached
    def ask(prompt: str) -> str:
        nonlocal call_count
        call_count += 1
        return "answer"

    ask("What is Python?")
    ask("Who wrote Hamlet?")
    assert call_count == 2


# ------------------------------------------------------------------ key arg resolution

def test_default_key_resolution_prompt(cache):
    """Parameter named 'prompt' is used automatically."""
    call_count = 0

    @cache.cached
    def ask(prompt: str) -> str:
        nonlocal call_count
        call_count += 1
        return "ok"

    ask("hello")
    ask("hello")
    assert call_count == 1


def test_explicit_key_arg(cache):
    """@cache.cached(key_arg=...) overrides auto-resolution."""
    call_count = 0

    @cache.cached(key_arg="query")
    def search(query: str, top_k: int = 5) -> str:
        nonlocal call_count
        call_count += 1
        return "results"

    search("python tutorials", top_k=3)
    search("python tutorials", top_k=10)  # different top_k, same query → cache hit
    assert call_count == 1


def test_invalid_key_arg_raises_at_decoration_time(cache):
    with pytest.raises(ValueError, match="not found"):

        @cache.cached(key_arg="nonexistent")
        def ask(prompt: str) -> str:
            return "ok"


# ------------------------------------------------------------------ non-str return

def test_non_str_return_passes_through_uncached(cache):
    """Functions returning non-str should be called every time."""
    call_count = 0

    @cache.cached
    def ask(prompt: str) -> dict:
        nonlocal call_count
        call_count += 1
        return {"answer": "42"}

    ask("meaning of life?")
    ask("meaning of life?")
    assert call_count == 2          # function called both times — not cached
    assert cache.stats()["hits"] == 0   # never a hit, non-str is never stored


# ------------------------------------------------------------------ stats

def test_stats_reflect_hits_and_misses(cache):
    @cache.cached
    def ask(prompt: str) -> str:
        return "answer"

    ask("question one")     # miss
    ask("question one")     # hit
    ask("question two")     # miss

    s = cache.stats()
    assert s["misses"] == 2
    assert s["hits"] == 1
    assert s["total"] == 3


# ------------------------------------------------------------------ flush

def test_flush_clears_cache(cache):
    @cache.cached
    def ask(prompt: str) -> str:
        return "answer"

    call_count = 0

    @cache.cached
    def ask2(prompt: str) -> str:
        nonlocal call_count
        call_count += 1
        return "answer"

    ask2("hello")
    assert call_count == 1

    cache.flush()

    ask2("hello")
    assert call_count == 2   # cache was flushed — function called again


# ------------------------------------------------------------------ direct get/set

def test_direct_set_and_get(cache):
    cache.set("What is 2+2?", "4")
    result = cache.get("What is 2+2?")
    assert result == "4"


def test_get_returns_none_on_miss(cache):
    result = cache.get("something not cached")
    assert result is None


# ------------------------------------------------------------------ fail-open

def test_fail_open_on_broken_backend():
    """If the backend raises, the decorated function is still called normally."""
    from unittest.mock import MagicMock

    broken = MagicMock(spec=MemoryBackend)
    broken.search_sync.side_effect = RuntimeError("backend down")
    broken.store_sync.side_effect = RuntimeError("backend down")
    broken.stats_sync.return_value = {"hits": 0, "misses": 0, "hit_rate": 0.0, "total": 0}

    cache = SemanticCache(backend=broken, embed_fn=dummy_embed)
    call_count = 0

    @cache.cached
    def ask(prompt: str) -> str:
        nonlocal call_count
        call_count += 1
        return "answer"

    result = ask("test")
    assert result == "answer"   # function was called despite backend error
    assert call_count == 1      # no exception propagated
