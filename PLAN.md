# SemanticCache — Library Implementation Plan

A configurable Python library that adds semantic caching to any function via a decorator. Wraps any LLM call (or any expensive function that takes text and returns text) and returns cached responses for semantically similar inputs.

---

## What This Library Does

```python
from semantic_cache import SemanticCache

cache = SemanticCache(redis_url="redis://localhost:6379", threshold=0.85)

@cache.cached
def ask(prompt: str) -> str:
    return openai_client.chat.completions.create(...).choices[0].message.content

ask("What is the capital of France?")   # MISS — calls the function
ask("Tell me the capital city of France")  # HIT  — returns from cache instantly
ask("What's France's capital?")             # HIT  — returns from cache instantly
```

The library handles embedding, similarity search, storage, and stats. The user owns the LLM call entirely.

---

## Design Principles

- **Zero mandatory dependencies** — the library core installs with nothing. Users opt in to fastembed, Redis, or numpy via extras.
- **Bring your own embedder** — any callable `(str) -> list[float]` works. Default is fastembed if installed.
- **Two backends** — `MemoryBackend` (in-process, zero deps, good for dev/testing) and `RedisBackend` (Redis Stack/ElastiCache, for production).
- **Decorator handles sync and async transparently** — `@cache.cached` works on both `def` and `async def` without any change to usage.
- **Fail-open always** — if the cache is unreachable for any reason, the wrapped function is called as normal. The user is never blocked.
- **No opinions on the LLM** — the library does not import or call any LLM API. The wrapped function is a black box.

---

## Public API

### Instantiation

```python
# Zero-config (in-memory backend, requires fastembed)
cache = SemanticCache()

# Redis backend (shorthand)
cache = SemanticCache(redis_url="redis://localhost:6379")

# Full control
cache = SemanticCache(
    backend=RedisBackend(url="redis://localhost:6379", namespace="myapp"),
    embed_fn=my_embedder,       # sync: (str) -> list[float]
    threshold=0.85,
    ttl=3600,
)
```

### Decorator — sync

```python
@cache.cached
def ask(prompt: str) -> str:
    ...

# Explicit key arg
@cache.cached(key_arg="query")
def search(query: str, top_k: int = 5) -> str:
    ...
```

### Decorator — async

```python
@cache.cached
async def ask(prompt: str) -> str:
    ...

@cache.cached(key_arg="message")
async def chat(message: str, history: list[dict]) -> str:
    ...
```

### Stats

```python
cache.stats()        # sync
await cache.astats() # async

# {"hits": 42, "misses": 18, "hit_rate": 0.70, "total": 60}
```

### Flush

```python
cache.flush()        # sync
await cache.aflush() # async
```

### Direct lookup / store (power users)

```python
result = cache.get(prompt)           # returns str | None
await cache.aget(prompt)

cache.set(prompt, response)
await cache.aset(prompt, response)
```

---

## Project Structure

```
semantic_cache/
├── semantic_cache/
│   ├── __init__.py              # Public exports
│   ├── cache.py                 # SemanticCache class + @cached decorator
│   ├── backends/
│   │   ├── __init__.py          # Re-exports MemoryBackend, RedisBackend
│   │   ├── base.py              # CacheBackend abstract base class
│   │   ├── memory.py            # In-process backend (no deps)
│   │   └── redis_backend.py     # Redis Stack / ElastiCache backend
│   └── embeddings.py            # DefaultEmbedder (fastembed) + type aliases
├── tests/
│   ├── conftest.py
│   ├── test_decorator_sync.py
│   ├── test_decorator_async.py
│   ├── test_memory_backend.py
│   ├── test_redis_backend.py    # skipped if Redis not available
│   └── test_embeddings.py
├── pyproject.toml
└── README.md
```

---

## Implementation

---

### Step 1 — `semantic_cache/backends/base.py`

Abstract base class that both backends implement. Defines the full interface for sync and async operations.

```python
from abc import ABC, abstractmethod

class CacheBackend(ABC):

    # --- Async interface (preferred) ---

    @abstractmethod
    async def search(
        self, vector: list[float], threshold: float
    ) -> tuple[str, float] | None:
        """
        Find the most similar cached entry.
        Returns (response, similarity_score) if similarity >= threshold, else None.
        """

    @abstractmethod
    async def store(
        self, vector: list[float], response: str, ttl: int | None = None
    ) -> None:
        """Store a new vector-response pair."""

    @abstractmethod
    async def clear(self) -> None:
        """Delete all cached entries and reset stats."""

    @abstractmethod
    async def stats(self) -> dict:
        """Return {"hits": int, "misses": int, "hit_rate": float, "total": int}."""

    @abstractmethod
    async def incr_hit(self) -> None: ...

    @abstractmethod
    async def incr_miss(self) -> None: ...

    # --- Sync interface ---

    @abstractmethod
    def search_sync(
        self, vector: list[float], threshold: float
    ) -> tuple[str, float] | None: ...

    @abstractmethod
    def store_sync(
        self, vector: list[float], response: str, ttl: int | None = None
    ) -> None: ...

    @abstractmethod
    def clear_sync(self) -> None: ...

    @abstractmethod
    def stats_sync(self) -> dict: ...

    @abstractmethod
    def incr_hit_sync(self) -> None: ...

    @abstractmethod
    def incr_miss_sync(self) -> None: ...
```

All methods must be safe to call repeatedly (idempotent where applicable). Both `search` and `search_sync` return the **response string** (not the vector), and the similarity score for informational use.

---

### Step 2 — `semantic_cache/backends/memory.py`

In-process backend. No dependencies beyond the standard library. Good for tests, local development, and scripts where Redis is not available.

**Storage:**
```python
@dataclass
class _Entry:
    vector: list[float]
    response: str
    created_at: float
    expires_at: float | None   # None means no TTL

_entries: list[_Entry]         # append-only; search is full linear scan
_hits: int
_misses: int
_lock: threading.Lock          # protects _entries for thread-safe sync usage
```

**Cosine similarity (pure Python fallback, numpy fast path):**
```python
def _cosine(a: list[float], b: list[float]) -> float:
    try:
        import numpy as np
        av, bv = np.array(a, dtype=np.float32), np.array(b, dtype=np.float32)
        denom = np.linalg.norm(av) * np.linalg.norm(bv)
        return float(np.dot(av, bv) / denom) if denom else 0.0
    except ImportError:
        # Pure Python fallback — correct but slower
        dot = sum(x * y for x, y in zip(a, b))
        mag = (sum(x*x for x in a)**0.5) * (sum(x*x for x in b)**0.5)
        return dot / mag if mag else 0.0
```

**`search_sync`:** iterate `_entries`, skip expired entries (check `expires_at`), compute cosine similarity for each, return the `(response, score)` pair with the highest score if it meets the threshold. Thread-safe via `_lock`.

**`search` (async):** acquires `asyncio.Lock` equivalent or delegates to a thread executor — since `MemoryBackend` has no I/O, the async methods simply call the sync methods directly (no executor needed).

**TTL enforcement:** checked lazily on `search` (skip expired entries) and periodically on `store` (trim entries whose `expires_at` has passed to avoid unbounded growth).

**`clear_sync`:** sets `_entries = []`, `_hits = 0`, `_misses = 0`.

---

### Step 3 — `semantic_cache/backends/redis_backend.py`

Redis backend using RediSearch vector index (HNSW). Requires `redis-stack-server` locally or ElastiCache with VSS enabled on AWS.

**Two clients, one backend:**
```python
class RedisBackend(CacheBackend):
    def __init__(self, url: str, namespace: str = "sc", dims: int = 256):
        self._url = url
        self._ns  = namespace
        self._dims = dims
        # Sync client — used by sync decorator path
        self._sync = redis.Redis.from_url(url, decode_responses=False)
        # Async client — used by async decorator path
        self._async: redis.asyncio.Redis | None = None   # lazy-init on first async call
```

Async client is lazy-initialised to avoid creating an event loop at import time.

**Index creation (`_ensure_index` / `_async_ensure_index`):**
```
FT.CREATE {ns}:idx ON HASH PREFIX 1 {ns}:vec:
  SCHEMA
    vec   VECTOR HNSW 6 TYPE FLOAT32 DIM {dims} DISTANCE_METRIC COSINE
    resp  TEXT NOSTEM
```
Catch `redis.exceptions.ResponseError` with message "Index already exists" — do not raise. Called once at first use.

**`store_sync`:**
```python
key = f"{ns}:vec:{uuid4()}"
pipe = client.pipeline()
pipe.hset(key, mapping={
    "vec":  np.array(vector, dtype=np.float32).tobytes(),
    "resp": response,
})
if ttl:
    pipe.expire(key, ttl)
pipe.execute()
```

**`search_sync`:**
```python
query = (
    f"*=>[KNN 1 @vec $BLOB AS score]"
)
result = client.ft(f"{ns}:idx").search(
    query,
    query_params={"BLOB": np.array(vector, dtype=np.float32).tobytes()},
)
# RediSearch returns distance (1 - similarity) in the "score" field
# Convert: similarity = 1 - distance
# Return (response, similarity) if similarity >= threshold
```

**Stats:** stored as Redis keys `{ns}:hits` and `{ns}:misses`, incremented with `INCR`. Simple, atomic, survives restarts.

**`clear_sync`:** scan for all `{ns}:vec:*` keys, delete in batches of 100 using UNLINK (non-blocking delete). Reset `{ns}:hits` and `{ns}:misses` to 0.

**Async methods:** mirror the sync methods but use `self._async` client and `await`.

---

### Step 4 — `semantic_cache/embeddings.py`

Embedding abstraction layer.

```python
# Type aliases
EmbedFn  = Callable[[str], list[float]]            # sync
AEmbedFn = Callable[[str], Awaitable[list[float]]] # async

class DefaultEmbedder:
    """
    Local embedder using fastembed (ONNX, no API key required).
    Model loaded once as a class-level singleton — shared across all
    SemanticCache instances in the same process.
    """
    _model = None   # class-level singleton

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
        if DefaultEmbedder._model is None:
            try:
                from fastembed import TextEmbedding
                DefaultEmbedder._model = TextEmbedding(model_name)
            except ImportError:
                raise ImportError(
                    "fastembed is required for the default embedder. "
                    "Install it with: pip install semantic-cache[default]"
                )

    def embed(self, text: str) -> list[float]:
        return list(DefaultEmbedder._model.embed([text]))[0].tolist()

    async def aembed(self, text: str) -> list[float]:
        # fastembed is CPU-bound ONNX — run in thread executor to avoid blocking
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.embed, text)
```

**Embedding dims by model** (for index creation):
```python
KNOWN_DIMS = {
    "BAAI/bge-small-en-v1.5": 384,
    "BAAI/bge-base-en-v1.5":  768,
    "sentence-transformers/all-MiniLM-L6-v2": 384,
}
```

If the user provides a custom `embed_fn`, they must also pass `dims` explicitly (or the library infers it by calling `embed_fn("probe")` once and measuring the length at init time).

---

### Step 5 — `semantic_cache/cache.py`

`SemanticCache` — the main class. Wires everything together and implements the decorator.

#### Constructor

```python
class SemanticCache:
    def __init__(
        self,
        *,
        # Backend — one of these three forms:
        backend: CacheBackend | None = None,   # explicit backend instance
        redis_url: str | None = None,           # shorthand: creates a RedisBackend
        # (if neither is set, creates a MemoryBackend)

        # Embedding
        embed_fn:  EmbedFn  | None = None,     # custom sync embedder
        aembed_fn: AEmbedFn | None = None,     # custom async embedder (optional upgrade)
        embedding_model: str = "BAAI/bge-small-en-v1.5",  # for default fastembed

        # Cache behaviour
        threshold: float = 0.85,               # cosine similarity cutoff (0.0 – 1.0)
        ttl: int | None = None,                # seconds; None = no expiry
        namespace: str = "sc",                 # key prefix (multi-tenant isolation)
    ):
```

**Precedence rules:**
- If `backend` is given, use it directly.
- Else if `redis_url` is given, create `RedisBackend(url=redis_url, namespace=namespace)`.
- Else create `MemoryBackend(namespace=namespace)`.

**Embedding precedence:**
- If `aembed_fn` is given, use it for async path; use `embed_fn` (or a sync wrapper) for sync path.
- If only `embed_fn` is given, use it for sync; wrap in `run_in_executor` for async.
- If neither is given, use `DefaultEmbedder` (fastembed).

**Dims inference:** call `embed_fn("probe")` once at init and measure `len(result)`. Cache result as `self._dims`. Used when creating the Redis index.

#### `@cached` decorator

```python
def cached(self, fn=None, *, key_arg: str | None = None):
    """
    Works in three forms:
        @cache.cached                        — no arguments
        @cache.cached(key_arg="my_param")   — with arguments
        cache.cached(fn)                    — programmatic
    """
    if fn is None:
        # Called with arguments: @cache.cached(key_arg=...)
        return functools.partial(self.cached, key_arg=key_arg)

    # Resolve which parameter is the cache key
    _key_param = _resolve_key_param(fn, key_arg)  # raises ValueError if not found

    if asyncio.iscoroutinefunction(fn):
        @functools.wraps(fn)
        async def async_wrapper(*args, **kwargs):
            prompt = _extract_key(_key_param, args, kwargs, fn)
            result = await self._aget(prompt)
            if result is not None:
                return result
            response = await fn(*args, **kwargs)
            if isinstance(response, str):
                # Fire-and-forget store — don't make user wait for cache write
                asyncio.create_task(self._aset(prompt, response))
            return response
        return async_wrapper
    else:
        @functools.wraps(fn)
        def sync_wrapper(*args, **kwargs):
            prompt = _extract_key(_key_param, args, kwargs, fn)
            result = self._get(prompt)
            if result is not None:
                return result
            response = fn(*args, **kwargs)
            if isinstance(response, str):
                self._set(prompt, response)  # synchronous — store before returning
            return response
        return sync_wrapper
```

**`_resolve_key_param(fn, key_arg)` logic:**
1. If `key_arg` is given, verify the parameter exists in `fn`'s signature → use it
2. Else scan `fn`'s parameters in order: first one named `prompt`, `query`, `message`, or `text`
3. Else the first parameter annotated as `str` (or with no annotation and positional)
4. If nothing found → raise `ValueError` at decoration time with a clear message

**Internal `_get` / `_aget`:**
```python
def _get(self, prompt: str) -> str | None:
    try:
        vector = self._embed(prompt)
        result = self._backend.search_sync(vector, self._threshold)
        if result:
            self._backend.incr_hit_sync()
            return result[0]   # response string
        self._backend.incr_miss_sync()
        return None
    except Exception as e:
        log.warning("SemanticCache lookup failed (fail-open): %s", e)
        return None

async def _aget(self, prompt: str) -> str | None:
    try:
        vector = await self._aembed(prompt)
        result = await self._backend.search(vector, self._threshold)
        if result:
            await self._backend.incr_hit()
            return result[0]
        await self._backend.incr_miss()
        return None
    except Exception as e:
        log.warning("SemanticCache lookup failed (fail-open): %s", e)
        return None
```

**`stats()` / `astats()`:**
```python
def stats(self) -> dict:
    return self._backend.stats_sync()

async def astats(self) -> dict:
    return await self._backend.stats()
```

**`flush()` / `aflush()`:**
```python
def flush(self) -> None:
    self._backend.clear_sync()

async def aflush(self) -> None:
    await self._backend.clear()
```

**`get()` / `set()` / `aget()` / `aset()` (direct access):**
For power users who want to bypass the decorator and manage cache entries manually.

---

### Step 6 — `semantic_cache/__init__.py`

```python
from .cache import SemanticCache
from .backends import MemoryBackend, RedisBackend

__all__ = ["SemanticCache", "MemoryBackend", "RedisBackend"]
__version__ = "0.1.0"
```

Users only need to import from `semantic_cache`. They should never need to import from submodules.

---

### Step 7 — `pyproject.toml`

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "semantic-cache"
version = "0.1.0"
description = "Semantic caching for LLM calls via a simple decorator"
requires-python = ">=3.10"
dependencies = []   # zero mandatory dependencies

[project.optional-dependencies]
# pip install semantic-cache[default]     -- local fastembed embedder + in-memory backend
default = [
    "fastembed>=0.3",
    "numpy>=1.24",
]
# pip install semantic-cache[redis]       -- Redis backend
redis = [
    "redis>=5.0",
    "numpy>=1.24",
]
# pip install semantic-cache[all]         -- everything
all = [
    "fastembed>=0.3",
    "redis>=5.0",
    "numpy>=1.24",
]
# pip install semantic-cache[dev]         -- dev + test tools
dev = [
    "fastembed>=0.3",
    "redis>=5.0",
    "numpy>=1.24",
    "pytest>=8.0",
    "pytest-asyncio>=0.23",
    "ruff>=0.4",
    "mypy>=1.9",
]

[project.urls]
Homepage = "https://github.com/yourname/semantic-cache"

[tool.pytest.ini_options]
asyncio_mode = "auto"

[tool.ruff.lint]
select = ["E", "F", "I"]
```

**Zero mandatory dependencies** is the headline. A user who provides their own `embed_fn` (e.g. calling OpenAI or Bedrock themselves) and uses `MemoryBackend` needs nothing beyond the standard library.

---

### Step 8 — Tests

#### `tests/conftest.py`

```python
import pytest

@pytest.fixture
def memory_backend():
    from semantic_cache.backends import MemoryBackend
    b = MemoryBackend()
    yield b
    b.clear_sync()

@pytest.fixture
def redis_backend():
    # Skip if Redis is not available
    pytest.importorskip("redis")
    from semantic_cache.backends import RedisBackend
    try:
        b = RedisBackend(url="redis://localhost:6379", namespace="test")
        b._sync.ping()   # raises if not reachable
    except Exception:
        pytest.skip("Redis not available")
    yield b
    b.clear_sync()

def dummy_embed(text: str) -> list[float]:
    """Deterministic fake embedder for tests — no fastembed required."""
    import hashlib, struct
    seed = int(hashlib.md5(text.encode()).hexdigest()[:8], 16)
    rng = __import__("random").Random(seed)
    raw = [rng.gauss(0, 1) for _ in range(16)]
    mag = sum(x*x for x in raw) ** 0.5
    return [x / mag for x in raw]   # L2-normalised, dim=16

def similar_embed(base: str, noise: float = 0.05) -> callable:
    """Returns an embedder that produces a slightly perturbed version of base."""
    base_vec = dummy_embed(base)
    def _embed(text: str) -> list[float]:
        import random
        perturbed = [x + random.gauss(0, noise) for x in base_vec]
        mag = sum(x*x for x in perturbed) ** 0.5
        return [x / mag for x in perturbed]
    return _embed
```

#### `tests/test_memory_backend.py`

Cover:
- Store one entry, search with same vector → hit (score = 1.0)
- Search with dissimilar vector → miss
- Search with similar vector above threshold → hit
- Search with similar vector below threshold → miss
- TTL: store with ttl=1, sleep 2s, search → miss (entry expired)
- Stats: hits and misses count correctly after mixed searches
- Clear: after clear, search returns miss; stats reset to 0
- Thread safety: 10 threads each storing and searching concurrently, no exceptions

#### `tests/test_decorator_sync.py`

```python
def test_cache_hit_on_similar_prompt(memory_backend):
    call_count = 0
    cache = SemanticCache(backend=memory_backend, embed_fn=dummy_embed, threshold=0.9)

    @cache.cached
    def ask(prompt: str) -> str:
        nonlocal call_count
        call_count += 1
        return f"answer to: {prompt}"

    r1 = ask("What is quantum entanglement?")
    r2 = ask("What is quantum entanglement?")    # exact same prompt
    assert call_count == 1
    assert r1 == r2

def test_cache_miss_on_different_prompt(memory_backend):
    # Distinct embeddings → miss regardless of threshold
    ...

def test_key_arg_resolution(memory_backend):
    # @cache.cached(key_arg="query") — uses "query" param not "prompt"
    ...

def test_non_str_return_not_cached(memory_backend):
    # function returning dict → passes through, no cache entry stored
    ...

def test_fail_open_on_backend_error():
    # backend raises on search_sync → function is called, no exception propagates
    ...

def test_stats_accuracy(memory_backend):
    # n calls with m hits → stats() returns correct counts
    ...
```

#### `tests/test_decorator_async.py`

Mirror of sync tests but with `async def ask(...)` and `await ask(...)`. Also cover:
- `asyncio.create_task` store is actually executed (yield control to allow task to run)
- `astats()` and `aflush()` work correctly
- Concurrent async calls to the same prompt don't cause double-stores

#### `tests/test_redis_backend.py`

Skipped if Redis not running. Covers:
- Same cases as `test_memory_backend.py` but through the real Redis client
- Index created idempotently (call `_ensure_index` twice, no error)
- Stats survive backend restart (Redis counters persist)
- Async methods (`search`, `store`, `clear`, `stats`) all pass

#### `tests/test_embeddings.py`

- `DefaultEmbedder.embed("hello")` returns a list of floats with the correct length
- Same input always returns same output (deterministic)
- Class-level singleton: two `DefaultEmbedder` instances share the model object, only one load
- `aembed` returns same result as `embed` (async wrapper parity)

---

### Step 9 — `README.md`

Sections:

1. **What it is** — one paragraph
2. **Install**
   ```bash
   pip install semantic-cache[default]   # local embeddings + in-memory backend
   pip install semantic-cache[redis]     # Redis backend (bring your own embedder)
   pip install semantic-cache[all]       # everything
   ```
3. **Quickstart (sync)**
4. **Quickstart (async)**
5. **Backends** — MemoryBackend vs RedisBackend, when to use each
6. **Embeddings** — default fastembed, custom embed_fn example
7. **Configuration options** — full constructor args table
8. **Key arg resolution** — how the decorator finds the cache key
9. **Stats and flush**
10. **Fail-open behaviour** — what happens when the backend is unreachable
11. **Running Redis locally** (one Docker command)
12. **Common issues** table

---

## Implementation Order

```
1.  semantic_cache/backends/base.py          — abstract base class
2.  semantic_cache/backends/memory.py        — in-memory backend
3.  tests/conftest.py                        — dummy embedder + fixtures
4.  tests/test_memory_backend.py             — validate backend before wiring decorator
5.  [ pytest tests/test_memory_backend.py — all pass ]
6.  semantic_cache/embeddings.py             — DefaultEmbedder (fastembed wrapper)
7.  tests/test_embeddings.py
8.  [ pytest tests/test_embeddings.py — all pass ]
9.  semantic_cache/cache.py                  — SemanticCache + @cached decorator
10. semantic_cache/__init__.py
11. tests/test_decorator_sync.py
12. tests/test_decorator_async.py
13. [ pytest tests/ -k "not redis" — all pass ]
14. semantic_cache/backends/redis_backend.py — Redis backend
15. tests/test_redis_backend.py
16. [ docker run -p 6379:6379 redis/redis-stack-server:latest ]
17. [ pytest tests/ — all pass including Redis tests ]
18. pyproject.toml                           — finalise extras + metadata
19. README.md
20. [ pip install -e .[all] && run README examples manually ]
21. [ ruff check . && mypy semantic_cache/ ]
```

---

## Behaviour Reference

### Threshold guidance

| Threshold | Behaviour |
|---|---|
| `0.95+` | Only matches very close paraphrases |
| `0.85` | Default — good balance for most LLM use cases |
| `0.75` | Broader matching — may return cached answers for loosely related questions |
| `< 0.70` | Usually too aggressive — unrelated questions may hit |

### Key arg resolution order

Given `@cache.cached` on a function `fn`:

1. `key_arg` explicitly passed to the decorator
2. Parameter named `prompt`, `query`, `message`, or `text` (first match)
3. First positional parameter with a `str` annotation
4. First positional parameter with no annotation
5. Nothing found → `ValueError` raised at **decoration time** (not call time)

### What gets cached

Only `str` return values are cached. If the wrapped function returns a non-string, the value is returned to the caller unchanged and nothing is stored. A `DEBUG`-level log entry is emitted.

### Async store behaviour

For `async def` wrapped functions, the cache write (`_aset`) is scheduled as an `asyncio.create_task`. This means:
- The response is returned to the caller immediately without waiting for the store
- If the event loop is closed before the task runs (e.g. in a short script), the store may not complete
- For scripts rather than long-running servers, call `await cache.aset(prompt, response)` manually instead

For `def` (sync) wrapped functions, the store is synchronous and completes before returning.

### Fail-open scope

Any exception raised inside `_get`, `_aget`, `_set`, or `_aset` is caught and logged at `WARNING` level. The wrapped function is always called on a lookup failure. Cache writes that fail are silently dropped (the user already has their response).
