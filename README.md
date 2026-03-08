# semantic-cache

Semantic caching for LLM calls via a decorator. Wraps any function that takes a string prompt and returns a string response — if a semantically similar prompt has been seen before, the cached response is returned instantly without calling the underlying function.

```python
@cache.cached
def ask(prompt: str) -> str:
    return openai_client.chat.completions.create(...).choices[0].message.content

ask("What is the capital of France?")      # MISS — calls your function (~800ms)
ask("Tell me the capital city of France")  # HIT  — returns from cache   (~5ms)
ask("What's France's capital?")            # HIT  — returns from cache   (~5ms)
```

---

## Table of Contents

1. [How it works](#how-it-works)
2. [Installation](#installation)
3. [Quickstart](#quickstart)
4. [Async support](#async-support)
5. [Backends](#backends)
6. [Embeddings](#embeddings)
7. [Configuration reference](#configuration-reference)
8. [Key argument resolution](#key-argument-resolution)
9. [Stats and flush](#stats-and-flush)
10. [Direct get / set](#direct-get--set)
11. [Fail-open behaviour](#fail-open-behaviour)
12. [Running Redis locally](#running-redis-locally)
13. [Common issues](#common-issues)

---

## How it works

1. The incoming prompt is embedded into a vector using a local embedding model.
2. The vector store is searched for the nearest cached vector using cosine similarity.
3. **Cache hit** — similarity meets the threshold → cached response is returned immediately.
4. **Cache miss** — your function is called, the response is stored for future use.

```
prompt
  │
  ├─ embed (local, ~15ms)
  │
  ├─ similarity search
  │   ├─ HIT  (score ≥ threshold) → return cached response
  │   └─ MISS                     → call your function → store result
  │
  └─ response
```

The library owns the embedding and the cache lookup. You own the LLM call.

---

## Installation

The core library has **zero mandatory dependencies**. Install only what you need.

```bash
# Local embeddings + in-memory cache (no Redis required — good for development)
pip install "semantic-cache[default]"

# Redis backend only (bring your own embedding function)
pip install "semantic-cache[redis]"

# Everything
pip install "semantic-cache[all]"
```

---

## Quickstart

### 1. In-memory cache (no setup required)

The simplest way to get started. The cache lives in process memory and resets when your program exits.

```python
from semantic_cache import SemanticCache

cache = SemanticCache()   # uses in-memory backend + local fastembed model

@cache.cached
def ask(prompt: str) -> str:
    # Replace with your actual LLM call
    return call_my_llm(prompt)

print(ask("What is machine learning?"))
# → MISS: your function is called, result stored

print(ask("Can you explain machine learning?"))
# → HIT: returned from cache without calling your function
```

### 2. Redis backend (for persistence across restarts)

```python
from semantic_cache import SemanticCache

cache = SemanticCache(redis_url="redis://localhost:6379")

@cache.cached
def ask(prompt: str) -> str:
    return call_my_llm(prompt)
```

The embedding dimensions are detected automatically. The Redis index is created on first use.

### 3. Custom embedding function

Use any embedder you already have — OpenAI, Cohere, Bedrock, or your own.

```python
import openai
from semantic_cache import SemanticCache

def embed(text: str) -> list[float]:
    response = openai.embeddings.create(
        model="text-embedding-3-small",
        input=text,
    )
    return response.data[0].embedding

cache = SemanticCache(
    redis_url="redis://localhost:6379",
    embed_fn=embed,
)

@cache.cached
def ask(prompt: str) -> str:
    return call_my_llm(prompt)
```

---

## Async support

`@cache.cached` works on `async def` functions with no changes to the decorator.

```python
import asyncio
from semantic_cache import SemanticCache

cache = SemanticCache()

@cache.cached
async def ask(prompt: str) -> str:
    return await call_my_llm_async(prompt)

async def main():
    r1 = await ask("What is the speed of light?")
    r2 = await ask("How fast does light travel?")   # hit
    print(r1 == r2)   # True

asyncio.run(main())
```

On a cache miss, the response is returned to the caller immediately. Writing to the cache happens in the background via `asyncio.create_task` so your await does not pay the storage cost.

---

## Backends

### `MemoryBackend` (default)

- No external dependencies
- Lives in process memory — resets on restart
- Thread-safe
- Good for: development, testing, short-lived scripts

```python
from semantic_cache import SemanticCache
from semantic_cache import MemoryBackend

# These two are equivalent:
cache = SemanticCache()
cache = SemanticCache(backend=MemoryBackend())
```

### `RedisBackend`

- Persists across restarts
- Shared across multiple processes / instances
- Requires `redis-stack-server` or any Redis with vector search enabled (ElastiCache OSS 7.1+)
- Good for: production, multi-instance deployments

```python
from semantic_cache import SemanticCache
from semantic_cache import RedisBackend

# Shorthand (dims detected automatically)
cache = SemanticCache(redis_url="redis://localhost:6379")

# Explicit (useful when dims are known ahead of time)
cache = SemanticCache(
    backend=RedisBackend(url="redis://localhost:6379", dims=384),
)
```

---

## Embeddings

### Default (fastembed, local)

Requires `pip install "semantic-cache[default]"`.

The model (`BAAI/bge-small-en-v1.5`, 384 dimensions) is loaded once and shared across all `SemanticCache` instances in the same process. No API key needed.

```python
cache = SemanticCache()   # fastembed loads automatically
```

### Custom embedding function

Pass any `(str) -> list[float]` callable as `embed_fn`. The function is called synchronously. For async functions, wrap it or use `run_in_executor` yourself.

```python
# Example: Amazon Bedrock
import boto3, json

bedrock = boto3.client("bedrock-runtime", region_name="us-east-1")

def embed(text: str) -> list[float]:
    body = json.dumps({"inputText": text, "dimensions": 256, "normalize": True})
    response = bedrock.invoke_model(modelId="amazon.titan-embed-text-v2:0", body=body)
    return json.loads(response["body"].read())["embedding"]

cache = SemanticCache(embed_fn=embed, redis_url="redis://localhost:6379")
```

```python
# Example: Cohere
import cohere

co = cohere.Client("your-api-key")

def embed(text: str) -> list[float]:
    return co.embed(texts=[text], model="embed-english-v3.0").embeddings[0]

cache = SemanticCache(embed_fn=embed)
```

When using a custom `embed_fn` with `RedisBackend`, dimensions are inferred automatically by calling `embed_fn("probe")` once at initialisation. You can skip this by creating the backend explicitly with the correct `dims`:

```python
cache = SemanticCache(
    backend=RedisBackend(url="redis://localhost:6379", dims=256),
    embed_fn=embed,
)
```

---

## Configuration reference

All options are keyword-only arguments to `SemanticCache`:

| Parameter | Type | Default | Description |
|---|---|---|---|
| `backend` | `CacheBackend` | `MemoryBackend()` | Explicit backend instance |
| `redis_url` | `str` | `None` | Shorthand to create a `RedisBackend` |
| `embed_fn` | `(str) -> list[float]` | `None` | Custom embedding function; defaults to fastembed |
| `threshold` | `float` | `0.85` | Minimum cosine similarity to count as a hit (0.0–1.0) |
| `ttl` | `int \| None` | `None` | Seconds before a cache entry expires; `None` = never |
| `namespace` | `str` | `"sc"` | Key prefix — isolates caches within a shared Redis instance |

### Threshold guidance

| Threshold | Effect |
|---|---|
| `0.95` | Only very close paraphrases hit. Rarely false-positives. |
| `0.85` | Default. Good balance for most LLM use cases. |
| `0.75` | Broader matching. May serve cached answers for loosely related prompts. |
| `< 0.70` | Usually too aggressive. Unrelated prompts may produce hits. |

### Example with all options set

```python
cache = SemanticCache(
    redis_url="redis://localhost:6379",
    embed_fn=my_embedder,
    threshold=0.90,
    ttl=3600,          # entries expire after 1 hour
    namespace="prod",  # keys stored as prod:vec:{uuid}
)
```

---

## Key argument resolution

The decorator needs to know which function parameter is the prompt to cache on. Resolution order:

1. **Explicit** — `@cache.cached(key_arg="param_name")`
2. **By name** — first parameter named `prompt`, `query`, `message`, or `text`
3. **By annotation** — first parameter annotated as `str`
4. **By position** — first positional parameter

Resolution happens at **decoration time**, not call time. If no suitable parameter is found, a `ValueError` is raised immediately when the decorator is applied.

```python
# Resolved automatically (parameter named "prompt")
@cache.cached
def ask(prompt: str) -> str: ...

# Resolved automatically (parameter named "query")
@cache.cached
def search(query: str, top_k: int = 5) -> str: ...

# Explicit override — caches on "message", ignores "history"
@cache.cached(key_arg="message")
def chat(message: str, history: list[dict]) -> str: ...

# Wrong key_arg — ValueError raised at decoration time, not at call time
@cache.cached(key_arg="typo")       # ← raises immediately
def ask(prompt: str) -> str: ...
```

---

## Stats and flush

### Stats

```python
# Sync
stats = cache.stats()

# Async
stats = await cache.astats()

# Output
{
    "hits":     42,
    "misses":   18,
    "hit_rate": 0.7,
    "total":    60,
}
```

### Flush (clear all cached entries)

```python
# Sync
cache.flush()

# Async
await cache.aflush()
```

Flush deletes all cache entries and resets hit/miss counters to zero.

---

## Direct get / set

Bypass the decorator to read or write cache entries manually.

```python
# Sync
cache.set("What is 2 + 2?", "4")
result = cache.get("What is 2 + 2?")   # "4"

# Async
await cache.aset("What is 2 + 2?", "4")
result = await cache.aget("What is 2 + 2?")   # "4"
```

`get` and `aget` return `None` on a miss.

---

## Fail-open behaviour

If the backend raises any exception during a lookup or store, the error is logged at `WARNING` level and the library fails open — the wrapped function is called as normal and the user receives a response. Cache write failures are silently dropped.

```
WARNING  semantic_cache.cache: SemanticCache.get failed (fail-open): Connection refused
```

This means a cache outage never causes your application to stop working.

---

## Running Redis locally

`RedisBackend` requires Redis with vector search support. The easiest way to run it locally is with Docker:

```bash
docker run -d -p 6379:6379 redis/redis-stack-server:latest
```

This starts `redis-stack-server`, which includes the RediSearch module needed for vector indexing. Plain `redis:latest` does **not** include RediSearch and will raise an error on index creation.

Verify it is running:

```bash
redis-cli ping   # PONG
```

---

## Common issues

| Problem | Fix |
|---|---|
| `ImportError: fastembed is required` | Run `pip install "semantic-cache[default]"` |
| `ModuleNotFoundError: redis` | Run `pip install "semantic-cache[redis]"` |
| `ResponseError: unknown command FT.CREATE` | Use `redis/redis-stack-server`, not plain `redis` |
| All queries are misses | Lower `threshold` (try `0.75`) — your phrasings may be more varied than expected |
| Unrelated questions are hitting | Raise `threshold` (try `0.92`) |
| `ValueError: @cached: cannot find a string parameter` | Add `@cache.cached(key_arg="param_name")` explicitly |
| Cache entries disappear unexpectedly | Check your `ttl` setting — `None` means no expiry |
| Stats reset after restart (MemoryBackend) | Switch to `RedisBackend` — it persists stats as Redis counters |
