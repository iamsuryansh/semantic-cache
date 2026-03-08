import asyncio
import functools
import inspect
import logging
from typing import Callable

from .backends.base import CacheBackend
from .backends.memory import MemoryBackend
from .backends.redis_backend import RedisBackend
from .embeddings import EmbedFn, FastEmbedder

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
#  Helpers                                                                    #
# --------------------------------------------------------------------------- #

def _find_key_param(fn: Callable, key_arg: str | None) -> str:
    """
    Resolve which parameter of `fn` holds the cache key (the prompt string).

    Resolution order:
      1. Explicit key_arg passed to @cache.cached(key_arg="...")
      2. Parameter named prompt / query / message / text
      3. First parameter with a str annotation
      4. First positional parameter (annotation-free)
    Raises ValueError at decoration time if nothing is found.
    """
    sig = inspect.signature(fn)
    params = list(sig.parameters.values())

    if key_arg:
        if key_arg not in sig.parameters:
            raise ValueError(
                f"@cached: '{key_arg}' not found in '{fn.__name__}' signature. "
                f"Available params: {[p.name for p in params]}"
            )
        return key_arg

    for name in ("prompt", "query", "message", "text"):
        if name in sig.parameters:
            return name

    for param in params:
        if param.annotation is str:
            return param.name

    if params:
        return params[0].name

    raise ValueError(
        f"@cached: cannot find a string parameter in '{fn.__name__}'. "
        f"Use @cache.cached(key_arg='param_name') to specify it explicitly."
    )


def _extract_key(param_name: str, fn: Callable, args: tuple, kwargs: dict) -> str:
    """Bind call arguments to the function signature and pull out the key value."""
    sig = inspect.signature(fn)
    bound = sig.bind(*args, **kwargs)
    bound.apply_defaults()
    value = bound.arguments.get(param_name)
    if not isinstance(value, str):
        raise TypeError(
            f"@cached: key parameter '{param_name}' must be a str, "
            f"got {type(value).__name__}"
        )
    return value


# --------------------------------------------------------------------------- #
#  SemanticCache                                                               #
# --------------------------------------------------------------------------- #

class SemanticCache:
    """
    Semantic cache that sits in front of any function.

    Usage
    -----
    # In-memory backend, local embeddings (zero config)
    cache = SemanticCache()

    # Redis backend (dims auto-detected from the embedder)
    cache = SemanticCache(redis_url="redis://localhost:6379")

    # Bring your own embedder and backend
    cache = SemanticCache(
        backend=RedisBackend(url="redis://...", dims=1536),
        embed_fn=openai_embed,
    )

    @cache.cached
    def ask(prompt: str) -> str:
        ...

    @cache.cached
    async def ask_async(prompt: str) -> str:
        ...
    """

    def __init__(
        self,
        *,
        backend: CacheBackend | None = None,
        redis_url: str | None = None,
        embed_fn: EmbedFn | None = None,
        threshold: float = 0.85,
        ttl: int | None = None,
        namespace: str = "sc",
    ) -> None:
        # --- Embedder ---------------------------------------------------------
        if embed_fn is not None:
            self._embed_fn = embed_fn
            self._embedder = None
        else:
            self._embedder = FastEmbedder()
            self._embed_fn = self._embedder.embed

        # --- Backend ----------------------------------------------------------
        if backend is not None:
            self._backend = backend
        elif redis_url is not None:
            # Probe dims by running one embedding so the index is created correctly
            probe = self._embed_fn("probe")
            self._backend = RedisBackend(
                url=redis_url, namespace=namespace, dims=len(probe)
            )
        else:
            self._backend = MemoryBackend()

        self._threshold = threshold
        self._ttl = ttl

    # --------------------------------------------------------------- embed

    def _embed(self, text: str) -> list[float]:
        return self._embed_fn(text)

    async def _aembed(self, text: str) -> list[float]:
        # If user supplied a plain sync callable, run it in a thread executor
        # so it doesn't block the event loop.
        if self._embedder is not None:
            return await self._embedder.aembed(text)
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self._embed_fn, text)

    # --------------------------------------------------------------- get / set

    def get(self, prompt: str) -> str | None:
        try:
            vector = self._embed(prompt)
            return self._backend.search_sync(vector, self._threshold)
        except Exception as exc:
            log.warning("SemanticCache.get failed (fail-open): %s", exc)
            return None

    def set(self, prompt: str, response: str) -> None:
        try:
            vector = self._embed(prompt)
            self._backend.store_sync(vector, response, self._ttl)
        except Exception as exc:
            log.warning("SemanticCache.set failed: %s", exc)

    async def aget(self, prompt: str) -> str | None:
        try:
            vector = await self._aembed(prompt)
            return await self._backend.search(vector, self._threshold)
        except Exception as exc:
            log.warning("SemanticCache.aget failed (fail-open): %s", exc)
            return None

    async def aset(self, prompt: str, response: str) -> None:
        try:
            vector = await self._aembed(prompt)
            await self._backend.store(vector, response, self._ttl)
        except Exception as exc:
            log.warning("SemanticCache.aset failed: %s", exc)

    # --------------------------------------------------------------- stats / flush

    def stats(self) -> dict:
        return self._backend.stats_sync()

    async def astats(self) -> dict:
        return await self._backend.stats()

    def flush(self) -> None:
        self._backend.clear_sync()

    async def aflush(self) -> None:
        await self._backend.clear()

    # --------------------------------------------------------------- decorator

    def cached(self, fn: Callable | None = None, *, key_arg: str | None = None):
        """
        Decorator that adds semantic caching to any function.

        Works on both sync and async functions:

            @cache.cached
            def ask(prompt: str) -> str: ...

            @cache.cached
            async def ask(prompt: str) -> str: ...

            @cache.cached(key_arg="query")
            def search(query: str, top_k: int = 5) -> str: ...
        """
        # Support both @cache.cached and @cache.cached(key_arg="...")
        if fn is None:
            return functools.partial(self.cached, key_arg=key_arg)

        key_param = _find_key_param(fn, key_arg)

        if asyncio.iscoroutinefunction(fn):
            return self._wrap_async(fn, key_param)
        return self._wrap_sync(fn, key_param)

    def _wrap_sync(self, fn: Callable, key_param: str) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            prompt = _extract_key(key_param, fn, args, kwargs)

            cached = self.get(prompt)
            if cached is not None:
                return cached

            response = fn(*args, **kwargs)

            if isinstance(response, str):
                self.set(prompt, response)

            return response

        return wrapper

    def _wrap_async(self, fn: Callable, key_param: str) -> Callable:
        @functools.wraps(fn)
        async def wrapper(*args, **kwargs):
            prompt = _extract_key(key_param, fn, args, kwargs)

            cached = await self.aget(prompt)
            if cached is not None:
                return cached

            response = await fn(*args, **kwargs)

            if isinstance(response, str):
                # Fire-and-forget: return to caller immediately,
                # write to cache in the background.
                asyncio.create_task(self.aset(prompt, response))

            return response

        return wrapper
