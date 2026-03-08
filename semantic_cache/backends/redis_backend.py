import logging
from uuid import uuid4

from .base import CacheBackend

log = logging.getLogger(__name__)


class RedisBackend(CacheBackend):
    """
    Redis backend using RediSearch HNSW vector index.
    Requires redis-stack-server locally or any Redis with VSS enabled.

    Each cache entry is stored as a Redis hash:
        Key:    {namespace}:vec:{uuid}
        Fields: vec  → float32[] as bytes
                resp → response text

    Stats are stored as plain Redis string counters:
        {namespace}:hits
        {namespace}:misses
    """

    def __init__(
        self,
        url: str = "redis://localhost:6379",
        namespace: str = "sc",
        dims: int = 384,
    ) -> None:
        self._url = url
        self._ns = namespace
        self._dims = dims
        self._index = f"{namespace}:idx"
        self._prefix = f"{namespace}:vec:"

        # Sync client — initialised eagerly
        import redis
        self._sync: redis.Redis = redis.Redis.from_url(url, decode_responses=False)

        # Async client — lazy, created on first async call to avoid
        # creating a Redis connection outside of an async context
        self._async = None

        self._index_ready = False   # ensure_index called once per client instance

    # --------------------------------------------------------------- index

    def _ensure_index_sync(self) -> None:
        if self._index_ready:
            return
        import redis.exceptions
        try:
            self._sync.ft(self._index).create_index(
                fields=self._index_fields(),
                definition=self._index_definition(),
            )
        except redis.exceptions.ResponseError as e:
            if "Index already exists" not in str(e):
                raise
        self._index_ready = True

    async def _ensure_index_async(self) -> None:
        if self._index_ready:
            return
        import redis.exceptions
        client = await self._get_async()
        try:
            await client.ft(self._index).create_index(
                fields=self._index_fields(),
                definition=self._index_definition(),
            )
        except redis.exceptions.ResponseError as e:
            if "Index already exists" not in str(e):
                raise
        self._index_ready = True

    def _index_fields(self):
        from redis.commands.search.field import TextField, VectorField
        return [
            VectorField(
                "vec",
                "HNSW",
                {
                    "TYPE": "FLOAT32",
                    "DIM": self._dims,
                    "DISTANCE_METRIC": "COSINE",
                },
            ),
            TextField("resp", no_stem=True),
        ]

    def _index_definition(self):
        from redis.commands.search.indexDefinition import IndexDefinition, IndexType
        return IndexDefinition(prefix=[self._prefix], index_type=IndexType.HASH)

    # --------------------------------------------------------------- internal

    async def _get_async(self):
        if self._async is None:
            import redis.asyncio
            self._async = redis.asyncio.Redis.from_url(
                self._url, decode_responses=False
            )
        return self._async

    def _to_bytes(self, vector: list[float]) -> bytes:
        import numpy as np
        return np.array(vector, dtype=np.float32).tobytes()

    def _parse_score(self, raw_score) -> float:
        """
        RediSearch returns cosine distance (1 - similarity).
        Convert to similarity so callers use a consistent scale.
        """
        return 1.0 - float(raw_score)

    # ------------------------------------------------------------------ sync

    def search_sync(self, vector: list[float], threshold: float) -> str | None:
        self._ensure_index_sync()
        from redis.commands.search.query import Query

        query = (
            Query("*=>[KNN 1 @vec $BLOB AS __score]")
            .return_fields("resp", "__score")
            .dialect(2)
        )
        results = self._sync.ft(self._index).search(
            query, query_params={"BLOB": self._to_bytes(vector)}
        )
        if not results.docs:
            self._sync.incr(f"{self._ns}:misses")
            return None

        doc = results.docs[0]
        similarity = self._parse_score(doc.__score)

        if similarity >= threshold:
            self._sync.incr(f"{self._ns}:hits")
            return doc.resp

        self._sync.incr(f"{self._ns}:misses")
        return None

    def store_sync(
        self, vector: list[float], response: str, ttl: int | None = None
    ) -> None:
        self._ensure_index_sync()
        key = f"{self._prefix}{uuid4()}"
        pipe = self._sync.pipeline()
        pipe.hset(key, mapping={"vec": self._to_bytes(vector), "resp": response})
        if ttl:
            pipe.expire(key, ttl)
        pipe.execute()

    def clear_sync(self) -> None:
        keys = self._sync.keys(f"{self._prefix}*")
        if keys:
            self._sync.delete(*keys)
        self._sync.set(f"{self._ns}:hits", 0)
        self._sync.set(f"{self._ns}:misses", 0)

    def stats_sync(self) -> dict:
        hits   = int(self._sync.get(f"{self._ns}:hits")   or 0)
        misses = int(self._sync.get(f"{self._ns}:misses") or 0)
        total  = hits + misses
        return {
            "hits": hits,
            "misses": misses,
            "hit_rate": round(hits / total, 4) if total else 0.0,
            "total": total,
        }

    # ----------------------------------------------------------------- async

    async def search(self, vector: list[float], threshold: float) -> str | None:
        await self._ensure_index_async()
        from redis.commands.search.query import Query

        client = await self._get_async()
        query = (
            Query("*=>[KNN 1 @vec $BLOB AS __score]")
            .return_fields("resp", "__score")
            .dialect(2)
        )
        results = await client.ft(self._index).search(
            query, query_params={"BLOB": self._to_bytes(vector)}
        )
        if not results.docs:
            await client.incr(f"{self._ns}:misses")
            return None

        doc = results.docs[0]
        similarity = self._parse_score(doc.__score)

        if similarity >= threshold:
            await client.incr(f"{self._ns}:hits")
            return doc.resp

        await client.incr(f"{self._ns}:misses")
        return None

    async def store(
        self, vector: list[float], response: str, ttl: int | None = None
    ) -> None:
        await self._ensure_index_async()
        client = await self._get_async()
        key = f"{self._prefix}{uuid4()}"
        pipe = client.pipeline()
        await pipe.hset(key, mapping={"vec": self._to_bytes(vector), "resp": response})
        if ttl:
            await pipe.expire(key, ttl)
        await pipe.execute()

    async def clear(self) -> None:
        client = await self._get_async()
        keys = await client.keys(f"{self._prefix}*")
        if keys:
            await client.delete(*keys)
        await client.set(f"{self._ns}:hits", 0)
        await client.set(f"{self._ns}:misses", 0)

    async def stats(self) -> dict:
        client = await self._get_async()
        hits   = int(await client.get(f"{self._ns}:hits")   or 0)
        misses = int(await client.get(f"{self._ns}:misses") or 0)
        total  = hits + misses
        return {
            "hits": hits,
            "misses": misses,
            "hit_rate": round(hits / total, 4) if total else 0.0,
            "total": total,
        }
