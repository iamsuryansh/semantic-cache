from abc import ABC, abstractmethod


class CacheBackend(ABC):
    """
    Interface every backend must implement.
    Both sync and async paths are required so the decorator can wrap
    plain `def` and `async def` functions without event-loop gymnastics.
    """

    # ------------------------------------------------------------------ sync

    @abstractmethod
    def search_sync(self, vector: list[float], threshold: float) -> str | None:
        """Return the cached response if a similar vector exists, else None."""

    @abstractmethod
    def store_sync(
        self, vector: list[float], response: str, ttl: int | None = None
    ) -> None:
        """Persist a vector/response pair."""

    @abstractmethod
    def clear_sync(self) -> None:
        """Delete all entries and reset stats."""

    @abstractmethod
    def stats_sync(self) -> dict:
        """Return {"hits": int, "misses": int, "hit_rate": float, "total": int}."""

    # ----------------------------------------------------------------- async

    @abstractmethod
    async def search(self, vector: list[float], threshold: float) -> str | None: ...

    @abstractmethod
    async def store(
        self, vector: list[float], response: str, ttl: int | None = None
    ) -> None: ...

    @abstractmethod
    async def clear(self) -> None: ...

    @abstractmethod
    async def stats(self) -> dict: ...
