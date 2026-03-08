import threading
import time
from dataclasses import dataclass, field

from .base import CacheBackend


def _cosine(a: list[float], b: list[float]) -> float:
    try:
        import numpy as np

        av = np.array(a, dtype=np.float32)
        bv = np.array(b, dtype=np.float32)
        denom = float(np.linalg.norm(av) * np.linalg.norm(bv))
        return float(np.dot(av, bv) / denom) if denom else 0.0
    except ImportError:
        dot = sum(x * y for x, y in zip(a, b))
        mag = (sum(x * x for x in a) ** 0.5) * (sum(x * x for x in b) ** 0.5)
        return dot / mag if mag else 0.0


@dataclass
class _Entry:
    vector: list[float]
    response: str
    created_at: float
    expires_at: float | None = None  # None = no expiry

    def is_expired(self) -> bool:
        return self.expires_at is not None and time.time() > self.expires_at


class MemoryBackend(CacheBackend):
    """
    In-process backend — no external dependencies.
    Stores everything in a list. Similarity search is a linear scan.
    Good for development, testing, and small caches.
    Thread-safe for sync usage via a Lock.
    """

    def __init__(self) -> None:
        self._entries: list[_Entry] = []
        self._hits = 0
        self._misses = 0
        self._lock = threading.Lock()

    # ------------------------------------------------------------------ sync

    def search_sync(self, vector: list[float], threshold: float) -> str | None:
        with self._lock:
            best_score = -1.0
            best_response: str | None = None

            for entry in self._entries:
                if entry.is_expired():
                    continue
                score = _cosine(vector, entry.vector)
                if score > best_score:
                    best_score = score
                    best_response = entry.response

            if best_score >= threshold:
                self._hits += 1
                return best_response

            self._misses += 1
            return None

    def store_sync(
        self, vector: list[float], response: str, ttl: int | None = None
    ) -> None:
        expires_at = time.time() + ttl if ttl else None
        entry = _Entry(
            vector=vector,
            response=response,
            created_at=time.time(),
            expires_at=expires_at,
        )
        with self._lock:
            self._entries.append(entry)
            self._purge_expired()

    def clear_sync(self) -> None:
        with self._lock:
            self._entries.clear()
            self._hits = 0
            self._misses = 0

    def stats_sync(self) -> dict:
        with self._lock:
            total = self._hits + self._misses
            return {
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": round(self._hits / total, 4) if total else 0.0,
                "total": total,
            }

    # ----------------------------------------------------------------- async
    # MemoryBackend has no I/O — async methods simply delegate to sync.

    async def search(self, vector: list[float], threshold: float) -> str | None:
        return self.search_sync(vector, threshold)

    async def store(
        self, vector: list[float], response: str, ttl: int | None = None
    ) -> None:
        self.store_sync(vector, response, ttl)

    async def clear(self) -> None:
        self.clear_sync()

    async def stats(self) -> dict:
        return self.stats_sync()

    # ------------------------------------------------------------ internal

    def _purge_expired(self) -> None:
        """Remove expired entries. Called inside the lock on every store."""
        self._entries = [e for e in self._entries if not e.is_expired()]
