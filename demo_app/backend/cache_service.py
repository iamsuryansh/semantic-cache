from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any

from openai import OpenAI

from semantic_cache import SemanticCache

from settings import Settings


@dataclass
class SessionState:
    session_id: str
    cache: SemanticCache
    history: list[dict[str, Any]] = field(default_factory=list)
    miss_count: int = 0
    avg_miss_latency_ms: float = 0.0
    avg_miss_cost_usd: float = 0.0


class DemoCacheService:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.openai = OpenAI(api_key=settings.openai_api_key)
        self.sessions: dict[str, SessionState] = {}

    def start_session(self, redis_url_override: str | None = None) -> dict[str, Any]:
        session_id = uuid.uuid4().hex[:12]
        redis_url = (redis_url_override or self.settings.redis_url).strip()
        namespace = f"demo:{session_id}"

        cache = SemanticCache(
            redis_url=redis_url,
            embed_fn=self._embed,
            threshold=self.settings.cache_threshold,
            ttl=self.settings.cache_ttl,
            namespace=namespace,
        )

        self.sessions[session_id] = SessionState(session_id=session_id, cache=cache)

        return {
            "session_id": session_id,
            "namespace": namespace,
            "redis_url": redis_url,
            "message": "Session started",
        }

    def ask(self, session_id: str, prompt: str) -> dict[str, Any]:
        session = self._get_session(session_id)

        started = time.perf_counter()
        cached_response = session.cache.get(prompt)

        if cached_response is not None:
            latency_ms = (time.perf_counter() - started) * 1000
            turn = {
                "prompt": prompt,
                "response": cached_response,
                "source": "hit_cache",
                "latency_ms": round(latency_ms, 2),
                "actual_cost_usd": 0.0,
                "estimated_saved_cost_usd": round(session.avg_miss_cost_usd, 6),
                "estimated_saved_time_ms": round(session.avg_miss_latency_ms, 2),
                "input_tokens": 0,
                "output_tokens": 0,
                "timestamp": int(time.time()),
            }
            session.history.append(turn)
            return self._with_stats(session, turn)

        completion = self.openai.chat.completions.create(
            model=self.settings.chat_model,
            messages=[{"role": "user", "content": prompt}],
        )

        response_text = completion.choices[0].message.content or ""
        session.cache.set(prompt, response_text)

        latency_ms = (time.perf_counter() - started) * 1000
        usage = completion.usage
        input_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        output_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        actual_cost = self._compute_cost_usd(input_tokens, output_tokens)

        session.miss_count += 1
        session.avg_miss_latency_ms = (
            (session.avg_miss_latency_ms * (session.miss_count - 1)) + latency_ms
        ) / session.miss_count
        session.avg_miss_cost_usd = (
            (session.avg_miss_cost_usd * (session.miss_count - 1)) + actual_cost
        ) / session.miss_count

        turn = {
            "prompt": prompt,
            "response": response_text,
            "source": "miss_openai",
            "latency_ms": round(latency_ms, 2),
            "actual_cost_usd": round(actual_cost, 6),
            "estimated_saved_cost_usd": 0.0,
            "estimated_saved_time_ms": 0.0,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "timestamp": int(time.time()),
        }
        session.history.append(turn)
        return self._with_stats(session, turn)

    def stats(self, session_id: str) -> dict[str, Any]:
        session = self._get_session(session_id)
        return self._build_stats(session)

    def reset(self, session_id: str) -> dict[str, Any]:
        session = self._get_session(session_id)
        session.cache.flush()
        session.history.clear()
        session.miss_count = 0
        session.avg_miss_latency_ms = 0.0
        session.avg_miss_cost_usd = 0.0
        return {"ok": True, "message": "Session reset"}

    def _embed(self, text: str) -> list[float]:
        embedding = self.openai.embeddings.create(
            model=self.settings.embed_model,
            input=text,
        )
        return list(embedding.data[0].embedding)

    def _compute_cost_usd(self, input_tokens: int, output_tokens: int) -> float:
        input_cost = (input_tokens / 1_000_000) * self.settings.input_cost_per_1m
        output_cost = (output_tokens / 1_000_000) * self.settings.output_cost_per_1m
        return input_cost + output_cost

    def _get_session(self, session_id: str) -> SessionState:
        session = self.sessions.get(session_id)
        if session is None:
            raise KeyError("Unknown session_id")
        return session

    def _with_stats(self, session: SessionState, turn: dict[str, Any]) -> dict[str, Any]:
        return {
            "turn": turn,
            "history": session.history,
            "stats": self._build_stats(session),
        }

    def _build_stats(self, session: SessionState) -> dict[str, Any]:
        cache_stats = session.cache.stats()
        total_actual_cost = sum(item["actual_cost_usd"] for item in session.history)
        total_saved_cost = sum(item["estimated_saved_cost_usd"] for item in session.history)
        total_latency = sum(item["latency_ms"] for item in session.history)
        total_saved_time = sum(item["estimated_saved_time_ms"] for item in session.history)

        return {
            "session_id": session.session_id,
            "cache_hits": cache_stats["hits"],
            "cache_misses": cache_stats["misses"],
            "hit_rate": round(cache_stats["hit_rate"], 4),
            "total_requests": cache_stats["total"],
            "total_actual_cost_usd": round(total_actual_cost, 6),
            "estimated_total_saved_cost_usd": round(total_saved_cost, 6),
            "total_latency_ms": round(total_latency, 2),
            "estimated_total_saved_time_ms": round(total_saved_time, 2),
        }
