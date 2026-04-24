from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


load_dotenv()


@dataclass(frozen=True)
class Settings:
    openai_api_key: str
    redis_url: str
    cache_threshold: float
    cache_ttl: int
    chat_model: str
    embed_model: str
    input_cost_per_1m: float
    output_cost_per_1m: float



def get_settings() -> Settings:
    openai_api_key = os.getenv("OPENAI_API_KEY", "").strip()
    redis_url = os.getenv("REDIS_URL", "redis://localhost:6379").strip()
    cache_threshold = float(os.getenv("CACHE_THRESHOLD", "0.82"))
    cache_ttl = int(os.getenv("CACHE_TTL", "86400"))
    chat_model = os.getenv("CHAT_MODEL", "gpt-4.1-mini").strip()
    embed_model = os.getenv("EMBED_MODEL", "text-embedding-3-small").strip()
    input_cost_per_1m = float(os.getenv("OPENAI_INPUT_COST_PER_1M", "0.40"))
    output_cost_per_1m = float(os.getenv("OPENAI_OUTPUT_COST_PER_1M", "1.60"))

    return Settings(
        openai_api_key=openai_api_key,
        redis_url=redis_url,
        cache_threshold=cache_threshold,
        cache_ttl=cache_ttl,
        chat_model=chat_model,
        embed_model=embed_model,
        input_cost_per_1m=input_cost_per_1m,
        output_cost_per_1m=output_cost_per_1m,
    )
