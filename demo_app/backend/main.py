from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from cache_service import DemoCacheService
from settings import get_settings


class StartSessionRequest(BaseModel):
    redis_url: str | None = None


class AskRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=4000)


settings = get_settings()
service = DemoCacheService(settings)

app = FastAPI(title="Semantic Cache Demo API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/session/start")
def start_session(payload: StartSessionRequest) -> dict[str, Any]:
    if not settings.openai_api_key:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY is not set")

    try:
        return service.start_session(redis_url_override=payload.redis_url)
    except Exception as exc:  # demo-friendly error surfacing
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/session/{session_id}/ask")
def ask(session_id: str, payload: AskRequest) -> dict[str, Any]:
    try:
        return service.ask(session_id=session_id, prompt=payload.prompt)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/session/{session_id}/stats")
def stats(session_id: str) -> dict[str, Any]:
    try:
        return service.stats(session_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/session/{session_id}/reset")
def reset(session_id: str) -> dict[str, Any]:
    try:
        return service.reset(session_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
