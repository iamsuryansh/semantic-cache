# Semantic Cache Demo App

Simple standalone demo for semantic caching with:
- FastAPI backend
- React frontend
- OpenAI for embeddings + chat
- Redis for cache storage

## Folder Structure

- `backend/` API and cache logic
- `frontend/` Demo UI

## 1) Start Redis

Use Redis Stack (required for vector search):

```bash
docker run -d --name redis-stack -p 6379:6379 redis/redis-stack-server:latest
```

## 2) Run Backend

```bash
cd backend
cp .env.example .env
# fill OPENAI_API_KEY in .env
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

## 3) Run Frontend

```bash
cd frontend
cp .env.example .env
npm install
npm run dev
```

Open http://localhost:5173

## Demo Script (60 seconds)

1. Click **Start Session**.
2. Click **Run Demo (2 Questions)**.
3. First turn should be `MISS` (`miss_openai`) and show non-zero actual cost.
4. Second turn should be `HIT` (`hit_cache`) and show lower latency with zero actual cost.
5. Check **Savings Summary** for hit rate and estimated saved cost/time.

## API Endpoints

- `POST /session/start`
- `POST /session/{session_id}/ask`
- `GET /session/{session_id}/stats`
- `POST /session/{session_id}/reset`

## Notes

- Optional Redis URL override is available in the UI start form.
- This is intentionally demo-focused: minimal architecture, full requested flow.
