# Suze — Run Guide

## Prerequisites

- Python 3.11+
- Node.js 18+
- NVIDIA API key (free at https://build.nvidia.com)

## Setup

### Backend

```bash
cd backend

# Create .env from template
cp .env.example .env
# Edit .env and set your NVIDIA_API_KEY

# Install dependencies
pip install -r requirements.txt
# Or: pip install fastapi uvicorn chromadb langgraph langchain-core rank-bm25 openai pydantic-settings python-dotenv python-multipart pymupdf httpx

# Run tests
pytest -v

# Start server
uvicorn src.main:app --reload --port 8000
```

### Frontend

```bash
cd frontend

# Install dependencies
npm install

# Development server (with API proxy to :8000)
npm run dev

# Production build
npm run build
```

## Usage

1. Start the backend (`uvicorn src.main:app --port 8000`)
2. Start the frontend (`npm run dev` from `frontend/`)
3. Open http://localhost:5173
4. Upload documents via the + button or type questions directly

## API Examples

```bash
# Health check
curl http://localhost:8000/api/v1/health

# Ingest text
curl -X POST http://localhost:8000/api/v1/ingest/text \
  -H "Content-Type: application/json" \
  -d '{"content": "Paris is the capital of France.", "source": "geo.txt"}'

# Chat
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the capital of France?"}'
```

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `AuthenticationError` on chat | Set `NVIDIA_API_KEY` in `backend/.env` |
| ChromaDB `NotFoundError` | Delete `backend/chroma_data/` and restart |
| Frontend can't reach backend | Ensure backend runs on port 8000 (Vite proxies `/api` there) |
