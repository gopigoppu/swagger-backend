# Swagger/OpenAPI Validation & Correction Platform

## Backend (FastAPI + LangGraph + Groq LLM)

### Setup
1. `cd swagger-backend`
2. `python -m venv venv && source venv/bin/activate`
3. `pip install -r requirements.txt`
4. Copy `.env.example` to `.env` and add your `GROQ_API_KEY`.

### Running
- `uvicorn main:app --reload`

### Endpoints
- `POST /upload` — Upload OpenAPI file or URL
- `POST /validate` — Validate OpenAPI spec
- `POST /llm-correct` — LLM correction pipeline (SSE)
- `POST /generate` — Generate OpenAPI spec from description

---
