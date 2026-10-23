# Company Knowledge Assistant

A Retrieval-Augmented Generation (RAG) API that answers employee questions **using only your company's own documents** (HR policies, SOPs, IT guides).

It runs **fully locally and for free** with no API keys or accounts: Ollama (llama3) writes the answers, `nomic-embed-text` creates embeddings, ChromaDB stores them, FastAPI + uvicorn serves the API, and LangChain connects the pieces.

## Features

- Upload PDF, TXT or Markdown files. They are chunked, embedded and indexed automatically.
- Ask questions and get a short answer **plus the source passages** it came from.
- Follow-up questions ("And for confirmed employees?") through per-session chat history.
- Says "I don't have that information" instead of guessing (distance guard + strict prompt).
- Liveness and readiness endpoints, consistent JSON errors, request IDs in every log line.
- Layered architecture with automated tests (Ollama and ChromaDB are faked in tests).

## Architecture

```
HTTP request
    │
API layer           app/api/             Routes, schemas, error handling, middleware
    │
Service layer       app/services/        Business logic: ingestion, retrieval, RAG, chat
    │
Repository layer    app/repositories/    The only code that touches the vector DB and chat history
    │
Infrastructure      app/infrastructure/  Clients for Ollama, ChromaDB and the file system
```

A layer only talks to the one directly below it, so swapping ChromaDB for another vector database
or the in-memory chat history for Redis means changing one repository, not the whole app.

**Upload flow:** file → validate and sanitize name → save → load → split into chunks → embed → store in ChromaDB.

**Question flow:** question → (rewrite into a standalone question if it is a follow-up) → embed → find the closest chunks →
drop chunks that are too far away → ask llama3 to answer from the remaining chunks only → return answer + sources.

## Requirements

- Python 3.10 or newer
- [Ollama](https://ollama.com) with these models:
```bash
  ollama pull llama3
  ollama pull nomic-embed-text
```
- About 8 GB of free RAM for llama3 8B

## Setup

```bash
git clone <your-repo-url>
cd knowledge-assistant
python -m venv venv
source venv/Scripts/activate      # Windows Git Bash. macOS/Linux: source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env              # then adjust values if needed
```

No credentials are needed. Start the Ollama app, then run:

```bash
uvicorn app.main:app --reload --port 8000
```

Open http://localhost:8000/docs for the interactive API documentation.

## Using the API

```bash
# Upload a document (use the sample file)
curl -F "file=@samples/company_handbook.txt" http://localhost:8000/api/v1/documents

# Ask a question
curl -X POST http://localhost:8000/api/v1/chat -H "Content-Type: application/json" \
  -d '{"question": "How many sick leave days do I get?", "session_id": "demo-1"}'

# Ask a follow-up (same session_id)
curl -X POST http://localhost:8000/api/v1/chat -H "Content-Type: application/json" \
  -d '{"question": "And what is the notice period for probation?", "session_id": "demo-1"}'
```

| Method | Path | Purpose |
|---|---|---|
| GET | `/api/v1/health` | Liveness: is the process running? |
| GET | `/api/v1/health/ready` | Readiness: are Ollama, both models and ChromaDB usable? (503 if not) |
| POST | `/api/v1/documents` | Upload and index a document (same name again = update) |
| GET | `/api/v1/documents` | List indexed documents with chunk counts |
| DELETE | `/api/v1/documents/{filename}` | Remove a document and its chunks |
| POST | `/api/v1/chat` | Ask a question (optional `session_id` for follow-ups) |
| DELETE | `/api/v1/chat/sessions/{session_id}` | Forget a conversation |

Errors always use the same shape:

```json
{"error": {"code": "not_found", "message": "Document 'x.txt' was not found.", "request_id": "8d3ddf7a9332"}}
```

## Configuration

Every setting is an environment variable or a line in `.env` (see `.env.example` for all of them).

| Variable | Default | Meaning |
|---|---|---|
| `LLM_MODEL` / `EMBEDDING_MODEL` | `llama3` / `nomic-embed-text` | Ollama models |
| `LLM_NUM_GPU` | unset | `0` forces CPU (use this if llama3 crashes with a CUDA error) |
| `CHROMA_PERSIST_DIR` | `data/chroma` | Where the vector database lives. Keep it outside OneDrive. |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `800` / `150` | Characters per chunk and overlap between chunks |
| `RETRIEVAL_TOP_K` | `4` | Chunks retrieved per question |
| `RETRIEVAL_MAX_DISTANCE` | `0.45` | Chunks farther than this are ignored. Lower is stricter. |
| `CORS_ALLOWED_ORIGINS` | `[]` | Browser origins allowed to call the API |

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

The tests replace Ollama and ChromaDB with fakes, so they run in seconds without Ollama installed.

## Docker (optional)

```bash
docker compose up --build -d
```

The container talks to Ollama running on your machine (`host.docker.internal:11434`). It has its own empty
knowledge base, so upload documents again. Stop with `docker compose down`.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `503 external_service_unavailable` | Start Ollama and check `/api/v1/health/ready` |
| `llama-server ... CUDA error` | Set `LLM_NUM_GPU=0` in `.env`, or update your GPU driver and Ollama |
| First answer takes a minute | Ollama is loading llama3 into memory. Later answers are faster. |
| Answers say "I don't have that information" too often | Raise `RETRIEVAL_MAX_DISTANCE` slightly (for example `0.5`) |
| Weird file-lock errors from ChromaDB | Move `CHROMA_PERSIST_DIR` out of OneDrive |

## Known limitations

- llama3 8B is small. It follows instructions well but can still misjudge edge cases, so keep `temperature` at 0.
- Chat history is in memory: it resets when the server restarts and is not shared between workers.
  Run a single worker, or implement `ConversationRepository` with Redis.
- No authentication or rate limiting. Put the API behind a gateway before exposing it.
- Scanned (image-only) PDFs are not supported because they contain no extractable text.
- The distance cut-off was tuned on a small handbook. Re-check it with your own documents.
- No per-document access control: every valid API key can query every indexed document. Do not index personal or confidential files.

## Evaluation

`evaluation/questions.json` holds 31 questions about `samples/company_handbook.txt`: in-scope facts, out-of-scope
questions that must be refused, and follow-up conversations. The checks are keyword based, so treat the score as
a trend, not a grade.

    export EVAL_API_KEY=<a user or admin key>
    python evaluation/run_evaluation.py                 # all cases (5 to 15 minutes on CPU)
    python evaluation/run_evaluation.py --filter notice # only matching cases
    python evaluation/run_evaluation.py --min-pass-rate 0.8   # exit code 1 below 80 percent

To add a case, append an entry to `questions.json`. `"must_contain": ["31 march|march 31"]` means the answer needs
that text, with `|` separating accepted phrasings. `"should_answer": false` means it must be refused.

Baseline (llama3 8B on CPU, chunk size 800): **<your passed>/31**, median response time <your seconds>s.