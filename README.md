# E-commerce Customer Support Agent

An AI-powered customer support agent for e-commerce stores. Customers can ask about order status, refunds, tracking, and store policies. The agent uses a tool-calling LLM (Groq/Llama) backed by a RAG pipeline over your policy documents, and knows when to escalate to a human.

## Features

- **AI chat agent** — answers order, refund, tracking, and policy questions naturally
- **Live order tools** — looks up real order data; verifies identity by matching order ID + email
- **RAG over policy docs** — searches your return, shipping, and payment policy documents; hedges on low-confidence matches
- **Human escalation** — creates a support ticket when it can't resolve something, or when the customer asks for a human
- **Streaming replies** — token-by-token streaming via Server-Sent Events (`/chat/stream`)
- **Per-message feedback** — thumbs up/down on every assistant reply
- **Analytics dashboard** — escalation rate, feedback score, top tools used, low-confidence queries
- **Admin panel** — upload policy PDFs, manage tickets, approve/reject refunds
- **Security** — customer verification throttling, admin token auth, no stack traces ever leaked to the client
- **Resilience** — retries on LLM failures, graceful fallbacks, transactional DB writes

## Tech stack

| Layer | Technology |
|---|---|
| Backend | FastAPI + Uvicorn |
| ORM / DB | SQLModel (SQLite by default, Postgres-ready) |
| Agent | LangChain tool-calling agent |
| LLM | Groq API (`llama-3.3-70b-versatile`) |
| Embeddings | HuggingFace `sentence-transformers/all-MiniLM-L6-v2` |
| Vector store | Chroma (local, persisted to disk) |
| Frontend | HTML + Tailwind CSS (CDN) + vanilla JS |
| Package manager | uv |

## Project structure

```
customer_support_agent/
├── app/
│   ├── main.py           # FastAPI app, lifespan, routers
│   ├── config.py         # Settings from .env
│   ├── db.py             # Engine + init_db
│   ├── seed.py           # Sample customers, orders, products
│   ├── models/           # SQLModel table definitions
│   ├── agent/
│   │   ├── agent.py      # ReAct tool loop, streaming, error handling
│   │   ├── tools.py      # LangChain tools (orders, refunds, RAG, tickets)
│   │   └── prompts.py    # System prompt
│   ├── rag/
│   │   ├── ingest.py     # Load → chunk → embed → persist to Chroma
│   │   └── retriever.py  # Similarity search with confidence scoring
│   └── api/
│       ├── chat.py       # POST /chat, POST /chat/stream
│       ├── orders.py     # PATCH /refunds/{id}
│       ├── tickets.py    # GET /tickets
│       ├── feedback.py   # POST /feedback
│       ├── analytics.py  # GET /analytics/summary
│       ├── knowledge_base.py  # POST /admin/knowledge-base/upload
│       └── deps.py       # Admin token verification
├── data/policies/        # Markdown policy docs used for RAG
├── chroma_db/            # Persisted vector store (auto-created)
├── frontend/
│   ├── index.html        # Customer chat UI
│   └── admin.html        # Admin panel
├── tests/
├── .env                  # Environment variables (not committed)
├── pyproject.toml
└── uv.lock
```

## Quick start

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) (`pip install uv`)
- A [Groq API key](https://console.groq.com)

### 1. Clone and install

```bash
git clone https://github.com/AHSANALI122/customer-support-agent.git
cd customer-support-agent
uv sync
```

### 2. Configure environment

Create a `.env` file in the project root:

```env
DATABASE_URL=sqlite:///./support_agent.db
GROQ_API_KEY=your_groq_api_key_here
MODEL_NAME=llama-3.3-70b-versatile
EMBED_MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
CHROMA_PATH=./chroma_db
ADMIN_TOKEN=your_strong_random_secret_here
CONFIDENCE_THRESHOLD=0.5
```

Generate a strong admin token:
```bash
openssl rand -hex 32
```

### 3. Seed the database

```bash
uv run python -m app.seed
```

### 4. Build the vector store

```bash
uv run python -m app.rag.ingest
```

### 5. Run the server

```bash
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000
```

| URL | Description |
|---|---|
| http://localhost:8000 | Customer chat UI |
| http://localhost:8000/admin.html | Admin panel |
| http://localhost:8000/docs | Interactive API docs |
| http://localhost:8000/health | Health check |

## API reference

### Customer endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/chat` | Send a message, get a reply |
| `POST` | `/chat/stream` | Streamed reply via Server-Sent Events |
| `POST` | `/feedback` | Submit thumbs up/down on a message |
| `GET` | `/health` | DB, LLM, and vector store status |

#### POST /chat

```json
// Request
{ "session_id": "uuid | null", "message": "string", "customer_email": "string | null" }

// Response
{ "session_id": "uuid", "reply": "string" }
```

#### POST /chat/stream

Same request shape. Returns `text/event-stream`:

```
event: token
data: {"text": "Sure"}

event: tool_call
data: {"tool": "get_order_status"}

event: done
data: {"session_id": "uuid", "full_reply": "string"}

event: error
data: {"message": "Something went wrong, please try again."}
```

### Admin endpoints

All require the header `X-Admin-Token: <your token>`.

| Method | Path | Description |
|---|---|---|
| `GET` | `/tickets` | List open support tickets |
| `PATCH` | `/refunds/{refund_id}` | Approve / reject / mark refunded |
| `GET` | `/analytics/summary` | Sessions, feedback, tools, low-confidence queries |
| `POST` | `/admin/knowledge-base/upload` | Upload a policy PDF to the knowledge base |

#### GET /analytics/summary

```
GET /analytics/summary?from=2024-01-01&to=2024-12-31
```

```json
{
  "total_sessions": 42,
  "total_messages": 187,
  "escalation_rate": 0.12,
  "feedback_positive_pct": 0.91,
  "top_tools_used": [{"tool": "get_order_status", "count": 55}],
  "low_confidence_queries": [{"query": "warranty on electronics", "top_score": 0.31}]
}
```

## Adding policy documents

Drop a Markdown file into `data/policies/` and re-run ingestion:

```bash
uv run python -m app.rag.ingest
```

Or upload a PDF directly from the admin panel at `/admin.html`.

## Running tests

```bash
uv run pytest
```

## Deployment notes

- **SQLite → Postgres:** change `DATABASE_URL` in `.env` to a Postgres connection string — no code changes needed.
- **Groq token limits:** the free tier is capped at 100k tokens/day. For production traffic, upgrade to the Groq Dev tier at [console.groq.com](https://console.groq.com/settings/billing).
- **Admin token:** use a random 32+ character secret. Never commit `.env` to version control.
- **Chroma persistence:** `chroma_db/` must be on a persistent volume in containerized deployments.
- **First-run order:** seed → ingest → start server. The server will refuse to start if the DB is unreachable.
