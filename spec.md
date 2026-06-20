# E-commerce Customer Support Agent — Feature Specification

## Overview

An AI-powered customer support agent for an e-commerce store. Customers can ask about order status, refunds, tracking, and general store policies (returns, shipping, payments). The agent uses Google's Gemini models (via the Gemini API / Google AI Studio) and decides on its own whether to query live order data or look up policy documents.

This spec is written feature by feature so it can be implemented one unit at a time in a spec-driven workflow: implement a feature, verify it against its acceptance criteria, then move to the next.

## Tech stack

- Backend framework: FastAPI
- ORM / DB layer: SQLModel (SQLite for development, swappable to Postgres later)
- Agent framework: LangChain (tool-calling agent)
- LLM: Google Gemini via the Gemini Developer API (Google AI Studio) — `gemini-3-flash` for general replies, `gemini-3.1-flash-lite` as a cheaper/faster option for simple lookups if you want to route by query complexity later
- Embeddings: `gemini-embedding-2` (or `-preview`), same Gemini API, no separate embedding service needed
- Vector store: Chroma (local, persisted to disk)
- Package manager: uv
- Frontend: static HTML + Tailwind CSS (CDN) + vanilla JS using `fetch`

**Note on this swap:** moving from local Ollama to the Gemini API removes the "run an LLM on your server" problem entirely from the WhatsApp/deployment discussion earlier — no GPU, no local model weights, the client's server just needs a `GOOGLE_API_KEY`. The trade-offs: it needs internet access at all times, scales by per-token billing instead of being free after setup, and customer messages are sent to Google's API rather than staying fully on your own infrastructure. Also worth knowing before you build on it: as of 2026 only Flash and Flash-Lite models are free-tier eligible (Pro models are paid-only), and enabling billing on a project removes its free tier entirely — for a real client, plan for paid usage from day one rather than assuming the free tier will hold.

## Project structure

```
support-agent/
├── pyproject.toml
├── uv.lock
├── .env
├── app/
│   ├── main.py
│   ├── config.py
│   ├── db.py
│   ├── models/
│   │   ├── customer.py
│   │   ├── order.py
│   │   ├── refund.py
│   │   ├── chat.py
│   │   └── ticket.py
│   ├── agent/
│   │   ├── tools.py
│   │   ├── agent.py
│   │   └── prompts.py
│   ├── rag/
│   │   ├── ingest.py
│   │   └── retriever.py
│   ├── api/
│   │   ├── chat.py
│   │   ├── orders.py
│   │   └── tickets.py
│   └── seed.py
├── data/
│   └── policies/        # markdown docs used for RAG
├── chroma_db/            # persisted vector store
└── frontend/
    └── index.html
```

## Data models (shared reference across features)

**Customer** — id (PK), name, email (unique), phone (nullable)

**Product** — id (PK), name, price, stock (reserved for a future stock-availability tool; no feature in this spec reads or writes it yet)

**Order** — id (PK), customer_id (FK), status (enum: pending, shipped, delivered, cancelled), tracking_number (nullable), total, created_at

**OrderItem** — id (PK), order_id (FK), product_id (FK), quantity, price

**RefundRequest** — id (PK), order_id (FK), reason, status (enum: requested, approved, rejected, refunded), created_at

**ChatSession** — id (UUID, PK), customer_id (FK, nullable), created_at

**ChatMessage** — id (PK), session_id (FK), role (enum: user, assistant, tool), content, created_at

**SupportTicket** — id (PK), session_id (FK), customer_id (FK, nullable), subject, status (enum: open, in_progress, resolved), created_at

**MessageFeedback** — id (PK), message_id (FK ChatMessage), rating (enum: up, down), comment (nullable), created_at

**RetrievalLog** — id (PK), session_id (FK, nullable), query, top_score (float), was_confident (bool), created_at

---

## Features

### F1 — Project setup & environment

**Goal:** scaffold the project so the server boots, the DB connects, and the Gemini API is reachable.

**Requirements**
- `uv init` the project; add dependencies: `fastapi`, `uvicorn`, `sqlmodel`, `langchain`, `langchain-google-genai`, `langchain-chroma`, `python-dotenv`
- `.env` provides `DATABASE_URL`, `GOOGLE_API_KEY`, `MODEL_NAME`, `EMBED_MODEL_NAME`, `CHROMA_PATH`, `ADMIN_TOKEN`
- `GET /health` checks DB connectivity, confirms the Gemini API key works (a minimal low-cost call, not a full chat request), and confirms the Chroma store at `CHROMA_PATH` exists and is readable — catches a missing or corrupted vector store before a customer hits it

**Acceptance criteria**
- [ ] `uv run uvicorn app.main:app` boots without errors
- [ ] `GET /health` returns `{"status": "ok", "db": "ok", "llm": "ok", "vector_store": "ok"}`

---

### F2 — Database models & seed data

**Goal:** all SQLModel tables exist and sample data is available for testing.

**Requirements**
- Implement all models listed in the Data Models section
- `app/seed.py` creates 3 customers, 5 products, 8 orders (mixed statuses), 2 refund requests

**Acceptance criteria**
- [ ] `uv run python -m app.seed` populates the database without errors
- [ ] Table schemas match the Data Models section exactly

---

### F3 — Knowledge base ingestion (RAG)

**Goal:** load store policy documents, chunk them, embed them, and persist to a local vector store.

**Requirements**
- At least 3 markdown documents in `data/policies/` (return policy, shipping policy, payment FAQ)
- `app/rag/ingest.py`: load → split (chunk size ~500, overlap ~50) → embed with `GoogleGenerativeAIEmbeddings` (`gemini-embedding-2`) → persist to `chroma_db/`
- Re-running ingestion should not duplicate existing chunks
- `app/rag/retriever.py` exposes `get_retriever()` returning top-k similarity results

**Acceptance criteria**
- [ ] `uv run python -m app.rag.ingest` builds or updates the vector store
- [ ] A test query against the retriever returns relevant chunks from the right document

---

### F4 — Customer data tools (order / refund / tracking)

**Goal:** LangChain tools that read and write live order data.

**Tools to implement**
- `list_orders_by_email(email)` → a short list of that customer's recent orders (id, status, total, date) — lets the agent answer "where's my order" even when the customer doesn't have their order ID handy
- `get_order_status(order_id, email)` → status, items, total
- `get_tracking_info(order_id, email)` → tracking number, status
- `get_refund_status(order_id, email)`
- `create_refund_request(order_id, email, reason)`
- `search_policy_docs(query)` — wraps the RAG retriever from F3

Every tool that touches a specific order requires `email` as a parameter and checks it against that order's customer before returning anything — this is what actually implements the verification rule described in F8; F8 states the requirement, this is where it's enforced.

**Refund eligibility & admin processing**
- `create_refund_request` only succeeds when the order's status is `delivered` — orders that are `pending`, `shipped`, or already `cancelled` are not refund-eligible, and the tool returns a clear explanation instead of silently creating a request
- Creating a request only ever sets its status to `requested`; nothing in the agent or tools advances it further than that
- **Admin endpoint:** `PATCH /refunds/{refund_id}` with body `{ "status": "approved" | "rejected" | "refunded" }` — this is the only way a request's status changes, since approving a refund is a human decision, not something the agent should ever do on its own. Protected by the admin auth in F14; don't expose this endpoint unauthenticated in the meantime

**Acceptance criteria**
- [ ] Each tool has a clear description so the LLM knows when to call it
- [ ] Tools return short, structured text — not raw DB objects
- [ ] An invalid `order_id` returns a graceful "not found" message, never an exception
- [ ] A mismatched `order_id` + `email` pair never returns order data — it asks the user to confirm both
- [ ] `list_orders_by_email` lets the agent resolve "where's my order" without the customer typing an order ID first
- [ ] Requesting a refund on a `pending` or already-`cancelled` order is rejected with an explanation, not silently created
- [ ] `PATCH /refunds/{refund_id}` updates the stored status and rejects calls without a valid admin token

---

### F5 — LangChain agent core

**Goal:** an agent that picks the right tool, stays in scope, and replies naturally.

**Requirements**
- System prompt defines persona, scope (e-commerce support only), and how to handle out-of-scope questions
- Tool-calling agent bound to `ChatGoogleGenerativeAI` (`gemini-3-flash`), with all F4 tools registered
- Conversation memory scoped per `session_id` — load the last **20 messages** (10 exchanges) from `ChatMessage` before each call; older history is dropped rather than summarized for now
- When `search_policy_docs` returns a low-confidence match (similarity below a configured threshold, default **0.5** — tune once real queries come in), the agent says it isn't fully sure rather than answering confidently off a weak match, and offers to escalate via `create_ticket` (F7) if the customer needs a definite answer

**Acceptance criteria**
- [ ] Asking "where is my order #123" triggers `get_order_status`/`get_tracking_info`
- [ ] Asking about the return policy triggers `search_policy_docs`
- [ ] The agent never invents order data when a tool returns "not found"
- [ ] A low-confidence policy match produces a hedged answer or an escalation offer, not a confident-sounding guess

---

### F6 — Chat API & session management

**Goal:** expose the agent over HTTP and persist every conversation.

**Endpoint:** `POST /chat`

Request:
```json
{ "session_id": "uuid | null", "message": "string", "customer_email": "string | null" }
```

Response:
```json
{ "session_id": "uuid", "reply": "string" }
```

**Requirements**
- A new `ChatSession` is created automatically if `session_id` is missing
- Both the user message and the assistant reply are stored in `ChatMessage`

**Acceptance criteria**
- [ ] First message with no `session_id` creates a new session
- [ ] Message history persists across server restarts (DB-backed, not in-memory)

---

### F7 — Human escalation / support ticket

**Goal:** create a ticket when the agent can't resolve something or the customer explicitly asks for a human.

**Requirements**
- A `create_ticket(session_id, subject)` tool the agent calls when escalation is needed
- Before creating a ticket, check whether an open (`status = open` or `in_progress`) ticket already exists for that `session_id` — if one does, don't create a duplicate, just confirm to the customer that their case is already with the team
- `GET /tickets` (admin use, protected by F14) lists open tickets with recent message context

**Acceptance criteria**
- [ ] Saying "I want to talk to a human" creates a ticket and the agent confirms this to the user
- [ ] Saying it again in the same session does not create a second ticket
- [ ] A ticket includes enough context (session_id + last few messages) for a human to pick up the case
- [ ] `GET /tickets` rejects requests without a valid admin token (see F14)

---

### F8 — Customer verification (lightweight)

**Goal:** stop one customer from viewing another customer's order data.

**Requirements**
- Order lookup requires `order_id` + the email used for that purchase to match
- On mismatch, the tool asks the user to confirm both values instead of returning data
- This rule is *implemented* inside F4's tool signatures (every order-specific tool takes `email` as a required parameter) — F8 exists as a named requirement so it's easy to point to and test on its own, not as separate code

**Acceptance criteria**
- [ ] Order details are never returned if the email doesn't match the order's customer
- [ ] Repeated mismatched attempts in a short window are throttled (see F13) rather than allowed unlimited guesses at order_id/email combinations

---

### F9 — Frontend chat widget

**Goal:** a simple, clean chat UI.

**Requirements**
- Single `index.html`, Tailwind via CDN, message list + input box
- A small email field above the chat (placeholder: "Email used for your order"), sent as `customer_email` on every `/chat` request — required for the agent's order/refund tools (F4, F8) to verify identity; general policy questions still work fine without it
- `fetch()` calls to `POST /chat`; `session_id` kept in a JS variable for the page session
- Loading indicator while waiting for a reply

**Acceptance criteria**
- [ ] User can send a message and see the agent's reply appended to the chat
- [ ] If the email field is empty and the customer asks an order-specific question, the agent's reply (via F4) asks for it instead of failing silently
- [ ] Chat auto-scrolls to the latest message
- [ ] A loading state is visible while the request is in flight

---

### F10 — Logging & basic evaluation (stretch)

**Goal:** visibility into what the agent is doing, for debugging.

**Requirements**
- Log every tool call (tool name, input, output, timestamp)

**Acceptance criteria**
- [ ] Tool calls are visible in logs with timestamps

---

### F11 — Streaming responses

**Goal:** the agent's reply appears token-by-token in the frontend instead of arriving all at once, so the chat feels live (similar to ChatGPT).

**Endpoint:** `POST /chat/stream`

Request: same shape as `POST /chat`
```json
{ "session_id": "uuid | null", "message": "string", "customer_email": "string | null" }
```

Response: `text/event-stream` (Server-Sent Events). Example event payloads:
```
event: token
data: {"text": "Sure"}

event: tool_call
data: {"tool": "get_order_status"}

event: done
data: {"session_id": "uuid", "full_reply": "string"}
```

**Requirements**
- FastAPI returns a `StreamingResponse` with media type `text/event-stream`
- The LangChain agent must stream via its async streaming interface (e.g. `astream_events`) rather than the blocking `invoke` call used in F6
- When the agent is calling a tool, emit a `tool_call` event so the frontend can show a "checking your order…" style state instead of just frozen silence
- Once generation finishes, emit a final `done` event containing the complete reply, and persist that complete reply to `ChatMessage` (same persistence rule as F6 — streaming changes delivery, not storage)
- If the client disconnects mid-stream, the server should stop generating and clean up the connection rather than continuing to burn LLM compute
- `POST /chat` (non-streaming, from F6) stays available — some integrations (e.g. WhatsApp webhooks) need a single final response, not a stream

**Frontend changes (extends F9)**
- Replace the single `fetch()` call with an `EventSource`-style reader (or `fetch` + `ReadableStream`, since `EventSource` only supports GET — confirm which approach fits the chosen endpoint method) that appends each `token` event's text to the current assistant bubble as it arrives
- Show a distinct "thinking" indicator while a `tool_call` event is active and no tokens have arrived yet
- Disable the input box while a stream is in progress; re-enable on `done`

**Acceptance criteria**
- [ ] Assistant text visibly appears in chunks in the browser, not all at once
- [ ] A tool-calling question (e.g. order status) shows a "checking…" state before tokens start streaming
- [ ] The final stored `ChatMessage` matches exactly what was streamed (no truncation or duplication)
- [ ] Closing the browser tab mid-stream does not leave the backend stuck generating indefinitely
- [ ] `POST /chat` still works unchanged for non-streaming consumers

**Dependencies:** F5 (agent core), F6 (chat API — this extends it, doesn't replace it), F9 (frontend widget)

---

### F12 — Feedback & analytics

**Goal:** capture quality signals from real conversations and give the store owner visibility into how the agent is performing.

**Part A — Per-message feedback**

Endpoint: `POST /feedback`
```json
{ "message_id": 123, "rating": "up | down", "comment": "string | null" }
```
- Stores a row in `MessageFeedback` linked to the exact `ChatMessage`
- Frontend (extends F9): thumbs up/down icons under each assistant bubble; clicking sends the request and shows a brief confirmation

**Part B — Retrieval gap logging**

- Every call to the `search_policy_docs` tool (from F4) logs a row to `RetrievalLog`: the query, the top similarity score, and a `was_confident` flag (score above a configured threshold)
- Low-confidence retrievals are the signal that the FAQ/policy content is missing something — this is for the store owner to review and add documents for, not for the agent to act on differently

**Part C — Admin analytics endpoint**

Endpoint (admin, protected by F14): `GET /analytics/summary?from=<date>&to=<date>`
```json
{
  "total_sessions": 0,
  "total_messages": 0,
  "escalation_rate": 0.0,
  "feedback_positive_pct": 0.0,
  "top_tools_used": [{"tool": "get_order_status", "count": 0}],
  "low_confidence_queries": [{"query": "string", "top_score": 0.0}]
}
```
- `escalation_rate` = tickets created (F7) ÷ total sessions in the period
- `feedback_positive_pct` = up-votes ÷ (up-votes + down-votes) in the period
- `top_tools_used` aggregated from tool-call logs (F10)
- `low_confidence_queries` pulled from `RetrievalLog` where `was_confident` is false

**Acceptance criteria**
- [ ] Submitting feedback on a message persists correctly and is linked to the right `message_id`
- [ ] A query that scores below the confidence threshold appears in `RetrievalLog` with `was_confident = false`
- [ ] `GET /analytics/summary` returns numbers that match what's in the seeded/test data for a known date range
- [ ] `GET /analytics/summary` rejects requests without a valid admin token (see F14); `POST /feedback` stays open to customers since it carries no sensitive data

**Dependencies:** F4 (tool calls to log), F6/F11 (messages to attach feedback to), F7 (ticket data for escalation rate), F9 (frontend feedback buttons)

---

### F13 — Error handling & resilience

**Goal:** the system fails gracefully at every layer instead of leaking stack traces, crashing the agent loop, or leaving the user with no response at all.

**API layer**
- A global FastAPI exception handler catches any unhandled exception, logs the full traceback server-side, and returns a generic `{"error": "Something went wrong, please try again."}` with a 500 status — never leak internal details to the client
- Pydantic validation errors (malformed request bodies) keep FastAPI's default clean 422 response with field-level detail — don't override these with the generic handler

**LLM / Gemini API layer**
- Wrap calls to the Gemini API in a timeout (e.g. 30s) with one retry on timeout or connection error
- Handle `429` (rate limit / quota exceeded) responses specifically — back off and retry once after a short delay rather than treating it like a generic failure; if it's still rate-limited, fall back to the apology message below
- If the API is still unreachable or over quota after the retry, return a fixed fallback message ("I'm having trouble connecting right now, please try again in a moment") instead of a raw exception, and log the failure
- `GET /health` (from F1) must reflect this honestly — if the Gemini API key is invalid or the service is down, the health check should show it

**Agent / tool layer**
- Every tool call is wrapped in try/except; a tool exception returns a structured "this lookup failed" string back to the agent instead of crashing the request — this generalizes F4's "order not found" handling to all error types, not just missing orders
- Set `max_iterations` (e.g. 5) on the agent executor so a confused agent can't loop forever calling tools; hitting the limit triggers a "let me get a human to help" response and creates a ticket (reuses F7)
- Repeated mismatched order/email attempts (F8) from the same session are throttled — e.g. after 5 mismatches in 10 minutes, stop trying lookups for that session and offer to create a ticket instead of allowing unlimited guesses

**Database layer**
- DB writes happen inside a transaction that rolls back cleanly on failure, instead of leaving a half-written state (e.g. the user's message saved but the assistant's reply never written)
- A DB connection failure at startup fails loudly rather than booting silently with a broken DB — `GET /health` should catch this too

**Streaming layer (extends F11)**
- If the agent errors mid-stream, emit an explicit `error` SSE event so the frontend shows a clear message instead of a stream that just stops with no explanation
```
event: error
data: {"message": "Something went wrong, please try again."}
```

**Frontend layer (extends F9)**
- A network failure (server unreachable) shows a retry-able error bubble instead of a spinner that's stuck forever
- A failed message is visibly marked as failed, with a way to resend it — never silently dropped

**Acceptance criteria**
- [ ] Simulating a Gemini API outage or a `429` rate-limit response still returns a clean, user-facing message — not a 500 with a stack trace
- [ ] A tool that throws an exception doesn't crash the `/chat` request; the agent recovers and responds
- [ ] An agent stuck in a tool-calling loop is cut off at `max_iterations` and escalates to a ticket instead of hanging
- [ ] Killing the DB connection mid-request doesn't leave partial or corrupted chat history
- [ ] A mid-stream failure in F11 sends a visible `error` event, not a silent disconnect
- [ ] No endpoint ever returns a raw Python traceback to the client
- [ ] Five consecutive mismatched order/email attempts in one session trigger a cool-down/ticket offer instead of allowing further guesses

**Dependencies:** cuts across F1, F4, F5, F6, F7, F9, F11 — best implemented as a hardening pass once F1–F11 exist and work in the happy path, right before handing the project to a real client.

---

### F14 — Admin authentication

**Goal:** close the loophole where `GET /tickets` (F7), `GET /analytics/summary` (F12), and `PATCH /refunds/{refund_id}` (F4) are reachable by anyone with the URL — none of them have any access control in the spec as originally written.

**Requirements**
- Simplest viable approach for this project's scale: a single shared admin token, not a full user/login system — `.env` provides `ADMIN_TOKEN` (already added to F1)
- Every admin endpoint requires a header `X-Admin-Token` matching `ADMIN_TOKEN`; a missing or wrong token returns `401` before any data is touched
- This is intentionally lightweight. If the client later wants multiple admin accounts with different roles or permissions, that's a separate, bigger feature — flag it as a future upgrade rather than building it now

**Acceptance criteria**
- [ ] Calling any admin endpoint without `X-Admin-Token` returns `401`, not the data
- [ ] Calling with the correct token returns the expected data
- [ ] The token is never referenced anywhere in the frontend (F9) — admin endpoints are not, and should never be, called from the customer-facing chat widget

**Dependencies:** protects F4's refund-approval endpoint, F7's ticket endpoint, and F12's analytics endpoint. Numbered last here because it was the gap found during review, but in practice, implement it before any of those three endpoints are exposed outside local development — don't treat the numbering as the only valid build order for this one.

---

## Recommended build order

F1 → F2 → F3 → F4 → F5 → F6 → F7 → F8 → F9 → F10 → F11 → F12 → F13 → F14

Implement one feature, verify it against its own acceptance criteria, then move to the next. This is what makes the process spec-driven: each section above is one self-contained, verifiable unit of work that Claude Code can be pointed at individually.

F11 and F12 are additive layers on top of F6 and F9 — they extend the existing chat endpoint and frontend rather than replacing them. F13 is a hardening pass across everything built so far. F14 closes an access-control gap found during a spec review (admin endpoints had no auth at all) — implement it alongside F7 and F12 in practice, even though it's listed last.
