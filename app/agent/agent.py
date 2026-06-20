"""LangChain tool-calling agent core (F5).

Wires the F4 customer tools to a Gemini chat model behind the system prompt,
loads recent per-session history from the database, and runs a turn.

Persisting messages is the chat API's job (F6); this module only reads history
and returns the assistant's reply.
"""

import asyncio
import json
import logging
import time
import uuid
from functools import lru_cache

import httpx
from google.genai.errors import ClientError
from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.errors import GraphRecursionError
from sqlmodel import Session

from app.agent.logging_callback import ToolLoggingCallback
from app.agent.prompts import build_system_prompt
from app.agent.tools import CUSTOMER_TOOLS, current_session_id
from app.config import settings
from app.db import engine
from app.models import ChatMessage, MessageRole

# Last 20 messages (~10 exchanges) carried as memory; older history is dropped
# rather than summarized for now (F5).
HISTORY_LIMIT = 20

FALLBACK_MESSAGE = (
    "I'm having trouble connecting right now, please try again in a moment."
)

_RECURSION_LIMIT_ERROR_SSE = (
    'event: error\ndata: {"message": "I wasn\'t able to resolve this in time.'
    ' A support ticket has been opened for you."}\n\n'
)
_GENERIC_ERROR_SSE = (
    'event: error\ndata: {"message": "Something went wrong, please try again."}\n\n'
)


@lru_cache(maxsize=1)
def get_agent():
    """Build (once) the tool-calling agent bound to the Gemini chat model."""
    llm = ChatGoogleGenerativeAI(
        model=settings.model_name,
        google_api_key=settings.google_api_key,
        temperature=0.3,
        timeout=settings.llm_timeout_seconds,
        max_retries=0,  # we own retry logic explicitly (F13)
    )
    return create_agent(llm, tools=CUSTOMER_TOOLS)


def _call_with_retry(fn, *args, **kwargs):
    """Call fn(*args, **kwargs), retrying once on timeout/connect/429 errors."""
    try:
        return fn(*args, **kwargs)
    except httpx.TimeoutException:
        logging.warning("LLM call timed out, retrying once…")
        time.sleep(1)
        return fn(*args, **kwargs)
    except httpx.ConnectError:
        logging.warning("LLM connection error, retrying once…")
        time.sleep(1)
        return fn(*args, **kwargs)
    except ClientError as e:
        if e.code == 429:
            logging.warning("LLM rate-limited (429), backing off and retrying…")
            time.sleep(2)
            return fn(*args, **kwargs)
        raise


def _escalate_on_loop(session_id: uuid.UUID) -> str:
    """Open a ticket when the agent hits max_iterations and return a reply."""
    from app.agent.tools import _open_ticket
    try:
        with Session(engine) as db:
            ticket_msg = _open_ticket(
                db,
                session_id,
                "Agent exceeded maximum steps — human review needed",
            )
    except Exception:
        logging.exception("Failed to open escalation ticket for session %s", session_id)
        ticket_msg = "Please contact our support team directly."
    return f"I wasn't able to fully resolve your question automatically. {ticket_msg}"


def _load_history(session_id: uuid.UUID) -> list:
    """Load the last HISTORY_LIMIT user/assistant messages for a session, in
    chronological order, as LangChain messages.

    Tool-role rows are skipped: they can't be safely replayed without the
    originating tool-call metadata, and F6 only persists user and assistant
    turns anyway.
    """
    from sqlmodel import select
    with Session(engine) as session:
        rows = session.exec(
            select(ChatMessage)
            .where(ChatMessage.session_id == session_id)
            .order_by(ChatMessage.created_at.desc())
            .limit(HISTORY_LIMIT)
        ).all()

    messages = []
    for row in reversed(rows):
        if row.role == MessageRole.user:
            messages.append(HumanMessage(content=row.content))
        elif row.role == MessageRole.assistant:
            messages.append(AIMessage(content=row.content))
    return messages


def run_agent(
    session_id: uuid.UUID | str,
    message: str,
    customer_email: str | None = None,
) -> str:
    """Run one agent turn for a session and return the assistant's reply text.

    Never raises — returns a fallback string on any LLM/network failure so the
    caller always gets a string it can return to the client (F13).
    """
    if isinstance(session_id, str):
        session_id = uuid.UUID(session_id)

    current_session_id.set(session_id)

    messages = [SystemMessage(content=build_system_prompt(customer_email))]
    messages.extend(_load_history(session_id))
    messages.append(HumanMessage(content=message))

    config = RunnableConfig(
        callbacks=[ToolLoggingCallback()],
        # LangGraph counts each node transition; each tool call ≈ 2 transitions.
        recursion_limit=settings.agent_max_iterations * 2,
    )

    try:
        result = _call_with_retry(
            get_agent().invoke, {"messages": messages}, config=config
        )
        return result["messages"][-1].content
    except GraphRecursionError:
        logging.warning("Agent hit recursion limit for session %s, escalating", session_id)
        return _escalate_on_loop(session_id)
    except Exception:
        logging.exception("LLM call failed for session %s", session_id)
        return FALLBACK_MESSAGE


async def stream_agent(
    session_id: uuid.UUID | str,
    message: str,
    customer_email: str | None = None,
):
    """Async generator yielding SSE-formatted strings for one agent turn.

    Emits four event types:
      token     — one chunk of assistant text
      tool_call — the agent is calling a named tool
      done      — generation finished; carries session_id and full_reply
      error     — unrecoverable failure; carries a user-facing message (F13)
    """
    if isinstance(session_id, str):
        session_id = uuid.UUID(session_id)

    current_session_id.set(session_id)

    messages = [SystemMessage(content=build_system_prompt(customer_email))]
    messages.extend(_load_history(session_id))
    messages.append(HumanMessage(content=message))

    config = RunnableConfig(
        callbacks=[ToolLoggingCallback()],
        recursion_limit=settings.agent_max_iterations * 2,
    )
    parts: list[str] = []

    try:
        async for event in get_agent().astream_events(
            {"messages": messages}, config=config, version="v2"
        ):
            kind = event["event"]

            if kind == "on_chat_model_stream":
                content = event["data"]["chunk"].content
                text = content if isinstance(content, str) else ""
                if text:
                    parts.append(text)
                    yield f'event: token\ndata: {json.dumps({"text": text})}\n\n'

            elif kind == "on_tool_start":
                tool_name = event.get("name", "")
                yield f'event: tool_call\ndata: {json.dumps({"tool": tool_name})}\n\n'

        full_reply = "".join(parts)
        yield f'event: done\ndata: {json.dumps({"session_id": str(session_id), "full_reply": full_reply})}\n\n'

    except GraphRecursionError:
        logging.warning("stream_agent hit recursion limit for session %s, escalating", session_id)
        escalation_msg = _escalate_on_loop(session_id)
        yield f'event: error\ndata: {json.dumps({"message": escalation_msg})}\n\n'

    except Exception:
        logging.exception("stream_agent failed for session %s", session_id)
        yield _GENERIC_ERROR_SSE
