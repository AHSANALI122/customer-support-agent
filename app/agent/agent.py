"""LangChain tool-calling agent core (F5).

Wires the F4 customer tools to a Gemini chat model behind the system prompt,
loads recent per-session history from the database, and runs a turn.

Persisting messages is the chat API's job (F6); this module only reads history
and returns the assistant's reply.
"""

import asyncio
import json
import uuid
from functools import lru_cache

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_google_genai import ChatGoogleGenerativeAI
from sqlmodel import Session, select

from app.agent.logging_callback import ToolLoggingCallback
from app.agent.prompts import build_system_prompt
from app.agent.tools import CUSTOMER_TOOLS, current_session_id
from app.config import settings
from app.db import engine
from app.models import ChatMessage, MessageRole

# Last 20 messages (~10 exchanges) carried as memory; older history is dropped
# rather than summarized for now (F5).
HISTORY_LIMIT = 20


@lru_cache(maxsize=1)
def get_agent():
    """Build (once) the tool-calling agent bound to the Gemini chat model.

    No fixed system_prompt is set here so each turn can supply its own system
    message carrying the session's email.
    """
    llm = ChatGoogleGenerativeAI(
        model=settings.model_name,
        google_api_key=settings.google_api_key,
        temperature=0.3,
    )
    return create_agent(llm, tools=CUSTOMER_TOOLS)


def _load_history(session_id: uuid.UUID) -> list:
    """Load the last HISTORY_LIMIT user/assistant messages for a session, in
    chronological order, as LangChain messages.

    Tool-role rows are skipped: they can't be safely replayed without the
    originating tool-call metadata, and F6 only persists user and assistant
    turns anyway.
    """
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
    """Run one agent turn for a session and return the assistant's reply text."""
    if isinstance(session_id, str):
        session_id = uuid.UUID(session_id)

    # Expose the session to the escalation tool (F7) without routing the UUID
    # through the LLM; the synchronous invoke below runs in this same context.
    current_session_id.set(session_id)

    messages = [SystemMessage(content=build_system_prompt(customer_email))]
    messages.extend(_load_history(session_id))
    messages.append(HumanMessage(content=message))

    result = get_agent().invoke(
        {"messages": messages},
        config=RunnableConfig(callbacks=[ToolLoggingCallback()]),
    )
    return result["messages"][-1].content


async def stream_agent(
    session_id: uuid.UUID | str,
    message: str,
    customer_email: str | None = None,
):
    """Async generator yielding SSE-formatted strings for one agent turn.

    Emits three event types:
      token     — one chunk of assistant text
      tool_call — the agent is calling a named tool
      done      — generation finished; carries session_id and full_reply
    """
    if isinstance(session_id, str):
        session_id = uuid.UUID(session_id)

    current_session_id.set(session_id)

    messages = [SystemMessage(content=build_system_prompt(customer_email))]
    messages.extend(_load_history(session_id))
    messages.append(HumanMessage(content=message))

    config = RunnableConfig(callbacks=[ToolLoggingCallback()])
    parts: list[str] = []

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
