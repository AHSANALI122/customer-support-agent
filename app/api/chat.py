"""Chat API & session management (F6).

Exposes the F5 tool-calling agent over HTTP at `POST /chat` and persists every
conversation to the database, so history survives server restarts.

A new `ChatSession` is created automatically when no `session_id` is supplied
(or when an unknown one is). Both the user message and the assistant reply are
stored in `ChatMessage`.
"""

import asyncio
import json
import uuid
from typing import Optional

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlmodel import Session, select

from app.agent.agent import run_agent, stream_agent
from app.db import engine
from app.models import ChatMessage, ChatSession, Customer, MessageRole

router = APIRouter()


class ChatRequest(BaseModel):
    # session_id typed as UUID so a malformed value gets a clean 422.
    session_id: Optional[uuid.UUID] = None
    message: str
    customer_email: Optional[str] = None


class ChatResponse(BaseModel):
    session_id: uuid.UUID
    reply: str


def _resolve_customer_id(session: Session, customer_email: Optional[str]) -> Optional[int]:
    """Map a customer email to a customer id, or None if unknown/absent.

    A session without a matching customer is fine — general policy questions
    work without identifying the customer; the order/refund tools (F4) do their
    own per-order email verification.
    """
    if not customer_email:
        return None
    customer = session.exec(
        select(Customer).where(Customer.email == customer_email)
    ).first()
    return customer.id if customer else None


def _ensure_session(
    session: Session,
    session_id: Optional[uuid.UUID],
    customer_email: Optional[str],
) -> ChatSession:
    """Return an existing session, or create one — minting a UUID when none was
    supplied, or honoring a client-supplied UUID that doesn't exist yet."""
    if session_id is not None:
        existing = session.get(ChatSession, session_id)
        if existing is not None:
            return existing

    chat_session = ChatSession(
        customer_id=_resolve_customer_id(session, customer_email),
    )
    if session_id is not None:
        chat_session.id = session_id
    session.add(chat_session)
    session.commit()
    session.refresh(chat_session)
    return chat_session


@router.post("/chat", response_model=ChatResponse)
def chat(body: ChatRequest):
    with Session(engine) as session:
        chat_session = _ensure_session(session, body.session_id, body.customer_email)
        session_id = chat_session.id

    # Run the agent before persisting the user message: run_agent loads recent
    # history from the DB and appends the current message itself, so writing the
    # user turn first would duplicate it in the model's context.
    reply = run_agent(session_id, body.message, body.customer_email)

    # Persist user message + assistant reply together so we never leave a user
    # turn saved without its reply.
    with Session(engine) as session:
        session.add(
            ChatMessage(
                session_id=session_id,
                role=MessageRole.user,
                content=body.message,
            )
        )
        session.add(
            ChatMessage(
                session_id=session_id,
                role=MessageRole.assistant,
                content=reply,
            )
        )
        session.commit()

    return ChatResponse(session_id=session_id, reply=reply)


@router.post("/chat/stream")
async def chat_stream(body: ChatRequest, request: Request):
    """Stream the agent reply token-by-token as Server-Sent Events.

    Event types: token | tool_call | done
    The complete reply is persisted to ChatMessage only after the done event,
    matching F6's atomicity guarantee (user + assistant saved together).
    """
    with Session(engine) as session:
        chat_session = _ensure_session(session, body.session_id, body.customer_email)
        session_id = chat_session.id

    async def event_generator():
        try:
            async for chunk in stream_agent(session_id, body.message, body.customer_email):
                if await request.is_disconnected():
                    return
                if chunk.startswith("event: done"):
                    data = json.loads(chunk.split("data: ", 1)[1])
                    full_reply = data.get("full_reply")
                    message_id = None
                    if full_reply is not None:
                        with Session(engine) as db_session:
                            db_session.add(
                                ChatMessage(
                                    session_id=session_id,
                                    role=MessageRole.user,
                                    content=body.message,
                                )
                            )
                            assistant_msg = ChatMessage(
                                session_id=session_id,
                                role=MessageRole.assistant,
                                content=full_reply,
                            )
                            db_session.add(assistant_msg)
                            db_session.commit()
                            db_session.refresh(assistant_msg)
                            message_id = assistant_msg.id
                    yield f"event: done\ndata: {json.dumps({**data, 'message_id': message_id})}\n\n"
                    continue
                yield chunk
        except asyncio.CancelledError:
            return

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
