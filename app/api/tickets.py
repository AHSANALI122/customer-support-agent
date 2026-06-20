"""Admin endpoint for human escalation / support tickets (F7).

Exposes `GET /tickets`, which lists the open (and in-progress) tickets the agent
has escalated, each with the last few chat messages from its session so a human
can pick up the case with enough context. Protected by the shared admin token
(F14) — the same dependency used for the refund-management endpoint.
"""

from fastapi import APIRouter, Depends
from sqlmodel import Session, select

from app.api.deps import verify_admin
from app.db import engine
from app.models import ChatMessage, SupportTicket, TicketStatus

router = APIRouter()

# How many recent messages to attach to each ticket as pick-up context.
_CONTEXT_MESSAGE_LIMIT = 5

# Tickets a human still needs to act on.
_ACTIVE_TICKET_STATUSES = [TicketStatus.open, TicketStatus.in_progress]


@router.get("/tickets", dependencies=[Depends(verify_admin)])
def list_tickets():
    """List active support tickets with recent conversation context (F7)."""
    with Session(engine) as session:
        tickets = session.exec(
            select(SupportTicket)
            .where(SupportTicket.status.in_(_ACTIVE_TICKET_STATUSES))
            .order_by(SupportTicket.created_at.desc())
        ).all()

        results = []
        for ticket in tickets:
            recent = session.exec(
                select(ChatMessage)
                .where(ChatMessage.session_id == ticket.session_id)
                .order_by(ChatMessage.created_at.desc())
                .limit(_CONTEXT_MESSAGE_LIMIT)
            ).all()
            results.append(
                {
                    "ticket_id": ticket.id,
                    "session_id": str(ticket.session_id),
                    "customer_id": ticket.customer_id,
                    "subject": ticket.subject,
                    "status": ticket.status.value,
                    "created_at": ticket.created_at,
                    "recent_messages": [
                        {
                            "role": m.role.value,
                            "content": m.content,
                            "created_at": m.created_at,
                        }
                        for m in reversed(recent)
                    ],
                }
            )
        return results
