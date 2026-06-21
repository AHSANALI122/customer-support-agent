"""Admin analytics summary endpoint (F12 Part C).

Protected by the shared admin token (F14). Returns aggregate quality signals
for a given date range: session/message counts, escalation rate, feedback
sentiment, top tools, and low-confidence retrieval queries.
"""

from datetime import date, datetime, timezone

from fastapi import APIRouter, Depends, Query
from sqlmodel import Session, func, select

from app.api.deps import verify_admin
from app.db import engine
from app.models import (
    ChatMessage,
    ChatSession,
    FeedbackRating,
    MessageFeedback,
    RetrievalLog,
    SupportTicket,
    ToolCallLog,
)

router = APIRouter()

_TOP_TOOLS_LIMIT = 10


def _to_dt_range(from_date: date, to_date: date):
    from_dt = datetime(from_date.year, from_date.month, from_date.day, 0, 0, 0, tzinfo=timezone.utc)
    to_dt = datetime(to_date.year, to_date.month, to_date.day, 23, 59, 59, tzinfo=timezone.utc)
    return from_dt, to_dt


@router.get("/analytics/summary", dependencies=[Depends(verify_admin)])
def analytics_summary(
    from_date: date = Query(..., alias="from"),
    to_date: date = Query(..., alias="to"),
):
    from_dt, to_dt = _to_dt_range(from_date, to_date)

    with Session(engine) as session:
        total_sessions = session.exec(
            select(func.count(ChatSession.id))
            .where(ChatSession.created_at >= from_dt)
            .where(ChatSession.created_at <= to_dt)
        ).one()

        total_messages = session.exec(
            select(func.count(ChatMessage.id))
            .where(ChatMessage.created_at >= from_dt)
            .where(ChatMessage.created_at <= to_dt)
        ).one()

        ticket_count = session.exec(
            select(func.count(SupportTicket.id))
            .where(SupportTicket.created_at >= from_dt)
            .where(SupportTicket.created_at <= to_dt)
        ).one()

        up_count = session.exec(
            select(func.count(MessageFeedback.id))
            .where(MessageFeedback.created_at >= from_dt)
            .where(MessageFeedback.created_at <= to_dt)
            .where(MessageFeedback.rating == FeedbackRating.up)
        ).one()

        down_count = session.exec(
            select(func.count(MessageFeedback.id))
            .where(MessageFeedback.created_at >= from_dt)
            .where(MessageFeedback.created_at <= to_dt)
            .where(MessageFeedback.rating == FeedbackRating.down)
        ).one()

        tool_rows = session.exec(
            select(ToolCallLog.tool_name, func.count(ToolCallLog.id).label("cnt"))
            .where(ToolCallLog.created_at >= from_dt)
            .where(ToolCallLog.created_at <= to_dt)
            .group_by(ToolCallLog.tool_name)
            .order_by(func.count(ToolCallLog.id).desc())
            .limit(_TOP_TOOLS_LIMIT)
        ).all()

        low_conf = session.exec(
            select(RetrievalLog)
            .where(RetrievalLog.created_at >= from_dt)
            .where(RetrievalLog.created_at <= to_dt)
            .where(RetrievalLog.was_confident == False)  # noqa: E712
            .order_by(RetrievalLog.created_at.desc())
        ).all()

    total_feedback = up_count + down_count
    escalation_rate = ticket_count / total_sessions if total_sessions > 0 else 0.0
    positive_pct = up_count / total_feedback if total_feedback > 0 else 0.0

    return {
        "total_sessions": total_sessions,
        "total_messages": total_messages,
        "escalation_rate": round(escalation_rate, 4),
        "feedback_positive_pct": round(positive_pct, 4),
        "top_tools_used": [{"tool": name, "count": cnt} for name, cnt in tool_rows],
        "low_confidence_queries": [
            {"query": r.query, "top_score": r.top_score} for r in low_conf
        ],
    }
