"""Per-message feedback endpoint (F12 Part A).

Customers submit thumbs-up/down ratings for individual assistant messages.
Open to customers — no admin auth required.
"""

from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from sqlmodel import Session

from app.db import engine
from app.models import ChatMessage, FeedbackRating, MessageFeedback

router = APIRouter()


class FeedbackRequest(BaseModel):
    message_id: int
    rating: FeedbackRating
    comment: Optional[str] = None


@router.post("/feedback", status_code=201)
def submit_feedback(body: FeedbackRequest):
    with Session(engine) as session:
        if session.get(ChatMessage, body.message_id) is None:
            raise HTTPException(status_code=404, detail="Message not found.")
        fb = MessageFeedback(
            message_id=body.message_id,
            rating=body.rating,
            comment=body.comment,
        )
        session.add(fb)
        session.commit()
        session.refresh(fb)
        return {"ok": True, "feedback_id": fb.id}
