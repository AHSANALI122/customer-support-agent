import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Optional
from sqlmodel import Field, SQLModel


class FeedbackRating(str, Enum):
    up = "up"
    down = "down"


class MessageFeedback(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    message_id: int = Field(foreign_key="chatmessage.id")
    rating: FeedbackRating
    comment: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class RetrievalLog(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    session_id: Optional[uuid.UUID] = Field(default=None, foreign_key="chatsession.id")
    query: str
    top_score: float
    was_confident: bool
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ToolCallLog(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    session_id: Optional[uuid.UUID] = Field(default=None, foreign_key="chatsession.id")
    tool_name: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
