import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Optional
from sqlmodel import Field, SQLModel


class MessageRole(str, Enum):
    user = "user"
    assistant = "assistant"
    tool = "tool"


class ChatSession(SQLModel, table=True):
    id: uuid.UUID = Field(default_factory=uuid.uuid4, primary_key=True)
    customer_id: Optional[int] = Field(default=None, foreign_key="customer.id")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ChatMessage(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    session_id: uuid.UUID = Field(foreign_key="chatsession.id")
    role: MessageRole
    content: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
