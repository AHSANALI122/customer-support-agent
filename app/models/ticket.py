import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Optional
from sqlmodel import Field, SQLModel


class TicketStatus(str, Enum):
    open = "open"
    in_progress = "in_progress"
    resolved = "resolved"


class SupportTicket(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    session_id: uuid.UUID = Field(foreign_key="chatsession.id")
    customer_id: Optional[int] = Field(default=None, foreign_key="customer.id")
    subject: str
    status: TicketStatus = TicketStatus.open
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
