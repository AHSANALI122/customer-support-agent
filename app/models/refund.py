from datetime import datetime, timezone
from enum import Enum
from typing import Optional
from sqlmodel import Field, SQLModel


class RefundStatus(str, Enum):
    requested = "requested"
    approved = "approved"
    rejected = "rejected"
    refunded = "refunded"


class RefundRequest(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    order_id: int = Field(foreign_key="order.id")
    reason: str
    status: RefundStatus = RefundStatus.requested
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
