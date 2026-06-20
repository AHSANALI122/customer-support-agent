import uuid
from datetime import datetime, timezone
from typing import Optional

from sqlmodel import Field, SQLModel


class VerificationThrottle(SQLModel, table=True):
    """Per-email counter of failed order/email verification attempts (F8).

    Keyed on the (lowercased) email being verified rather than the chat
    session, so an attacker can't reset the limit by starting a fresh session.
    The email is the thing being guessed against, so this is what we throttle.
    """

    email: str = Field(primary_key=True)
    mismatch_count: int = Field(default=0)
    last_mismatch_at: Optional[datetime] = Field(default=None)
    # True once an escalation ticket has been opened for the current lockout, so
    # continued guessing (even across rotated sessions) can't spawn duplicate
    # tickets. Reset when the window lapses or a verification succeeds.
    ticket_opened: bool = Field(default=False)
