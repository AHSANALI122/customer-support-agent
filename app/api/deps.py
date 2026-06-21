from typing import Optional

from fastapi import Header, HTTPException

from app.config import settings


def verify_admin(
    x_admin_token: Optional[str] = Header(default=None, alias="X-Admin-Token"),
) -> None:
    """Reject any request without a valid admin token (F14)."""
    if not settings.admin_token or x_admin_token != settings.admin_token:
        raise HTTPException(status_code=401, detail="Invalid or missing admin token.")
