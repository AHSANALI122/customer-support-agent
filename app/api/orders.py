"""Admin endpoints for order/refund management (F4).

Currently exposes the refund-status update endpoint, which is the only way
a refund request's status advances past `requested` — approving a refund is
a human decision, never something the agent does on its own. Protected by
the shared admin token (F14).
"""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlmodel import Session

from app.api.deps import verify_admin
from app.db import engine
from app.models import RefundRequest, RefundStatus

router = APIRouter()


class RefundUpdate(BaseModel):
    # Typed as RefundStatus so FastAPI returns a clean 422 for invalid values.
    status: RefundStatus


@router.patch("/refunds/{refund_id}", dependencies=[Depends(verify_admin)])
def update_refund(refund_id: int, body: RefundUpdate):
    # The agent only ever creates requests in the `requested` state; this
    # endpoint advances them, but can't reset one back to the initial state.
    if body.status == RefundStatus.requested:
        raise HTTPException(
            status_code=400,
            detail="Refund status can't be set back to 'requested'.",
        )
    with Session(engine) as session:
        refund = session.get(RefundRequest, refund_id)
        if refund is None:
            raise HTTPException(
                status_code=404, detail=f"Refund request #{refund_id} not found."
            )
        refund.status = body.status
        session.add(refund)
        session.commit()
        session.refresh(refund)
        return {
            "refund_id": refund.id,
            "order_id": refund.order_id,
            "status": refund.status.value,
        }
