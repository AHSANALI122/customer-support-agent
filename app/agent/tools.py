"""LangChain tools that read and write live order data (F4) and escalate to a
human (F7).

Every tool that touches a specific order takes `email` as a required
parameter and verifies it against the order's customer before returning
anything — this enforces the lightweight verification rule from F8.
"""

import contextvars
import logging
import uuid
from datetime import datetime, timedelta, timezone

from langchain_core.tools import tool
from sqlmodel import Session, select

from app.config import settings
from app.db import engine
from app.models import (
    ChatSession,
    Customer,
    Order,
    OrderItem,
    OrderStatus,
    Product,
    RefundRequest,
    RefundStatus,
    RetrievalLog,
    SupportTicket,
    TicketStatus,
    VerificationThrottle,
)
from app.rag.retriever import search_with_scores

# Set by run_agent (F5) before each turn so escalation can tie a ticket to the
# active session without the LLM ever handling the internal UUID (F7).
current_session_id: contextvars.ContextVar[uuid.UUID | None] = contextvars.ContextVar(
    "current_session_id", default=None
)

# Generic message used whenever an email doesn't match an order. It never
# reveals which value was wrong, so it can't be used to probe for valid
# order IDs or emails.
MISMATCH_MSG = (
    "I couldn't match that order ID with that email. Please confirm both the "
    "order ID and the email address used for the purchase."
)

# Returned once an email has used up its verification attempts (F8). The
# message stays generic so it never confirms whether the email is even real.
THROTTLE_MSG = (
    "For your security I've paused order lookups for this email after several "
    "failed verification attempts."
)


def _throttle_key(email: str) -> str:
    """Normalise the email used as the throttle key. Lower-cased so trivial
    case changes can't be used to dodge the limit; empty when no email was
    supplied, in which case the caller skips throttling entirely."""
    return (email or "").strip().lower()


def _is_throttled(session: Session, key: str) -> bool:
    """True if this email has hit the failed-attempt limit within the window.

    Keyed on the email being verified rather than the chat session, so starting
    a fresh session does not reset the limit (F8). A window that has fully
    elapsed since the last failure is treated as not throttled, so a legitimate
    customer is never locked out indefinitely.
    """
    if not key:
        return False
    entry = session.get(VerificationThrottle, key)
    if entry is None or entry.last_mismatch_at is None:
        return False
    if entry.mismatch_count < settings.verification_max_attempts:
        return False
    window = timedelta(minutes=settings.verification_window_minutes)
    last = entry.last_mismatch_at
    if last.tzinfo is None:
        last = last.replace(tzinfo=timezone.utc)
    return datetime.now(timezone.utc) - last < window


def _record_failure(session: Session, key: str) -> None:
    """Count a failed order/email verification for an email, resetting the
    counter when the previous failure fell outside the rolling window."""
    if not key:
        return
    now = datetime.now(timezone.utc)
    entry = session.get(VerificationThrottle, key)
    if entry is None:
        entry = VerificationThrottle(email=key, mismatch_count=1, last_mismatch_at=now)
    else:
        last = entry.last_mismatch_at
        if last is not None and last.tzinfo is None:
            last = last.replace(tzinfo=timezone.utc)
        window = timedelta(minutes=settings.verification_window_minutes)
        if last is None or now - last >= window:
            # A fresh lockout window starts: clear the per-lockout ticket flag.
            entry.mismatch_count = 1
            entry.ticket_opened = False
        else:
            entry.mismatch_count += 1
        entry.last_mismatch_at = now
    session.add(entry)
    session.commit()


def _clear_failures(session: Session, key: str) -> None:
    """Reset the failed-attempt counter after a successful verification."""
    if not key:
        return
    entry = session.get(VerificationThrottle, key)
    if entry is None or (entry.mismatch_count == 0 and not entry.ticket_opened):
        return
    entry.mismatch_count = 0
    entry.last_mismatch_at = None
    entry.ticket_opened = False
    session.add(entry)
    session.commit()


def _try_open_throttle_ticket(
    session: Session, key: str, session_id: uuid.UUID | None
) -> str | None:
    """Open one escalation ticket per email lockout. Returns the ticket message
    the first time it's called during a lockout, or None if a ticket was already
    opened for this lockout (even from a different, rotated session) or there's
    no session to attach it to."""
    if session_id is None:
        return None
    entry = session.get(VerificationThrottle, key)
    if entry is None or entry.ticket_opened:
        return None
    msg = _open_ticket(
        session,
        session_id,
        "Identity verification needed — repeated failed order lookups",
    )
    entry.ticket_opened = True
    session.add(entry)
    session.commit()
    return msg


def _throttled_response(
    session: Session, key: str, session_id: uuid.UUID | None
) -> str | None:
    """If this email is currently locked out, count the attempt, open one
    escalation ticket per lockout, and return the message to show the customer.
    Returns None when the email is not throttled (the caller proceeds normally).
    Shared by every email-based lookup so none of them is an unthrottled hole."""
    if not _is_throttled(session, key):
        return None
    # Refresh the window so continued guessing keeps the email locked.
    _record_failure(session, key)
    ticket_msg = _try_open_throttle_ticket(session, key, session_id)
    if ticket_msg:
        return f"{THROTTLE_MSG} {ticket_msg}"
    return THROTTLE_MSG


def _verify_order(session: Session, order_id: int, email: str):
    """Resolve and verify an order against the supplied email.

    Returns (order, None) on success or (None, error_message) otherwise.
    A genuinely non-existent order_id yields a specific "not found"
    message; any email mismatch yields the generic MISMATCH_MSG.

    Repeated failures (not-found or mismatch) within the configured window are
    throttled per email: once the limit is hit, further failed lookups return
    THROTTLE_MSG instead of leaking which value was wrong, so order_id/email
    combinations can't be guessed without limit (F8). Keying on the email rather
    than the chat session means an attacker can't reset the limit by starting a
    new session. A genuinely correct order_id + email still verifies and clears
    the throttle immediately, so a legitimate customer is never locked out by
    their own earlier typos.
    """
    session_id = current_session_id.get()
    key = _throttle_key(email)
    order = session.get(Order, order_id)
    if order is not None:
        customer = session.exec(
            select(Customer).where(Customer.email == email)
        ).first()
        if customer is not None and order.customer_id == customer.id:
            _clear_failures(session, key)
            return order, None

    # Verification failed (unknown order or email mismatch). A correct
    # order_id + email above bypasses this, so a legitimate customer is never
    # blocked by the lockout — only continued failures are.
    throttled = _throttled_response(session, key, session_id)
    if throttled is not None:
        return None, throttled
    _record_failure(session, key)
    if order is None:
        return None, (
            f"I couldn't find an order with ID #{order_id}. "
            "Please double-check the order number."
        )
    return None, MISMATCH_MSG


@tool
def list_orders_by_email(email: str) -> str:
    """List a customer's recent orders (id, status, total, date) for the
    email used at checkout. Use this when a customer asks about their
    order(s) but hasn't provided a specific order ID."""
    # This tool hands back order data from an email alone, so it is the weakest
    # link (F8 #2): it must honour the per-email lockout, otherwise it could be
    # used to enumerate a victim's order IDs or to sidestep a lockout on the
    # order-specific tools. Unlike _verify_order there's no stronger proof to
    # bypass with, so we check the throttle up front before returning anything.
    try:
        session_id = current_session_id.get()
        key = _throttle_key(email)
        with Session(engine) as session:
            throttled = _throttled_response(session, key, session_id)
            if throttled is not None:
                return throttled
            customer = session.exec(
                select(Customer).where(Customer.email == email)
            ).first()
            if customer is None:
                # A non-existent account is a failed verification attempt — count it
                # so the email can't be probed without limit.
                _record_failure(session, key)
                return (
                    "I couldn't find any account with that email. Please confirm "
                    "the email address used for your purchase."
                )
            orders = session.exec(
                select(Order)
                .where(Order.customer_id == customer.id)
                .order_by(Order.created_at.desc())
                .limit(10)
            ).all()
            if not orders:
                return "I couldn't find any orders associated with that email."
            lines = [f"Found {len(orders)} recent order(s):"]
            for o in orders:
                lines.append(
                    f"- Order #{o.id} | {o.status.value} | "
                    f"${o.total:.2f} | {o.created_at.date()}"
                )
            return "\n".join(lines)
    except Exception:
        logging.exception("list_orders_by_email failed")
        return "I wasn't able to complete that lookup right now. Please try again in a moment."


@tool
def get_order_status(order_id: int, email: str) -> str:
    """Get the status, line items, and total for a specific order. Requires
    the order ID and the email used for the purchase to verify identity."""
    try:
        with Session(engine) as session:
            order, err = _verify_order(session, order_id, email)
            if err:
                return err
            items = session.exec(
                select(OrderItem).where(OrderItem.order_id == order.id)
            ).all()
            lines = [f"Order #{order.id} — status: {order.status.value}"]
            if items:
                lines.append("Items:")
                for it in items:
                    product = session.get(Product, it.product_id)
                    name = product.name if product else f"Product #{it.product_id}"
                    lines.append(f"- {name} x{it.quantity} @ ${it.price:.2f}")
            lines.append(f"Total: ${order.total:.2f}")
            return "\n".join(lines)
    except Exception:
        logging.exception("get_order_status failed")
        return "I wasn't able to complete that lookup right now. Please try again in a moment."


@tool
def get_tracking_info(order_id: int, email: str) -> str:
    """Get the tracking number and shipping status for a specific order.
    Requires the order ID and the email used for the purchase."""
    try:
        with Session(engine) as session:
            order, err = _verify_order(session, order_id, email)
            if err:
                return err
            if not order.tracking_number:
                return (
                    f"Order #{order.id} doesn't have a tracking number yet. "
                    f"Its current status is {order.status.value}."
                )
            return (
                f"Order #{order.id} — tracking number: {order.tracking_number} "
                f"(status: {order.status.value})."
            )
    except Exception:
        logging.exception("get_tracking_info failed")
        return "I wasn't able to complete that lookup right now. Please try again in a moment."


@tool
def get_refund_status(order_id: int, email: str) -> str:
    """Check whether a refund request exists for an order and report its
    current status. Requires the order ID and the email used for the
    purchase."""
    try:
        with Session(engine) as session:
            order, err = _verify_order(session, order_id, email)
            if err:
                return err
            refund = session.exec(
                select(RefundRequest).where(RefundRequest.order_id == order.id)
            ).first()
            if refund is None:
                return f"There's no refund request on file for order #{order.id}."
            return (
                f"Refund for order #{order.id} — status: {refund.status.value} "
                f"(requested {refund.created_at.date()})."
            )
    except Exception:
        logging.exception("get_refund_status failed")
        return "I wasn't able to complete that lookup right now. Please try again in a moment."


@tool
def create_refund_request(order_id: int, email: str, reason: str) -> str:
    """Create a refund request for an order. Only delivered orders are
    eligible for a refund. Requires the order ID, the email used for the
    purchase, and a reason for the refund."""
    try:
        with Session(engine) as session:
            order, err = _verify_order(session, order_id, email)
            if err:
                return err
            if order.status != OrderStatus.delivered:
                explanations = {
                    OrderStatus.pending: "it hasn't shipped yet",
                    OrderStatus.shipped: "it's still in transit and hasn't been delivered",
                    OrderStatus.cancelled: "it was cancelled",
                }
                why = explanations.get(
                    order.status, f"its status is {order.status.value}"
                )
                return (
                    f"Order #{order.id} isn't eligible for a refund because {why}. "
                    "Refunds can only be requested for delivered orders."
                )
            existing = session.exec(
                select(RefundRequest).where(RefundRequest.order_id == order.id)
            ).first()
            if existing is not None:
                return (
                    f"A refund request already exists for order #{order.id} "
                    f"(status: {existing.status.value}). No new request was created."
                )
            refund = RefundRequest(
                order_id=order.id, reason=reason, status=RefundStatus.requested
            )
            session.add(refund)
            session.commit()
            session.refresh(refund)
            return (
                f"Refund request #{refund.id} submitted for order #{order.id}. "
                "Our team will review it and follow up."
            )
    except Exception:
        logging.exception("create_refund_request failed")
        return "I wasn't able to complete that lookup right now. Please try again in a moment."


# Prefix the agent watches for to recognise a weak policy match. When it sees
# this, it hedges and offers escalation instead of answering confidently (F5).
LOW_CONFIDENCE_PREFIX = "LOW_CONFIDENCE"


@tool
def search_policy_docs(query: str) -> str:
    """Search the store's policy documents (returns, shipping, payments, and
    other general store policies) for information relevant to the customer's
    question."""
    try:
        results = search_with_scores(query)
        if not results:
            _log_retrieval(query, top_score=0.0, was_confident=False)
            return "I couldn't find any relevant policy information for that question."
        top_score = results[0][1]
        was_confident = top_score >= settings.confidence_threshold
        _log_retrieval(query, top_score=top_score, was_confident=was_confident)
        chunks = "\n\n---\n\n".join(doc.page_content for doc, _ in results)
        if not was_confident:
            return (
                f"{LOW_CONFIDENCE_PREFIX}: the closest policy match scored "
                f"{top_score:.2f}, below the {settings.confidence_threshold} "
                "confidence threshold. Treat the excerpts below as uncertain — do "
                "not state them as definitive:\n\n" + chunks
            )
        return chunks
    except Exception:
        logging.exception("search_policy_docs failed")
        return "I wasn't able to complete that lookup right now. Please try again in a moment."


def _log_retrieval(query: str, top_score: float, was_confident: bool) -> None:
    with Session(engine) as db:
        db.add(RetrievalLog(
            session_id=current_session_id.get(),
            query=query,
            top_score=top_score,
            was_confident=was_confident,
        ))
        db.commit()


# Ticket statuses that count as an active, unresolved case. A session already
# holding one of these shouldn't spawn a duplicate ticket (F7).
_ACTIVE_TICKET_STATUSES = [TicketStatus.open, TicketStatus.in_progress]


def _open_ticket(session: Session, session_id: uuid.UUID, subject: str) -> str:
    """Open a support ticket for a session, reusing the F7 duplicate guard so a
    session never spawns a second active ticket. Operates on the caller's DB
    session. Shared by the create_ticket tool and the verification throttle."""
    existing = session.exec(
        select(SupportTicket)
        .where(SupportTicket.session_id == session_id)
        .where(SupportTicket.status.in_(_ACTIVE_TICKET_STATUSES))
    ).first()
    if existing is not None:
        return (
            f"Your case is already with our support team (ticket "
            f"#{existing.id}). A human will follow up — there's no need to "
            "open another ticket."
        )
    chat_session = session.get(ChatSession, session_id)
    customer_id = chat_session.customer_id if chat_session else None
    ticket = SupportTicket(
        session_id=session_id,
        customer_id=customer_id,
        subject=subject,
        status=TicketStatus.open,
    )
    session.add(ticket)
    session.commit()
    session.refresh(ticket)
    return (
        f"I've opened support ticket #{ticket.id} for you. A member of our "
        "team will review the conversation and follow up."
    )


@tool
def create_ticket(subject: str) -> str:
    """Escalate the conversation to a human support agent by opening a support
    ticket. Call this when the customer explicitly asks to talk to a human, or
    when you cannot resolve their issue with the other tools. `subject` should
    be a short one-line summary of what the customer needs help with."""
    try:
        session_id = current_session_id.get()
        if session_id is None:
            # No active session context — never raise; report failure as text so the
            # agent can apologise rather than crash the turn (F4 convention).
            return (
                "I couldn't open a support ticket right now. Please try again in a "
                "moment."
            )
        with Session(engine) as session:
            return _open_ticket(session, session_id, subject)
    except Exception:
        logging.exception("create_ticket failed")
        return "I wasn't able to complete that lookup right now. Please try again in a moment."


# Registered together for the agent core (F5); create_ticket added for F7.
CUSTOMER_TOOLS = [
    list_orders_by_email,
    get_order_status,
    get_tracking_info,
    get_refund_status,
    create_refund_request,
    search_policy_docs,
    create_ticket,
]
