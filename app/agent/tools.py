"""LangChain tools that read and write live order data (F4).

Every tool that touches a specific order takes `email` as a required
parameter and verifies it against the order's customer before returning
anything — this enforces the lightweight verification rule from F8.
"""

from langchain_core.tools import tool
from sqlmodel import Session, select

from app.config import settings
from app.db import engine
from app.models import (
    Customer,
    Order,
    OrderItem,
    OrderStatus,
    Product,
    RefundRequest,
    RefundStatus,
)
from app.rag.retriever import search_with_scores

# Generic message used whenever an email doesn't match an order. It never
# reveals which value was wrong, so it can't be used to probe for valid
# order IDs or emails.
MISMATCH_MSG = (
    "I couldn't match that order ID with that email. Please confirm both the "
    "order ID and the email address used for the purchase."
)


def _verify_order(session: Session, order_id: int, email: str):
    """Resolve and verify an order against the supplied email.

    Returns (order, None) on success or (None, error_message) otherwise.
    A genuinely non-existent order_id yields a specific "not found"
    message; any email mismatch yields the generic MISMATCH_MSG.
    """
    order = session.get(Order, order_id)
    if order is None:
        return None, (
            f"I couldn't find an order with ID #{order_id}. "
            "Please double-check the order number."
        )
    customer = session.exec(
        select(Customer).where(Customer.email == email)
    ).first()
    if customer is None or order.customer_id != customer.id:
        return None, MISMATCH_MSG
    return order, None


@tool
def list_orders_by_email(email: str) -> str:
    """List a customer's recent orders (id, status, total, date) for the
    email used at checkout. Use this when a customer asks about their
    order(s) but hasn't provided a specific order ID."""
    with Session(engine) as session:
        customer = session.exec(
            select(Customer).where(Customer.email == email)
        ).first()
        if customer is None:
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


@tool
def get_order_status(order_id: int, email: str) -> str:
    """Get the status, line items, and total for a specific order. Requires
    the order ID and the email used for the purchase to verify identity."""
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


@tool
def get_tracking_info(order_id: int, email: str) -> str:
    """Get the tracking number and shipping status for a specific order.
    Requires the order ID and the email used for the purchase."""
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


@tool
def get_refund_status(order_id: int, email: str) -> str:
    """Check whether a refund request exists for an order and report its
    current status. Requires the order ID and the email used for the
    purchase."""
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


@tool
def create_refund_request(order_id: int, email: str, reason: str) -> str:
    """Create a refund request for an order. Only delivered orders are
    eligible for a refund. Requires the order ID, the email used for the
    purchase, and a reason for the refund."""
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


# Prefix the agent watches for to recognise a weak policy match. When it sees
# this, it hedges and offers escalation instead of answering confidently (F5).
LOW_CONFIDENCE_PREFIX = "LOW_CONFIDENCE"


@tool
def search_policy_docs(query: str) -> str:
    """Search the store's policy documents (returns, shipping, payments, and
    other general store policies) for information relevant to the customer's
    question."""
    results = search_with_scores(query)
    if not results:
        return "I couldn't find any relevant policy information for that question."
    top_score = results[0][1]
    chunks = "\n\n---\n\n".join(doc.page_content for doc, _ in results)
    if top_score < settings.confidence_threshold:
        return (
            f"{LOW_CONFIDENCE_PREFIX}: the closest policy match scored "
            f"{top_score:.2f}, below the {settings.confidence_threshold} "
            "confidence threshold. Treat the excerpts below as uncertain — do "
            "not state them as definitive:\n\n" + chunks
        )
    return chunks


# Registered together for the agent core (F5).
CUSTOMER_TOOLS = [
    list_orders_by_email,
    get_order_status,
    get_tracking_info,
    get_refund_status,
    create_refund_request,
    search_policy_docs,
]
