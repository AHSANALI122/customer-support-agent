from datetime import datetime, timedelta, timezone
from sqlmodel import Session, select
from app.db import engine, init_db
from app.models import (
    Customer, Product, Order, OrderItem, OrderStatus,
    RefundRequest, RefundStatus,
)


def seed():
    init_db()

    with Session(engine) as session:
        # --- Customers (idempotent by email) ---
        customers_data = [
            {"name": "Alice Johnson", "email": "alice@example.com", "phone": "555-0101"},
            {"name": "Bob Smith", "email": "bob@example.com", "phone": "555-0102"},
            {"name": "Carol White", "email": "carol@example.com", "phone": None},
        ]
        customers = []
        for data in customers_data:
            existing = session.exec(select(Customer).where(Customer.email == data["email"])).first()
            if existing:
                customers.append(existing)
            else:
                c = Customer(**data)
                session.add(c)
                session.flush()
                customers.append(c)

        # --- Products (idempotent by name) ---
        products_data = [
            {"name": "Wireless Headphones", "price": 79.99, "stock": 50},
            {"name": "USB-C Hub", "price": 34.99, "stock": 120},
            {"name": "Mechanical Keyboard", "price": 129.99, "stock": 30},
            {"name": "Webcam 1080p", "price": 59.99, "stock": 75},
            {"name": "Laptop Stand", "price": 44.99, "stock": 200},
        ]
        products = []
        for data in products_data:
            existing = session.exec(select(Product).where(Product.name == data["name"])).first()
            if existing:
                products.append(existing)
            else:
                p = Product(**data)
                session.add(p)
                session.flush()
                products.append(p)

        session.commit()
        # Refresh to get IDs
        for c in customers:
            session.refresh(c)
        for p in products:
            session.refresh(p)

        # --- Orders (skip if customer already has orders) ---
        existing_orders = session.exec(select(Order)).all()
        if existing_orders:
            print(f"Seed data already present ({len(existing_orders)} orders found). Skipping orders.")
            return

        now = datetime.now(timezone.utc)

        orders_spec = [
            # (customer_idx, status, tracking_number, days_ago, [(product_idx, qty)])
            (0, OrderStatus.pending,   None,          1,  [(0, 1), (1, 2)]),
            (0, OrderStatus.pending,   None,          2,  [(2, 1)]),
            (1, OrderStatus.shipped,   "TRK100001",   5,  [(3, 1), (4, 1)]),
            (1, OrderStatus.shipped,   "TRK100002",   7,  [(0, 2)]),
            (2, OrderStatus.delivered, None,          14, [(1, 1), (2, 1)]),
            (0, OrderStatus.delivered, None,          20, [(4, 3)]),
            (1, OrderStatus.delivered, None,          30, [(3, 1)]),
            (2, OrderStatus.cancelled, None,          10, [(0, 1)]),
        ]

        created_orders = []
        for cust_idx, status, tracking, days_ago, items in orders_spec:
            total = sum(products[prod_idx].price * qty for prod_idx, qty in items)
            order = Order(
                customer_id=customers[cust_idx].id,
                status=status,
                tracking_number=tracking,
                total=round(total, 2),
                created_at=now - timedelta(days=days_ago),
            )
            session.add(order)
            session.flush()

            for prod_idx, qty in items:
                item = OrderItem(
                    order_id=order.id,
                    product_id=products[prod_idx].id,
                    quantity=qty,
                    price=products[prod_idx].price,
                )
                session.add(item)

            created_orders.append(order)

        session.flush()

        # --- Refund requests on 2 delivered orders ---
        delivered_orders = [o for o in created_orders if o.status == OrderStatus.delivered]
        refund_specs = [
            (delivered_orders[0], "Item arrived damaged"),
            (delivered_orders[1], "Wrong item received"),
        ]
        for order, reason in refund_specs:
            refund = RefundRequest(
                order_id=order.id,
                reason=reason,
                status=RefundStatus.requested,
            )
            session.add(refund)

        session.commit()
        print(f"Seeded: {len(customers)} customers, {len(products)} products, "
              f"{len(created_orders)} orders, 2 refund requests.")


if __name__ == "__main__":
    seed()
