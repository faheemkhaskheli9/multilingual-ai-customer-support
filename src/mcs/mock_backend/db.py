"""Persistence for the mock order/invoice backend (issue #3).

Production target is PostgreSQL (README §3); the boundary is a SQLAlchemy
engine URL read from ``DATABASE_URL``, defaulting to a local SQLite file so
the service and its tests run with zero external services. Swapping to
Postgres is a connection-string change, not a code change — the same
convention used elsewhere in this portfolio (see ai-rfp-generator/src/
ai_rfp_generator/db.py).

Sample data mirrors the synthetic in-memory records in
``mcs.backends.orders``/``mcs.backends.invoices`` (same ids, same
generic/non-proprietary content) so the two stay swappable without callers
noticing a data change.
"""

from __future__ import annotations

import os

from sqlalchemy import Float, ForeignKey, Integer, String, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, relationship, sessionmaker


class Base(DeclarativeBase):
    pass


class Order(Base):
    __tablename__ = "orders"

    order_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    status: Mapped[str] = mapped_column(String(32))
    carrier: Mapped[str | None] = mapped_column(String(64), nullable=True)
    eta: Mapped[str | None] = mapped_column(String(32), nullable=True)


class Invoice(Base):
    __tablename__ = "invoices"

    order_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    invoice_date: Mapped[str] = mapped_column(String(32))
    tax: Mapped[float] = mapped_column(Float)

    line_items: Mapped[list["InvoiceLineItem"]] = relationship(
        back_populates="invoice", cascade="all, delete-orphan", order_by="InvoiceLineItem.id"
    )


class InvoiceLineItem(Base):
    __tablename__ = "invoice_line_items"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    invoice_order_id: Mapped[str] = mapped_column(ForeignKey("invoices.order_id"), nullable=False)
    description: Mapped[str] = mapped_column(String(255))
    amount: Mapped[float] = mapped_column(Float)

    invoice: Mapped[Invoice] = relationship(back_populates="line_items")


def make_engine(database_url: str | None = None):
    database_url = database_url or os.environ.get("DATABASE_URL", "sqlite:///./mock_backend.db")
    connect_args = {"check_same_thread": False} if database_url.startswith("sqlite") else {}
    engine = create_engine(database_url, connect_args=connect_args)
    Base.metadata.create_all(engine)
    return engine


def make_session_factory(engine) -> sessionmaker[Session]:
    return sessionmaker(bind=engine, expire_on_commit=False)


# Synthetic sample data — no employer/client content — mirroring
# mcs.backends.orders/invoices so both stay swappable with the same ids.
_SAMPLE_ORDERS = [
    Order(order_id="1234", status="shipped", carrier="GenericPost", eta="2026-09-10"),
    Order(order_id="5678", status="processing", carrier=None, eta=None),
    Order(order_id="9012", status="delivered", carrier="GenericPost", eta="2026-09-01"),
    Order(order_id="3456", status="cancelled", carrier=None, eta=None),
]

_SAMPLE_INVOICES: list[tuple[Invoice, list[InvoiceLineItem]]] = [
    (
        Invoice(order_id="1234", invoice_date="2026-09-08", tax=2.24),
        [
            InvoiceLineItem(description="Wireless mouse", amount=19.99),
            InvoiceLineItem(description="USB-C cable", amount=8.00),
        ],
    ),
    (
        Invoice(order_id="9012", invoice_date="2026-08-30", tax=2.76),
        [InvoiceLineItem(description="Desk lamp", amount=34.50)],
    ),
]


def seed(session: Session) -> None:
    """Populate the database with sample orders/invoices, idempotently.

    Safe to call repeatedly (e.g. on every service start): existing rows are
    left untouched rather than duplicated, keyed by ``order_id``.
    """
    existing_order_ids = {row.order_id for row in session.query(Order.order_id).all()}
    for order in _SAMPLE_ORDERS:
        if order.order_id not in existing_order_ids:
            session.add(Order(order_id=order.order_id, status=order.status, carrier=order.carrier, eta=order.eta))

    existing_invoice_ids = {row.order_id for row in session.query(Invoice.order_id).all()}
    for invoice, line_items in _SAMPLE_INVOICES:
        if invoice.order_id not in existing_invoice_ids:
            new_invoice = Invoice(order_id=invoice.order_id, invoice_date=invoice.invoice_date, tax=invoice.tax)
            new_invoice.line_items = [
                InvoiceLineItem(description=item.description, amount=item.amount) for item in line_items
            ]
            session.add(new_invoice)

    session.commit()
