"""In-memory mock invoice backend.

All data here is synthetic and generic — no employer/client data. Mirrors
``backends.orders``: a stable public surface (``get_invoice``) that issue #3
can swap for a real database without touching the tool layer. Invoices are
keyed by the same order/account number used for order-status lookups, since
that is the identifier the user actually has on hand.
"""

from __future__ import annotations

from dataclasses import dataclass


class InvoiceNotFoundError(LookupError):
    """Raised when an order/account number has no associated invoice."""

    def __init__(self, order_id: str) -> None:
        super().__init__(f"invoice for {order_id!r} not found")
        self.order_id = order_id


@dataclass(frozen=True)
class LineItem:
    description: str
    amount: float


@dataclass(frozen=True)
class Invoice:
    order_id: str
    date: str
    line_items: tuple[LineItem, ...]
    tax: float

    @property
    def subtotal(self) -> float:
        return round(sum(item.amount for item in self.line_items), 2)

    @property
    def total(self) -> float:
        return round(self.subtotal + self.tax, 2)


# Synthetic sample data, keyed by the same order id used in `backends.orders`
# so a single number resolves both order status and invoice details.
_INVOICES: dict[str, Invoice] = {
    "1234": Invoice(
        "1234",
        "2026-09-08",
        (LineItem("Wireless mouse", 19.99), LineItem("USB-C cable", 8.00)),
        tax=2.24,
    ),
    "9012": Invoice(
        "9012",
        "2026-08-30",
        (LineItem("Desk lamp", 34.50),),
        tax=2.76,
    ),
}


def normalize_order_id(raw: str) -> str:
    """Strip a leading ``#`` and surrounding whitespace from a user-supplied id."""

    return raw.strip().lstrip("#").strip()


def get_invoice(order_id: str) -> dict[str, object]:
    """Return an invoice record for ``order_id``.

    Raises :class:`InvoiceNotFoundError` for unknown ids rather than returning
    a sentinel, so the caller must decide explicitly how to present "not
    found" (the agent tool wrapper turns it into a structured
    ``{"found": False}`` payload the LLM can phrase safely).
    """

    key = normalize_order_id(order_id)
    if not key:
        raise InvoiceNotFoundError(order_id)
    try:
        invoice = _INVOICES[key]
    except KeyError as exc:
        raise InvoiceNotFoundError(order_id) from exc
    return {
        "found": True,
        "order_id": invoice.order_id,
        "date": invoice.date,
        "line_items": [
            {"description": item.description, "amount": item.amount}
            for item in invoice.line_items
        ],
        "subtotal": invoice.subtotal,
        "tax": invoice.tax,
        "total": invoice.total,
    }
