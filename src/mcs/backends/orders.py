"""In-memory mock order backend.

All data here is synthetic and generic — no employer/client data. The public
surface (``get_order_status``) is what the agent's ``get_order_status`` tool
calls; keep its signature stable so issue #3 can swap the implementation for a
real database without touching the tool layer.
"""

from __future__ import annotations

from dataclasses import dataclass


class OrderNotFoundError(LookupError):
    """Raised when an order id does not exist in the backend."""

    def __init__(self, order_id: str) -> None:
        super().__init__(f"order {order_id!r} not found")
        self.order_id = order_id


@dataclass(frozen=True)
class Order:
    order_id: str
    status: str
    carrier: str | None
    eta: str | None


# Synthetic sample data. Keys are strings so callers never have to guess at
# int-vs-str; real order numbers are frequently non-numeric.
_ORDERS: dict[str, Order] = {
    "1234": Order("1234", "shipped", "GenericPost", "2026-09-10"),
    "5678": Order("5678", "processing", None, None),
    "9012": Order("9012", "delivered", "GenericPost", "2026-09-01"),
    "3456": Order("3456", "cancelled", None, None),
}


def normalize_order_id(raw: str) -> str:
    """Strip a leading ``#`` and surrounding whitespace from a user-supplied id."""

    return raw.strip().lstrip("#").strip()


def get_order_status(order_id: str) -> dict[str, object]:
    """Return a status record for ``order_id``.

    Raises :class:`OrderNotFoundError` for unknown ids rather than returning a
    sentinel, so the caller must decide explicitly how to present "not found"
    (the agent tool wrapper turns it into a structured ``{"found": False}``
    payload the LLM can phrase safely).
    """

    key = normalize_order_id(order_id)
    if not key:
        raise OrderNotFoundError(order_id)
    try:
        order = _ORDERS[key]
    except KeyError as exc:
        raise OrderNotFoundError(order_id) from exc
    return {
        "found": True,
        "order_id": order.order_id,
        "status": order.status,
        "carrier": order.carrier,
        "eta": order.eta,
    }
