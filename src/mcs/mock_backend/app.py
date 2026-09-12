"""Mock order/invoice backend service (issue #3, Phase 1).

A standalone FastAPI service backed by PostgreSQL (SQLite locally/in tests —
see ``db.make_engine``), seeded with synthetic sample data, so the agent's
tool-calling flow can be built and tested against something that looks like
a real order/invoice system without any live production dependency.

Response shapes intentionally mirror ``mcs.backends.orders.get_order_status``
/ ``mcs.backends.invoices.get_invoice`` (``found``/``order_id``/... fields),
so a later issue can point the agent's tools at this HTTP service instead of
the in-process dict-backed mock without changing what callers see.

    uvicorn mcs.mock_backend.app:app --port 8100 --reload

The database connection/seed is deliberately *not* done at import time (an
importable module must stay a pure, cheap declaration — see the portfolio's
robustness rule on import-time side effects): it's created lazily, on first
use of a given ``$DATABASE_URL``. The env var is resolved into the cache key
itself (rather than read once and memoized behind it), so a test that
changes ``$DATABASE_URL`` between cases gets its own fresh engine instead of
silently reusing a stale one.
"""

from __future__ import annotations

import os

from fastapi import FastAPI, HTTPException

from .db import Invoice, Order, make_engine, make_session_factory, seed

app = FastAPI(title="Mock Order/Invoice Backend", version="0.1.0")

_session_factories: dict[str, object] = {}


def get_session_factory():
    """Return the session factory for the current ``$DATABASE_URL``, creating
    (and seeding) it on first use — cached per resolved URL, not globally, so
    the resolution depends only on that one input."""
    database_url = os.environ.get("DATABASE_URL", "sqlite:///./mock_backend.db")
    factory = _session_factories.get(database_url)
    if factory is None:
        engine = make_engine(database_url)
        factory = make_session_factory(engine)
        with factory() as session:
            seed(session)
        _session_factories[database_url] = factory
    return factory


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/orders/{order_id}")
def get_order(order_id: str) -> dict[str, object]:
    with get_session_factory()() as session:
        order = session.get(Order, order_id)
        if order is None:
            raise HTTPException(status_code=404, detail=f"order {order_id!r} not found")
        return {
            "found": True,
            "order_id": order.order_id,
            "status": order.status,
            "carrier": order.carrier,
            "eta": order.eta,
        }


@app.get("/invoices/{order_id}")
def get_invoice(order_id: str) -> dict[str, object]:
    with get_session_factory()() as session:
        invoice = session.get(Invoice, order_id)
        if invoice is None:
            raise HTTPException(status_code=404, detail=f"invoice for {order_id!r} not found")
        subtotal = round(sum(item.amount for item in invoice.line_items), 2)
        return {
            "found": True,
            "order_id": invoice.order_id,
            "date": invoice.invoice_date,
            "line_items": [
                {"description": item.description, "amount": item.amount}
                for item in invoice.line_items
            ],
            "subtotal": subtotal,
            "tax": invoice.tax,
            "total": round(subtotal + invoice.tax, 2),
        }
