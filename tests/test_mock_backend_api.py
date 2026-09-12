"""Integration tests for the mock order/invoice backend (issue #3)."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client(tmp_path, monkeypatch):
    """A TestClient wired to a fresh, seeded, throwaway SQLite DB per test."""
    db_path = tmp_path / "mock_backend.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_path}")

    from mcs.mock_backend.app import app

    return TestClient(app)


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_get_known_order_returns_seeded_data(client):
    response = client.get("/orders/1234")
    assert response.status_code == 200
    body = response.json()
    assert body == {
        "found": True,
        "order_id": "1234",
        "status": "shipped",
        "carrier": "GenericPost",
        "eta": "2026-09-10",
    }


def test_get_order_with_no_carrier_or_eta(client):
    response = client.get("/orders/5678")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "processing"
    assert body["carrier"] is None
    assert body["eta"] is None


def test_unknown_order_is_404(client):
    response = client.get("/orders/does-not-exist")
    assert response.status_code == 404


def test_get_known_invoice_returns_seeded_line_items_and_totals(client):
    response = client.get("/invoices/1234")
    assert response.status_code == 200
    body = response.json()
    assert body["order_id"] == "1234"
    assert len(body["line_items"]) == 2
    assert body["subtotal"] == pytest.approx(27.99)
    assert body["tax"] == pytest.approx(2.24)
    assert body["total"] == pytest.approx(30.23)


def test_order_with_no_invoice_is_404(client):
    # order 5678 exists but has no invoice in the seed data
    response = client.get("/invoices/5678")
    assert response.status_code == 404


def test_unknown_invoice_is_404(client):
    response = client.get("/invoices/does-not-exist")
    assert response.status_code == 404


def test_seeding_is_idempotent_across_requests(client):
    # two requests against the same DB must not duplicate/alter seeded rows
    first = client.get("/orders/1234").json()
    second = client.get("/orders/1234").json()
    assert first == second


def test_seed_can_be_called_directly_without_duplicating_rows(tmp_path):
    from mcs.mock_backend.db import Order, make_engine, make_session_factory, seed

    engine = make_engine(f"sqlite:///{tmp_path / 'direct.db'}")
    session_factory = make_session_factory(engine)

    with session_factory() as session:
        seed(session)
        seed(session)  # calling twice must not duplicate rows
        count = session.query(Order).count()

    assert count == 4  # the four sample orders, not eight
