import pytest

from mcs.backends.invoices import (
    InvoiceNotFoundError,
    get_invoice,
    normalize_order_id,
)


def test_known_invoice_returns_line_items_and_total():
    rec = get_invoice("1234")
    assert rec["found"] is True
    assert rec["order_id"] == "1234"
    assert rec["line_items"] == [
        {"description": "Wireless mouse", "amount": 19.99},
        {"description": "USB-C cable", "amount": 8.00},
    ]
    assert rec["subtotal"] == pytest.approx(27.99)
    assert rec["total"] == pytest.approx(rec["subtotal"] + rec["tax"])


def test_leading_hash_is_stripped():
    assert normalize_order_id("#1234") == "1234"
    assert get_invoice("#1234")["order_id"] == "1234"


def test_unknown_order_raises():
    with pytest.raises(InvoiceNotFoundError):
        get_invoice("0000")


def test_blank_order_raises():
    with pytest.raises(InvoiceNotFoundError):
        get_invoice("   ")


def test_order_with_no_invoice_raises():
    # 5678 exists in the order backend but has no invoice on file.
    with pytest.raises(InvoiceNotFoundError):
        get_invoice("5678")
