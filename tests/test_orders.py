import pytest

from mcs.backends.orders import (
    OrderNotFoundError,
    get_order_status,
    normalize_order_id,
)


def test_known_order_returns_status():
    rec = get_order_status("1234")
    assert rec["found"] is True
    assert rec["status"] == "shipped"
    assert rec["carrier"] == "GenericPost"


def test_leading_hash_is_stripped():
    assert normalize_order_id("#1234") == "1234"
    assert get_order_status("#1234")["order_id"] == "1234"


def test_unknown_order_raises():
    with pytest.raises(OrderNotFoundError):
        get_order_status("0000")


def test_blank_order_raises():
    with pytest.raises(OrderNotFoundError):
        get_order_status("   ")
