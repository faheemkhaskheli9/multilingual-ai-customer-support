from fastapi.testclient import TestClient

from mcs.api import app

client = TestClient(app)


def test_health():
    assert client.get("/health").json() == {"status": "ok"}


def test_chat_order_status():
    resp = client.post("/chat", json={"message": "where is my order #1234?"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["tool_calls"] == ["get_order_status"]
    assert "1234" in body["reply"]
    assert body["stop_reason"] == "completed"


def test_chat_rejects_empty_message():
    assert client.post("/chat", json={"message": ""}).status_code == 422


def test_chat_invoice_lookup():
    resp = client.post("/chat", json={"message": "invoice for order 1234"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["tool_calls"] == ["get_invoice"]
    assert "1234" in body["reply"]


def test_chat_session_threads_multiturn_tax_followup():
    session_id = "test-session-tax"
    first = client.post(
        "/chat", json={"message": "invoice for order 1234", "session_id": session_id}
    )
    assert first.json()["tool_calls"] == ["get_invoice"]

    followup = client.post(
        "/chat", json={"message": "what about the tax on that?", "session_id": session_id}
    )
    body = followup.json()
    assert body["tool_calls"] == []
    assert "2.24" in body["reply"]


def test_chat_without_session_id_has_no_cross_request_context():
    resp = client.post("/chat", json={"message": "what about the tax on that?"})
    body = resp.json()
    assert body["tool_calls"] == []
    assert "order or account number" in body["reply"].lower()
