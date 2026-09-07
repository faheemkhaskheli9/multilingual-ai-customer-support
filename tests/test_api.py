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
