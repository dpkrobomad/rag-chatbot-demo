from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_health():
    r = client.get("/api/health")
    assert r.status_code == 200 and r.json()["chunks"] > 0


def test_retrieves_right_section(monkeypatch):
    monkeypatch.setenv("LLM_BASE_URL", "http://127.0.0.1:9")  # force extractive mode
    r = client.post("/api/ask", json={"question": "How long do refunds take?"})
    data = r.json()
    assert r.status_code == 200
    assert data["mode"] == "extractive"
    assert data["sources"][0]["heading"] == "Returns and refunds"


def test_no_match():
    r = client.post("/api/ask", json={"question": "zebra quantum xylophone"})
    assert r.json()["mode"] == "no-match" and r.json()["sources"] == []


def test_validation():
    assert client.post("/api/ask", json={"question": ""}).status_code == 422
