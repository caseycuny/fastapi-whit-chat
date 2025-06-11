from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}

def test_initialize_chat():
    response = client.post("/initialize_chat", json={
        "assignment_id": 1,
        "assistant_id": "test_assistant"
    })
    assert response.status_code == 200
    assert "thread_id" in response.json()
