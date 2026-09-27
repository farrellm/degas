from fastapi.testclient import TestClient

from degas import __version__
from degas.app import app


def test_health() -> None:
    resp = TestClient(app).get("/api/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok", "version": __version__}
