from fastapi.testclient import TestClient

from degas_worker import __version__
from degas_worker.app import app


def test_health() -> None:
    resp = TestClient(app).get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok", "version": __version__}
