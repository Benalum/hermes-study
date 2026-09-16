from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient

from hermes_study.app import create_app
from hermes_study.config import Settings


def test_api_smoke(tmp_path: Path):
    settings = Settings(data_dir=tmp_path, host="127.0.0.1", port=8787)
    app = create_app(settings)
    client = TestClient(app)
    assert client.get("/").status_code == 200
    r = client.post("/api/courses", json={"name": "Chemistry 1", "code": "CHEM 1215"})
    assert r.status_code == 200
    course = r.json()
    assert course["name"] == "Chemistry 1"
    listing = client.get("/api/courses").json()
    assert listing[0]["id"] == course["id"]
