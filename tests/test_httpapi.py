from starlette.testclient import TestClient
from utils import test_record_paths

from speech2text_api.httpapi import app


def test_transcribe_default_api() -> None:
    client = TestClient(app)
    with open(test_record_paths["default"], "rb") as in_file:
        response = client.post("/transcribe/", files={"file": in_file})

    assert response.status_code == 200
    assert "testovací" in response.json()["transcript"]


def test_transcribe_en_api() -> None:
    client = TestClient(app)
    with open(test_record_paths["en"], "rb") as in_file:
        response = client.post("/transcribe/en/", files={"file": in_file})

    assert response.status_code == 200
    assert "evacuation" in response.json()["transcript"].lower()


def test_transcribe_cs_api() -> None:
    client = TestClient(app)
    with open(test_record_paths["cs"], "rb") as in_file:
        response = client.post("/transcribe/cs/", files={"file": in_file})

    assert response.status_code == 200
    assert "testovací" in response.json()["transcript"]
