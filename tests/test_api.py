from fastapi.testclient import TestClient
from pneumonia.api import app

def test_read_root():
    with TestClient(app) as client:
        response = client.get("/")
        assert response.status_code == 200
        assert response.json() == {"Welcome":"To the Pneumonia Detection API"}

def test_pred_empty_upload():
    with TestClient(app) as client:
        response = client.post("/pred/", files={"data": ("", b"")})
        assert response.status_code == 422

def test_pred_valid_image():
    with TestClient(app) as client:
        path = "tests/dummy_data/raw/train/NORMAL/IM-0119-0001.jpeg"
        with open(path, "rb") as f:
            resp = client.post(
                "/pred/",
                files={"data": ("IM-0119-0001.jpeg", f, "image/jpeg")},
                headers={"accept": "application/json"},  
            )
        assert resp.status_code == 200
        json_response = resp.json()
        assert isinstance(json_response["sigmoid"], float)
        assert isinstance(json_response["pred"], str)