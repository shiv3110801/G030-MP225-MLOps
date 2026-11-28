import json
from unittest.mock import Mock

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.main import create_app


@pytest.fixture
def payload():
    return {
        "rooms": 3,
        "members": 4,
        "ac_hours_per_day": 0,
        "avg_temp_c": 28.5,
        "house_area_sqft": 900,
        "solar_kw": 8.5,
        "connection": "domestic",
        "feeder": "Feeder-1",
    }


@pytest.fixture
def service():
    model = Mock()
    model.predict.return_value = pd.Series([123.5])
    loader = Mock(return_value=model)
    with TestClient(create_app(loader)) as client:
        yield client, loader, model


def test_health_does_not_load_model(service):
    client, loader, _ = service

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    loader.assert_not_called()


def test_predict_passes_one_row_in_model_feature_order(service, payload):
    client, loader, model = service

    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    assert response.json() == {"prediction": 123.5}
    loader.assert_called_once_with()
    model.predict.assert_called_once()
    frame = model.predict.call_args.args[0]
    assert isinstance(frame, pd.DataFrame)
    assert frame.columns.tolist() == list(payload)
    assert frame.to_dict(orient="records") == [payload]


@pytest.mark.parametrize(
    ("field", "value"),
    [("solar_kw", 8.51), ("ac_hours_per_day", -0.01), ("record_id", "unexpected")],
)
def test_invalid_payload_does_not_reach_model(service, payload, field, value):
    client, loader, model = service

    response = client.post("/predict", json={**payload, field: value})

    assert response.status_code == 422
    loader.assert_not_called()
    model.predict.assert_not_called()


def test_predict_writes_one_json_line_to_stdout(service, payload, capsys):
    client, _, _ = service

    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    output = capsys.readouterr()
    lines = output.out.splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0]) == {"event": "prediction", "prediction": 123.5}


def test_missing_model_returns_503_without_breaking_health(payload):
    loader = Mock(side_effect=FileNotFoundError)
    with TestClient(create_app(loader)) as client:
        prediction = client.post("/predict", json=payload)
        health = client.get("/health")

    assert prediction.status_code == 503
    assert prediction.json() == {"detail": "Prediction model is unavailable"}
    assert health.status_code == 200
    loader.assert_called_once_with()
