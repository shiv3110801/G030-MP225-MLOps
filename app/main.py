import json
import logging
from pathlib import Path
from threading import Lock

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "model.joblib"


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    rooms: int
    members: int
    ac_hours_per_day: float = Field(ge=0)
    avg_temp_c: float
    house_area_sqft: float
    solar_kw: float = Field(le=8.5)
    connection: str
    feeder: str


class PredictionFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        return json.dumps({"event": record.getMessage(), "prediction": record.prediction})


class StdoutHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        print(self.format(record), flush=True)


prediction_logger = logging.getLogger("mp225.predictions")
prediction_logger.setLevel(logging.INFO)
prediction_logger.propagate = False
if not prediction_logger.handlers:
    handler = StdoutHandler()
    handler.setFormatter(PredictionFormatter())
    prediction_logger.addHandler(handler)


def load_model():
    return joblib.load(MODEL_PATH)


def create_app(model_loader=None) -> FastAPI:
    service = FastAPI()
    loader = load_model if model_loader is None else model_loader
    model = None
    model_lock = Lock()

    @service.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @service.post("/predict")
    def predict(payload: PredictionRequest) -> dict[str, float]:
        nonlocal model
        with model_lock:
            if model is None:
                try:
                    model = loader()
                except FileNotFoundError as error:
                    raise HTTPException(status_code=503, detail="Prediction model is unavailable") from error

        frame = pd.DataFrame([payload.model_dump()], columns=list(PredictionRequest.model_fields))
        prediction = float(model.predict(frame)[0])
        prediction_logger.info("prediction", extra={"prediction": prediction})
        return {"prediction": prediction}

    return service


app = create_app()
