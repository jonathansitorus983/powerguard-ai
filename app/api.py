from pathlib import Path
from typing import Dict
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "app" / "models"

app = FastAPI(title="PowerGuard AI API", version="1.0")
model = joblib.load(MODEL_DIR / "power_forecaster.joblib")
anomaly_model = joblib.load(MODEL_DIR / "anomaly_detector.joblib")
features = joblib.load(MODEL_DIR / "features.joblib")

class TelemetryInput(BaseModel):
    gpu_utilization: float = Field(..., ge=0, le=1)
    memory_utilization: float = Field(..., ge=0, le=1)
    temperature_c: float
    active_jobs: int
    queued_jobs: int
    energy_price_per_kwh: float
    hour: float = Field(..., ge=0, le=24)
    day_of_week: int = Field(..., ge=0, le=6)

@app.get("/")
def home() -> Dict[str, str]:
    return {"message": "PowerGuard AI is running."}

@app.post("/predict")
def predict_power(payload: TelemetryInput):
    row = pd.DataFrame([payload.model_dump()])[features]
    predicted_power_kw = float(model.predict(row)[0])
    anomaly_score = int(anomaly_model.predict(pd.DataFrame([{
        "power_kw": predicted_power_kw,
        "gpu_utilization": payload.gpu_utilization,
        "temperature_c": payload.temperature_c,
        "active_jobs": payload.active_jobs,
    }]))[0])
    risk_level = "High" if predicted_power_kw > 165 or anomaly_score == -1 else "Normal"
    estimated_hourly_cost = predicted_power_kw * payload.energy_price_per_kwh
    return {
        "predicted_power_kw": round(predicted_power_kw, 2),
        "risk_level": risk_level,
        "estimated_hourly_cost_usd": round(estimated_hourly_cost, 2),
    }
