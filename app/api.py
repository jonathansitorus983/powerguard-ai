from pathlib import Path
from typing import Dict, Optional
import os
import json
import hashlib

import joblib
import pandas as pd
from fastapi import Depends, FastAPI, Header, HTTPException, Response
from pydantic import BaseModel, Field
from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST

try:
    import redis
except ImportError:  # makes local setup still work if Redis is not installed
    redis = None

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "app" / "models"
API_KEY = os.getenv("POWERGUARD_API_KEY", "dev-powerguard-key")
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379/0")

app = FastAPI(title="PowerGuard AI API", version="2.0")
model = joblib.load(MODEL_DIR / "power_forecaster.joblib")
anomaly_model = joblib.load(MODEL_DIR / "anomaly_detector.joblib")
features = joblib.load(MODEL_DIR / "features.joblib")

redis_client = None
if redis is not None:
    try:
        redis_client = redis.Redis.from_url(REDIS_URL, decode_responses=True)
        redis_client.ping()
    except Exception:
        redis_client = None

PREDICTIONS = Counter("powerguard_predictions_total", "Total prediction requests served")
CACHE_HITS = Counter("powerguard_cache_hits_total", "Prediction responses served from Redis cache")
PREDICTION_LATENCY = Histogram("powerguard_prediction_latency_seconds", "Prediction latency in seconds")
POWER_GAUGE = Gauge("powerguard_latest_predicted_power_kw", "Latest predicted GPU power in kW")
RISK_GAUGE = Gauge("powerguard_latest_risk_flag", "Latest risk flag, 1 means high risk")

class TelemetryInput(BaseModel):
    gpu_utilization: float = Field(..., ge=0, le=1)
    memory_utilization: float = Field(..., ge=0, le=1)
    temperature_c: float
    active_jobs: int
    queued_jobs: int
    energy_price_per_kwh: float
    hour: float = Field(..., ge=0, le=24)
    day_of_week: int = Field(..., ge=0, le=6)
    workload_type: Optional[str] = "mixed_ai_workload"

def require_api_key(x_api_key: str = Header(default="")):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
    return True

def cache_key(payload: TelemetryInput) -> str:
    raw = json.dumps(payload.model_dump(), sort_keys=True)
    return "powerguard:prediction:" + hashlib.sha256(raw.encode()).hexdigest()

@app.get("/")
def home() -> Dict[str, str]:
    return {"message": "PowerGuard AI API is running.", "docs": "/docs", "metrics": "/metrics"}

@app.get("/health")
def health():
    return {"status": "ok", "redis_enabled": redis_client is not None}

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.post("/predict", dependencies=[Depends(require_api_key)])
@PREDICTION_LATENCY.time()
def predict_power(payload: TelemetryInput):
    PREDICTIONS.inc()
    key = cache_key(payload)
    if redis_client is not None:
        cached = redis_client.get(key)
        if cached:
            CACHE_HITS.inc()
            return json.loads(cached)

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

    POWER_GAUGE.set(predicted_power_kw)
    RISK_GAUGE.set(1 if risk_level == "High" else 0)

    result = {
        "predicted_power_kw": round(predicted_power_kw, 2),
        "risk_level": risk_level,
        "workload_type": payload.workload_type,
        "estimated_hourly_cost_usd": round(estimated_hourly_cost, 2),
        "cache_status": "miss" if redis_client is not None else "disabled",
    }
    if redis_client is not None:
        redis_client.setex(key, 60, json.dumps(result))
    return result
