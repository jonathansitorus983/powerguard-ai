from pathlib import Path
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "gpu_telemetry.csv"
MODEL_DIR = ROOT / "app" / "models"
MODEL_DIR.mkdir(exist_ok=True)

df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
df["hour"] = df["timestamp"].dt.hour + df["timestamp"].dt.minute / 60
df["day_of_week"] = df["timestamp"].dt.dayofweek

features = [
    "gpu_utilization",
    "memory_utilization",
    "temperature_c",
    "active_jobs",
    "queued_jobs",
    "energy_price_per_kwh",
    "hour",
    "day_of_week",
]

target = "power_kw"
X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

model = RandomForestRegressor(
    n_estimators=240,
    max_depth=14,
    min_samples_leaf=3,
    random_state=42,
    n_jobs=-1,
)
model.fit(X_train, y_train)
preds = model.predict(X_test)

mae = mean_absolute_error(y_test, preds)
r2 = r2_score(y_test, preds)

anomaly_model = IsolationForest(contamination=0.025, random_state=42)
anomaly_model.fit(df[["power_kw", "gpu_utilization", "temperature_c", "active_jobs"]])

joblib.dump(model, MODEL_DIR / "power_forecaster.joblib")
joblib.dump(anomaly_model, MODEL_DIR / "anomaly_detector.joblib")
joblib.dump(features, MODEL_DIR / "features.joblib")

metrics = {
    "mae_kw": round(float(mae), 2),
    "r2": round(float(r2), 4),
    "training_rows": int(len(X_train)),
    "test_rows": int(len(X_test)),
}
joblib.dump(metrics, MODEL_DIR / "metrics.joblib")
print(metrics)
