"""Optional advanced sequence model for GPU power forecasting.

Install advanced deps first:
    python -m pip install -r requirements-advanced.txt
Then run:
    python app/train_lstm_model.py
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "gpu_telemetry.csv"
MODEL_DIR = ROOT / "app" / "models"
MODEL_DIR.mkdir(exist_ok=True)

FEATURES = [
    "gpu_utilization",
    "memory_utilization",
    "temperature_c",
    "active_jobs",
    "queued_jobs",
    "energy_price_per_kwh",
]
TARGET = "power_kw"
LOOKBACK = 24

class PowerLSTM(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden_dim, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        output, _ = self.lstm(x)
        return self.head(output[:, -1, :]).squeeze(-1)


def make_windows(values, targets, lookback):
    xs, ys = [], []
    for i in range(lookback, len(values)):
        xs.append(values[i - lookback:i])
        ys.append(targets[i])
    return np.array(xs), np.array(ys)


df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"]).sort_values("timestamp")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[FEATURES])
y = df[TARGET].values.astype("float32")

X_seq, y_seq = make_windows(X_scaled, y, LOOKBACK)
split = int(len(X_seq) * 0.8)
X_train, X_test = X_seq[:split], X_seq[split:]
y_train, y_test = y_seq[:split], y_seq[split:]

model = PowerLSTM(input_dim=len(FEATURES))
optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=1e-4)
loss_fn = nn.L1Loss()

X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.float32)
X_test_t = torch.tensor(X_test, dtype=torch.float32)

for epoch in range(30):
    model.train()
    optimizer.zero_grad()
    preds = model(X_train_t)
    loss = loss_fn(preds, y_train_t)
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 10 == 0:
        print(f"epoch={epoch + 1}, train_mae={loss.item():.2f}")

model.eval()
with torch.no_grad():
    test_preds = model(X_test_t).numpy()
mae = mean_absolute_error(y_test, test_preds)

torch.save(model.state_dict(), MODEL_DIR / "lstm_power_forecaster.pt")
joblib.dump({"features": FEATURES, "lookback": LOOKBACK, "mae_kw": round(float(mae), 2)}, MODEL_DIR / "lstm_metrics.joblib")
joblib.dump(scaler, MODEL_DIR / "lstm_scaler.joblib")
print({"lstm_mae_kw": round(float(mae), 2), "saved_to": str(MODEL_DIR)})
