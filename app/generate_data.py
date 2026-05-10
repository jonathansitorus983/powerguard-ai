import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
DATA_DIR.mkdir(exist_ok=True)

np.random.seed(42)

# 14 days of 5-minute telemetry ending now so the dashboard feels live.
n = 24 * 14 * 12
timestamps = pd.date_range(end=pd.Timestamp.now().floor("5min"), periods=n, freq="5min")

hour = (timestamps.hour + timestamps.minute / 60).to_numpy()
weekday = timestamps.dayofweek.to_numpy()

workload_types = np.array([
    "llm_inference",
    "computer_vision_training",
    "batch_embeddings",
    "multimodal_inference",
    "distributed_training",
])
workload_probs = [0.34, 0.20, 0.24, 0.12, 0.10]
workload_type = np.random.choice(workload_types, size=n, p=workload_probs)

workload_intensity = {
    "llm_inference": 0.82,
    "computer_vision_training": 0.74,
    "batch_embeddings": 0.58,
    "multimodal_inference": 0.78,
    "distributed_training": 0.92,
}
intensity = np.array([workload_intensity[w] for w in workload_type])

business_cycle = 0.44 + 0.34 * np.sin((hour - 8) / 24 * 2 * np.pi)
weekly_factor = np.where(weekday < 5, 1.0, 0.72)
random_load = np.random.normal(0, 0.08, n)

cluster_gpu_util = np.clip((business_cycle * weekly_factor * intensity + random_load + 0.18), 0.05, 0.99)
memory_util = np.clip(cluster_gpu_util * 0.80 + np.random.normal(0, 0.07, n), 0.05, 0.97)
temperature_c = 36 + 45 * cluster_gpu_util + np.random.normal(0, 2.6, n)
active_jobs = np.random.poisson(8 + 60 * cluster_gpu_util)
queued_jobs = np.random.poisson(2 + 20 * np.maximum(cluster_gpu_util - 0.64, 0))

base_power_kw = 18
power_kw = (
    base_power_kw
    + 147 * cluster_gpu_util
    + 0.44 * temperature_c
    + 0.20 * active_jobs
    + 8 * (workload_type == "distributed_training")
    + 5 * (workload_type == "llm_inference")
    + np.random.normal(0, 5, n)
)

spike_indices = np.random.choice(np.arange(n), size=45, replace=False)
power_kw[spike_indices] += np.random.uniform(35, 80, size=len(spike_indices))

affected = np.zeros(n, dtype=int)
affected[spike_indices] = 1
cooling_risk = ((temperature_c > 75) & (cluster_gpu_util > 0.78)).astype(int)
energy_price_per_kwh = 0.11 + 0.045 * ((hour >= 14) & (hour <= 20)).astype(float)
estimated_cost_per_5min = power_kw * (5 / 60) * energy_price_per_kwh

telemetry = pd.DataFrame({
    "timestamp": timestamps,
    "workload_type": workload_type,
    "gpu_utilization": cluster_gpu_util,
    "memory_utilization": memory_util,
    "temperature_c": temperature_c,
    "active_jobs": active_jobs,
    "queued_jobs": queued_jobs,
    "power_kw": power_kw,
    "cooling_risk": cooling_risk,
    "synthetic_spike_event": affected,
    "energy_price_per_kwh": energy_price_per_kwh,
    "estimated_cost_per_5min": estimated_cost_per_5min,
})

telemetry.to_csv(DATA_DIR / "gpu_telemetry.csv", index=False)
print(f"Saved telemetry to {DATA_DIR / 'gpu_telemetry.csv'}")
