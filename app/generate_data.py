import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
DATA_DIR.mkdir(exist_ok=True)

np.random.seed(42)

n = 24 * 14 * 12  # 14 days, 5-min intervals
timestamps = timestamps = pd.date_range(
    end=pd.Timestamp.now(),
    periods=n,
    freq="5min"
)

hour = (timestamps.hour + timestamps.minute / 60).to_numpy()
weekday = timestamps.dayofweek.to_numpy()

# Workload cycles: daytime peaks and weekly variation
business_cycle = 0.45 + 0.35 * np.sin((hour - 8) / 24 * 2 * np.pi)
weekly_factor = np.where(weekday < 5, 1.0, 0.72)
random_load = np.random.normal(0, 0.08, n)

cluster_gpu_util = np.clip((business_cycle * weekly_factor + random_load), 0.05, 0.98)
memory_util = np.clip(cluster_gpu_util * 0.82 + np.random.normal(0, 0.07, n), 0.05, 0.96)
temperature_c = 38 + 42 * cluster_gpu_util + np.random.normal(0, 2.5, n)
active_jobs = np.random.poisson(10 + 55 * cluster_gpu_util)
queued_jobs = np.random.poisson(2 + 16 * np.maximum(cluster_gpu_util - 0.68, 0))

# Simulated GPU power: physically plausible relationship with utilization and temperature
base_power_kw = 18
power_kw = (
    base_power_kw
    + 145 * cluster_gpu_util
    + 0.42 * temperature_c
    + 0.18 * active_jobs
    + np.random.normal(0, 5, n)
)

# Inject rare spike events
spike_indices = np.random.choice(np.arange(n), size=35, replace=False)
power_kw[spike_indices] += np.random.uniform(35, 75, size=len(spike_indices))

cooling_risk = ((temperature_c > 75) & (cluster_gpu_util > 0.78)).astype(int)
energy_price_per_kwh = 0.11 + 0.04 * ((hour >= 14) & (hour <= 20)).astype(float)
estimated_cost_per_5min = power_kw * (5 / 60) * energy_price_per_kwh

telemetry = pd.DataFrame({
    "timestamp": timestamps,
    "gpu_utilization": cluster_gpu_util,
    "memory_utilization": memory_util,
    "temperature_c": temperature_c,
    "active_jobs": active_jobs,
    "queued_jobs": queued_jobs,
    "power_kw": power_kw,
    "cooling_risk": cooling_risk,
    "energy_price_per_kwh": energy_price_per_kwh,
    "estimated_cost_per_5min": estimated_cost_per_5min,
})

telemetry.to_csv(DATA_DIR / "gpu_telemetry.csv", index=False)
print(f"Saved telemetry to {DATA_DIR / 'gpu_telemetry.csv'}")
