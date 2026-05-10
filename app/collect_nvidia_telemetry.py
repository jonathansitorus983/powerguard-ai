"""Collect real NVIDIA telemetry when available.

This script uses nvidia-smi so it works without requiring a Python GPU SDK.
If no NVIDIA GPU is available, it exits cleanly and tells you to use generate_data.py.
"""
from __future__ import annotations

import csv
import subprocess
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "nvidia_live_telemetry.csv"
DATA_PATH.parent.mkdir(exist_ok=True)

QUERY = [
    "nvidia-smi",
    "--query-gpu=timestamp,index,utilization.gpu,utilization.memory,temperature.gpu,power.draw,memory.used,memory.total",
    "--format=csv,noheader,nounits",
]

HEADER = [
    "collected_at",
    "gpu_index",
    "gpu_utilization",
    "memory_utilization",
    "temperature_c",
    "power_kw",
    "memory_used_mb",
    "memory_total_mb",
]


def collect_once() -> list[dict]:
    result = subprocess.run(QUERY, capture_output=True, text=True, check=True)
    rows = []
    for line in result.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 8:
            continue
        _, gpu_index, gpu_util, mem_util, temp, power_w, mem_used, mem_total = parts[:8]
        rows.append({
            "collected_at": datetime.utcnow().isoformat(),
            "gpu_index": int(gpu_index),
            "gpu_utilization": float(gpu_util) / 100,
            "memory_utilization": float(mem_util) / 100,
            "temperature_c": float(temp),
            "power_kw": float(power_w) / 1000,
            "memory_used_mb": float(mem_used),
            "memory_total_mb": float(mem_total),
        })
    return rows


def append_rows(rows: list[dict]) -> None:
    new_file = not DATA_PATH.exists()
    with DATA_PATH.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=HEADER)
        if new_file:
            writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    try:
        while True:
            rows = collect_once()
            append_rows(rows)
            print(f"Collected {len(rows)} GPU telemetry row(s) into {DATA_PATH}")
            time.sleep(5)
    except FileNotFoundError:
        print("nvidia-smi was not found. Use app/generate_data.py for synthetic telemetry or run this on an NVIDIA GPU machine.")
    except subprocess.CalledProcessError as exc:
        print("NVIDIA telemetry collection failed:", exc.stderr or exc)
