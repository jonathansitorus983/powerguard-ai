import json
from pathlib import Path

import pandas as pd
from kafka import KafkaConsumer

ROOT = Path(__file__).resolve().parents[2]
OUT_PATH = ROOT / "data" / "streamed_gpu_telemetry.csv"
TOPIC = "gpu-telemetry"

consumer = KafkaConsumer(
    TOPIC,
    bootstrap_servers="localhost:9092",
    auto_offset_reset="earliest",
    enable_auto_commit=True,
    value_deserializer=lambda v: json.loads(v.decode("utf-8")),
)

buffer = []
for msg in consumer:
    event = msg.value
    buffer.append(event)
    if len(buffer) >= 25:
        existing = pd.read_csv(OUT_PATH) if OUT_PATH.exists() else pd.DataFrame()
        updated = pd.concat([existing, pd.DataFrame(buffer)], ignore_index=True)
        updated.to_csv(OUT_PATH, index=False)
        print(f"wrote {len(buffer)} events to {OUT_PATH}")
        buffer.clear()
