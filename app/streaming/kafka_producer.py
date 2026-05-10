import json
import time
from pathlib import Path

import pandas as pd
from kafka import KafkaProducer

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "gpu_telemetry.csv"
TOPIC = "gpu-telemetry"

producer = KafkaProducer(
    bootstrap_servers="localhost:9092",
    value_serializer=lambda v: json.dumps(v, default=str).encode("utf-8"),
)

df = pd.read_csv(DATA_PATH)
for _, row in df.iterrows():
    event = row.to_dict()
    producer.send(TOPIC, event)
    print("sent", event.get("timestamp"), event.get("workload_type"))
    time.sleep(0.05)
producer.flush()
