# PowerGuard AI

PowerGuard AI is an AI infrastructure analytics platform that simulates how modern AI data centers monitor GPU-heavy workloads, forecast power demand, and detect operational risk.

This upgraded version includes production-style features such as API authentication, Redis-ready prediction caching, Prometheus metrics, Kafka streaming scaffolding, Docker deployment, optional NVIDIA telemetry collection, and an optional LSTM forecasting model.

## What It Solves

AI workloads such as LLM inference, computer vision training, batch embedding generation, multimodal inference, and distributed training can create unpredictable GPU utilization, power, and cooling spikes. PowerGuard AI helps teams forecast energy demand, detect abnormal workload behavior, and monitor infrastructure KPIs from one dashboard and API layer.

## Features

- Streamlit executive dashboard for GPU observability
- FastAPI prediction endpoint with API key authentication
- Random Forest power forecasting model
- Isolation Forest anomaly detection model
- Redis-ready caching for repeated prediction requests
- Prometheus `/metrics` endpoint for observability
- Kafka producer and consumer scaffolding for streaming telemetry
- Docker and Docker Compose support
- Optional NVIDIA `nvidia-smi` telemetry collector
- Optional LSTM forecasting model for sequence prediction

## Tech Stack

Python, FastAPI, Streamlit, Scikit-learn, Pandas, Plotly, Redis, Prometheus, Kafka, Docker, PyTorch optional, GitHub

## Local Setup

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Generate telemetry data:

```bash
python app/generate_data.py
```

Train the forecasting and anomaly models:

```bash
python app/train_model.py
```

Run the dashboard:

```bash
python -m streamlit run app/dashboard.py
```

Run the API:

```bash
python -m uvicorn app.api:app --reload
```

Open API docs:

```text
http://127.0.0.1:8000/docs
```

Use this API key in the `x-api-key` header:

```text
dev-powerguard-key
```

## Docker Setup

Run the dashboard, API, Redis, Prometheus, Kafka, and Zookeeper:

```bash
docker compose up --build
```

Services:

- Dashboard: `http://localhost:8501`
- API: `http://localhost:8000/docs`
- Prometheus: `http://localhost:9090`
- Redis: `localhost:6379`
- Kafka: `localhost:9092`

## Prometheus Metrics

The API exposes:

```text
http://localhost:8000/metrics
```

Tracked metrics include:

- `powerguard_predictions_total`
- `powerguard_cache_hits_total`
- `powerguard_prediction_latency_seconds`
- `powerguard_latest_predicted_power_kw`
- `powerguard_latest_risk_flag`

## Kafka Streaming

Start Docker Compose first, then run:

```bash
python app/streaming/kafka_producer.py
```

In another terminal:

```bash
python app/streaming/kafka_consumer.py
```

This simulates real-time GPU telemetry ingestion from the generated dataset.

## NVIDIA Telemetry

On a machine with NVIDIA drivers installed:

```bash
python app/collect_nvidia_telemetry.py
```

This writes real `nvidia-smi` telemetry into:

```text
data/nvidia_live_telemetry.csv
```

If no NVIDIA GPU is available, use the synthetic generator.

## Optional LSTM Forecasting

Install advanced dependencies:

```bash
python -m pip install -r requirements-advanced.txt
```

Train LSTM model:

```bash
python app/train_lstm_model.py
```

This saves:

```text
app/models/lstm_power_forecaster.pt
app/models/lstm_metrics.joblib
app/models/lstm_scaler.joblib
```


