# PowerGuard AI

AI infrastructure analytics platform that forecasts short-term GPU power demand, detects abnormal power spikes, and translates telemetry into business-friendly operational insights.

## Why this project matters
AI workloads are expensive because GPU clusters consume large amounts of power and cooling. PowerGuard AI acts like a decision-support system for AI infrastructure teams by forecasting GPU power demand and surfacing cost, utilization, and risk metrics.

## Features
- Generate realistic synthetic GPU telemetry
- Train a power forecasting model
- Detect high-risk power spikes
- Serve predictions through FastAPI
- Visualize forecasts, utilization, temperature, and estimated cost in Streamlit
- Export resume-ready metrics and executive insights

## Tech Stack
Python, pandas, scikit-learn, FastAPI, Streamlit, Plotly, joblib

## Quick Start

```bash
pip install -r requirements.txt
python app/generate_data.py
python app/train_model.py
streamlit run app/dashboard.py
```

Optional API:

```bash
uvicorn app.api:app --reload
```

Then visit:

```text
http://127.0.0.1:8000/docs
```

## Project Framing for MIS / Business Analytics
This is not just a machine learning project. It is an AI infrastructure analytics and decision-support platform focused on forecasting operational demand, estimating compute cost, and improving resource planning.
