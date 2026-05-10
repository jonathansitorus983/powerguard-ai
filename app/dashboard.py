from pathlib import Path
import joblib
import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "gpu_telemetry.csv"
MODEL_DIR = ROOT / "app" / "models"

st.set_page_config(page_title="PowerGuard AI", layout="wide")
st.title("PowerGuard AI")
st.caption("AI infrastructure analytics for GPU power forecasting, cost visibility, and operational risk monitoring.")

df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
model = joblib.load(MODEL_DIR / "power_forecaster.joblib")
features = joblib.load(MODEL_DIR / "features.joblib")
metrics = joblib.load(MODEL_DIR / "metrics.joblib")

df["hour"] = df["timestamp"].dt.hour + df["timestamp"].dt.minute / 60
df["day_of_week"] = df["timestamp"].dt.dayofweek
df["predicted_power_kw"] = model.predict(df[features])
df["forecast_error_kw"] = df["power_kw"] - df["predicted_power_kw"]
df["risk_flag"] = (df["predicted_power_kw"] > 165) | ((df["temperature_c"] > 75) & (df["gpu_utilization"] > 0.78))

total_cost = df["estimated_cost_per_5min"].sum()
avg_power = df["power_kw"].mean()
peak_power = df["power_kw"].max()
risk_events = int(df["risk_flag"].sum())

col1, col2, col3, col4 = st.columns(4)
col1.metric("Avg Power", f"{avg_power:.1f} kW")
col2.metric("Peak Power", f"{peak_power:.1f} kW")
col3.metric("Estimated 14-Day Cost", f"${total_cost:,.0f}")
col4.metric("Risk Events", f"{risk_events}")

st.subheader("Executive Summary")
st.write(
    f"PowerGuard AI forecasts GPU power demand with an MAE of {metrics['mae_kw']} kW and R² of {metrics['r2']}. "
    "The platform identifies periods where utilization, temperature, and queued jobs create elevated power and cooling risk. "
    "For an MIS or Business Analytics portfolio, this project demonstrates how predictive analytics can improve AI infrastructure planning, cost control, and operational decision-making."
)

st.subheader("Power Forecast vs Actual Usage")
fig = px.line(df, x="timestamp", y=["power_kw", "predicted_power_kw"], labels={"value": "Power kW", "timestamp": "Time"})
st.plotly_chart(fig, use_container_width=True)

st.subheader("GPU Utilization vs Power Demand")
fig2 = px.scatter(df, x="gpu_utilization", y="power_kw", color="temperature_c", hover_data=["active_jobs", "queued_jobs"])
st.plotly_chart(fig2, use_container_width=True)

st.subheader("Hourly Cost Pattern")
hourly = df.set_index("timestamp").resample("1h")["estimated_cost_per_5min"].sum().reset_index()
fig3 = px.bar(hourly, x="timestamp", y="estimated_cost_per_5min", labels={"estimated_cost_per_5min": "Estimated Cost ($)"})
st.plotly_chart(fig3, use_container_width=True)

st.subheader("Risk Events")
risk_df = df[df["risk_flag"]]
st.dataframe(risk_df[["timestamp", "gpu_utilization", "temperature_c", "active_jobs", "queued_jobs", "predicted_power_kw"]].head(50), use_container_width=True)
