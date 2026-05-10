from pathlib import Path
import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "gpu_telemetry.csv"
MODEL_DIR = ROOT / "app" / "models"

st.set_page_config(
    page_title="PowerGuard AI",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------- Styling ----------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background: radial-gradient(circle at top left, #1b2a41 0%, #0b1020 35%, #050814 100%);
        color: #f8fafc;
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f172a 0%, #020617 100%);
        border-right: 1px solid rgba(148, 163, 184, 0.18);
    }

    .hero-card {
        padding: 2rem 2.2rem;
        border-radius: 28px;
        background: linear-gradient(135deg, rgba(239,68,68,0.22), rgba(234,179,8,0.10), rgba(59,130,246,0.12));
        border: 1px solid rgba(255,255,255,0.14);
        box-shadow: 0 24px 80px rgba(0,0,0,0.35);
        margin-bottom: 1.2rem;
    }

    .hero-title {
        font-size: 3.2rem;
        font-weight: 850;
        letter-spacing: -0.06em;
        margin-bottom: 0.25rem;
        background: linear-gradient(90deg, #ffffff, #facc15, #ef4444);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .hero-subtitle {
        color: #cbd5e1;
        font-size: 1.05rem;
        max-width: 980px;
        line-height: 1.65;
    }

    .pill-row {
        display: flex;
        gap: 0.6rem;
        flex-wrap: wrap;
        margin-top: 1rem;
    }

    .pill {
        padding: 0.42rem 0.75rem;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 700;
        color: #e2e8f0;
        background: rgba(15, 23, 42, 0.75);
        border: 1px solid rgba(255,255,255,0.12);
    }

    .metric-card {
        padding: 1.2rem 1.25rem;
        border-radius: 22px;
        background: rgba(15, 23, 42, 0.82);
        border: 1px solid rgba(148, 163, 184, 0.20);
        box-shadow: 0 16px 45px rgba(0,0,0,0.25);
        min-height: 135px;
    }

    .metric-label {
        color: #94a3b8;
        font-size: 0.78rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }

    .metric-value {
        color: #ffffff;
        font-size: 2rem;
        font-weight: 850;
        margin-top: 0.35rem;
        letter-spacing: -0.04em;
    }

    .metric-note {
        color: #cbd5e1;
        font-size: 0.82rem;
        margin-top: 0.25rem;
    }

    .section-card {
        padding: 1.3rem 1.45rem;
        border-radius: 24px;
        background: rgba(15, 23, 42, 0.72);
        border: 1px solid rgba(148, 163, 184, 0.18);
        box-shadow: 0 12px 40px rgba(0,0,0,0.23);
        margin: 1rem 0;
    }

    .section-title {
        color: #f8fafc;
        font-size: 1.35rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        margin-bottom: 0.35rem;
    }

    .section-copy {
        color: #cbd5e1;
        font-size: 0.95rem;
        line-height: 1.7;
    }

    .insight-grid {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 0.8rem;
        margin-top: 0.9rem;
    }

    .insight-box {
        padding: 1rem;
        border-radius: 18px;
        background: rgba(2, 6, 23, 0.65);
        border: 1px solid rgba(148, 163, 184, 0.15);
    }

    .insight-heading {
        color: #facc15;
        font-weight: 800;
        font-size: 0.9rem;
        margin-bottom: 0.3rem;
    }

    .insight-text {
        color: #cbd5e1;
        font-size: 0.83rem;
        line-height: 1.55;
    }

    div[data-testid="stMetric"] {
        background: rgba(15, 23, 42, 0.72);
        padding: 1rem;
        border-radius: 18px;
        border: 1px solid rgba(148, 163, 184, 0.16);
    }

    div[data-testid="stDataFrame"] {
        border-radius: 20px;
        overflow: hidden;
        border: 1px solid rgba(148, 163, 184, 0.18);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------- Data ----------
@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH, parse_dates=["timestamp"])

@st.cache_resource
def load_model_assets():
    model = joblib.load(MODEL_DIR / "power_forecaster.joblib")
    features = joblib.load(MODEL_DIR / "features.joblib")
    metrics = joblib.load(MODEL_DIR / "metrics.joblib")
    return model, features, metrics

df = load_data()
model, features, metrics = load_model_assets()

df["hour"] = df["timestamp"].dt.hour + df["timestamp"].dt.minute / 60
df["day_of_week"] = df["timestamp"].dt.dayofweek
df["predicted_power_kw"] = model.predict(df[features])
df["forecast_error_kw"] = df["power_kw"] - df["predicted_power_kw"]
df["risk_flag"] = (df["predicted_power_kw"] > 165) | ((df["temperature_c"] > 75) & (df["gpu_utilization"] > 0.78))
df["workload_label"] = df.get("workload_type", "mixed_ai_workload")

# ---------- Sidebar Controls ----------
st.sidebar.markdown("## ⚡ PowerGuard AI")
st.sidebar.caption("AI infrastructure forecasting and workload risk monitoring")

min_date = df["timestamp"].min().date()
max_date = df["timestamp"].max().date()
selected_dates = st.sidebar.date_input(
    "Date range",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date,
)

if isinstance(selected_dates, tuple) and len(selected_dates) == 2:
    start_date, end_date = selected_dates
    mask = (df["timestamp"].dt.date >= start_date) & (df["timestamp"].dt.date <= end_date)
    filtered = df.loc[mask].copy()
else:
    filtered = df.copy()

if "workload_type" in filtered.columns:
    workloads = sorted(filtered["workload_type"].dropna().unique())
    selected_workloads = st.sidebar.multiselect("Workload type", workloads, default=workloads)
    filtered = filtered[filtered["workload_type"].isin(selected_workloads)]

risk_threshold = st.sidebar.slider("Power risk threshold, kW", 120, 220, 165, 5)
filtered["risk_flag"] = (filtered["predicted_power_kw"] > risk_threshold) | ((filtered["temperature_c"] > 75) & (filtered["gpu_utilization"] > 0.78))

# ---------- KPI calculations ----------
total_cost = filtered["estimated_cost_per_5min"].sum()
avg_power = filtered["power_kw"].mean()
peak_power = filtered["power_kw"].max()
risk_events = int(filtered["risk_flag"].sum())
avg_temp = filtered["temperature_c"].mean()
avg_util = filtered["gpu_utilization"].mean() * 100
mae = float(metrics.get("mae_kw", 0))
r2 = float(metrics.get("r2", 0))

# ---------- Hero ----------
st.markdown(
    """
    <div class="hero-card">
        <div class="hero-title">PowerGuard AI</div>
        <div class="hero-subtitle">
            Executive-grade AI infrastructure analytics for forecasting GPU power demand, detecting abnormal workload behavior,
            and improving visibility into energy cost, thermal risk, and compute utilization across modern AI workloads.
        </div>
        <div class="pill-row">
            <div class="pill">LLM Inference</div>
            <div class="pill">Computer Vision Training</div>
            <div class="pill">Batch Embeddings</div>
            <div class="pill">GPU Telemetry</div>
            <div class="pill">FastAPI + Streamlit</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------- Metrics ----------
metric_cols = st.columns(5)
metric_data = [
    ("Avg Power", f"{avg_power:.1f} kW", "Mean GPU cluster draw"),
    ("Peak Power", f"{peak_power:.1f} kW", "Highest observed load"),
    ("14-Day Cost", f"${total_cost:,.0f}", "Estimated energy spend"),
    ("Risk Events", f"{risk_events:,}", "Thermal or power alerts"),
    ("Model MAE", f"{mae:.2f} kW", f"R² = {r2:.2f}"),
]
for col, (label, value, note) in zip(metric_cols, metric_data):
    with col:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{value}</div>
                <div class="metric-note">{note}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ---------- Executive Summary ----------
st.markdown(
    f"""
    <div class="section-card">
        <div class="section-title">Executive Summary</div>
        <div class="section-copy">
            PowerGuard AI identifies when AI workloads are likely to create energy-cost and cooling risk. Across the selected window,
            the platform observed an average utilization of <b>{avg_util:.1f}%</b>, average temperature of <b>{avg_temp:.1f}°C</b>,
            and <b>{risk_events:,}</b> risk events above the configured threshold. The forecasting model supports infrastructure planning
            by estimating short-term GPU power demand before workloads create expensive utilization spikes.
        </div>
        <div class="insight-grid">
            <div class="insight-box">
                <div class="insight-heading">Business Problem</div>
                <div class="insight-text">AI workloads create unpredictable power spikes that increase cloud/data-center cost and cooling risk.</div>
            </div>
            <div class="insight-box">
                <div class="insight-heading">Analytics Solution</div>
                <div class="insight-text">Forecast power demand using GPU utilization, temperature, memory pressure, and queue depth telemetry.</div>
            </div>
            <div class="insight-box">
                <div class="insight-heading">Operational Value</div>
                <div class="insight-text">Help teams schedule workloads, detect anomalies, and make better infrastructure capacity decisions.</div>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------- Charts ----------
plot_template = "plotly_dark"

left, right = st.columns([1.35, 1])
with left:
    st.markdown('<div class="section-title">Power Forecast vs Actual Usage</div>', unsafe_allow_html=True)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=filtered["timestamp"],
        y=filtered["power_kw"],
        mode="lines",
        name="Actual Power",
        line=dict(width=2.5, color="#ef4444"),
    ))
    fig.add_trace(go.Scatter(
        x=filtered["timestamp"],
        y=filtered["predicted_power_kw"],
        mode="lines",
        name="Predicted Power",
        line=dict(width=2.5, color="#facc15", dash="dot"),
    ))
    fig.update_layout(
        template=plot_template,
        height=430,
        margin=dict(l=20, r=20, t=25, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.45)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis_title="Time",
        yaxis_title="Power Demand, kW",
    )
    st.plotly_chart(fig, use_container_width=True)

with right:
    st.markdown('<div class="section-title">Workload Risk Mix</div>', unsafe_allow_html=True)
    if "workload_type" in filtered.columns:
        workload_risk = filtered.groupby("workload_type")["risk_flag"].sum().reset_index().sort_values("risk_flag", ascending=False)
        fig_workload = px.bar(
            workload_risk,
            x="risk_flag",
            y="workload_type",
            orientation="h",
            template=plot_template,
            labels={"risk_flag": "Risk Events", "workload_type": "Workload"},
            color="risk_flag",
            color_continuous_scale=["#22c55e", "#facc15", "#ef4444"],
        )
        fig_workload.update_layout(
            height=430,
            margin=dict(l=20, r=20, t=25, b=20),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(15,23,42,0.45)",
            coloraxis_showscale=False,
        )
        st.plotly_chart(fig_workload, use_container_width=True)
    else:
        st.info("Workload type column not found in dataset.")

c1, c2 = st.columns(2)
with c1:
    st.markdown('<div class="section-title">GPU Utilization vs Power Demand</div>', unsafe_allow_html=True)
    fig2 = px.scatter(
        filtered,
        x="gpu_utilization",
        y="power_kw",
        color="temperature_c",
        size="queued_jobs" if "queued_jobs" in filtered.columns else None,
        hover_data=[col for col in ["active_jobs", "queued_jobs", "predicted_power_kw"] if col in filtered.columns],
        template=plot_template,
        color_continuous_scale=["#38bdf8", "#facc15", "#ef4444"],
        labels={"gpu_utilization": "GPU Utilization", "power_kw": "Power Demand, kW", "temperature_c": "Temp °C"},
    )
    fig2.update_layout(
        height=430,
        margin=dict(l=20, r=20, t=25, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.45)",
    )
    st.plotly_chart(fig2, use_container_width=True)

with c2:
    st.markdown('<div class="section-title">Hourly Cost Pattern</div>', unsafe_allow_html=True)
    hourly = filtered.set_index("timestamp").resample("1h")["estimated_cost_per_5min"].sum().reset_index()
    fig3 = px.area(
        hourly,
        x="timestamp",
        y="estimated_cost_per_5min",
        template=plot_template,
        labels={"estimated_cost_per_5min": "Estimated Cost, $", "timestamp": "Time"},
    )
    fig3.update_traces(line_color="#facc15", fillcolor="rgba(250, 204, 21, 0.25)")
    fig3.update_layout(
        height=430,
        margin=dict(l=20, r=20, t=25, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15,23,42,0.45)",
    )
    st.plotly_chart(fig3, use_container_width=True)

# ---------- Risk Table ----------
st.markdown('<div class="section-title">High-Risk Telemetry Events</div>', unsafe_allow_html=True)
risk_df = filtered[filtered["risk_flag"]].copy()
show_cols = [
    "timestamp",
    "workload_type" if "workload_type" in risk_df.columns else None,
    "gpu_utilization",
    "temperature_c",
    "memory_utilization" if "memory_utilization" in risk_df.columns else None,
    "active_jobs" if "active_jobs" in risk_df.columns else None,
    "queued_jobs" if "queued_jobs" in risk_df.columns else None,
    "power_kw",
    "predicted_power_kw",
]
show_cols = [c for c in show_cols if c is not None and c in risk_df.columns]
st.dataframe(risk_df[show_cols].head(75), use_container_width=True, hide_index=True)

st.caption("PowerGuard AI is a portfolio project simulating AI data-center observability, forecasting, and infrastructure decision support.")
