import os
import re
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


st.set_page_config(
    page_title="Energy Intelligence Dashboard",
    page_icon="grid",
    layout="wide",
    initial_sidebar_state="expanded",
)


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
PLOTS_DIR = OUTPUT_DIR / "plots"
ADVANCED_PLOTS_DIR = PLOTS_DIR / "advanced"


st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=Source+Serif+4:wght@500;700&display=swap');

    :root {
        --bg: #f4efe6;
        --panel: rgba(255, 250, 242, 0.88);
        --panel-strong: rgba(255, 248, 237, 0.98);
        --ink: #1b1f1d;
        --muted: #5f655f;
        --ink-soft: #32413b;
        --line: rgba(32, 40, 36, 0.12);
        --gold: #d6942a;
        --teal: #0f766e;
        --forest: #204b43;
        --coral: #c95f3d;
        --shadow: 0 18px 55px rgba(32, 40, 36, 0.08);
    }

    .stApp {
        background:
            radial-gradient(circle at top left, rgba(214, 148, 42, 0.18), transparent 26%),
            radial-gradient(circle at top right, rgba(15, 118, 110, 0.16), transparent 24%),
            linear-gradient(180deg, #f7f2e8 0%, var(--bg) 100%);
        color: var(--ink);
        font-family: "Space Grotesk", "Segoe UI", sans-serif;
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f2f29 0%, #081b18 100%);
        border-right: 1px solid rgba(219, 248, 235, 0.18);
        overflow-x: hidden !important;
        font-family: "Space Grotesk", "Segoe UI", sans-serif;
    }

    [data-testid="stSidebar"] * {
        color: #eef8f2 !important;
        box-sizing: border-box;
        max-width: 100%;
    }

    [data-testid="stSidebar"] [data-testid="stIconMaterial"] {
        font-family: "Material Symbols Rounded", "Material Icons", sans-serif !important;
        letter-spacing: normal !important;
        white-space: nowrap !important;
    }

    [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
    [data-testid="stSidebar"] .stCaption {
        color: #d7ebe2 !important;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1280px;
    }

    h1, h2, h3 {
        color: var(--ink);
        font-family: "Source Serif 4", Georgia, serif !important;
        letter-spacing: -0.02em;
    }

    h1 {
        font-size: 3.2rem !important;
        line-height: 1.02;
    }

    .hero-shell {
        position: relative;
        overflow: hidden;
        background:
            linear-gradient(135deg, rgba(255,248,237,0.95) 0%, rgba(248,239,225,0.92) 46%, rgba(236,245,241,0.9) 100%);
        border: 1px solid rgba(32, 40, 36, 0.08);
        border-radius: 28px;
        padding: 2.6rem 2.8rem 2.3rem 2.8rem;
        box-shadow: var(--shadow);
        margin-bottom: 1.4rem;
    }

    .hero-shell::after {
        content: "";
        position: absolute;
        right: -90px;
        top: -90px;
        width: 280px;
        height: 280px;
        border-radius: 50%;
        background: radial-gradient(circle, rgba(15,118,110,0.22), rgba(15,118,110,0) 70%);
    }

    .eyebrow {
        display: inline-block;
        margin-bottom: 0.9rem;
        padding: 0.35rem 0.8rem;
        border-radius: 999px;
        background: rgba(32, 75, 67, 0.08);
        border: 1px solid rgba(32, 75, 67, 0.12);
        color: var(--forest);
        font-size: 0.78rem;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        font-weight: 700;
    }

    .hero-subtitle {
        max-width: 760px;
        font-size: 1.06rem;
        line-height: 1.7;
        color: var(--ink-soft);
        margin-top: 0.75rem;
    }

    .hero-strip {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 0.9rem;
        margin-top: 1.5rem;
    }

    .hero-chip {
        background: rgba(255,255,255,0.85);
        border: 1px solid rgba(32, 40, 36, 0.08);
        border-radius: 18px;
        padding: 1rem 1.1rem;
        backdrop-filter: blur(8px);
    }

    .hero-chip-label {
        font-size: 0.72rem;
        color: #3c4742;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.3rem;
    }

    .hero-chip-value {
        font-size: 1.15rem;
        font-weight: 700;
        color: var(--ink);
    }

    .section-title {
        margin-top: 0.5rem;
        margin-bottom: 0.4rem;
        font-size: 1.7rem;
        font-family: "Source Serif 4", Georgia, serif;
    }

    .section-copy {
        color: #3b4742;
        margin-bottom: 1rem;
    }

    .stat-card {
        background: var(--panel);
        border: 1px solid var(--line);
        border-radius: 24px;
        padding: 1.15rem 1.2rem 1rem 1.2rem;
        box-shadow: var(--shadow);
        min-height: 160px;
    }

    .stat-card .label {
        font-size: 0.76rem;
        color: var(--muted);
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.55rem;
    }

    .stat-card .value {
        font-size: 1.9rem;
        line-height: 1.05;
        font-weight: 700;
        margin-bottom: 0.55rem;
        color: var(--ink);
    }

    .stat-card .caption {
        color: #3d4843;
        font-size: 0.92rem;
        line-height: 1.45;
    }

    .panel {
        background: var(--panel);
        border: 1px solid var(--line);
        border-radius: 24px;
        padding: 1.25rem 1.35rem;
        box-shadow: var(--shadow);
    }

    .panel-strong {
        background: var(--panel-strong);
    }

    .mini-note {
        font-size: 0.88rem;
        color: #39443f;
    }

    .pill-row {
        display: flex;
        flex-wrap: wrap;
        gap: 0.55rem;
        margin-top: 0.75rem;
    }

    .pill {
        padding: 0.45rem 0.72rem;
        border-radius: 999px;
        background: rgba(15, 118, 110, 0.08);
        border: 1px solid rgba(15, 118, 110, 0.12);
        color: var(--forest);
        font-size: 0.82rem;
        font-weight: 600;
    }

    .recommend-card {
        background: linear-gradient(135deg, rgba(255,247,235,0.96), rgba(238,245,242,0.96));
        border: 1px solid rgba(32, 40, 36, 0.08);
        border-radius: 22px;
        padding: 1rem 1.1rem;
        min-height: 124px;
    }

    .recommend-card .num {
        width: 2rem;
        height: 2rem;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        border-radius: 999px;
        background: var(--forest);
        color: #fffdf7;
        font-weight: 700;
        margin-bottom: 0.8rem;
    }

    .recommend-card .text {
        color: var(--ink);
        line-height: 1.45;
    }

    .gallery-caption {
        margin-top: 0.55rem;
        margin-bottom: 1rem;
        color: #3c4742;
        font-size: 0.9rem;
    }

    .panel p, .panel li, .panel strong, .panel span {
        color: #1f2924 !important;
    }

    [data-testid="stSidebar"] > div,
    [data-testid="stSidebar"] section,
    [data-testid="stSidebar"] [data-testid="stSidebarContent"] {
        overflow-x: hidden !important;
    }

    [data-testid="stSidebar"] .block-container {
        padding-left: 0.85rem !important;
        padding-right: 0.85rem !important;
        overflow-x: hidden !important;
    }

    .stDataFrame, [data-testid="stImage"], [data-testid="stMetric"] {
        border-radius: 20px;
        overflow: hidden;
    }

    [data-baseweb="tab-list"] {
        gap: 0.5rem;
        margin-bottom: 1rem;
    }

    [data-baseweb="tab"] {
        background: rgba(255,255,255,0.5);
        border: 1px solid rgba(32, 40, 36, 0.08);
        border-radius: 999px;
        padding: 0.55rem 1rem;
        color: #1c2521 !important;
        font-weight: 600;
    }

    [aria-selected="true"][data-baseweb="tab"] {
        background: #17342f !important;
        color: #f8f3ea !important;
    }

    .sidebar-card {
        width: 100%;
        max-width: 100%;
        padding: 0.95rem 1rem;
        border-radius: 18px;
        background: rgba(5, 35, 29, 0.72);
        border: 1px solid rgba(223, 249, 239, 0.24);
        box-shadow: inset 0 1px 0 rgba(255,255,255,0.1);
        margin-bottom: 0.8rem;
        overflow: hidden;
    }

    .sidebar-card .sidebar-label {
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        opacity: 1;
        color: #bddad0 !important;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }

    .sidebar-card .sidebar-value {
        font-size: 1.15rem;
        font-weight: 700;
        color: #f7fff9 !important;
        text-shadow: 0 1px 1px rgba(0,0,0,0.25);
        overflow-wrap: anywhere;
        word-break: break-word;
    }

    .sidebar-card .sidebar-sub {
        margin-top: 0.18rem;
        font-size: 0.9rem;
        color: #d4eee4 !important;
        font-weight: 600;
        overflow-wrap: anywhere;
        word-break: break-word;
    }

    .accuracy-board {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 0.9rem;
        margin-top: 1rem;
        margin-bottom: 0.9rem;
    }

    .accuracy-card {
        border-radius: 18px;
        border: 1px solid rgba(32, 40, 36, 0.1);
        background: rgba(255, 252, 247, 0.93);
        padding: 0.95rem 1rem;
        box-shadow: 0 10px 24px rgba(28, 35, 32, 0.08);
    }

    .accuracy-card .k {
        font-size: 0.73rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #48625c;
        margin-bottom: 0.3rem;
        font-weight: 700;
    }

    .accuracy-card .v {
        font-size: 1.3rem;
        color: #15211d;
        font-weight: 800;
    }

    .accuracy-card .c {
        font-size: 0.86rem;
        color: #44524d;
        margin-top: 0.2rem;
    }

    @media (max-width: 960px) {
        .hero-strip {
            grid-template-columns: 1fr;
        }
        .accuracy-board {
            grid-template-columns: 1fr;
        }
        h1 {
            font-size: 2.3rem !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_processed_data():
    path = OUTPUT_DIR / "processed_data.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, parse_dates=["Datetime"])
    return df


@st.cache_data(show_spinner=False)
def load_anomalies():
    path = OUTPUT_DIR / "anomalies_detected.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, parse_dates=["Datetime"])
    return df


@st.cache_data(show_spinner=False)
def load_metrics_text():
    path = OUTPUT_DIR / "metrics.txt"
    if not path.exists():
        return None
    return path.read_text(encoding="utf-8", errors="ignore")


@st.cache_data(show_spinner=False)
def load_recommendations():
    path = OUTPUT_DIR / "recommendations.txt"
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    recommendations = []
    for line in lines:
        stripped = line.strip()
        if re.match(r"^\d+\.", stripped):
            recommendations.append(re.sub(r"^\d+\.\s*", "", stripped))
    return recommendations


@st.cache_data(show_spinner=False)
def load_lstm_metrics():
    path = OUTPUT_DIR / "lstm" / "metadata.pkl"
    if not path.exists():
        return {}
    try:
        data = joblib.load(path)
        if isinstance(data, dict):
            return data
    except Exception:
        return {}
    return {}


def parse_metrics(metrics_text):
    values = {}
    if not metrics_text:
        return values

    patterns = {
        "training_samples": r"Training Samples\s*:\s*([\d,]+)",
        "test_samples": r"Test Samples\s*:\s*([\d,]+)",
        "features": r"Features\s*:\s*([\d,]+)",
        "test_mae": r"Test MAE\s*:\s*([\d,.]+)",
        "test_rmse": r"Test RMSE\s*:\s*([\d,.]+)",
        "test_r2": r"Test R(?:2|\^2|²|Â²|Ã‚Â²) Score\s*:\s*([\d.]+)",
        "total_records": r"Total Records\s*:\s*([\d,]+)",
        "anomalies_detected": r"Anomalies Detected\s*:\s*([\d,]+)\s*\(([\d.]+)%\)",
        "estimated_wastage": r"Estimated Wastage\s*:\s*([\d,]+\.\d+)",
        "estimated_cost": r"Estimated Cost\s*:\s*\$([\d,]+\.\d+)",
        "annual_savings": r"Annual Savings Potential\s*:\s*\$([\d,]+\.\d+)",
        "residual_flagged": r"Residual Flagged\s*:\s*([\d,]+)",
        "residual_threshold": r"Residual Threshold\s*:\s*([\d,.]+)",
    }

    for key, pattern in patterns.items():
        match = re.search(pattern, metrics_text)
        if not match:
            continue
        if key == "anomalies_detected":
            values["anomalies_detected"] = match.group(1)
            values["anomaly_pct"] = match.group(2)
        else:
            values[key] = match.group(1)

    feature_matches = re.findall(r"^\s{2}([a-z_]+):\s*([\d.]+)$", metrics_text, flags=re.MULTILINE)
    if feature_matches:
        feature_df = pd.DataFrame(feature_matches, columns=["feature", "importance"])
        feature_df["importance"] = feature_df["importance"].astype(float)
        values["feature_importance"] = feature_df.sort_values("importance", ascending=False)

    return values


def human_currency(value):
    if value is None:
        return "N/A"
    number = float(str(value).replace(",", ""))
    if number >= 1_000_000_000:
        return f"${number / 1_000_000_000:.2f}B"
    if number >= 1_000_000:
        return f"${number / 1_000_000:.2f}M"
    return f"${number:,.0f}"


def human_number(value):
    if value is None:
        return "N/A"
    number = float(str(value).replace(",", ""))
    if number >= 1_000_000:
        return f"{number / 1_000_000:.2f}M"
    if number >= 1_000:
        return f"{number / 1_000:.1f}K"
    return f"{number:,.0f}"


def format_r2(value):
    if value is None:
        return "N/A"
    try:
        return f"{float(value):.4f}"
    except (TypeError, ValueError):
        return str(value)


def r2_percent(value):
    if value is None:
        return "N/A"
    try:
        return f"{float(value) * 100:.2f}%"
    except (TypeError, ValueError):
        return "N/A"


def render_stat_card(label, value, caption):
    st.markdown(
        f"""
        <div class="stat-card">
            <div class="label">{label}</div>
            <div class="value">{value}</div>
            <div class="caption">{caption}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_plot_card(plot_path, caption):
    if plot_path.exists():
        st.image(str(plot_path), use_container_width=True)
        st.markdown(f'<div class="gallery-caption">{caption}</div>', unsafe_allow_html=True)
    else:
        st.info(f"Missing plot: {plot_path.name}")


processed_df = load_processed_data()
anomaly_df = load_anomalies()
metrics_text = load_metrics_text()
recommendations = load_recommendations()
metrics = parse_metrics(metrics_text)
lstm_metrics = load_lstm_metrics()

missing_items = []
if processed_df is None:
    missing_items.append("processed_data.csv")
if metrics_text is None:
    missing_items.append("metrics.txt")
if anomaly_df is None:
    missing_items.append("anomalies_detected.csv")

if missing_items:
    st.error(
        "Dashboard inputs are incomplete. Run `python main.py --dataset PJME_hourly.csv --no-dashboard` first."
    )
    st.stop()


rf_r2_display = format_r2(metrics.get("test_r2"))
rf_r2_pct = r2_percent(metrics.get("test_r2"))
lstm_r2_display = format_r2(lstm_metrics.get("r2"))
lstm_r2_pct = r2_percent(lstm_metrics.get("r2"))
lstm_device = str(lstm_metrics.get("device", "N/A")).upper()


records = len(processed_df)
date_min = processed_df["Datetime"].min()
date_max = processed_df["Datetime"].max()
avg_consumption = processed_df["energy_consumption"].mean()
peak_hour = int(processed_df.groupby("hour")["energy_consumption"].mean().idxmax())
peak_hour_value = processed_df.groupby("hour")["energy_consumption"].mean().max()
top_day = int(processed_df.groupby("dayofweek")["energy_consumption"].mean().idxmax())
top_day_name = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"][top_day]
monthly_avg = (
    processed_df.assign(month_label=processed_df["Datetime"].dt.strftime("%b %Y"))
    .groupby("month_label")["energy_consumption"]
    .mean()
    .tail(12)
)

anomaly_by_type = (
    anomaly_df["anomaly_type"].value_counts().rename_axis("anomaly_type").reset_index(name="count")
)
anomaly_timeline = (
    anomaly_df.assign(month=anomaly_df["Datetime"].dt.to_period("M").astype(str))
    .groupby("month")
    .size()
    .rename("count")
)
top_anomalies = anomaly_df.sort_values("energy_consumption", ascending=False).head(12)


with st.sidebar:
    st.markdown("## Control Room")
    st.markdown(
        """
        <div class="sidebar-card">
            <div class="sidebar-label">Run Status</div>
            <div class="sidebar-value">Outputs Ready</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""
        <div class="sidebar-card">
            <div class="sidebar-label">Date Range</div>
            <div class="sidebar-value">{date_min.date()} to {date_max.date()}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""
        <div class="sidebar-card">
            <div class="sidebar-label">Records Analyzed</div>
            <div class="sidebar-value">{records:,}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""
        <div class="sidebar-card">
            <div class="sidebar-label">Model Accuracy</div>
            <div class="sidebar-value">RF R2 {rf_r2_display}</div>
            <div class="sidebar-sub">LSTM R2 {lstm_r2_display}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        f"""
        <div class="sidebar-card">
            <div class="sidebar-label">Anomaly Rate</div>
            <div class="sidebar-value">{metrics.get("anomaly_pct", "N/A")}%</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.caption("Use this dashboard after every new pipeline run to compare datasets and output quality.")


st.markdown(
    f"""
    <div class="hero-shell">
        <div class="eyebrow">Energy Analytics Major Project</div>
        <h1>Energy Intelligence Dashboard</h1>
        <div class="hero-subtitle">
            A presentation-grade view of demand forecasting, anomaly detection, and estimated savings.
            This version turns the raw pipeline outputs into a cleaner operational story for demos, viva, and project review.
        </div>
        <div class="hero-strip">
            <div class="hero-chip">
                <div class="hero-chip-label">Records Modeled</div>
                <div class="hero-chip-value">{records:,}</div>
            </div>
            <div class="hero-chip">
                <div class="hero-chip-label">Forecast Quality</div>
                <div class="hero-chip-value">RF R2 {rf_r2_display}</div>
            </div>
            <div class="hero-chip">
                <div class="hero-chip-label">Savings Potential</div>
                <div class="hero-chip-value">{human_currency(metrics.get("annual_savings"))}/year</div>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <div class="accuracy-board">
        <div class="accuracy-card">
            <div class="k">Random Forest Accuracy</div>
            <div class="v">R2 {rf_r2_display}</div>
            <div class="c">Approx. fit quality: {rf_r2_pct}</div>
        </div>
        <div class="accuracy-card">
            <div class="k">LSTM Accuracy</div>
            <div class="v">R2 {lstm_r2_display}</div>
            <div class="c">Approx. fit quality: {lstm_r2_pct} | Device: {lstm_device}</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)


tabs = st.tabs(["Overview", "Visual Lab", "Anomaly Center", "Recommendations"])


with tabs[0]:
    st.markdown('<div class="section-title">Executive Snapshot</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-copy">The dashboard leads with the numbers that matter most in a project presentation: scale, prediction quality, anomalies, and financial impact.</div>',
        unsafe_allow_html=True,
    )

    stat_cols = st.columns(4)
    with stat_cols[0]:
        render_stat_card(
            "Average Load",
            f"{avg_consumption:,.0f} MW",
            f"Across {records:,} cleaned hourly observations.",
        )
    with stat_cols[1]:
        render_stat_card(
            "Anomalies Detected",
            human_number(metrics.get("anomalies_detected")),
            f"{metrics.get('anomaly_pct', 'N/A')}% of all modeled records were flagged.",
        )
    with stat_cols[2]:
        render_stat_card(
            "Estimated Waste Cost",
            human_currency(metrics.get("estimated_cost")),
            "Potential cost impact of abnormal high-consumption behavior.",
        )
    with stat_cols[3]:
        render_stat_card(
            "Residual Alerts",
            human_number(metrics.get("residual_flagged")),
            f"Predictions above the {metrics.get('residual_threshold', 'N/A')} MW residual threshold.",
        )

    left, right = st.columns([1.05, 0.95], gap="large")

    with left:
        st.markdown('<div class="panel panel-strong">', unsafe_allow_html=True)
        st.subheader("Operational Highlights")
        st.markdown(
            f"""
            The strongest demand driver in this model is **hour of day**, followed by **month**. The highest average load
            appears around **{peak_hour:02d}:00**, with a typical value near **{peak_hour_value:,.0f} MW**. The heaviest weekday load
            trend occurs on **{top_day_name}**.
            """,
        )
        st.markdown(
            """
            <div class="pill-row">
                <div class="pill">Random Forest Forecasting</div>
                <div class="pill">Dual-Method Anomaly Detection</div>
                <div class="pill">Chronological Train/Test Split</div>
                <div class="pill">Energy Cost Estimation</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)

        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.subheader("Recent Monthly Average Load")
        st.line_chart(monthly_avg, height=260, use_container_width=True, color="#0f766e")
        st.markdown(
            '<div class="mini-note">This view gives the presentation a stronger narrative than a raw CSV preview. It shows how average demand moves across the most recent months in the dataset.</div>',
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.subheader("Model Feature Importance")
        feature_df = metrics.get("feature_importance")
        if feature_df is not None and not feature_df.empty:
            feature_chart = feature_df.set_index("feature")
            st.bar_chart(feature_chart, height=320, color="#c95f3d")
        else:
            st.info("Feature importance data is not available.")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)

        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.subheader("Dataset Preview")
        preview_cols = ["Datetime", "energy_consumption", "hour", "dayofweek", "month", "is_peak_hour"]
        st.dataframe(
            processed_df[preview_cols].head(12),
            use_container_width=True,
            hide_index=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)


with tabs[1]:
    st.markdown('<div class="section-title">Visual Storyboard</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-copy">The visual lab organizes the saved plots like a polished presentation deck instead of a long unstructured image dump.</div>',
        unsafe_allow_html=True,
    )

    st.subheader("Core Visuals")
    col1, col2 = st.columns(2, gap="large")
    with col1:
        render_plot_card(PLOTS_DIR / "trend.png", "Long-range trend view of energy consumption over time.")
        render_plot_card(PLOTS_DIR / "predictions.png", "Forecast quality check comparing actual and predicted values.")
    with col2:
        render_plot_card(PLOTS_DIR / "hourly_pattern.png", "Average hourly demand pattern with peak-hour emphasis.")

    st.subheader("Advanced Analytics Gallery")
    adv_cols = st.columns(2, gap="large")
    advanced_plot_info = [
        ("anomaly_scatter.png", "Detected anomalies across the timeline."),
        ("heatmap_hour_day.png", "Hourly and weekday interaction heatmap."),
        ("monthly_boxplot.png", "Monthly load distribution spread."),
        ("anomaly_breakdown.png", "Anomaly categories and their hourly concentration."),
        ("zscore_analysis.png", "Z-score distribution and outlier thresholds."),
        ("weekday_weekend.png", "Weekday versus weekend consumption behavior."),
    ]
    for idx, (filename, caption) in enumerate(advanced_plot_info):
        with adv_cols[idx % 2]:
            render_plot_card(ADVANCED_PLOTS_DIR / filename, caption)


with tabs[2]:
    st.markdown('<div class="section-title">Anomaly Center</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-copy">This section isolates unusual behavior so you can show what the model flags, where those events cluster, and which records deserve investigation first.</div>',
        unsafe_allow_html=True,
    )

    anomaly_cols = st.columns([0.95, 1.05], gap="large")
    with anomaly_cols[0]:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.subheader("Anomaly Type Distribution")
        st.bar_chart(anomaly_by_type.set_index("anomaly_type"), height=300, color="#d6942a")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div style='height: 1rem;'></div>", unsafe_allow_html=True)

        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.subheader("Monthly Anomaly Volume")
        st.area_chart(anomaly_timeline, height=280, use_container_width=True, color="#0f766e")
        st.markdown("</div>", unsafe_allow_html=True)

    with anomaly_cols[1]:
        st.markdown('<div class="panel panel-strong">', unsafe_allow_html=True)
        st.subheader("Highest-Load Anomalous Records")
        display_anomalies = top_anomalies.copy()
        display_anomalies["Datetime"] = display_anomalies["Datetime"].dt.strftime("%Y-%m-%d %H:%M")
        st.dataframe(
            display_anomalies[["Datetime", "energy_consumption", "z_score", "anomaly_type"]],
            use_container_width=True,
            hide_index=True,
        )
        st.markdown(
            '<div class="mini-note">These are the most extreme anomalous readings by demand magnitude. In a presentation, this is the table to discuss operational risk.</div>',
            unsafe_allow_html=True,
        )
        st.markdown("</div>", unsafe_allow_html=True)


with tabs[3]:
    st.markdown('<div class="section-title">Optimization Actions</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="section-copy">The recommendations panel translates model outputs into project-ready operational actions and clean summary text.</div>',
        unsafe_allow_html=True,
    )

    action_cols = st.columns(2, gap="large")

    with action_cols[0]:
        st.markdown('<div class="panel panel-strong">', unsafe_allow_html=True)
        st.subheader("Recommended Actions")
        if recommendations:
            rec_cols = st.columns(2, gap="medium")
            for idx, recommendation in enumerate(recommendations):
                with rec_cols[idx % 2]:
                    st.markdown(
                        f"""
                        <div class="recommend-card">
                            <div class="num">{idx + 1}</div>
                            <div class="text">{recommendation}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
        else:
            st.info("Recommendations file is not available.")
        st.markdown("</div>", unsafe_allow_html=True)

    with action_cols[1]:
        st.markdown('<div class="panel">', unsafe_allow_html=True)
        st.subheader("Project Summary")
        st.markdown(
            f"""
            - **Dataset window:** {date_min.date()} to {date_max.date()}
            - **Training samples:** {metrics.get('training_samples', 'N/A')}
            - **Testing samples:** {metrics.get('test_samples', 'N/A')}
            - **Forecast error:** MAE {metrics.get('test_mae', 'N/A')} MW, RMSE {metrics.get('test_rmse', 'N/A')} MW
            - **Potential annual savings:** {human_currency(metrics.get('annual_savings'))}
            - **Peak demand period:** {peak_hour:02d}:00
            """
        )
        if metrics_text:
            with st.expander("Open raw metrics text"):
                st.text(metrics_text)
        st.markdown("</div>", unsafe_allow_html=True)
