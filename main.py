from __future__ import annotations

import argparse
import subprocess
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler

warnings.filterwarnings("ignore")
plt.switch_backend("Agg")

RF_FEATURES = ["hour", "dayofweek", "month", "is_peak_hour", "is_weekend"]
LSTM_FEATURES = ["energy_consumption", "hour", "dayofweek", "month", "is_peak_hour"]


@dataclass
class RFResult:
    model: RandomForestRegressor
    split_idx: int
    y_train: pd.Series
    y_test: pd.Series
    y_pred_train: np.ndarray
    y_pred_test: np.ndarray
    y_pred_all: np.ndarray
    feature_importance: dict[str, float]
    metrics: dict[str, float]


@dataclass
class AnomalyResult:
    full_df: pd.DataFrame
    anomaly_df: pd.DataFrame
    summary: dict[str, float]
    recommendations: list[str]


def log(message: str = "") -> None:
    print(message, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Energy consumption pipeline: training, anomaly detection, plots, LSTM, dashboard outputs."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="PJME_hourly.csv",
        help="Dataset path or filename. If filename, searched recursively in the project.",
    )
    parser.add_argument("--data-dir", type=str, default="data")
    parser.add_argument("--archive-dir", type=str, default="archive__1_")
    parser.add_argument("--output-dir", type=str, default="outputs")
    parser.add_argument("--no-dashboard", action="store_true", help="Do not launch Streamlit at the end.")
    parser.add_argument("--skip-lstm", action="store_true", help="Skip LSTM training.")
    parser.add_argument("--epochs", type=int, default=8, help="LSTM epochs.")
    parser.add_argument("--batch-size", type=int, default=512, help="LSTM batch size.")
    parser.add_argument("--sample-size", type=int, default=50000, help="Maximum train rows for LSTM.")
    parser.add_argument("--sequence-length", type=int, default=24, help="LSTM sequence length.")
    parser.add_argument(
        "--lstm-train-end",
        type=str,
        default="2015-12-31 23:59:59",
        help="Date cutoff used for LSTM date split.",
    )
    parser.add_argument(
        "--residual-quantile",
        type=float,
        default=0.95,
        help="Residual quantile threshold for anomaly flagging.",
    )
    parser.add_argument("--cost-per-mwh", type=float, default=120.0, help="Cost conversion for wastage estimates.")
    parser.add_argument("--dashboard-port", type=int, default=8501)
    return parser.parse_args()


def ensure_directories(output_dir: Path) -> None:
    (output_dir / "plots").mkdir(parents=True, exist_ok=True)
    (output_dir / "plots" / "advanced").mkdir(parents=True, exist_ok=True)
    (output_dir / "lstm").mkdir(parents=True, exist_ok=True)


def find_dataset(project_root: Path, dataset_arg: str, data_dir: str, archive_dir: str) -> Path:
    explicit = Path(dataset_arg)
    if explicit.exists():
        return explicit.resolve()

    if not explicit.is_absolute():
        maybe_local = (project_root / explicit).resolve()
        if maybe_local.exists():
            return maybe_local

    search_roots = [project_root / archive_dir, project_root / data_dir, project_root]
    matches: list[Path] = []
    for root in search_roots:
        if not root.exists():
            continue
        if explicit.name:
            matches.extend([p for p in root.rglob(explicit.name) if p.is_file()])

    if not matches and dataset_arg == "PJME_hourly.csv":
        fallback_patterns = ["*hourly*.csv", "*.csv"]
        for pattern in fallback_patterns:
            for root in search_roots:
                if root.exists():
                    matches.extend([p for p in root.rglob(pattern) if p.is_file()])
            if matches:
                break

    if not matches:
        raise FileNotFoundError(f"Dataset not found: {dataset_arg}")

    unique = {str(p.resolve()): p.resolve() for p in matches}
    chosen = sorted(unique.values(), key=lambda p: p.stat().st_size, reverse=True)[0]
    return chosen


def infer_columns(raw_df: pd.DataFrame) -> tuple[str, str]:
    cols = [str(c) for c in raw_df.columns]
    datetime_col = next(
        (c for c in cols if any(token in c.lower() for token in ["datetime", "date", "time"])),
        cols[0],
    )

    if "energy_consumption" in cols:
        return datetime_col, "energy_consumption"

    candidate_scores: list[tuple[str, float]] = []
    for col in cols:
        if col == datetime_col:
            continue
        numeric = pd.to_numeric(raw_df[col], errors="coerce")
        candidate_scores.append((col, float(numeric.notna().mean())))

    if not candidate_scores:
        raise ValueError("No usable energy column found.")

    candidate_scores.sort(key=lambda item: item[1], reverse=True)
    energy_col = candidate_scores[0][0]
    return datetime_col, energy_col


def load_dataset(dataset_path: Path) -> tuple[pd.DataFrame, str, str]:
    raw_df = pd.read_csv(dataset_path, low_memory=False)
    if raw_df.shape[1] < 2:
        raise ValueError(f"Dataset must contain at least 2 columns: {dataset_path}")

    datetime_col, energy_col = infer_columns(raw_df)
    df = pd.DataFrame(
        {
            "Datetime": pd.to_datetime(raw_df[datetime_col], errors="coerce"),
            "energy_consumption": pd.to_numeric(raw_df[energy_col], errors="coerce"),
        }
    )
    before = len(df)
    df = df.dropna(subset=["Datetime", "energy_consumption"])
    df = df.sort_values("Datetime").drop_duplicates(subset="Datetime").reset_index(drop=True)
    after = len(df)

    if after < 1000:
        raise ValueError(f"Dataset has too few valid rows after cleaning ({after}).")

    log(f"Rows loaded: {before:,} | usable rows: {after:,}")
    return df, datetime_col, energy_col


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["hour"] = out["Datetime"].dt.hour
    out["dayofweek"] = out["Datetime"].dt.dayofweek
    out["month"] = out["Datetime"].dt.month
    out["is_peak_hour"] = out["hour"].between(17, 21).astype(int)
    out["is_weekend"] = (out["dayofweek"] >= 5).astype(int)
    return out


def train_random_forest(df: pd.DataFrame, output_dir: Path) -> RFResult:
    X = df[RF_FEATURES].copy()
    y = df["energy_consumption"].copy()

    split_idx = int(len(df) * 0.8)
    split_idx = max(1, min(split_idx, len(df) - 1))

    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=20,
        min_samples_split=4,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    y_pred_all = model.predict(X)

    metrics = {
        "train_mae": float(mean_absolute_error(y_train, y_pred_train)),
        "test_mae": float(mean_absolute_error(y_test, y_pred_test)),
        "train_rmse": float(np.sqrt(mean_squared_error(y_train, y_pred_train))),
        "test_rmse": float(np.sqrt(mean_squared_error(y_test, y_pred_test))),
        "train_r2": float(r2_score(y_train, y_pred_train)),
        "test_r2": float(r2_score(y_test, y_pred_test)),
        "train_samples": int(len(y_train)),
        "test_samples": int(len(y_test)),
        "features": int(len(RF_FEATURES)),
    }

    feature_importance = {
        name: float(value) for name, value in zip(RF_FEATURES, model.feature_importances_)
    }
    feature_importance = dict(sorted(feature_importance.items(), key=lambda item: item[1], reverse=True))

    joblib.dump(model, output_dir / "model.pkl")
    return RFResult(
        model=model,
        split_idx=split_idx,
        y_train=y_train,
        y_test=y_test,
        y_pred_train=y_pred_train,
        y_pred_test=y_pred_test,
        y_pred_all=y_pred_all,
        feature_importance=feature_importance,
        metrics=metrics,
    )


def classify_anomaly(row: pd.Series, residual_threshold: float) -> str:
    if not bool(row["is_anomaly"]):
        return "Normal"
    if row["residual"] >= residual_threshold:
        return "Demand Surge"
    if row["residual"] <= -residual_threshold:
        return "Demand Drop"
    if row["is_peak_hour"] == 1 and row["z_score"] > 2:
        return "Peak Hour Spike"
    return "Pattern Outlier"


def detect_anomalies(
    df: pd.DataFrame,
    y_pred_all: np.ndarray,
    residual_quantile: float,
    cost_per_mwh: float,
) -> AnomalyResult:
    out = df.copy()
    out["predicted_rf"] = y_pred_all
    out["residual"] = out["energy_consumption"] - out["predicted_rf"]

    hour_mean = out.groupby("hour")["energy_consumption"].transform("mean")
    hour_std = out.groupby("hour")["energy_consumption"].transform("std").replace(0, np.nan)
    hour_std = hour_std.fillna(out["energy_consumption"].std())
    out["hour_mean"] = hour_mean
    out["z_score"] = ((out["energy_consumption"] - out["hour_mean"]) / hour_std).fillna(0.0)

    residual_threshold = float(np.quantile(np.abs(out["residual"]), residual_quantile))
    out["residual_flag"] = np.abs(out["residual"]) > residual_threshold
    out["z_flag"] = np.abs(out["z_score"]) > 2.0
    out["is_anomaly"] = out["residual_flag"] | out["z_flag"]
    out["anomaly_type"] = out.apply(classify_anomaly, axis=1, residual_threshold=residual_threshold)
    out["wastage_mw"] = np.where(
        out["is_anomaly"] & (out["energy_consumption"] > out["hour_mean"]),
        out["energy_consumption"] - out["hour_mean"],
        0.0,
    )
    out["cost_usd"] = out["wastage_mw"] * cost_per_mwh

    anomaly_df = out.loc[
        out["is_anomaly"],
        [
            "Datetime",
            "hour",
            "energy_consumption",
            "hour_mean",
            "z_score",
            "residual",
            "anomaly_type",
            "is_peak_hour",
            "is_weekend",
            "wastage_mw",
            "cost_usd",
        ],
    ].copy()

    total_records = len(out)
    anomaly_count = int(out["is_anomaly"].sum())
    anomaly_pct = (anomaly_count / total_records * 100.0) if total_records else 0.0
    estimated_wastage = float(out["wastage_mw"].sum())
    estimated_cost = float(out["cost_usd"].sum())

    days_span = max((out["Datetime"].max() - out["Datetime"].min()).days + 1, 1)
    annual_savings = estimated_cost * (365.0 / days_span)
    residual_flagged = int(out["residual_flag"].sum())

    summary = {
        "total_records": int(total_records),
        "anomalies_detected": anomaly_count,
        "anomaly_pct": float(anomaly_pct),
        "estimated_wastage": estimated_wastage,
        "estimated_cost": estimated_cost,
        "annual_savings": float(annual_savings),
        "residual_flagged": residual_flagged,
        "residual_threshold": float(residual_threshold),
    }

    recommendations = build_recommendations(out, summary)
    return AnomalyResult(full_df=out, anomaly_df=anomaly_df, summary=summary, recommendations=recommendations)


def build_recommendations(df: pd.DataFrame, summary: dict[str, float]) -> list[str]:
    recommendations: list[str] = []
    peak_anomalies = int(df.loc[df["is_anomaly"] & (df["is_peak_hour"] == 1)].shape[0])
    weekend_anomalies = int(df.loc[df["is_anomaly"] & (df["is_weekend"] == 1)].shape[0])
    surge_count = int(df.loc[df["anomaly_type"] == "Demand Surge"].shape[0])

    if peak_anomalies > 0:
        recommendations.append(
            f"{peak_anomalies:,} anomalies occurred in peak hours. Shift flexible loads outside 17:00-21:00."
        )
    if weekend_anomalies > 0:
        recommendations.append(
            f"{weekend_anomalies:,} anomalies appeared on weekends. Audit idle equipment and HVAC schedules."
        )
    if surge_count > 0:
        recommendations.append(
            f"{surge_count:,} high-demand surges were detected. Add threshold alerts and staged load shedding."
        )

    recommendations.append(
        f"Estimated annual savings potential is ${summary['annual_savings']:,.2f} based on detected avoidable load."
    )
    recommendations.append("Review top anomalous timestamps weekly and track a reduction target month over month.")
    return recommendations[:5]


def write_recommendations(output_dir: Path, recommendations: list[str]) -> None:
    lines = ["OPTIMIZATION RECOMMENDATIONS", "=" * 40, ""]
    lines.extend([f"{idx}. {rec}" for idx, rec in enumerate(recommendations, start=1)])
    (output_dir / "recommendations.txt").write_text("\n".join(lines), encoding="utf-8")


def write_metrics_file(
    output_dir: Path,
    dataset_name: str,
    rf: RFResult,
    anomaly: AnomalyResult,
) -> None:
    lines = [
        "RANDOM FOREST METRICS",
        "=" * 50,
        f"Dataset: {dataset_name}",
        "",
        f"Training Samples: {rf.metrics['train_samples']:,}",
        f"Test Samples: {rf.metrics['test_samples']:,}",
        f"Features: {rf.metrics['features']}",
        "",
        f"Train MAE: {rf.metrics['train_mae']:.2f}",
        f"Test MAE: {rf.metrics['test_mae']:.2f}",
        f"Train RMSE: {rf.metrics['train_rmse']:.2f}",
        f"Test RMSE: {rf.metrics['test_rmse']:.2f}",
        f"Train R2 Score: {rf.metrics['train_r2']:.4f}",
        f"Test R2 Score: {rf.metrics['test_r2']:.4f}",
        "",
        "Feature Importance:",
    ]

    for feature, importance in rf.feature_importance.items():
        lines.append(f"  {feature}: {importance:.4f}")

    lines.extend(
        [
            "",
            "ANOMALY SUMMARY",
            "=" * 50,
            f"Total Records: {anomaly.summary['total_records']:,}",
            f"Anomalies Detected: {anomaly.summary['anomalies_detected']:,} ({anomaly.summary['anomaly_pct']:.2f}%)",
            f"Estimated Wastage: {anomaly.summary['estimated_wastage']:,.2f}",
            f"Estimated Cost: ${anomaly.summary['estimated_cost']:,.2f}",
            f"Annual Savings Potential: ${anomaly.summary['annual_savings']:,.2f}",
            f"Residual Flagged: {anomaly.summary['residual_flagged']:,}",
            f"Residual Threshold: {anomaly.summary['residual_threshold']:.2f}",
        ]
    )

    metrics_text = "\n".join(lines)
    (output_dir / "metrics.txt").write_text(metrics_text, encoding="utf-8")
    (output_dir / "rf_metrics.txt").write_text(metrics_text, encoding="utf-8")


def save_prediction_scenarios(output_dir: Path, model: RandomForestRegressor) -> None:
    scenarios = [
        {
            "scenario": "Weekday Morning Peak",
            "hour": 8,
            "dayofweek": 1,
            "month": 6,
            "is_peak_hour": 0,
            "is_weekend": 0,
        },
        {
            "scenario": "Weekday Evening Peak",
            "hour": 19,
            "dayofweek": 2,
            "month": 7,
            "is_peak_hour": 1,
            "is_weekend": 0,
        },
        {
            "scenario": "Weekend Afternoon",
            "hour": 14,
            "dayofweek": 6,
            "month": 8,
            "is_peak_hour": 0,
            "is_weekend": 1,
        },
        {
            "scenario": "Night Off-Peak",
            "hour": 2,
            "dayofweek": 3,
            "month": 1,
            "is_peak_hour": 0,
            "is_weekend": 0,
        },
        {
            "scenario": "Holiday Morning",
            "hour": 10,
            "dayofweek": 5,
            "month": 12,
            "is_peak_hour": 0,
            "is_weekend": 1,
        },
    ]

    rows: list[dict[str, Any]] = []
    for scenario in scenarios:
        row = pd.DataFrame([scenario])[RF_FEATURES]
        prediction = float(model.predict(row)[0])
        payload = scenario.copy()
        payload["rf_prediction"] = prediction
        rows.append(payload)

    pd.DataFrame(rows).to_csv(output_dir / "prediction_scenarios.csv", index=False)


def create_plots(df: pd.DataFrame, rf: RFResult, output_dir: Path) -> None:
    plots_dir = output_dir / "plots"
    advanced_dir = plots_dir / "advanced"
    advanced_dir.mkdir(parents=True, exist_ok=True)

    sns.set_theme(style="whitegrid")

    sample_every = max(1, len(df) // 12000)
    trend_df = df.iloc[::sample_every]
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.plot(trend_df["Datetime"], trend_df["energy_consumption"], color="#2e86de", linewidth=1.0)
    ax.set_title("Energy Consumption Trend")
    ax.set_xlabel("Datetime")
    ax.set_ylabel("Energy Consumption (MW)")
    plt.tight_layout()
    plt.savefig(plots_dir / "trend.png", dpi=160, bbox_inches="tight")
    plt.close()

    hourly = df.groupby("hour")["energy_consumption"].mean()
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(hourly.index, hourly.values, color="#16a085", alpha=0.85)
    ax.axvspan(17, 21, color="orange", alpha=0.15, label="Peak Hour Window")
    ax.set_title("Average Hourly Consumption Pattern")
    ax.set_xlabel("Hour")
    ax.set_ylabel("Average MW")
    ax.set_xticks(range(0, 24, 2))
    ax.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "hourly_pattern.png", dpi=160, bbox_inches="tight")
    plt.close()

    sample = min(700, len(rf.y_test))
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.plot(
        rf.y_test.index[-sample:],
        rf.y_test.values[-sample:],
        label="Actual",
        color="#3498db",
        linewidth=1.2,
    )
    ax.plot(
        rf.y_test.index[-sample:],
        rf.y_pred_test[-sample:],
        label=f"Predicted (R2={rf.metrics['test_r2']:.3f})",
        color="#e74c3c",
        linewidth=1.2,
        linestyle="--",
    )
    ax.set_title("Random Forest: Test Predictions vs Actual")
    ax.set_xlabel("Sample Index")
    ax.set_ylabel("Energy Consumption (MW)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "predictions.png", dpi=160, bbox_inches="tight")
    plt.close()

    scatter_df = df.iloc[::max(1, len(df) // 25000)].copy()
    normal_df = scatter_df[~scatter_df["is_anomaly"]]
    anomaly_df = scatter_df[scatter_df["is_anomaly"]]
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.scatter(normal_df["Datetime"], normal_df["energy_consumption"], s=6, alpha=0.35, label="Normal", color="#2980b9")
    ax.scatter(anomaly_df["Datetime"], anomaly_df["energy_consumption"], s=16, alpha=0.85, label="Anomaly", color="#c0392b")
    ax.set_title("Anomaly Detection Timeline")
    ax.set_xlabel("Datetime")
    ax.set_ylabel("Energy Consumption (MW)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(advanced_dir / "anomaly_scatter.png", dpi=200, bbox_inches="tight")
    plt.close()

    heatmap_data = df.pivot_table(index="dayofweek", columns="hour", values="energy_consumption", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(14, 6))
    sns.heatmap(heatmap_data, cmap="YlOrRd", ax=ax)
    ax.set_title("Average Consumption Heatmap by Day and Hour")
    ax.set_xlabel("Hour")
    ax.set_ylabel("Day of Week (0=Mon)")
    plt.tight_layout()
    plt.savefig(advanced_dir / "heatmap_hour_day.png", dpi=200, bbox_inches="tight")
    plt.close()

    month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    monthly_df = df.copy()
    monthly_df["month_name"] = monthly_df["month"].map(lambda m: month_names[m - 1])
    fig, ax = plt.subplots(figsize=(14, 6))
    sns.boxplot(data=monthly_df, x="month_name", y="energy_consumption", ax=ax, palette="Set3")
    ax.set_title("Monthly Consumption Distribution")
    ax.set_xlabel("Month")
    ax.set_ylabel("Energy Consumption (MW)")
    plt.tight_layout()
    plt.savefig(advanced_dir / "monthly_boxplot.png", dpi=200, bbox_inches="tight")
    plt.close()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    anomaly_counts = df.loc[df["is_anomaly"], "anomaly_type"].value_counts()
    if not anomaly_counts.empty:
        ax1.pie(anomaly_counts.values, labels=anomaly_counts.index, autopct="%1.1f%%", startangle=140)
        ax1.set_title("Anomaly Type Share")
    else:
        ax1.text(0.5, 0.5, "No anomalies", ha="center", va="center")
        ax1.set_title("Anomaly Type Share")

    anomaly_by_hour = df.loc[df["is_anomaly"]].groupby("hour").size()
    ax2.bar(anomaly_by_hour.index, anomaly_by_hour.values, color="#f39c12", alpha=0.85)
    ax2.set_title("Anomaly Count by Hour")
    ax2.set_xlabel("Hour")
    ax2.set_ylabel("Count")
    ax2.set_xticks(range(0, 24, 2))
    plt.tight_layout()
    plt.savefig(advanced_dir / "anomaly_breakdown.png", dpi=200, bbox_inches="tight")
    plt.close()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))
    ax1.hist(df["z_score"], bins=120, color="#3498db", alpha=0.8)
    ax1.axvline(-2, color="orange", linestyle="--", linewidth=1.2, label="-2 threshold")
    ax1.axvline(2, color="orange", linestyle="--", linewidth=1.2, label="+2 threshold")
    ax1.set_title("Z-score Distribution")
    ax1.set_xlabel("Z-score")
    ax1.set_ylabel("Frequency")
    ax1.legend()

    z_sample = df.iloc[::max(1, len(df) // 15000)]
    colors = np.where(z_sample["is_anomaly"], "#e74c3c", "#2e86de")
    ax2.scatter(z_sample["Datetime"], z_sample["z_score"], c=colors, s=6, alpha=0.5)
    ax2.axhline(-2, color="orange", linestyle="--", linewidth=1.2)
    ax2.axhline(2, color="orange", linestyle="--", linewidth=1.2)
    ax2.set_title("Z-score Over Time")
    ax2.set_xlabel("Datetime")
    ax2.set_ylabel("Z-score")
    plt.tight_layout()
    plt.savefig(advanced_dir / "zscore_analysis.png", dpi=200, bbox_inches="tight")
    plt.close()

    weekday = df[df["is_weekend"] == 0].groupby("hour")["energy_consumption"].mean()
    weekend = df[df["is_weekend"] == 1].groupby("hour")["energy_consumption"].mean()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    ax1.plot(weekday.index, weekday.values, marker="o", label="Weekday", color="#2e86de")
    ax1.plot(weekend.index, weekend.values, marker="s", label="Weekend", color="#e67e22")
    ax1.axvspan(17, 21, alpha=0.12, color="gold")
    ax1.set_title("Weekday vs Weekend Profile")
    ax1.set_xlabel("Hour")
    ax1.set_ylabel("Average Consumption (MW)")
    ax1.legend()
    ax1.set_xticks(range(0, 24, 2))

    bars = ax2.bar(
        ["Weekday", "Weekend"],
        [weekday.mean(), weekend.mean()],
        color=["#2e86de", "#e67e22"],
        alpha=0.85,
    )
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2.0, height, f"{height:.0f} MW", ha="center", va="bottom")
    ax2.set_title("Average Demand Comparison")
    ax2.set_ylabel("Consumption (MW)")
    plt.tight_layout()
    plt.savefig(advanced_dir / "weekday_weekend.png", dpi=200, bbox_inches="tight")
    plt.close()


def create_sequences(data: np.ndarray, seq_len: int) -> tuple[np.ndarray, np.ndarray]:
    X, y = [], []
    for idx in range(len(data) - seq_len):
        X.append(data[idx : idx + seq_len])
        y.append(data[idx + seq_len, 0])
    if not X:
        return np.empty((0, seq_len, data.shape[1]), dtype=np.float32), np.empty((0,), dtype=np.float32)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def train_lstm(
    df: pd.DataFrame,
    rf: RFResult,
    output_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any] | None:
    if args.skip_lstm:
        log("LSTM skipped by flag.")
        return None

    try:
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset
    except Exception:
        log("PyTorch not available. LSTM skipped.")
        return None

    if len(df) < (args.sequence_length + 500):
        log("Dataset too small for LSTM sequence training. LSTM skipped.")
        return None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"LSTM device: {device}")

    train_end = pd.Timestamp(args.lstm_train_end)
    use_date_split = bool((df["Datetime"] > train_end).any() and (df["Datetime"] <= train_end).sum() > args.sequence_length)

    if use_date_split:
        df_train = df[df["Datetime"] <= train_end].tail(args.sample_size).copy()
        df_test = df[df["Datetime"] > train_end].copy()
        if len(df_test) < 24:
            use_date_split = False
        else:
            train_raw = df_train[LSTM_FEATURES].values
            test_raw = df_test[LSTM_FEATURES].values
            scaler = MinMaxScaler()
            train_scaled = scaler.fit_transform(train_raw)
            test_scaled = scaler.transform(test_raw)
            X_train, y_train = create_sequences(train_scaled, args.sequence_length)
            test_seed = np.vstack([train_scaled[-args.sequence_length :], test_scaled])
            X_test, y_test = create_sequences(test_seed, args.sequence_length)
            test_dates = df_test["Datetime"].values[: len(y_test)]
    if not use_date_split:
        df_sample = df.tail(args.sample_size).copy()
        raw = df_sample[LSTM_FEATURES].values
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(raw)
        X_all, y_all = create_sequences(scaled, args.sequence_length)
        split = int(len(X_all) * 0.8)
        split = max(1, min(split, len(X_all) - 1))
        X_train, y_train = X_all[:split], y_all[:split]
        X_test, y_test = X_all[split:], y_all[split:]
        date_values = df_sample["Datetime"].values
        test_dates = date_values[split + args.sequence_length : split + args.sequence_length + len(y_test)]

    if len(X_train) == 0 or len(X_test) == 0:
        log("Not enough LSTM sequences after split. LSTM skipped.")
        return None

    train_ds = TensorDataset(
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)

    class EnergyLSTM(nn.Module):
        def __init__(self, input_size: int, hidden_size: int = 64) -> None:
            super().__init__()
            self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=1, batch_first=True)
            self.head = nn.Sequential(
                nn.Linear(hidden_size, 32),
                nn.ReLU(),
                nn.Linear(32, 1),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            out, _ = self.lstm(x)
            out = out[:, -1, :]
            return self.head(out)

    model = EnergyLSTM(input_size=X_train.shape[2]).to(device)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    log(f"LSTM sequences -> train: {len(X_train):,}, test: {len(X_test):,}, epochs: {args.epochs}")
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        for batch_X, batch_y in train_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()
            pred = model(batch_X).squeeze()
            loss = loss_fn(pred, batch_y)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item())
        if (epoch + 1) in {1, args.epochs} or (epoch + 1) % max(1, args.epochs // 3) == 0:
            log(f"  epoch {epoch + 1}/{args.epochs}  loss={running_loss / len(train_loader):.6f}")

    model.eval()
    test_ds = TensorDataset(torch.tensor(X_test, dtype=torch.float32))
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    pred_scaled: list[np.ndarray] = []
    with torch.no_grad():
        for (batch_X,) in test_loader:
            pred = model(batch_X.to(device)).cpu().numpy().reshape(-1)
            pred_scaled.append(pred)
    y_pred_scaled = np.concatenate(pred_scaled) if pred_scaled else np.array([], dtype=np.float32)

    n_features = len(LSTM_FEATURES)
    pred_dummy = np.zeros((len(y_pred_scaled), n_features), dtype=np.float32)
    pred_dummy[:, 0] = y_pred_scaled
    y_pred = scaler.inverse_transform(pred_dummy)[:, 0]

    actual_dummy = np.zeros((len(y_test), n_features), dtype=np.float32)
    actual_dummy[:, 0] = y_test
    y_true = scaler.inverse_transform(actual_dummy)[:, 0]

    lstm_metrics = {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "train_sequences": int(len(X_train)),
        "test_sequences": int(len(X_test)),
        "device": str(device),
        "use_date_split": bool(use_date_split),
    }

    torch.save(model.state_dict(), output_dir / "lstm" / "lstm_model.pth")
    joblib.dump(scaler, output_dir / "lstm" / "scaler.pkl")
    joblib.dump(
        {
            "sequence_length": args.sequence_length,
            "features": LSTM_FEATURES,
            "sample_size": args.sample_size,
            **lstm_metrics,
        },
        output_dir / "lstm" / "metadata.pkl",
    )

    fig, ax = plt.subplots(figsize=(16, 6))
    compare_n = min(350, len(y_pred), len(rf.y_pred_test), len(rf.y_test))
    if compare_n > 0:
        ax.plot(rf.y_test.values[-compare_n:], label=f"RF actual (R2={rf.metrics['test_r2']:.3f})", color="#2e86de")
        ax.plot(rf.y_pred_test[-compare_n:], label="RF predicted", color="#e74c3c", linestyle="--")
        ax.plot(y_pred[:compare_n], label=f"LSTM predicted (R2={lstm_metrics['r2']:.3f})", color="#27ae60")
    ax.set_title("Model Comparison: Random Forest vs LSTM")
    ax.set_xlabel("Sample")
    ax.set_ylabel("Energy Consumption (MW)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "plots" / "advanced" / "model_comparison.png", dpi=200, bbox_inches="tight")
    plt.close()

    if len(test_dates) == len(y_pred):
        fig, ax = plt.subplots(figsize=(18, 6))
        ax.plot(test_dates, y_true, label="Actual", color="#2e86de", linewidth=1.2)
        ax.plot(test_dates, y_pred, label="LSTM Forecast", color="#e74c3c", linewidth=1.2, linestyle="--")
        ax.fill_between(test_dates, y_true, y_pred, alpha=0.12, color="gray")
        ax.set_title("LSTM Forecast vs Actual")
        ax.set_xlabel("Date")
        ax.set_ylabel("Energy Consumption (MW)")
        ax.legend()
        plt.tight_layout()
        plt.savefig(output_dir / "plots" / "advanced" / "lstm_forecast_vs_actual.png", dpi=200, bbox_inches="tight")
        plt.close()

        pred_df = pd.DataFrame(
            {
                "Datetime": test_dates,
                "Actual_MW": y_true,
                "LSTM_Pred_MW": y_pred,
                "Error_MW": y_true - y_pred,
                "AbsError_MW": np.abs(y_true - y_pred),
            }
        )
        pred_df.to_csv(output_dir / "lstm" / "lstm_predictions_vs_actual.csv", index=False)

    comparison_lines = [
        "MODEL COMPARISON: RANDOM FOREST vs LSTM",
        "=" * 60,
        "",
        f"{'Metric':<15} {'Random Forest':<20} {'LSTM':<20}",
        "-" * 60,
        f"{'MAE (MW)':<15} {rf.metrics['test_mae']:<20.2f} {lstm_metrics['mae']:<20.2f}",
        f"{'RMSE (MW)':<15} {rf.metrics['test_rmse']:<20.2f} {lstm_metrics['rmse']:<20.2f}",
        f"{'R2 Score':<15} {rf.metrics['test_r2']:<20.4f} {lstm_metrics['r2']:<20.4f}",
        "",
        f"Date split used: {use_date_split}",
        f"LSTM device: {device}",
    ]
    (output_dir / "model_comparison.txt").write_text("\n".join(comparison_lines), encoding="utf-8")

    return {
        "metrics": lstm_metrics,
        "y_true": y_true,
        "y_pred": y_pred,
    }


def launch_dashboard(project_root: Path, port: int) -> None:
    app_path = project_root / "app.py"
    if not app_path.exists():
        log("Dashboard file app.py not found; skipping launch.")
        return
    cmd = [sys.executable, "-m", "streamlit", "run", str(app_path), "--server.port", str(port)]
    subprocess.Popen(cmd, cwd=str(project_root), shell=False)
    log(f"Dashboard launched on port {port}.")


def run_pipeline(args: argparse.Namespace) -> None:
    project_root = Path(__file__).resolve().parent
    output_dir = (project_root / args.output_dir).resolve()
    ensure_directories(output_dir)

    log("=" * 80)
    log("ENERGY CONSUMPTION PIPELINE")
    log("=" * 80)

    dataset_path = find_dataset(project_root, args.dataset, args.data_dir, args.archive_dir)
    log(f"Dataset selected: {dataset_path}")

    df, datetime_col, energy_col = load_dataset(dataset_path)
    log(f"Inferred columns -> datetime: '{datetime_col}', energy: '{energy_col}'")

    df = add_features(df)
    df.to_csv(output_dir / "processed_data.csv", index=False)

    log("Training Random Forest...")
    rf = train_random_forest(df, output_dir)
    log(
        "RF metrics -> "
        f"MAE {rf.metrics['test_mae']:.2f}, RMSE {rf.metrics['test_rmse']:.2f}, R2 {rf.metrics['test_r2']:.4f}"
    )

    log("Running anomaly detection...")
    anomaly = detect_anomalies(
        df=df,
        y_pred_all=rf.y_pred_all,
        residual_quantile=args.residual_quantile,
        cost_per_mwh=args.cost_per_mwh,
    )
    anomaly.anomaly_df.to_csv(output_dir / "anomalies_detected.csv", index=False)
    log(
        "Anomalies -> "
        f"{anomaly.summary['anomalies_detected']:,} ({anomaly.summary['anomaly_pct']:.2f}%), "
        f"annual savings ${anomaly.summary['annual_savings']:,.2f}"
    )

    write_recommendations(output_dir, anomaly.recommendations)
    write_metrics_file(output_dir, dataset_path.name, rf, anomaly)
    save_prediction_scenarios(output_dir, rf.model)

    plot_df = anomaly.full_df.copy()
    create_plots(plot_df, rf, output_dir)
    log("Saved all core and advanced plots.")

    lstm_result = train_lstm(plot_df, rf, output_dir, args)
    if lstm_result is not None:
        r2_value = lstm_result["metrics"]["r2"]
        log(f"LSTM trained successfully (R2={r2_value:.4f}).")
    else:
        log("LSTM not trained in this run.")

    log("=" * 80)
    log("PIPELINE COMPLETE")
    log("=" * 80)
    log(f"Model file: {output_dir / 'model.pkl'}")
    log(f"Metrics file: {output_dir / 'metrics.txt'}")
    log(f"Anomaly file: {output_dir / 'anomalies_detected.csv'}")
    log(f"Plots dir: {output_dir / 'plots'}")
    log(f"Scenarios file: {output_dir / 'prediction_scenarios.csv'}")

    if not args.no_dashboard:
        launch_dashboard(project_root, args.dashboard_port)


if __name__ == "__main__":
    try:
        run_pipeline(parse_args())
    except Exception as exc:
        log(f"ERROR: {exc}")
        sys.exit(1)
