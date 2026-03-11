from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd


def main() -> None:
    base_dir = Path(__file__).resolve().parent
    model_path = base_dir / "outputs" / "model.pkl"
    if not model_path.exists():
        raise FileNotFoundError("Model not found. Run `python main.py --no-dashboard` first.")

    model = joblib.load(model_path)
    feature_names = list(getattr(model, "feature_names_in_", ["hour", "dayofweek", "month", "is_peak_hour", "is_weekend"]))

    scenarios = [
        {"name": "Weekday Morning (09:00)", "hour": 9, "dayofweek": 0, "month": 7, "is_peak_hour": 0, "is_weekend": 0},
        {"name": "Weekday Afternoon (14:00)", "hour": 14, "dayofweek": 2, "month": 7, "is_peak_hour": 0, "is_weekend": 0},
        {"name": "Evening Peak (18:00)", "hour": 18, "dayofweek": 3, "month": 7, "is_peak_hour": 1, "is_weekend": 0},
        {"name": "Night Off-Peak (22:00)", "hour": 22, "dayofweek": 4, "month": 7, "is_peak_hour": 0, "is_weekend": 0},
        {"name": "Weekend Afternoon (15:00)", "hour": 15, "dayofweek": 5, "month": 7, "is_peak_hour": 0, "is_weekend": 1},
    ]

    print("=" * 64)
    print("ENERGY CONSUMPTION PREDICTION RESULTS")
    print("=" * 64)

    predictions = []
    for scenario in scenarios:
        row = {feature: scenario.get(feature, 0) for feature in feature_names}
        pred = float(model.predict(pd.DataFrame([row], columns=feature_names))[0])
        predictions.append(pred)
        print(f"{scenario['name']:<30} -> {pred:,.2f} MW")

    print("\nSummary:")
    print(f"Average predicted consumption: {sum(predictions) / len(predictions):,.2f} MW")
    best_idx = max(range(len(predictions)), key=lambda idx: predictions[idx])
    print(f"Highest demand scenario: {scenarios[best_idx]['name']} ({predictions[best_idx]:,.2f} MW)")
    print(f"Lowest demand: {min(predictions):,.2f} MW")


if __name__ == "__main__":
    main()
