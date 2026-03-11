# Energy Consumption Intelligence Dashboard

End-to-end energy analytics project with:
- Random Forest demand forecasting
- LSTM forecasting (GPU-ready with CUDA)
- Hybrid anomaly detection (z-score + residual threshold)
- Cost/savings estimation
- Presentation-grade Streamlit dashboard

## Latest Results (PJME dataset)

From the most recent full run (`March 11, 2026`):
- Random Forest test score: `R2 = 0.6543` (~65.43%)
- LSTM test score: `R2 = 0.9849` (~98.49%)
- Total anomalies detected: `11,040` (`7.59%`)
- Estimated annual savings potential: `$709,245,341.30`

## Dashboard

Main app:
- `app.py`

Live local URL:
- `http://127.0.0.1:8501`

The dashboard includes:
- RF vs LSTM accuracy cards
- Executive KPI view
- anomaly intelligence center
- recommendations panel
- advanced visual gallery

## Project Structure

```text
energy consumption project/
  app.py
  main.py
  predict.py
  requirements.txt
  requirements-train.txt
  outputs/
    metrics.txt
    processed_data.csv
    anomalies_detected.csv
    recommendations.txt
    plots/
    lstm/
```

## Quick Start

### 1) Create environment

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

### 2) Run dashboard

```bash
streamlit run app.py --server.port 8501
```

## Train / Rebuild Outputs

```bash
pip install -r requirements-train.txt
python main.py --dataset PJME_hourly.csv --no-dashboard
```

Useful flags:
- `--skip-lstm` to skip deep learning
- `--epochs 12` to train longer
- `--sample-size 50000` to control LSTM train window

## GPU Setup (Windows, NVIDIA)

If you want CUDA LSTM training:

```bash
pip install --upgrade --force-reinstall torch --index-url https://download.pytorch.org/whl/cu124
```

Quick check:

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No CUDA')"
```

## Streamlit Community Cloud Deployment

1. Push this repo to GitHub.
2. Open `https://share.streamlit.io/` (or `https://streamlit.io/cloud`).
3. Click **New app**.
4. Select repository + branch.
5. Set **Main file path** to `app.py`.
6. Deploy.

Because this project already includes generated outputs in `outputs/`, the app can render immediately after deploy.

## Prediction Script

```bash
python predict.py
```

This uses `outputs/model.pkl` and prints scenario-based RF predictions.

## Notes

- `main.py` is productionized for Windows-safe logging and robust dataset discovery.
- `app.py` has contrast fixes for sidebar/content readability and explicit model accuracy display.
- If outputs are missing, run:

```bash
python main.py --dataset PJME_hourly.csv --no-dashboard
```
