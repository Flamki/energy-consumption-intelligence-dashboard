# Deployment Runbook

## A) GitHub Push

From project root:

```bash
git init
git add .
git commit -m "Initial production-ready energy dashboard"
gh repo create <repo-name> --public --source . --push
```

If repo already exists:

```bash
git remote add origin https://github.com/<user>/<repo>.git
git branch -M main
git push -u origin main
```

## B) Streamlit Community Cloud

1. Open `https://share.streamlit.io/`.
2. Sign in with GitHub.
3. Click **New app**.
4. Select:
   - Repository: your pushed repo
   - Branch: `main`
   - Main file path: `app.py`
5. Click **Deploy**.

## C) Post-Deploy Check

Validate:
- Dashboard loads without missing-file errors.
- Accuracy cards show:
  - RF R2
  - LSTM R2
- All tabs render charts and tables.

## D) Optional GPU Re-Training (local)

```bash
pip install --upgrade --force-reinstall torch --index-url https://download.pytorch.org/whl/cu124
python main.py --dataset PJME_hourly.csv --no-dashboard
```
