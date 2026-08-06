# QuantApp backend container.
# Serves the API; the weekly job runs as a separate cron service from the
# same image (see render.yaml).
FROM python:3.12-slim

WORKDIR /app

# libgomp1 is required by LightGBM at import time.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir lightgbm xgboost scikit-learn scipy yfinance

# Backend code + prebuilt snapshots (backtest.json / features.json are static
# between quarterly retrains; rankings.json is refreshed by the cron job).
COPY . .

# The engine subset the weekly job needs: config, feature code, saved model.
# Copied into the image at build time from engine/ (see deploy notes in README).
ENV ENGINE_DIR=/app/engine

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
