# SupChainMate API — FastAPI over the existing engines.
FROM python:3.11-slim

WORKDIR /app

# Build tools for any packages without a manylinux wheel (kept minimal).
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install dependencies first for layer caching.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App source (includes demo data + the api/, modules/, ai/, planner/, brain/ packages).
COPY . .

ENV PYTHONUNBUFFERED=1 \
    AUTH_ENABLED=true \
    DATABASE_URL=sqlite:///data/supchainmate_app.db

EXPOSE 8000

# Production server. Binds to $PORT when the platform sets one (Render, etc.),
# else 8000 for local/docker-compose. Override DATABASE_URL / JWT_SECRET /
# FRONTEND_ORIGIN via the environment.
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
