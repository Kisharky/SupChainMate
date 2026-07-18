"""
config.py
Central configuration for SupChainMate: paths, environment lookup,
model identifiers, tunable thresholds, and logging.

Modules import from here instead of hard-coding constants or re-implementing
.env parsing. Values that users tune live here; pure domain constants
(e.g. Z-score tables, emission factors) stay with their domain modules.
"""

from __future__ import annotations

import logging
import os
from typing import Optional

# ── Paths ──────────────────────────────────────────────────────────────────────

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
DB_PATH = os.path.join(DATA_DIR, "supchainmate.db")

DEMO_ORDERS = os.path.join(DATA_DIR, "olist_orders.csv")
DEMO_DELIVERY = os.path.join(DATA_DIR, "olist_orders_dataset.csv")
DEMO_CUSTOMERS = os.path.join(DATA_DIR, "olist_customers_dataset.csv")
DEMO_GEO = os.path.join(DATA_DIR, "olist_geolocation_dataset.csv")

# ── Environment / secrets ──────────────────────────────────────────────────────

_ENV_FILES = [os.path.join(BASE_DIR, ".env"), ".env", "logistics-ai-dashboard/.env"]


def get_env(name: str) -> Optional[str]:
    """Environment variable, falling back to a .env file entry."""
    val = os.environ.get(name)
    if val:
        return val
    for path in _ENV_FILES:
        try:
            if os.path.exists(path):
                with open(path) as f:
                    for line in f:
                        line = line.strip()
                        if line.startswith(name + "="):
                            return line.split("=", 1)[1].strip()
        except OSError:
            continue
    return None


# ── Model identifiers & API endpoints ─────────────────────────────────────────

GROQ_MODEL = "llama-3.3-70b-versatile"
GROQ_MODEL_FAST = "llama-3.1-8b-instant"
NVIDIA_CHAT_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
NVIDIA_CUOPT_URL = "https://integrate.api.nvidia.com/v1/nvidia/cuopt"
NVIDIA_DEEPSEEK_MODEL = "deepseek-ai/deepseek-v4-pro"
SHOPIFY_API_VERSION = "2024-01"
HTTP_TIMEOUT = 30

# ── Tunable thresholds ─────────────────────────────────────────────────────────

# Control tower: an open shipment is AT RISK in the top decile of ML delay
# probability, with a floor so a uniformly low-risk fleet flags nothing.
AT_RISK_PERCENTILE = 90
AT_RISK_FLOOR_PCT = 15.0

# Invoice scanner: tolerated deviation between invoice total and recorded cost
RATE_TOLERANCE = 0.10

# Forecasting: backtest window for the model tournament
HOLDOUT_DAYS = 28

# SKU intelligence: engine cap (top-N SKUs by volume)
MAX_SKUS = 200

# ── Logging ────────────────────────────────────────────────────────────────────

_LOG_FORMAT = "%(asctime)s %(levelname)s [%(name)s] %(message)s"
_configured = False


def get_logger(name: str) -> logging.Logger:
    """Project logger; configures the root handler once."""
    global _configured
    if not _configured:
        level = (get_env("SUPCHAINMATE_LOG_LEVEL") or "INFO").upper()
        logging.basicConfig(level=getattr(logging, level, logging.INFO),
                            format=_LOG_FORMAT)
        _configured = True
    return logging.getLogger(name)
