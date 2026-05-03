#!/bin/bash
# SupChainMate — Quick Start
set -e

echo "Installing dependencies..."
pip install -r requirements.txt

echo "Launching SupChainMate..."
cd logistics-ai-dashboard
streamlit run app.py
