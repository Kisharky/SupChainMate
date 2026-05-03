# SupChainMate — Quick Start (Windows)
Write-Host "Installing dependencies..." -ForegroundColor Cyan
pip install -r logistics-ai-dashboard\requirements.txt

Write-Host "Launching SupChainMate..." -ForegroundColor Green
py -m streamlit run logistics-ai-dashboard\app.py
