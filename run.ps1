# ──────────────────────────────────────────────────────────────────────
#  run.ps1  —  Start FastAPI + Streamlit (Windows PowerShell)
# ──────────────────────────────────────────────────────────────────────
#  Usage:  .\run.ps1
# ──────────────────────────────────────────────────────────────────────

Write-Host "`n🚀  Launching 360° Surveillance System …`n" -ForegroundColor Cyan

# 1. Start FastAPI in the background
Write-Host "  ➜  Starting FastAPI server on http://localhost:8000" -ForegroundColor Green
$env:PYTHONPATH = (Get-Location).Path
Start-Process -NoNewWindow -FilePath "python" -ArgumentList "-m", "uvicorn", "api.server:app", "--host", "0.0.0.0", "--port", "8000"

Start-Sleep -Seconds 3   # give the backend a moment

# 2. Start Streamlit (foreground so you see its logs)
Write-Host "  ➜  Starting Streamlit dashboard on http://localhost:8501" -ForegroundColor Green
$env:PYTHONPATH = (Get-Location).Path
streamlit run frontend/dashboard.py --server.port 8501
