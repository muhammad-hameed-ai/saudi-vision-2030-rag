@echo off
echo =======================================================
echo   Saudi Vision 2030 - RAG Pipeline Bootstrapper
echo =======================================================
echo.

echo [1/4] Starting Qdrant Docker container...
docker run -d --name vision2030-qdrant-local -p 6333:6333 -p 6334:6334 -v "%cd%\qdrant_storage:/qdrant/storage" qdrant/qdrant 2>nul
if %errorlevel% neq 0 (
    echo        Container exists. Restarting it...
    docker start vision2030-qdrant-local
)

echo.
echo [2/4] Waiting for Qdrant engine to become healthy...
:WAIT_LOOP
powershell -Command "try { $r = Invoke-WebRequest -Uri 'http://localhost:6333/healthz' -UseBasicParsing -TimeoutSec 2; if ($r.StatusCode -eq 200) { exit 0 } else { exit 1 } } catch { exit 1 }"
if %errorlevel% neq 0 (
    echo        Qdrant not ready yet... retrying in 2s
    timeout /t 2 /nobreak >nul
    goto WAIT_LOOP
)
echo        Qdrant is ONLINE and accepting connections.

echo.
echo [3/4] Launching FastAPI backend in a new terminal...
start "Vision2030-API" cmd /k "call .\venv\Scripts\activate.bat && uvicorn src.api:app --host 127.0.0.1 --port 8000 --reload"

echo.
echo [4/4] Opening dashboard in default browser...
timeout /t 3 /nobreak >nul
start http://127.0.0.1:8000

echo.
echo =======================================================
echo   All systems operational. You may close this window.
echo =======================================================
