@echo off
REM ============================================================
REM  Indonesian Law RAG - ONE-CLICK LAUNCHER
REM  Starts: Docker/Milvus -> Backend :8000 -> Frontend :3000
REM  Usage: double-click this file. DON'T close the 2 black windows.
REM ============================================================
setlocal
cd /d "%~dp0"
title Indonesian Law RAG - Launcher

echo.
echo  ==================================================
echo   Indonesian Law RAG - Starting all servers
echo  ==================================================
echo.

REM ---------- [0] Check python venv ----------
if not exist ".venv\Scripts\python.exe" (
  echo  [FATAL] .venv not found in this folder.
  echo          This file must sit at the project root - next to pipeline/-.
  pause
  exit /b 1
)

REM ---------- [1] Docker Desktop ----------
echo  [1/4] Checking Docker Desktop ...
docker info >nul 2>&1
if errorlevel 1 (
  echo.
  echo  [WARNING] Docker Desktop is OFF / Engine not running.
  echo            Consequence: HNSW/IVF/FLAT index methods unavailable.
  echo            Still works: exact-numpy + BM25 + hybrid-exact.
  echo.
  echo            To enable Milvus: open Docker Desktop,
  echo            wait for "Engine running", then re-run
  echo            this file OR manually:  cd C:\milvus  ^&  docker compose up -d
  echo.
  set /p CONTINUE="  Continue WITHOUT Milvus? [Y/N]: "
  if /i not "%CONTINUE%"=="Y" (
    echo  Cancelled. Start Docker Desktop first, then retry.
    echo.
    echo  Press any key to close ...
    pause >nul
    exit /b 1
  )
  echo  Continuing without Milvus ...
  goto SKIP_MILVUS
)

echo  Docker is up. Starting Milvus ...
cd /d C:\milvus
docker compose up -d
if errorlevel 1 (
  echo  [WARNING] "docker compose up -d" failed. Continuing without Milvus.
  cd /d "%~dp0"
  goto SKIP_MILVUS
)
echo  Waiting for Milvus - probing port 19530 every 5s, max 90s- ...
set TRY=0
:WAIT_MILVUS
powershell -noprofile -command "if ((Test-NetConnection 127.0.0.1 -Port 19530 -WarningAction SilentlyContinue).TcpTestSucceeded) { exit 0 } else { exit 1 }" >nul 2>&1
if not errorlevel 1 goto MILVUS_OK
set /a TRY+=1
if %TRY% GEQ 18 goto MILVUS_SLOW
timeout /t 5 /nobreak >nul
goto WAIT_MILVUS
:MILVUS_SLOW
echo  [WARNING] Port 19530 not open after 90s.
echo            First start can be slow - leave it running and continue;
echo            on "Connection refused", wait then re-run this script.
goto MILVUS_CONT
:MILVUS_OK
echo  Milvus reachable on port 19530.
:MILVUS_CONT
docker ps --format "  {{.Names}} - {{.Status}}" | findstr /i "milvus etcd minio"
cd /d "%~dp0"

:SKIP_MILVUS
echo.

REM ---------- [2] Python deps (once, skipped when present) ----------
echo  [2/4] Checking backend deps ...
".venv\Scripts\python.exe" -c "import fastapi, uvicorn" >nul 2>&1
if errorlevel 1 (
  echo  fastapi/uvicorn missing - installing requirements, one-time, slow ...
  ".venv\Scripts\python.exe" -m pip install -r requirements.txt
  if errorlevel 1 (
    echo  [FATAL] pip install failed. Check internet, then retry.
    pause
    exit /b 1
  )
) else (
  echo  Dependencies OK.
)
echo.

REM ---------- [3] Backend :8000 ----------
echo  [3/4] Backend http://127.0.0.1:8000 ...
netstat -ano | findstr ":8000" | findstr "LISTENING" >nul
if not errorlevel 1 (
  echo  [WARNING] Port 8000 in use - old backend still running-. Skipping.
  echo            For a stale backend: kill it first
  echo            -netstat -ano ^| findstr ":8000" -- taskkill /PID ... /F-, then retry.
) else (
  start "RAG Backend :8000" /d "%~dp0webapp\backend" cmd /k ..\..\\.venv\Scripts\python.exe -m uvicorn app:app --port 8000
  echo  Backend started in a new window. Wait for "Application startup complete".
  timeout /t 8 /nobreak >nul
)
echo.

REM ---------- [4] Frontend :3000 ----------
echo  [4/4] Frontend http://localhost:3000 ...
where node >nul 2>&1
if errorlevel 1 (
  echo  [FATAL] node/npm not found. Install Node.js LTS from nodejs.org, then retry.
  pause
  exit /b 1
)
if not exist "webapp\frontend\node_modules" (
  echo  node_modules missing - npm install, one-time, slow ...
  cd /d "%~dp0webapp\frontend"
  call npm install
  if errorlevel 1 (
    echo  [FATAL] npm install failed. Check internet, then retry.
    pause
    exit /b 1
  )
  cd /d "%~dp0"
)
netstat -ano | findstr ":3000" | findstr "LISTENING" >nul
if not errorlevel 1 (
  echo  [WARNING] Port 3000 in use - old frontend still running-. Skipping.
) else (
  start "RAG Frontend :3000" /d "%~dp0webapp\frontend" cmd /k npm run dev
  echo  Frontend started in a new window. Wait for "Ready".
)
echo.
echo  ==================================================
echo   DONE. Keep those 2 black windows open while using.
echo   Open: http://localhost:3000
echo   Backend check: http://127.0.0.1:8000/api/health  -must be ok:true-
echo  ==================================================
echo.
echo  Press any key to close this window ...
pause >nul
