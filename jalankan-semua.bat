@echo off
REM ============================================================
REM  RAG Hukum Indonesia - LAUNCHER SATU KLIK
REM  Menyalakan: Docker/Milvus -> Backend :8000 -> Frontend :3000
REM  Cara pakai: klik 2x file ini. JANGAN tutup 2 jendela hitam yg terbuka.
REM ============================================================
setlocal
cd /d "%~dp0"
title RAG Hukum Indonesia - Launcher

echo.
echo  ==================================================
echo   RAG Hukum Indonesia - Menyalakan semua server
echo  ==================================================
echo.

REM ---------- [0] Cek python venv ----------
if not exist ".venv\Scripts\python.exe" (
  echo  [FATAL] .venv tidak ketemu di folder ini.
  echo          Pastikan file ini ada di root project -sejajar folder pipeline-.
  pause
  exit /b 1
)

REM ---------- [1] Docker Desktop ----------
echo  [1/4] Cek Docker Desktop ...
docker info >nul 2>&1
if errorlevel 1 (
  echo.
  echo  [WARNING] Docker Desktop MATI / belum Engine running.
  echo            Akibat: metode index HNSW/IVF/FLAT tidak bisa dipakai.
  echo            Yang tetap jalan: exact-numpy + BM25 + hybrid-exact.
  echo.
  echo            Untuk menyalakan Milvus: buka aplikasi Docker Desktop,
  echo            tunggu "Engine running", lalu tutup-tidak perlu, jalankan
  echo            lagi file ini ATAU manual:  cd C:\milvus  ^&  docker compose up -d
  echo.
  set /p LANJUT="  Lanjut TANPA Milvus? [Y/N]: "
  if /i not "%LANJUT%"=="Y" (
    echo  Dibatalkan. Nyalakan Docker Desktop dulu lalu ulangi.
    echo.
    echo  Tekan tombol apa saja untuk menutup ...
    pause >nul
    exit /b 1
  )
  echo  Lanjut tanpa Milvus ...
  goto SKIP_MILVUS
)

echo  Docker hidup. Menyalakan Milvus ...
cd /d C:\milvus
docker compose up -d
if errorlevel 1 (
  echo  [WARNING] "docker compose up -d" gagal. Lanjut tanpa Milvus.
  cd /d "%~dp0"
  goto SKIP_MILVUS
)
echo  Menunggu Milvus siap -cek port 19530 tiap 5 detik, maks 90 detik- ...
set COBA=0
:TUNGGU_MILVUS
powershell -noprofile -command "if ((Test-NetConnection 127.0.0.1 -Port 19530 -WarningAction SilentlyContinue).TcpTestSucceeded) { exit 0 } else { exit 1 }" >nul 2>&1
if not errorlevel 1 goto MILVUS_OK
set /a COBA+=1
if %COBA% GEQ 18 goto MILVUS_LAMA
timeout /t 5 /nobreak >nul
goto TUNGGU_MILVUS
:MILVUS_LAMA
echo  [WARNING] Port 19530 belum terbuka setelah 90 detik.
echo            Start pertama memang bisa lama - biarkan jalan, lanjut dulu;
echo            kalau nanti ada "Connection refused", tunggu lalu ulangi script ini.
goto MILVUS_LANJUT
:MILVUS_OK
echo  Milvus terjangkau di port 19530.
:MILVUS_LANJUT
docker ps --format "  {{.Names}} - {{.Status}}" | findstr /i "milvus etcd minio"
cd /d "%~dp0"

:SKIP_MILVUS
echo.

REM ---------- [2] Dependensi python (sekali saja, dilewati bila sudah ada) ----------
echo  [2/4] Cek dependensi backend ...
".venv\Scripts\python.exe" -c "import fastapi, uvicorn" >nul 2>&1
if errorlevel 1 (
  echo  fastapi/uvicorn belum ada - install requirements, agak lama, sekali saja ...
  ".venv\Scripts\python.exe" -m pip install -r requirements.txt
  if errorlevel 1 (
    echo  [FATAL] pip install gagal. Cek koneksi internet lalu ulangi.
    pause
    exit /b 1
  )
) else (
  echo  Dependensi OK.
)
echo.

REM ---------- [3] Backend :8000 ----------
echo  [3/4] Backend http://127.0.0.1:8000 ...
netstat -ano | findstr ":8000" | findstr "LISTENING" >nul
if not errorlevel 1 (
  echo  [WARNING] Port 8000 sudah dipakai -backend lama masih jalan-. Lewati.
  echo            Kalau itu backend versi lama: bunuh dulu
  echo            -netstat -ano ^| findstr ":8000" -- taskkill /PID ... /F-, lalu ulangi.
) else (
  start "RAG Backend :8000" /d "%~dp0webapp\backend" cmd /k ..\..\\.venv\Scripts\python.exe -m uvicorn app:app --port 8000
  echo  Backend dinyalakan di jendela baru. Tunggu tulisan "Application startup complete".
  timeout /t 8 /nobreak >nul
)
echo.

REM ---------- [4] Frontend :3000 ----------
echo  [4/4] Frontend http://localhost:3000 ...
where node >nul 2>&1
if errorlevel 1 (
  echo  [FATAL] node/npm tidak ketemu. Install Node.js LTS dulu dari nodejs.org, lalu ulangi.
  pause
  exit /b 1
)
if not exist "webapp\frontend\node_modules" (
  echo  node_modules belum ada - npm install sekali saja, agak lama ...
  cd /d "%~dp0webapp\frontend"
  call npm install
  if errorlevel 1 (
    echo  [FATAL] npm install gagal. Cek koneksi internet lalu ulangi.
    pause
    exit /b 1
  )
  cd /d "%~dp0"
)
netstat -ano | findstr ":3000" | findstr "LISTENING" >nul
if not errorlevel 1 (
  echo  [WARNING] Port 3000 sudah dipakai -frontend lama masih jalan-. Lewati.
) else (
  start "RAG Frontend :3000" /d "%~dp0webapp\frontend" cmd /k npm run dev
  echo  Frontend dinyalakan di jendela baru. Tunggu tulisan "Ready".
)
echo.
echo  ==================================================
echo   SELESAI. Jangan tutup 2 jendela hitam itu selama memakai.
echo   Buka : http://localhost:3000
echo   Cek backend: http://127.0.0.1:8000/api/health  -harus ok:true-
echo  ==================================================
echo.
echo  Tekan tombol apa saja untuk menutup jendela ini ...
pause >nul
