@echo off
setlocal EnableDelayedExpansion

rem AlgoTrade-X - Start Backend + Frontend on Windows.
rem Usage: double-click this file or run launchers\start.bat.

set "PROJECT_DIR=%~dp0.."
for %%I in ("%PROJECT_DIR%") do set "PROJECT_DIR=%%~fI"
set "VENV_PYTHON=%PROJECT_DIR%\venv\Scripts\python.exe"
set "VENV_UVICORN=%PROJECT_DIR%\venv\Scripts\uvicorn.exe"
set "VENV_STREAMLIT=%PROJECT_DIR%\venv\Scripts\streamlit.exe"
set "API_PORT=8000"
set "DASH_PORT=8501"
set "HEALTH_URL=http://localhost:%API_PORT%/health"
set "MAX_WAIT=120"
set "POLL_INTERVAL=2"

if not exist "%VENV_PYTHON%" (
    echo [ERROR] Project venv not found at "%VENV_PYTHON%".
    echo [INFO]  Create it with: python -m venv venv
    pause
    exit /b 1
)

for /f "tokens=*" %%v in ('"%VENV_PYTHON%" --version 2^>^&1') do echo [INFO] Using %%v from project venv

"%VENV_PYTHON%" -c "import fastapi, uvicorn, streamlit" >nul 2>&1
if errorlevel 1 (
    echo [INFO] Installing required packages into project venv...
    "%VENV_PYTHON%" -m pip install -r "%PROJECT_DIR%\requirements-dev.txt"
    if errorlevel 1 (
        echo [ERROR] Dependency installation failed.
        pause
        exit /b 1
    )
)

echo.
echo [1/3] Starting API backend on port %API_PORT%...
if exist "%VENV_UVICORN%" (
    start "AlgoTrade-X API" cmd /k "cd /d "%PROJECT_DIR%" && "%VENV_UVICORN%" src.api.main:app --host 0.0.0.0 --port %API_PORT%"
) else (
    start "AlgoTrade-X API" cmd /k "cd /d "%PROJECT_DIR%" && "%VENV_PYTHON%" -m uvicorn src.api.main:app --host 0.0.0.0 --port %API_PORT%"
)

echo [2/3] Waiting for API to be ready...
set /a ELAPSED=0

:WAIT_LOOP
    timeout /t %POLL_INTERVAL% /nobreak >nul
    set /a ELAPSED+=POLL_INTERVAL

    curl -s -o nul -w "%%{http_code}" %HEALTH_URL% 2>nul | findstr /x "200" >nul
    if not errorlevel 1 goto :API_UP

    "%VENV_PYTHON%" -c "import urllib.request; urllib.request.urlopen('%HEALTH_URL%', timeout=2)" >nul 2>&1
    if not errorlevel 1 goto :API_UP

    if !ELAPSED! geq %MAX_WAIT% (
        echo [WARN] API did not respond after %MAX_WAIT%s; starting dashboard anyway.
        goto :START_DASH
    )

    echo        ... waiting (!ELAPSED!s / %MAX_WAIT%s)
    goto :WAIT_LOOP

:API_UP
echo [OK]  API is up and healthy after %ELAPSED%s.

:START_DASH
echo [3/3] Starting Streamlit dashboard on port %DASH_PORT%...
if exist "%VENV_STREAMLIT%" (
    start "AlgoTrade-X Dashboard" cmd /k "cd /d "%PROJECT_DIR%" && "%VENV_STREAMLIT%" run src/dashboard/app.py --server.port %DASH_PORT% --server.address localhost --server.headless false"
) else (
    start "AlgoTrade-X Dashboard" cmd /k "cd /d "%PROJECT_DIR%" && "%VENV_PYTHON%" -m streamlit run src/dashboard/app.py --server.port %DASH_PORT% --server.address localhost --server.headless false"
)

timeout /t 4 /nobreak >nul

set "FIREFOX="
for %%F in (
    "%ProgramFiles%\Mozilla Firefox\firefox.exe"
    "%ProgramFiles(x86)%\Mozilla Firefox\firefox.exe"
    "%LocalAppData%\Mozilla Firefox\firefox.exe"
) do (
    if exist %%F set "FIREFOX=%%~fF"
)

if not "%FIREFOX%"=="" (
    echo [INFO] Opening dashboard in Firefox...
    start "" "%FIREFOX%" "http://localhost:%DASH_PORT%"
) else (
    echo [INFO] Firefox not found; opening in default browser...
    start "" "http://localhost:%DASH_PORT%"
)

echo.
echo AlgoTrade-X is running
echo API:        http://localhost:%API_PORT%
echo Dashboard:  http://localhost:%DASH_PORT%
echo API docs:   http://localhost:%API_PORT%/docs
echo.
echo Close either terminal window to stop.
echo.

endlocal
