@echo off
title Supplier Search Agent

echo Checking Docker...

docker info >nul 2>&1
if errorlevel 1 (
    echo Docker not running. Starting Docker Desktop...
    start "" "C:\Program Files\Docker\Docker\Docker Desktop.exe"
    
    echo Waiting for Docker...
    :wait_docker
    docker info >nul 2>&1
    if errorlevel 1 (
        timeout /t 2 >nul
        goto wait_docker
    )
)

echo Docker is ready.

echo Starting Docker Compose (if needed)
docker compose up -d >nul 2>&1

echo Checking Ollama...

curl http://localhost:11434 >nul 2>&1
if errorlevel 1 (
    echo Ollama not running. Starting Ollama app...
    
    start "" "%APPDATA%\Microsoft\Windows\Start Menu\Programs\Ollama.lnk"
    
    timeout /t 5 >nul
) else (
    echo Ollama already running.
)

echo Choose mode:
echo 1 - python main.py
echo 2 - langgraph dev
echo.

set /p choice=Select (1/2): 

if "%choice%"=="1" (
    python main.py
)

if "%choice%"=="2" (
    langgraph dev
)

echo.
pause