@echo off
TITLE Crystal Seed Interface
echo Initializing Crystal Environment...
echo.

:: Check for dependencies
pip install -r seed_ai/requirements.txt >nul 2>&1

:: Run the seed
python seed_ai/standalone_seed.py

:: Keep window open if it crashes or exits
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [ERROR] The seed process terminated unexpectedly.
    pause
) else (
    echo.
    echo [SYSTEM] Session ended.
    pause
)
