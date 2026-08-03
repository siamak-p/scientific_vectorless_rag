@echo off
REM ===========================================================================
REM  Scientific RAG - Windows installer
REM
REM  Double-click this file (or run it from a Command Prompt) to set up
REM  everything the application needs on a Windows machine:
REM    1. Verifies Python 3.10+ is installed.
REM    2. Creates an isolated virtual environment in ".venv".
REM    3. Installs every Python dependency from src\requirements.txt.
REM    4. Creates a ".env" file from the template, if one does not exist yet.
REM    5. Initialises the local SQLite database and application folders.
REM    6. Creates "Run Scientific RAG.bat" and, optionally, a Desktop shortcut.
REM
REM  It does not require Administrator rights and never touches system Python.
REM ===========================================================================

setlocal enabledelayedexpansion
cd /d "%~dp0\.."
set "PROJECT_ROOT=%CD%"

echo(
echo ===============================================
echo   Scientific RAG - Installer
echo ===============================================
echo(

REM --- 1. Locate a usable Python interpreter -------------------------------
set "PYTHON_CMD="
where py >nul 2>nul
if %errorlevel%==0 (
    py -3 -c "import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)" >nul 2>nul
    if !errorlevel!==0 set "PYTHON_CMD=py -3"
)
if not defined PYTHON_CMD (
    where python >nul 2>nul
    if %errorlevel%==0 (
        python -c "import sys; sys.exit(0 if sys.version_info >= (3,10) else 1)" >nul 2>nul
        if !errorlevel!==0 set "PYTHON_CMD=python"
    )
)

if not defined PYTHON_CMD (
    echo [ERROR] Python 3.10 or newer was not found on this machine.
    echo         Install it from https://www.python.org/downloads/ and make sure
    echo         "Add python.exe to PATH" is checked during setup, then run this
    echo         installer again.
    echo(
    pause
    exit /b 1
)

echo [OK] Using interpreter: %PYTHON_CMD%
%PYTHON_CMD% --version

REM --- 2. Create the virtual environment -----------------------------------
if exist "%PROJECT_ROOT%\.venv\Scripts\python.exe" (
    echo [OK] Virtual environment already exists, skipping creation.
) else (
    echo(
    echo Creating virtual environment in .venv ...
    %PYTHON_CMD% -m venv "%PROJECT_ROOT%\.venv"
    if errorlevel 1 (
        echo [ERROR] Failed to create the virtual environment.
        pause
        exit /b 1
    )
)

set "VENV_PY=%PROJECT_ROOT%\.venv\Scripts\python.exe"

REM --- 3. Install dependencies ----------------------------------------------
echo(
echo Upgrading pip ...
"%VENV_PY%" -m pip install --upgrade pip >nul

echo Installing dependencies from src\requirements.txt (this can take a few minutes) ...
"%VENV_PY%" -m pip install -r "%PROJECT_ROOT%\src\requirements.txt"
if errorlevel 1 (
    echo [ERROR] Dependency installation failed. Check your internet connection
    echo         and re-run this installer.
    pause
    exit /b 1
)

REM --- 4. Prepare the .env file ----------------------------------------------
if not exist "%PROJECT_ROOT%\.env" (
    echo(
    echo Creating .env from .env.example ...
    copy /y "%PROJECT_ROOT%\.env.example" "%PROJECT_ROOT%\.env" >nul
) else (
    echo [OK] .env already exists, leaving it untouched.
)

REM --- 5. Initialise storage --------------------------------------------------
echo(
echo Initialising the application database ...
"%VENV_PY%" "%PROJECT_ROOT%\init_database.py"
if errorlevel 1 (
    echo [ERROR] Database initialisation failed.
    pause
    exit /b 1
)

REM --- 6. Create the launcher --------------------------------------------------
set "LAUNCHER=%PROJECT_ROOT%\Run Scientific RAG.bat"
(
    echo @echo off
    echo cd /d "%%~dp0"
    echo ".venv\Scripts\python.exe" app.py
    echo pause
) > "%LAUNCHER%"
echo [OK] Created "%LAUNCHER%"

REM --- 7. Optional Desktop shortcut --------------------------------------------
REM  Skipped when launched from the Inno Setup wizard ("/inno"): the wizard
REM  already manages shortcuts through its own "Create a desktop shortcut"
REM  task, so offering it twice here would be confusing.
if /i "%~1"=="/inno" goto :skip_shortcut

echo(
set /p MAKE_SHORTCUT="Create a Desktop shortcut? [Y/n] "
if /i not "%MAKE_SHORTCUT%"=="n" (
    powershell -NoProfile -Command ^
        "$s = (New-Object -ComObject WScript.Shell).CreateShortcut(\"$env:USERPROFILE\Desktop\Scientific RAG.lnk\"); ^
         $s.TargetPath = '%LAUNCHER%'; ^
         $s.WorkingDirectory = '%PROJECT_ROOT%'; ^
         $s.IconLocation = '%VENV_PY%'; ^
         $s.Save()"
    echo [OK] Desktop shortcut created.
)
:skip_shortcut

echo(
echo ===============================================
echo   Installation complete.
echo   Start the app with "Run Scientific RAG.bat"
echo   or the new Desktop shortcut.
echo(
echo   Open the Settings page in the app, or edit
echo   ".env", to add your LLM/search provider keys.
echo ===============================================
echo(
if /i "%~1"=="/inno" goto :eof
pause
endlocal
