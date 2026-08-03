@echo off
REM ===========================================================================
REM  Scientific RAG - Windows uninstaller
REM
REM  Removes the virtual environment and generated launcher/shortcut from this
REM  copy of the project. It does NOT delete your source files, and it does
REM  NOT delete your data (chats, settings, cached papers) which lives in
REM  %LOCALAPPDATA%\ScientificRAG - remove that folder yourself if you also
REM  want to erase all application data.
REM ===========================================================================

setlocal
cd /d "%~dp0\.."
set "PROJECT_ROOT=%CD%"

echo(
echo ===============================================
echo   Scientific RAG - Uninstaller
echo ===============================================
echo(
echo This will remove:
echo   - "%PROJECT_ROOT%\.venv"
echo   - "%PROJECT_ROOT%\Run Scientific RAG.bat"
echo   - The Desktop shortcut "Scientific RAG.lnk" (if present)
echo(
echo Your application data in %%LOCALAPPDATA%%\ScientificRAG (chats, settings,
echo cached papers, logs) is left untouched.
echo(
set /p CONFIRM="Continue? [y/N] "
if /i not "%CONFIRM%"=="y" (
    echo Aborted.
    exit /b 0
)

if exist "%PROJECT_ROOT%\.venv" (
    echo Removing virtual environment ...
    rmdir /s /q "%PROJECT_ROOT%\.venv"
)

if exist "%PROJECT_ROOT%\Run Scientific RAG.bat" (
    del /q "%PROJECT_ROOT%\Run Scientific RAG.bat"
)

if exist "%USERPROFILE%\Desktop\Scientific RAG.lnk" (
    del /q "%USERPROFILE%\Desktop\Scientific RAG.lnk"
)

echo(
echo Done. The source folder and your application data were kept.
echo To also erase application data, delete:
echo   %%LOCALAPPDATA%%\ScientificRAG
echo(
pause
endlocal
