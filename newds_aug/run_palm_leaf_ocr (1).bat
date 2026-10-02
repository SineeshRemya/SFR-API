@echo off
REM ==========================================================
REM Palm Leaf OCR launcher (LeafOCR conda environment)
REM Double-click this file (or a shortcut to it) to start the app.
REM ==========================================================

setlocal enabledelayedexpansion

REM --- Settings you might want to change ---
set ENV_NAME=LeafOCR
set PROJECT_DIR=D:\sfr-system

cd /d "%PROJECT_DIR%"
set PYTHONNOUSERSITE=1

REM Ask conda itself where it's installed, instead of hardcoding a
REM drive/folder. This makes the script work even if Anaconda is on
REM a different drive, or if this .bat is copied to another machine.
for /f "delims=" %%i in ('conda info --base') do set CONDA_ROOT=%%i

set PYEXE=%CONDA_ROOT%\envs\%ENV_NAME%\python.exe

if not exist "%PYEXE%" (
    echo Could not find the "%ENV_NAME%" environment's python.exe at:
    echo   %PYEXE%
    echo Check that the environment name and CONDA_ROOT are correct.
    pause
    exit /b 1
)

echo Starting Palm Leaf OCR using: %PYEXE%
"%PYEXE%" app.py

REM Keep the window open if it crashes, so you can read the error
pause
