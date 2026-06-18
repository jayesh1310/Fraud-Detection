@echo off
title Fraud Detection — Jupyter Lab
echo =====================================================
echo   Financial Fraud Detection — Jupyter Launcher
echo =====================================================
echo.

:: Add Python to PATH
set PATH=%PATH%;%LOCALAPPDATA%\Programs\Python\Python312;%LOCALAPPDATA%\Programs\Python\Python312\Scripts

:: Set confirmed JAVA_HOME for PySpark
set JAVA_HOME=C:\Program Files\Microsoft\jdk-21.0.11.10-hotspot
set PATH=%PATH%;%JAVA_HOME%\bin

:: Suppress noisy Spark/Hadoop warnings (optional)
set PYSPARK_PYTHON=python
set HADOOP_HOME=%~dp0
set SPARK_LOCAL_IP=127.0.0.1

echo Java  : %JAVA_HOME%
echo Python: Loaded from PATH
echo Project folder: %~dp0
echo.
echo Starting Jupyter Lab — your browser will open automatically.
echo Press Ctrl+C to stop the server when done.
echo.

:: Launch Jupyter Lab in the project folder
cd /d "%~dp0"
python -m jupyter lab

pause
