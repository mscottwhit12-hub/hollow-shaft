@echo off
cd /d "%~dp0"
py shaft_optimizer.py
if errorlevel 1 pause
