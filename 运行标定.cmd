@echo off
setlocal
set "PYTHONPATH=%~dp0src"
py -m probe_calibration.cli %*
if errorlevel 1 pause
