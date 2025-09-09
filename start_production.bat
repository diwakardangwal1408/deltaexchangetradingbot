@echo off
REM BTC Options Trader - Production Server Startup
REM Uses Waitress WSGI server for better performance

echo Starting BTC Options Trader in Production Mode...
echo.

REM Set environment variables for production
set HOST=0.0.0.0
set PORT=5000

REM Start the production server
python run_production.py

pause