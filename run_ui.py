#!/usr/bin/env python3
"""
Launch script for BTC Options Trader Web UI
Provides a simple way to start the Flask application with proper configuration
"""

import os
import sys
import webbrowser
import time
import threading
from app import app

def check_dependencies():
    """Check if all required dependencies are installed"""
    required_modules = [
        'flask', 'pandas', 'numpy', 'yfinance', 'requests', 'waitress'
    ]
    
    missing_modules = []
    
    for module in required_modules:
        try:
            __import__(module)
        except ImportError:
            missing_modules.append(module)
    
    if missing_modules:
        print("ERROR: Missing required dependencies:")
        for module in missing_modules:
            print(f"   - {module}")
        print("\nInstall missing dependencies with:")
        print("   pip install -r requirements_flask.txt")
        return False
    
    return True

def check_config():
    """Check if configuration file exists"""
    if not os.path.exists('application.config'):
        print("WARNING: Configuration file not found!")
        print("   The application will create a default application.config file")
        print("   Configure your settings via the web interface")
        return True  # Allow startup, config will be created automatically
    return True

def open_browser(url):
    """Open browser after a short delay"""
    time.sleep(2)  # Wait for Flask to start
    try:
        webbrowser.open(url)
        print(f"Opening browser: {url}")
    except Exception as e:
        print(f"Could not open browser automatically: {e}")
        print(f"Please open your browser and go to: {url}")

def main():
    """Main function to start the web UI"""
    print("BTC Options Trader - Web UI")
    print("=" * 50)
    
    # Check dependencies
    print("Checking dependencies...")
    if not check_dependencies():
        sys.exit(1)
    
    print("SUCCESS: All dependencies found")
    
    # Check configuration
    print("Checking configuration...")
    if not check_config():
        print("ERROR: Please run setup first")
        sys.exit(1)
    
    print("SUCCESS: Configuration file found")
    
    # Determine host and port
    host = os.getenv('FLASK_HOST', '127.0.0.1')
    port = int(os.getenv('FLASK_PORT', 5000))
    debug = os.getenv('FLASK_DEBUG', 'false').lower() == 'true'
    
    url = f"http://{host}:{port}"
    
    print(f"\nStarting Web UI...")
    print(f"   URL: {url}")
    print(f"   Host: {host}")
    print(f"   Port: {port}")
    print(f"   Debug: {debug}")
    print(f"\nFeatures available:")
    print("   - Dashboard: Real-time trading status")
    print("   - Settings: API configuration and risk management")
    print("   - Trades: Trade history and performance")
    print("   - Logs: Real-time trading logs")
    print(f"\nSecurity Notes:")
    print("   - UI runs locally on your machine")
    print("   - API credentials stored locally only")
    print("   - Paper trading enabled by default")
    
    print(f"\nAccess the UI at: {url}")
    print("   Press Ctrl+C to stop the server")
    print("-" * 50)
    
    # Start browser in background thread
    if host in ['127.0.0.1', 'localhost'] and not debug:
        browser_thread = threading.Thread(target=open_browser, args=(url,))
        browser_thread.daemon = True
        browser_thread.start()
    
    try:
        # Use production WSGI server (Waitress) instead of Flask dev server
        if debug:
            print("DEBUG mode: Using Flask development server")
            app.run(
                host=host,
                port=port,
                debug=debug,
                use_reloader=False,
                threaded=True
            )
        else:
            print("PRODUCTION mode: Using Waitress WSGI server")
            from waitress import serve
            serve(
                app, 
                host=host, 
                port=port, 
                threads=6,  # Handle multiple concurrent requests
                connection_limit=100,
                cleanup_interval=30,
                channel_timeout=120
            )
    except KeyboardInterrupt:
        print("\n\nWeb UI stopped")
    except Exception as e:
        print(f"\nERROR: Error starting web UI: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()