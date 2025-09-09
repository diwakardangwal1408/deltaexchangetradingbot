#!/usr/bin/env python3
"""
Production deployment script for BTC Options Trader
Uses Waitress WSGI server for better performance and stability
"""

import os
import sys
from waitress import serve
from app import app
from logger_config import get_logger
from config_manager import config_manager

def main():
    """Run the application in production mode using Waitress WSGI server"""
    
    # Initialize logger
    try:
        config = config_manager.get_all_config()
        logging_config = config.get('logging', {})
        console_level = logging_config.get('console_level', 'INFO')
        log_file = logging_config.get('log_file', 'delta_btc_trading.log')
        logger = get_logger('production_server', console_level, log_file)
    except Exception as e:
        logger = get_logger('production_server', 'INFO', 'delta_btc_trading.log')
        logger.warning(f"Could not load logging config, using defaults: {e}")
    
    # Configuration
    host = os.getenv('HOST', '0.0.0.0')  # Allow external connections
    port = int(os.getenv('PORT', 5000))
    
    logger.info("BTC Options Trader - Production Server")
    logger.info("=" * 50)
    logger.info(f"Server: Waitress WSGI")
    logger.info(f"Host: {host}")
    logger.info(f"Port: {port}")
    logger.info(f"URL: http://{host}:{port}")
    logger.info(f"Threads: 6")
    logger.info(f"Connection Limit: 100")
    logger.info("=" * 50)
    logger.info("Starting production server...")
    logger.info("Press Ctrl+C to stop")
    
    try:
        serve(
            app,
            host=host,
            port=port,
            threads=6,                    # Handle multiple concurrent requests
            connection_limit=100,         # Max concurrent connections
            cleanup_interval=30,          # Cleanup idle connections
            channel_timeout=120,          # Request timeout (2 minutes)
            max_request_body_size=10485760,  # 10MB max request size
            expose_tracebacks=False,      # Don't expose tracebacks in production
            ident=None                    # Don't expose server identity
        )
    except KeyboardInterrupt:
        logger.info("\nProduction server stopped")
    except Exception as e:
        logger.error(f"ERROR: Failed to start production server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()