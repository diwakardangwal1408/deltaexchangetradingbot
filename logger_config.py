"""
Centralized Logging Configuration for BTC Trading System
Provides configurable console and file logging with different levels
"""
import logging
import logging.handlers
import os
from datetime import datetime


class TradingLogger:
    """
    Centralized logging configuration for the trading system.
    
    Features:
    - File logging: All logs (DEBUG level) for complete audit trail
    - Console logging: Configurable level (INFO for limited, DEBUG for detailed)
    - Log rotation: Prevents log files from growing too large
    - Timestamp formatting: Clear timestamps for all log entries
    """
    
    _loggers = {}  # Class variable to store configured loggers
    
    @classmethod
    def setup_logger(cls, name: str, console_level: str = "INFO", log_file: str = None):
        """
        Set up a logger with both file and console handlers.
        
        Args:
            name: Logger name (usually __name__ of the module)
            console_level: Console logging level ("DEBUG", "INFO", "WARNING", "ERROR")
            log_file: Optional custom log file name (defaults to delta_btc_trading.log)
        
        Returns:
            logging.Logger: Configured logger instance
        """
        
        # Return existing logger if already configured
        if name in cls._loggers:
            return cls._loggers[name]
        
        # Create logger
        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG)  # Logger accepts all levels
        
        # Clear any existing handlers to avoid duplicates
        logger.handlers.clear()
        
        # Set default log file
        if log_file is None:
            log_file = "delta_btc_trading.log"
        
        # Create formatters
        detailed_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        console_formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%H:%M:%S'
        )
        
        # File handler - All logs with rotation
        try:
            file_handler = logging.handlers.RotatingFileHandler(
                log_file,
                maxBytes=10*1024*1024,  # 10MB max file size
                backupCount=5,  # Keep 5 backup files
                encoding='utf-8'
            )
            file_handler.setLevel(logging.DEBUG)  # File gets all logs
            file_handler.setFormatter(detailed_formatter)
            logger.addHandler(file_handler)
        except Exception as e:
            print(f"Warning: Could not create file handler for {log_file}: {e}")
        
        # Console handler - Configurable level
        console_handler = logging.StreamHandler()
        
        # Convert string level to logging constant
        numeric_level = getattr(logging, console_level.upper(), logging.INFO)
        console_handler.setLevel(numeric_level)
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
        
        # Store configured logger
        cls._loggers[name] = logger
        
        # Log the logger configuration
        logger.debug(f"Logger '{name}' configured - Console: {console_level}, File: DEBUG")
        
        return logger
    
    @classmethod
    def get_logger(cls, name: str):
        """
        Get an existing logger or create a new one with default settings.
        
        Args:
            name: Logger name
            
        Returns:
            logging.Logger: Logger instance
        """
        if name not in cls._loggers:
            return cls.setup_logger(name)
        return cls._loggers[name]
    
    @classmethod
    def set_console_level(cls, level: str):
        """
        Change console logging level for all configured loggers.
        
        Args:
            level: New console logging level ("DEBUG", "INFO", "WARNING", "ERROR")
        """
        numeric_level = getattr(logging, level.upper(), logging.INFO)
        
        for logger_name, logger in cls._loggers.items():
            for handler in logger.handlers:
                if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
                    handler.setLevel(numeric_level)
                    logger.debug(f"Console level changed to {level} for logger '{logger_name}'")
    
    @classmethod
    def log_system_info(cls, logger):
        """
        Log system startup information.
        
        Args:
            logger: Logger instance to use
        """
        logger.info("="*60)
        logger.info("BTC Trading System - Logging Started")
        logger.info(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info(f"Log file: delta_btc_trading.log")
        logger.info("="*60)


def get_logger(name: str, console_level: str = "INFO", log_file: str = None):
    """
    Convenience function to get a configured logger.
    
    Args:
        name: Logger name (usually __name__)
        console_level: Console logging level ("DEBUG", "INFO", "WARNING", "ERROR")  
        log_file: Optional custom log file name
        
    Returns:
        logging.Logger: Configured logger instance
    """
    return TradingLogger.setup_logger(name, console_level, log_file)


def set_console_level(level: str):
    """
    Convenience function to change console logging level for all loggers.
    
    Args:
        level: New console logging level ("DEBUG", "INFO", "WARNING", "ERROR")
    """
    TradingLogger.set_console_level(level)


# Example usage:
if __name__ == "__main__":
    # Test the logging configuration
    logger = get_logger(__name__, "DEBUG")
    
    logger.debug("This is a debug message - shows in file and console (if DEBUG level)")
    logger.info("This is an info message - shows in file and console")
    logger.warning("This is a warning message - shows in file and console")
    logger.error("This is an error message - shows in file and console")
    
    # Change console level to INFO
    set_console_level("INFO")
    logger.debug("This debug message only goes to file now")
    logger.info("This info message goes to both file and console")