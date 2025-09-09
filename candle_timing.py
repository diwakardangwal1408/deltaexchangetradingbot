#!/usr/bin/env python3
"""
Candle Timing Utilities for BTC Trading Bot
Handles proper candle timing synchronization with trading platforms like TradingView
"""

from datetime import datetime, timedelta
import pytz
from typing import Tuple, Optional
from config_manager import config_manager


def get_trading_timezone():
    """Get the configured trading timezone"""
    config = config_manager.get_all_config()
    timezone_str = config.get('trading_timing', {}).get('timezone', 'Asia/Kolkata')
    return pytz.timezone(timezone_str)


def get_trading_start_time():
    """Get the configured trading start time as datetime.time"""
    config = config_manager.get_all_config()
    start_time_str = config.get('trading_timing', {}).get('trading_start_time', '17:30')
    hour, minute = map(int, start_time_str.split(':'))
    return datetime.now().replace(hour=hour, minute=minute, second=0, microsecond=0).time()


def get_candle_close_times(interval_minutes: int, trading_start_time: datetime.time, timezone: pytz.BaseTzInfo) -> Tuple[datetime, datetime]:
    """
    Calculate the last candle close time and next candle close time for given interval
    
    Args:
        interval_minutes: Candle interval in minutes (3 for 3M, 60 for 1H)
        trading_start_time: Trading start time (e.g., 17:30)
        timezone: Trading timezone
    
    Returns:
        Tuple of (last_candle_close, next_candle_close) as timezone-aware datetime objects
    """
    now = datetime.now(timezone)
    
    # Get today's trading start time in the trading timezone
    today_start = now.replace(
        hour=trading_start_time.hour, 
        minute=trading_start_time.minute, 
        second=0, 
        microsecond=0
    )
    
    # If we haven't reached today's trading start time, use yesterday's schedule
    if now < today_start:
        today_start = today_start - timedelta(days=1)
    
    # Calculate how many intervals have passed since trading start
    elapsed_time = now - today_start
    elapsed_minutes = int(elapsed_time.total_seconds() / 60)
    
    # Find the number of complete intervals that have passed
    complete_intervals = elapsed_minutes // interval_minutes
    
    # Calculate last candle close time
    last_candle_close = today_start + timedelta(minutes=complete_intervals * interval_minutes)
    
    # Calculate next candle close time
    next_candle_close = last_candle_close + timedelta(minutes=interval_minutes)
    
    return last_candle_close, next_candle_close


def is_candle_closed(interval_minutes: int) -> bool:
    """
    Check if a candle has closed for the given interval
    
    Args:
        interval_minutes: Candle interval in minutes (3 for 3M, 60 for 1H)
        
    Returns:
        True if the candle has closed, False otherwise
    """
    timezone = get_trading_timezone()
    trading_start_time = get_trading_start_time()
    
    last_close, next_close = get_candle_close_times(interval_minutes, trading_start_time, timezone)
    now = datetime.now(timezone)
    
    # Candle is closed if we've passed the close time by at least 1 second
    return now >= (last_close + timedelta(seconds=1))


def get_last_candle_close_time(interval_minutes: int) -> datetime:
    """
    Get the timestamp of the last completed candle
    
    Args:
        interval_minutes: Candle interval in minutes (3 for 3M, 60 for 1H)
        
    Returns:
        Timezone-aware datetime of the last candle close
    """
    timezone = get_trading_timezone()
    trading_start_time = get_trading_start_time()
    
    last_close, _ = get_candle_close_times(interval_minutes, trading_start_time, timezone)
    return last_close


def get_next_candle_close_time(interval_minutes: int) -> datetime:
    """
    Get the timestamp when the next candle will close
    
    Args:
        interval_minutes: Candle interval in minutes (3 for 3M, 60 for 1H)
        
    Returns:
        Timezone-aware datetime of the next candle close
    """
    timezone = get_trading_timezone()
    trading_start_time = get_trading_start_time()
    
    _, next_close = get_candle_close_times(interval_minutes, trading_start_time, timezone)
    return next_close


def seconds_until_next_candle_close(interval_minutes: int) -> int:
    """
    Get seconds remaining until the next candle closes
    
    Args:
        interval_minutes: Candle interval in minutes (3 for 3M, 60 for 1H)
        
    Returns:
        Number of seconds until next candle close
    """
    next_close = get_next_candle_close_time(interval_minutes)
    now = datetime.now(get_trading_timezone())
    
    remaining = next_close - now
    return max(0, int(remaining.total_seconds()))


def format_candle_times_display(trading_start_time: str) -> Tuple[str, str]:
    """
    Format candle close times for display in settings page
    
    Args:
        trading_start_time: Trading start time string (e.g., "17:30")
        
    Returns:
        Tuple of (3m_times_str, 1h_times_str) for display
    """
    hour, minute = map(int, trading_start_time.split(':'))
    
    # Generate 3M times (first few)
    times_3m = []
    current_minute = minute
    current_hour = hour
    
    for i in range(5):  # Show first 5 times
        current_minute += 3
        if current_minute >= 60:
            current_minute -= 60
            current_hour += 1
            if current_hour >= 24:
                current_hour = 0
        times_3m.append(f"{current_hour:02d}:{current_minute:02d}")
    
    times_3m_str = ", ".join(times_3m) + "..."
    
    # Generate 1H times (first few)
    times_1h = []
    current_hour = hour + 1  # First 1H candle closes 1 hour after start
    if current_hour >= 24:
        current_hour = 0
        
    for i in range(4):  # Show first 4 times
        times_1h.append(f"{current_hour:02d}:{minute:02d}")
        current_hour += 1
        if current_hour >= 24:
            current_hour = 0
    
    times_1h_str = ", ".join(times_1h) + "..."
    
    return times_3m_str, times_1h_str


def validate_trading_timing_config() -> bool:
    """
    Validate that the trading timing configuration is valid
    
    Returns:
        True if configuration is valid, False otherwise
    """
    try:
        # Test timezone
        timezone = get_trading_timezone()
        
        # Test trading start time
        start_time = get_trading_start_time()
        
        # Test candle calculations
        last_3m, next_3m = get_candle_close_times(3, start_time, timezone)
        last_1h, next_1h = get_candle_close_times(60, start_time, timezone)
        
        return True
    except Exception as e:
        print(f"Trading timing configuration error: {e}")
        return False


if __name__ == "__main__":
    # Test the functions
    print("Testing candle timing functions:")
    print(f"Trading timezone: {get_trading_timezone()}")
    print(f"Trading start time: {get_trading_start_time()}")
    print(f"3M candle closed: {is_candle_closed(3)}")
    print(f"1H candle closed: {is_candle_closed(60)}")
    print(f"Last 3M close: {get_last_candle_close_time(3)}")
    print(f"Next 3M close: {get_next_candle_close_time(3)}")
    print(f"Seconds until next 3M: {seconds_until_next_candle_close(3)}")
    print(f"Seconds until next 1H: {seconds_until_next_candle_close(60)}")
    
    times_3m, times_1h = format_candle_times_display("17:30")
    print(f"3M display times: {times_3m}")
    print(f"1H display times: {times_1h}")