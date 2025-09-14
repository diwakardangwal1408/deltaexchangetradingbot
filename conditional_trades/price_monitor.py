#!/usr/bin/env python3
"""
Price Monitor Service
Background service for monitoring BTC price and triggering conditional trades
"""

import time
import threading
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Callable, List
import logging

from delta_exchange_client import DeltaExchangeClient
from .conditional_trade_manager import ConditionalTradeManager
from logger_config import get_logger


class PriceMonitor:
    """
    Background service that monitors BTC price and triggers conditional trades
    Runs in a separate thread with configurable monitoring intervals
    """
    
    def __init__(self, 
                 delta_client: DeltaExchangeClient,
                 trade_manager: ConditionalTradeManager,
                 monitor_interval: int = 5):
        """
        Initialize price monitor
        
        Args:
            delta_client: Delta Exchange client for price fetching
            trade_manager: Conditional trade manager for trade execution
            monitor_interval: Price check interval in seconds (default: 5)
        """
        self.logger = get_logger(__name__)
        self.delta_client = delta_client
        self.trade_manager = trade_manager
        self.monitor_interval = monitor_interval
        
        # Monitoring state
        self._is_running = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        
        # Price tracking
        self._current_price = 0.0
        self._last_price_update: Optional[datetime] = None
        self._price_history = []  # Keep last 100 price points
        self._price_lock = threading.Lock()
        
        # Statistics
        self._stats = {
            'total_price_updates': 0,
            'total_trades_triggered': 0,
            'successful_executions': 0,
            'failed_executions': 0,
            'last_error': None,
            'uptime_start': None
        }
        
        # Callbacks for external notifications
        self._price_update_callbacks = []
        self._trade_execution_callbacks = []
        
        self.logger.info(f"PriceMonitor initialized with {monitor_interval}s interval")
    
    def start(self):
        """Start the price monitoring service"""
        if self._is_running:
            self.logger.warning("Price monitor is already running")
            return
        
        self._is_running = True
        self._stop_event.clear()
        self._stats['uptime_start'] = datetime.now()
        
        # Start monitoring thread
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            name="PriceMonitor",
            daemon=True
        )
        self._monitor_thread.start()
        
        self.logger.info("Price monitor started")
    
    def stop(self):
        """Stop the price monitoring service"""
        if not self._is_running:
            self.logger.warning("Price monitor is not running")
            return
        
        self.logger.info("Stopping price monitor...")
        
        self._is_running = False
        self._stop_event.set()
        
        # Wait for monitor thread to finish
        if self._monitor_thread:
            self._monitor_thread.join(timeout=10)
            if self._monitor_thread.is_alive():
                self.logger.warning("Monitor thread did not stop gracefully")
        
        self.logger.info("Price monitor stopped")
    
    def is_running(self) -> bool:
        """Check if price monitor is currently running"""
        return self._is_running
    
    def get_current_price(self) -> float:
        """Get the current BTC price"""
        with self._price_lock:
            return self._current_price
    
    def get_last_update(self) -> Optional[datetime]:
        """Get timestamp of last price update"""
        with self._price_lock:
            return self._last_price_update
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get monitoring statistics"""
        stats = self._stats.copy()
        
        if stats['uptime_start']:
            uptime = datetime.now() - stats['uptime_start']
            stats['uptime_seconds'] = uptime.total_seconds()
            stats['uptime_formatted'] = str(uptime).split('.')[0]  # Remove microseconds
        
        with self._price_lock:
            stats['current_price'] = self._current_price
            stats['last_update'] = self._last_price_update.isoformat() if self._last_price_update else None
            stats['price_history_count'] = len(self._price_history)
        
        stats['active_trades_count'] = self.trade_manager.get_trade_count()
        
        return stats
    
    def add_price_update_callback(self, callback: Callable[[float, datetime], None]):
        """Add callback function to be called on price updates"""
        self._price_update_callbacks.append(callback)
    
    def add_trade_execution_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """Add callback function to be called on trade executions"""
        self._trade_execution_callbacks.append(callback)
    
    def force_price_update(self) -> Dict[str, Any]:
        """Force an immediate price update (for testing/debugging)"""
        try:
            new_price = self._fetch_btc_price()
            if new_price:
                self._update_price(new_price)
                return {"success": True, "price": new_price, "timestamp": datetime.now().isoformat()}
            else:
                return {"success": False, "error": "Failed to fetch price"}
        except Exception as e:
            self.logger.error(f"Error in force price update: {e}")
            return {"success": False, "error": str(e)}
    
    def _monitor_loop(self):
        """Main monitoring loop that runs in background thread"""
        self.logger.info("Price monitoring loop started")
        
        while not self._stop_event.is_set():
            try:
                # Fetch current BTC price
                new_price = self._fetch_btc_price()
                
                if new_price and new_price > 0:
                    # Update price and trigger trade checks
                    self._update_price(new_price)
                    self._check_and_execute_trades(new_price)
                    
                    # Update statistics
                    self._stats['total_price_updates'] += 1
                    self._stats['last_error'] = None
                
                else:
                    self.logger.warning("Failed to fetch valid BTC price")
                    self._stats['last_error'] = "Invalid price fetched"
            
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                self._stats['last_error'] = str(e)
            
            # Wait for next interval or stop signal
            self._stop_event.wait(self.monitor_interval)
        
        self.logger.info("Price monitoring loop ended")
    
    def _fetch_btc_price(self) -> Optional[float]:
        """Fetch current BTC price from Delta Exchange using same method as dashboard"""
        try:
            # Use the same method as the dashboard's BTC price API
            price = self.delta_client.get_current_btc_price()
            
            if price and price > 0:
                return float(price)
            else:
                self.logger.warning("Invalid BTC price received")
                return None
        
        except Exception as e:
            self.logger.error(f"Error fetching BTC price: {e}")
            return None
    
    def _update_price(self, new_price: float):
        """Update current price and maintain price history"""
        now = datetime.now()
        
        with self._price_lock:
            self._current_price = new_price
            self._last_price_update = now
            
            # Add to price history (keep last 100 points)
            self._price_history.append({
                'price': new_price,
                'timestamp': now
            })
            
            # Trim history to last 100 points
            if len(self._price_history) > 100:
                self._price_history = self._price_history[-100:]
        
        # Notify callbacks
        for callback in self._price_update_callbacks:
            try:
                callback(new_price, now)
            except Exception as e:
                self.logger.error(f"Error in price update callback: {e}")
    
    def _check_and_execute_trades(self, current_price: float):
        """Check conditional trades and execute if conditions are met"""
        try:
            # Get trades that should be executed
            execution_results = self.trade_manager.check_and_execute_trades(current_price)
            
            # Process execution results
            for result in execution_results:
                self._stats['total_trades_triggered'] += 1
                
                try:
                    # Wait for execution to complete (with timeout)
                    execution_result = result['future'].result(timeout=30)
                    
                    if execution_result.get('success'):
                        self._stats['successful_executions'] += 1
                        self.logger.info(f"Successfully executed conditional trade: {execution_result}")
                        
                        # Notify execution callbacks
                        for callback in self._trade_execution_callbacks:
                            try:
                                callback(execution_result)
                            except Exception as e:
                                self.logger.error(f"Error in trade execution callback: {e}")
                    
                    else:
                        self._stats['failed_executions'] += 1
                        self.logger.error(f"Failed to execute conditional trade: {execution_result}")
                
                except Exception as e:
                    self._stats['failed_executions'] += 1
                    self.logger.error(f"Exception during trade execution: {e}")
        
        except Exception as e:
            self.logger.error(f"Error checking and executing trades: {e}")
    
    def get_price_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent price history"""
        with self._price_lock:
            history = self._price_history[-limit:] if limit else self._price_history
            return [
                {
                    'price': point['price'],
                    'timestamp': point['timestamp'].isoformat()
                }
                for point in history
            ]
    
    def __enter__(self):
        """Context manager entry"""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.stop()