#!/usr/bin/env python3
"""
Data Manager for BTC Trading Bot
Manages loading, caching, and serving historical data from local CSV files
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import threading
import time
import concurrent.futures
from threading import Lock

from data_downloader import DataDownloader
from logger_config import get_logger


class DataManager:
    """Manages in-memory historical trading data for fast backtesting"""
    
    def __init__(self):
        """Initialize the data manager"""
        self.logger = get_logger(__name__)
        
        # Data directory
        self.data_dir = os.path.join(os.path.dirname(__file__), 'data')
        
        # In-memory data storage
        self._data_cache = {}
        self._indicators_cache = {}  # Pre-calculated indicators for ultra-fast backtesting
        self._cache_lock = threading.Lock()
        self._last_loaded = {}
        
        # Data downloader for updates
        self.downloader = DataDownloader()
        
        # Supported timeframes
        self.timeframes = {
            '1m': 'BTCUSD_1m.csv',
            '3m': 'BTCUSD_3m.csv', 
            '1h': 'BTCUSD_1h.csv'
        }
        
        self.logger.info("DataManager initialized")
    
    def preload_parallel(self, force_reload: bool = False, max_workers: int = 3) -> bool:
        """
        Pre-load all data in parallel threads for ultra-fast server startup
        
        Args:
            force_reload: Force reload even if already cached
            max_workers: Maximum number of parallel threads (default: 3 for 1M, 3M, 1H)
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.logger.info("=== Starting PARALLEL data loading for ultra-fast startup ===")
            start_time = time.time()
            
            # First, load raw data in parallel
            raw_load_success = self._load_raw_data_parallel(force_reload, max_workers)
            if not raw_load_success:
                self.logger.error("Failed to load raw data in parallel")
                return False
                
            raw_load_time = time.time() - start_time
            self.logger.info(f"✅ Raw data loaded in parallel: {raw_load_time:.2f} seconds")
            
            # Then, pre-calculate indicators in parallel
            indicators_start = time.time()
            indicators_success = self._calculate_indicators_parallel(max_workers)
            if not indicators_success:
                self.logger.error("Failed to calculate indicators in parallel")
                return False
                
            indicators_time = time.time() - indicators_start
            total_time = time.time() - start_time
            
            self.logger.info(f"✅ Indicators calculated in parallel: {indicators_time:.2f} seconds")
            self.logger.info(f"🚀 TOTAL PARALLEL STARTUP TIME: {total_time:.2f} seconds (vs ~{total_time*3:.1f}s sequential)")
            self.logger.info("=== Parallel data loading complete - Ready for ultra-fast backtesting ===")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error in parallel data loading: {e}")
            return False
    
    def _load_raw_data_parallel(self, force_reload: bool, max_workers: int) -> bool:
        """Load raw CSV data for all timeframes in parallel"""
        try:
            timeframes = ['1m', '3m', '1h']
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all timeframe loading tasks
                future_to_timeframe = {
                    executor.submit(self._load_single_timeframe, tf, force_reload): tf 
                    for tf in timeframes
                }
                
                # Collect results
                results = {}
                for future in concurrent.futures.as_completed(future_to_timeframe):
                    timeframe = future_to_timeframe[future]
                    try:
                        success = future.result()
                        results[timeframe] = success
                        status = "✅ SUCCESS" if success else "❌ FAILED"
                        self.logger.info(f"Thread completed: {timeframe} - {status}")
                    except Exception as e:
                        self.logger.error(f"Thread failed for {timeframe}: {e}")
                        results[timeframe] = False
            
            # Check if all succeeded
            successful = sum(1 for success in results.values() if success)
            total = len(results)
            self.logger.info(f"Parallel raw data loading: {successful}/{total} timeframes loaded")
            
            return all(results.values())
            
        except Exception as e:
            self.logger.error(f"Error in parallel raw data loading: {e}")
            return False
    
    def _load_single_timeframe(self, timeframe: str, force_reload: bool) -> bool:
        """Load a single timeframe (designed to run in separate thread)"""
        try:
            thread_name = threading.current_thread().name
            self.logger.info(f"[{thread_name}] Loading {timeframe} data...")
            
            # Check if already loaded and not forcing reload
            if not force_reload and timeframe in self._data_cache:
                self.logger.debug(f"[{thread_name}] {timeframe} data already loaded")
                return True
            
            csv_file = os.path.join(self.data_dir, self.timeframes[timeframe])
            
            if not os.path.exists(csv_file):
                self.logger.warning(f"[{thread_name}] CSV file not found: {csv_file}")
                return False
            
            # Load CSV data
            self.logger.debug(f"[{thread_name}] Reading CSV: {csv_file}")
            df = pd.read_csv(csv_file)
            
            if df.empty:
                self.logger.error(f"[{thread_name}] Empty CSV file: {csv_file}")
                return False
            
            # Process the data
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp')
            df = df.sort_index()
            
            # Rename columns to match expected format
            df = df.rename(columns={
                'open': 'Open',
                'high': 'High', 
                'low': 'Low',
                'close': 'Close',
                'volume': 'Volume'
            })
            
            # Ensure all columns are float
            for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                df[col] = df[col].astype(float)
            
            # Thread-safe cache update
            with self._cache_lock:
                self._data_cache[timeframe] = df
                self._last_loaded[timeframe] = datetime.now()
            
            self.logger.info(f"[{thread_name}] ✅ {timeframe}: {len(df)} candles loaded ({df.memory_usage(deep=True).sum()/1024/1024:.1f} MB)")
            self.logger.debug(f"[{thread_name}] Date range: {df.index.min()} to {df.index.max()}")
            
            return True
            
        except Exception as e:
            thread_name = threading.current_thread().name
            self.logger.error(f"[{thread_name}] Error loading {timeframe} data: {e}")
            return False
    
    def _calculate_indicators_parallel(self, max_workers: int) -> bool:
        """Calculate indicators for all timeframes in parallel"""
        try:
            # Only calculate indicators for timeframes that need them
            indicator_tasks = [
                ('1m', self._prepare_1m_data),      # Just store raw data
                ('3m', self._calculate_3m_indicators),  # Heavy computation
                ('1h', self._calculate_1h_indicators)   # Medium computation
            ]
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all indicator calculation tasks
                future_to_timeframe = {}
                
                for timeframe, calc_function in indicator_tasks:
                    if timeframe in self._data_cache:
                        future = executor.submit(calc_function, self._data_cache[timeframe].copy())
                        future_to_timeframe[future] = timeframe
                
                # Collect results
                results = {}
                for future in concurrent.futures.as_completed(future_to_timeframe):
                    timeframe = future_to_timeframe[future]
                    try:
                        df_with_indicators = future.result()
                        
                        # Thread-safe cache update
                        with self._cache_lock:
                            self._indicators_cache[timeframe] = df_with_indicators
                        
                        results[timeframe] = True
                        self.logger.info(f"✅ {timeframe} indicators calculated: {len(df_with_indicators)} candles")
                    except Exception as e:
                        self.logger.error(f"❌ {timeframe} indicators failed: {e}")
                        results[timeframe] = False
            
            successful = sum(1 for success in results.values() if success)
            total = len(results)
            self.logger.info(f"Parallel indicator calculation: {successful}/{total} completed")
            
            return all(results.values())
            
        except Exception as e:
            self.logger.error(f"Error in parallel indicator calculation: {e}")
            return False
    
    def _prepare_1m_data(self, df):
        """Prepare 1M data (no indicators needed, just return raw data)"""
        thread_name = threading.current_thread().name
        self.logger.debug(f"[{thread_name}] Preparing 1M data (no indicators needed)")
        return df
    
    def preload_with_indicators(self, force_reload: bool = False) -> bool:
        """
        Pre-load all data into memory with indicators pre-calculated for ultra-fast backtesting
        
        Args:
            force_reload: Force reload even if already cached
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.logger.info("=== Pre-loading all data with indicators for ultra-fast backtesting ===")
            
            # Load raw data first
            load_results = self.load_all_data(force_reload)
            if not all(load_results.values()):
                self.logger.error("Failed to load raw data")
                return False
            
            # Pre-calculate indicators for each timeframe
            with self._cache_lock:
                for timeframe in ['1m', '3m', '1h']:  # ALL timeframes for comprehensive backtesting
                    if timeframe in self._data_cache:
                        self.logger.info(f"Pre-calculating indicators for {timeframe} timeframe...")
                        
                        df = self._data_cache[timeframe].copy()
                        
                        if timeframe == '1m':
                            # Store 1M data for precise exit criteria (no additional indicators needed)
                            self._indicators_cache[timeframe] = df
                            self.logger.info(f"✅ 1M data loaded for precise exits: {len(df)} candles")
                            
                        elif timeframe == '3m':
                            # Pre-calculate 3M indicators (performance bottleneck)
                            df_with_indicators = self._calculate_3m_indicators(df)
                            self._indicators_cache[timeframe] = df_with_indicators
                            self.logger.info(f"✅ 3M indicators pre-calculated: {len(df_with_indicators)} candles")
                            
                        elif timeframe == '1h':
                            # Pre-calculate 1H indicators
                            df_with_indicators = self._calculate_1h_indicators(df)
                            self._indicators_cache[timeframe] = df_with_indicators
                            self.logger.info(f"✅ 1H indicators pre-calculated: {len(df_with_indicators)} candles")
            
            self.logger.info("=== All data and indicators loaded into memory successfully ===")
            return True
            
        except Exception as e:
            self.logger.error(f"Error pre-loading data with indicators: {e}")
            return False
    
    def _calculate_3m_indicators(self, df):
        """Pre-calculate all 3M indicators using strategy logic"""
        try:
            from btc_multi_timeframe_strategy import BTCMultiTimeframeStrategy
            
            # Initialize strategy temporarily for indicator calculations
            strategy = BTCMultiTimeframeStrategy("dummy", "dummy", paper_trading=True)
            
            # Calculate all Dashboard 3M indicators
            df_with_indicators = strategy.calculate_dashboard_technical_indicators(df.copy())
            
            self.logger.info(f"3M indicators calculated: VWAP, SAR, ATR, Price Action, Total Score")
            return df_with_indicators
            
        except Exception as e:
            self.logger.error(f"Error calculating 3M indicators: {e}")
            return df
    
    def _calculate_1h_indicators(self, df):
        """Pre-calculate all 1H indicators using strategy logic"""
        try:
            from btc_multi_timeframe_strategy import BTCMultiTimeframeStrategy
            
            # Initialize strategy temporarily for indicator calculations
            strategy = BTCMultiTimeframeStrategy("dummy", "dummy", paper_trading=True)
            
            # Calculate all Dashboard 1H indicators
            df_with_indicators = strategy.calculate_dashboard_higher_timeframe_indicators(df.copy())
            
            self.logger.info(f"1H indicators calculated: Fisher Transform, TSI, Pivot Points, Dow Theory")
            return df_with_indicators
            
        except Exception as e:
            self.logger.error(f"Error calculating 1H indicators: {e}")
            return df
    
    def get_data_with_indicators(self, timeframe: str, start_date: Optional[str] = None, end_date: Optional[str] = None) -> Optional[pd.DataFrame]:
        """
        Get pre-calculated data with indicators for ultra-fast backtesting
        
        Args:
            timeframe: '1m', '3m', or '1h'
            start_date: Start date in YYYY-MM-DD format (optional)
            end_date: End date in YYYY-MM-DD format (optional)
            
        Returns:
            DataFrame with OHLCV data and pre-calculated indicators
        """
        if timeframe not in ['1m', '3m', '1h']:
            self.logger.warning(f"Data not pre-loaded for {timeframe}, falling back to raw data")
            return self.get_data(timeframe, start_date, end_date)
        
        with self._cache_lock:
            if timeframe not in self._indicators_cache:
                self.logger.warning(f"No pre-calculated indicators for {timeframe}, run preload_with_indicators() first")
                return self.get_data(timeframe, start_date, end_date)
            
            df = self._indicators_cache[timeframe]
            
            # Apply date filtering if specified
            if start_date:
                try:
                    start_dt = pd.to_datetime(start_date)
                    df = df[df.index >= start_dt]
                except Exception as e:
                    self.logger.warning(f"Invalid start_date format: {start_date}")
            
            if end_date:
                try:
                    end_dt = pd.to_datetime(end_date) + timedelta(days=1)  # Include end date
                    df = df[df.index < end_dt]
                except Exception as e:
                    self.logger.warning(f"Invalid end_date format: {end_date}")
            
            self.logger.debug(f"Retrieved {len(df)} {timeframe} candles with pre-calculated indicators")
            return df
    
    def load_timeframe_data(self, timeframe: str, force_reload: bool = False) -> bool:
        """
        Load data for a specific timeframe into memory
        
        Args:
            timeframe: '1m', '3m', or '1h'
            force_reload: Force reload even if already cached
            
        Returns:
            True if successful, False otherwise
        """
        if timeframe not in self.timeframes:
            self.logger.error(f"Unsupported timeframe: {timeframe}")
            return False
        
        # Check if already loaded and not forcing reload
        if not force_reload and timeframe in self._data_cache:
            self.logger.debug(f"{timeframe} data already loaded in memory")
            return True
        
        csv_file = os.path.join(self.data_dir, self.timeframes[timeframe])
        
        if not os.path.exists(csv_file):
            self.logger.warning(f"CSV file not found: {csv_file}")
            return False
        
        try:
            self.logger.info(f"Loading {timeframe} data from {csv_file}...")
            
            # Load CSV data
            df = pd.read_csv(csv_file)
            
            if df.empty:
                self.logger.error(f"Empty CSV file: {csv_file}")
                return False
            
            # Process the data
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp')
            df = df.sort_index()  # Ensure chronological order
            
            # Rename columns to match expected format
            df = df.rename(columns={
                'open': 'Open',
                'high': 'High', 
                'low': 'Low',
                'close': 'Close',
                'volume': 'Volume'
            })
            
            # Ensure all columns are float
            for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                df[col] = df[col].astype(float)
            
            # Cache the data with thread safety
            with self._cache_lock:
                self._data_cache[timeframe] = df
                self._last_loaded[timeframe] = datetime.now()
            
            self.logger.info(f"Successfully loaded {len(df)} {timeframe} candles into memory")
            self.logger.info(f"Data range: {df.index.min()} to {df.index.max()}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading {timeframe} data: {e}")
            return False
    
    def load_all_data(self, force_reload: bool = False) -> Dict[str, bool]:
        """
        Load all timeframe data into memory
        
        Args:
            force_reload: Force reload even if already cached
            
        Returns:
            Dictionary with success status for each timeframe
        """
        results = {}
        
        self.logger.info("=== Loading Historical Data into Memory ===")
        
        for timeframe in self.timeframes.keys():
            results[timeframe] = self.load_timeframe_data(timeframe, force_reload)
        
        # Summary
        successful = sum(1 for success in results.values() if success)
        total = len(results)
        
        self.logger.info(f"=== Data Loading Complete: {successful}/{total} timeframes loaded ===")
        
        return results
    
    def get_data(self, timeframe: str, start_date: Optional[str] = None, end_date: Optional[str] = None) -> Optional[pd.DataFrame]:
        """
        Get historical data for a timeframe with on-demand loading (no caching)
        
        Args:
            timeframe: '1m', '3m', or '1h'
            start_date: Start date in YYYY-MM-DD format (optional)
            end_date: End date in YYYY-MM-DD format (optional)
            
        Returns:
            DataFrame with OHLCV data or None if not available
        """
        if timeframe not in self.timeframes:
            self.logger.error(f"Unsupported timeframe: {timeframe}")
            return None
        
        csv_file = os.path.join(self.data_dir, self.timeframes[timeframe])
        
        if not os.path.exists(csv_file):
            self.logger.warning(f"CSV file not found: {csv_file}")
            return None
        
        try:
            # Load data directly from CSV (on-demand, no caching)
            self.logger.debug(f"Loading {timeframe} data on-demand from CSV...")
            df = pd.read_csv(csv_file)
            
            if df.empty:
                self.logger.error(f"Empty CSV file: {csv_file}")
                return None
            
            # Process the data
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.set_index('timestamp')
            df = df.sort_index()
            
            # Rename columns to match expected format
            df = df.rename(columns={
                'open': 'Open',
                'high': 'High', 
                'low': 'Low',
                'close': 'Close',
                'volume': 'Volume'
            })
            
            # Ensure all columns are float
            for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                df[col] = df[col].astype(float)
            
            # Apply date filtering if specified
            if start_date:
                try:
                    start_dt = pd.to_datetime(start_date)
                    df = df[df.index >= start_dt]
                except Exception as e:
                    self.logger.warning(f"Invalid start_date format: {start_date}")
            
            if end_date:
                try:
                    end_dt = pd.to_datetime(end_date) + timedelta(days=1)  # Include end date
                    df = df[df.index < end_dt]
                except Exception as e:
                    self.logger.warning(f"Invalid end_date format: {end_date}")
            
            self.logger.info(f"Loaded {len(df)} {timeframe} candles for date range {start_date} to {end_date}")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading {timeframe} data on-demand: {e}")
            return None
    
    def get_multi_timeframe_data(self, start_date: Optional[str] = None, end_date: Optional[str] = None) -> Dict[str, pd.DataFrame]:
        """
        Get data for all timeframes (equivalent to delta_client.get_multi_timeframe_data)
        
        Args:
            start_date: Start date in YYYY-MM-DD format (optional)
            end_date: End date in YYYY-MM-DD format (optional)
            
        Returns:
            Dictionary with DataFrames for each timeframe
        """
        data = {}
        
        for timeframe in ['3m', '1h']:  # Main timeframes used in backtesting
            df = self.get_data(timeframe, start_date, end_date)
            if df is not None:
                data[timeframe] = df
            else:
                self.logger.warning(f"Could not get {timeframe} data")
        
        return data
    
    def get_data_info(self) -> Dict:
        """Get information about loaded data"""
        info = {
            'loaded_timeframes': list(self._data_cache.keys()),
            'load_times': {tf: dt.isoformat() for tf, dt in self._last_loaded.items()},
            'data_ranges': {},
            'memory_usage': {}
        }
        
        with self._cache_lock:
            for timeframe, df in self._data_cache.items():
                info['data_ranges'][timeframe] = {
                    'start': df.index.min().isoformat(),
                    'end': df.index.max().isoformat(),
                    'count': len(df)
                }
                
                info['memory_usage'][timeframe] = {
                    'memory_mb': df.memory_usage(deep=True).sum() / 1024 / 1024,
                    'shape': df.shape
                }
        
        return info
    
    def get_available_date_range(self) -> Dict[str, Optional[str]]:
        """
        Get the available date range for backtesting by reading CSV metadata
        
        Returns:
            Dictionary with start_date and end_date in YYYY-MM-DD format
        """
        try:
            csv_3m = os.path.join(self.data_dir, self.timeframes['3m'])
            csv_1h = os.path.join(self.data_dir, self.timeframes['1h'])
            
            if not os.path.exists(csv_3m) or not os.path.exists(csv_1h):
                self.logger.warning("Required CSV files not found")
                return {'start_date': None, 'end_date': None}
            
            # Read just the first and last rows to get date range efficiently
            def get_date_range_from_csv(csv_file):
                df = pd.read_csv(csv_file, usecols=['timestamp'])
                if df.empty:
                    return None, None
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                return df['timestamp'].min(), df['timestamp'].max()
            
            # Get date ranges for both timeframes
            start_3m, end_3m = get_date_range_from_csv(csv_3m)
            start_1h, end_1h = get_date_range_from_csv(csv_1h)
            
            if None in [start_3m, end_3m, start_1h, end_1h]:
                return {'start_date': None, 'end_date': None}
            
            # Use the most restrictive range (latest start, earliest end)
            start_date = max(start_3m, start_1h)
            end_date = min(end_3m, end_1h)
            
            return {
                'start_date': start_date.strftime('%Y-%m-%d'),
                'end_date': end_date.strftime('%Y-%m-%d')
            }
                
        except Exception as e:
            self.logger.error(f"Error getting available date range: {e}")
            return {'start_date': None, 'end_date': None}
    
    def ensure_data_available(self, max_age_hours: int = 1) -> bool:
        """
        Ensure fresh data is available, download if necessary
        
        Args:
            max_age_hours: Maximum acceptable age of data in hours
            
        Returns:
            True if data is available and fresh, False otherwise
        """
        self.logger.info("Checking data availability...")
        
        # Check what needs updating (daily check)
        needs_update = self.downloader.should_update_data(max_age_hours=24)  # Daily updates
        
        # Do incremental updates for stale data
        any_updated = False
        for timeframe, needs in needs_update.items():
            if needs:
                self.logger.info(f"Performing daily incremental update for {timeframe}...")
                if self.downloader.download_incremental_update(timeframe):
                    any_updated = True
                else:
                    self.logger.error(f"Failed to incrementally update {timeframe} data")
                    return False
        
        if any_updated:
            self.logger.info("Incremental data update completed")
        else:
            self.logger.info("No data updates needed")
        
        # Verify all required CSV files exist
        required_timeframes = ['3m', '1h']  # Main ones for backtesting
        for timeframe in required_timeframes:
            csv_file = os.path.join(self.data_dir, self.timeframes[timeframe])
            if not os.path.exists(csv_file):
                self.logger.error(f"Required CSV file for {timeframe} not available")
                return False
        
        self.logger.info("Data availability check complete - all required data loaded")
        return True
    
    def background_update_loop(self, update_interval_minutes: int = 1440):  # Default: daily (1440 minutes)
        """
        Background thread function to periodically update data with incremental updates
        
        Args:
            update_interval_minutes: How often to check for updates (default: daily)
        """
        self.logger.info(f"Starting background data update loop (every {update_interval_minutes} minutes)")
        
        while True:
            try:
                time.sleep(update_interval_minutes * 60)
                self.logger.info("Starting scheduled daily incremental update...")
                
                # Perform daily incremental updates
                results = self.downloader.daily_update_all_timeframes()
                
                successful = sum(1 for success in results.values() if success)
                total = len(results)
                self.logger.info(f"Daily incremental update complete: {successful}/{total} timeframes updated")
                
            except Exception as e:
                self.logger.error(f"Error in background update loop: {e}")
                time.sleep(3600)  # Wait an hour before retry
    
    def start_background_updates(self, update_interval_minutes: int = 1440):  # Default: daily
        """Start background data updates in a separate thread"""
        update_thread = threading.Thread(
            target=self.background_update_loop,
            args=(update_interval_minutes,),
            daemon=True
        )
        update_thread.start()
        self.logger.info("Background daily incremental updates started")


# Global data manager instance
_data_manager = None

def get_data_manager() -> DataManager:
    """Get the global data manager instance (singleton pattern)"""
    global _data_manager
    if _data_manager is None:
        _data_manager = DataManager()
    return _data_manager


if __name__ == "__main__":
    # CLI interface for testing
    import sys
    
    dm = DataManager()
    
    if len(sys.argv) > 1:
        command = sys.argv[1].lower()
        
        if command == 'load':
            # Load all data
            results = dm.load_all_data()
            print("Load results:", results)
            
        elif command == 'info':
            # Show data info
            info = dm.get_data_info()
            print("\n=== Data Info ===")
            for timeframe in info['loaded_timeframes']:
                range_info = info['data_ranges'][timeframe]
                memory_info = info['memory_usage'][timeframe]
                print(f"{timeframe}: {range_info['count']} candles")
                print(f"  Range: {range_info['start']} to {range_info['end']}")
                print(f"  Memory: {memory_info['memory_mb']:.1f} MB")
            
        elif command == 'test':
            # Test data retrieval
            print("Testing data retrieval...")
            dm.ensure_data_available()
            
            data = dm.get_multi_timeframe_data('2024-08-01', '2024-09-01')
            for timeframe, df in data.items():
                print(f"{timeframe}: {len(df)} candles from {df.index.min()} to {df.index.max()}")
        
        elif command == 'parallel':
            # Test parallel loading
            print("Testing parallel loading...")
            start_time = time.time()
            success = dm.preload_parallel()
            total_time = time.time() - start_time
            
            if success:
                print(f"SUCCESS: Parallel loading completed in {total_time:.2f} seconds")
                # Show memory usage
                info = dm.get_data_info()
                total_memory = 0
                for tf in info['memory_usage'].values():
                    total_memory += tf['memory_mb']
                print(f"Total memory usage: {total_memory:.1f} MB")
            else:
                print("FAILED: Parallel loading failed")
        
        else:
            print("Usage: python data_manager.py [load|info|test|parallel]")
    else:
        print("Usage: python data_manager.py [load|info|test|parallel]")