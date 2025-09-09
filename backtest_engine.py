import pandas as pd
import numpy as np
import logging
from datetime import datetime, timedelta
import time
from typing import Dict, List, Optional, Tuple
import json

from delta_exchange_client import DeltaExchangeClient
from btc_multi_timeframe_strategy import BTCMultiTimeframeStrategy
from config_manager import config_manager
from candle_timing import get_last_candle_close_time
from data_manager import get_data_manager


class BacktestEngine:
    """
    Backtesting engine that uses existing trading logic and settings
    to test strategies against historical data from Delta Exchange
    """
    
    def __init__(self, delta_client=None, strategy=None):
        """Initialize backtesting engine with current settings
        
        Args:
            delta_client: Existing DeltaExchangeClient instance to reuse (optional)
            strategy: Existing BTCMultiTimeframeStrategy instance to reuse (optional)
        """
        self.config = config_manager.get_all_config()
        
        # Reuse existing clients if provided, otherwise create new ones
        if delta_client:
            self.delta_client = delta_client
        else:
            # Initialize Delta Exchange client for historical data
            self.delta_client = DeltaExchangeClient(
                api_key=self.config['api_key'],
                api_secret=self.config['api_secret'],
                paper_trading=True  # Always use paper trading for backtesting
            )
        
        if strategy:
            self.strategy = strategy
        else:
            # Initialize strategy with current settings
            self.strategy = BTCMultiTimeframeStrategy(
                api_key=self.config['api_key'],
                api_secret=self.config['api_secret'],
                paper_trading=True
            )
        
        # Backtesting parameters from config
        self.initial_capital = float(self.config.get('portfolio_size', 1000.0))
        self.position_size_usd = float(self.config.get('position_size_usd', 500.0))
        self.max_positions = int(self.config.get('max_positions', 2))
        self.max_daily_loss = float(self.config.get('max_daily_loss', 100.0))
        
        # Dollar-based risk management
        dollar_risk = self.config.get('dollar_based_risk', {})
        self.stop_loss_usd = float(dollar_risk.get('stop_loss_usd', 100.0))
        self.take_profit_usd = float(dollar_risk.get('take_profit_usd', 200.0))
        self.trailing_stop_usd = float(dollar_risk.get('trailing_stop_usd', 50.0))
        self.quick_profit_usd = float(dollar_risk.get('quick_profit_usd', 60.0))
        self.max_risk_usd = float(dollar_risk.get('max_risk_usd', 150.0))
        
        # ATR-based risk management
        atr_config = self.config.get('atr_exits', {})
        self.atr_exits_enabled = atr_config.get('enabled', False)
        self.atr_period = int(atr_config.get('atr_period', 14))
        self.stop_loss_atr_multiplier = float(atr_config.get('stop_loss_atr_multiplier', 2.0))
        self.take_profit_atr_multiplier = float(atr_config.get('take_profit_atr_multiplier', 3.0))
        self.trailing_atr_multiplier = float(atr_config.get('trailing_atr_multiplier', 1.5))
        self.buffer_zone_atr_multiplier = float(atr_config.get('buffer_zone_atr_multiplier', 0.3))
        self.volume_threshold_percentile = float(atr_config.get('volume_threshold_percentile', 70))
        self.hunting_zone_offset = float(atr_config.get('hunting_zone_offset', 5))
        
        # Setup logging first
        from logger_config import get_logger
        self.logger = get_logger(__name__)
        
        # Futures strategy settings
        futures_config = self.config.get('futures_strategy', {})
        self.futures_enabled = futures_config.get('enabled', True)
        self.long_signal_threshold = int(futures_config.get('long_signal_threshold', 11))
        self.short_signal_threshold = int(futures_config.get('short_signal_threshold', -12))
        self.leverage = int(futures_config.get('leverage', 10))
        self.min_signal_strength = int(futures_config.get('min_signal_strength', 4))
        
        self.min_time_between_trades = int(futures_config.get('min_time_between_trades', 3600))
        
        # Backtesting state
        self.reset_backtest()
    
    def reset_backtest(self):
        """Reset backtesting state for new run"""
        self.current_capital = self.initial_capital
        self.current_positions = {}
        self.closed_positions = []
        self.equity_curve = []
        self.drawdown_curve = []
        self.daily_returns = []
        self.trades_history = []
        self.peak_capital = self.initial_capital
        self.max_drawdown = 0.0
        self.last_trade_time = None
        self.daily_pnl = 0.0
        self.current_date = None
        
    def fetch_historical_data(self, from_date: str, to_date: str, progress_callback=None) -> Dict:
        """
        Fetch historical data from Delta Exchange for the specified date range
        
        Args:
            from_date: Start date in YYYY-MM-DD format
            to_date: End date in YYYY-MM-DD format
            progress_callback: Function to call with progress updates
            
        Returns:
            Dictionary containing 1h and 3m historical data
        """
        try:
            if progress_callback:
                progress_callback({"step": "Fetching 1h data", "progress": 10})
            
            # Use local CSV data instead of API calls for much faster backtesting
            
            # Get the data manager instance  
            data_manager = get_data_manager()
            
            if progress_callback:
                progress_callback({"step": "Ensuring data availability", "progress": 20})
            
            # Ensure fresh data is available
            if not data_manager.ensure_data_available():
                raise Exception("Could not ensure data availability")
            
            if progress_callback:
                progress_callback({"step": "Pre-loading data in parallel threads for ultra-fast startup", "progress": 40})
            
            # Pre-load all data with indicators using parallel threads for maximum performance
            if not data_manager.preload_parallel():
                self.logger.warning("Failed to pre-load data in parallel, falling back to sequential loading")
                if not data_manager.preload_with_indicators():
                    self.logger.warning("Sequential fallback also failed, falling back to on-demand calculation")
            
            if progress_callback:
                progress_callback({"step": "Retrieving pre-calculated data", "progress": 70})
            
            # Get data with pre-calculated indicators for ultra-fast backtesting
            df_1m = data_manager.get_data_with_indicators('1m', from_date, to_date)  # For precise exits!
            df_3m = data_manager.get_data_with_indicators('3m', from_date, to_date)
            df_1h = data_manager.get_data_with_indicators('1h', from_date, to_date)
            
            # Fallback to regular data if indicators not available
            if df_1m is None:
                df_1m = data_manager.get_data('1m', from_date, to_date)
            if df_3m is None:
                df_3m = data_manager.get_data('3m', from_date, to_date)
            if df_1h is None:
                df_1h = data_manager.get_data('1h', from_date, to_date)
            
            if df_1m is None or df_3m is None or df_1h is None:
                raise Exception("Could not retrieve historical data from local storage")
            
            if df_1m.empty or df_3m.empty or df_1h.empty:
                raise Exception("Retrieved data is empty for the specified date range")
            
            self.logger.info(f"Loaded historical data: {len(df_1h)} 1h candles, {len(df_3m)} 3m candles, {len(df_1m)} 1m candles")
            self.logger.info(f"1M data range: {df_1m.index.min()} to {df_1m.index.max()}")
            self.logger.info(f"3M data range: {df_3m.index.min()} to {df_3m.index.max()}")
            self.logger.info(f"1H data range: {df_1h.index.min()} to {df_1h.index.max()}")
            
            if progress_callback:
                progress_callback({"step": "Data ready", "progress": 100})
            
            return {
                '1m': df_1m,  # Critical for precise exit timing!
                '1h': df_1h,
                '3m': df_3m,
                'from_date': from_date,
                'to_date': to_date
            }
            
        except Exception as e:
            self.logger.error(f"Error fetching historical data: {e}")
            raise e
    
    def _process_candle_data(self, raw_data: List) -> pd.DataFrame:
        """Convert raw candle data to DataFrame with proper columns"""
        if not raw_data:
            return pd.DataFrame()
            
        df = pd.DataFrame(raw_data)
        
        # Ensure we have the required columns
        required_columns = ['time', 'open', 'high', 'low', 'close', 'volume']
        for col in required_columns:
            if col not in df.columns:
                self.logger.error(f"Missing column {col} in historical data")
                return pd.DataFrame()
        
        # Convert timestamp to datetime
        df['datetime'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('datetime', inplace=True)
        
        # Ensure numeric columns
        numeric_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Rename columns to uppercase to match strategy expectations
        df = df.rename(columns={
            'open': 'Open', 
            'high': 'High', 
            'low': 'Low', 
            'close': 'Close', 
            'volume': 'Volume'
        })
        
        return df.sort_index()
    
    def _calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range (ATR) for the given data"""
        if len(df) < period:
            return pd.Series([0] * len(df), index=df.index)
        
        # Calculate True Range components
        high_low = df['High'] - df['Low']
        high_close = np.abs(df['High'] - df['Close'].shift())
        low_close = np.abs(df['Low'] - df['Close'].shift())
        
        # True Range is max of the three
        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        
        # ATR is exponential moving average of True Range
        atr = true_range.ewm(span=period).mean()
        
        return atr.fillna(0)
    
    def _is_high_volume_period(self, df: pd.DataFrame, current_idx: int) -> bool:
        """Check if current period has high volume (potential stop hunting)"""
        if current_idx < 20:  # Need minimum data
            return False
        
        # Get recent volume data (last 20 periods)
        recent_volumes = df['Volume'].iloc[max(0, current_idx-20):current_idx+1]
        current_volume = df['Volume'].iloc[current_idx]
        
        # Check if current volume is above threshold percentile
        volume_percentile = np.percentile(recent_volumes, self.volume_threshold_percentile)
        
        return current_volume > volume_percentile
    
    def _avoid_hunting_zones(self, price: float, proposed_level: float, side: str, atr_value: float) -> float:
        """Adjust stop/profit levels to avoid common hunting zones"""
        buffer = atr_value * self.buffer_zone_atr_multiplier
        
        # Round numbers (psychological levels) to avoid
        round_levels = []
        price_magnitude = 10 ** (len(str(int(price))) - 2)  # e.g., for 45000, magnitude = 1000
        
        # Generate round levels near the proposed level
        base_round = round(proposed_level / price_magnitude) * price_magnitude
        round_levels.extend([
            base_round - price_magnitude,
            base_round,
            base_round + price_magnitude
        ])
        
        # Half-round levels (e.g., 45500 when price is 45000+)
        half_magnitude = price_magnitude / 2
        base_half = round(proposed_level / half_magnitude) * half_magnitude
        round_levels.append(base_half)
        
        # Check if proposed level is too close to any round level
        adjusted_level = proposed_level
        
        for round_level in round_levels:
            distance = abs(proposed_level - round_level)
            
            if distance < self.hunting_zone_offset:  # Too close to round number
                # Move level away from round number based on trade side
                if side == 'long':
                    # For long trades, move stop below and profit above round numbers
                    if proposed_level < price:  # Stop loss
                        adjusted_level = round_level - buffer
                    else:  # Take profit
                        adjusted_level = round_level + buffer
                else:  # short
                    # For short trades, move stop above and profit below round numbers  
                    if proposed_level > price:  # Stop loss
                        adjusted_level = round_level + buffer
                    else:  # Take profit
                        adjusted_level = round_level - buffer
                break
        
        return adjusted_level
    
    def _calculate_anti_hunt_exits(self, trade: Dict, df_3m: pd.DataFrame, current_idx: int) -> Dict:
        """Calculate ATR-based exit levels with stop-hunt protection"""
        try:
            if not self.atr_exits_enabled:
                return self._calculate_dollar_exits(trade)  # Fallback to dollar-based
            
            # Get ATR values for 3M timeframe
            atr_series = self._calculate_atr(df_3m, self.atr_period)
            
            if current_idx >= len(atr_series) or current_idx < self.atr_period:
                return self._calculate_dollar_exits(trade)  # Fallback if insufficient data
            
            current_atr = atr_series.iloc[current_idx]
            current_price = trade['entry_price']
            side = trade['side']
            
            # Check for high volume period (potential hunting)
            is_high_volume = self._is_high_volume_period(df_3m, current_idx)
            
            # Base ATR multipliers
            stop_multiplier = self.stop_loss_atr_multiplier
            profit_multiplier = self.take_profit_atr_multiplier
            trail_multiplier = self.trailing_atr_multiplier
            
            # Increase multipliers during high volume periods to avoid hunting
            if is_high_volume:
                stop_multiplier *= 1.3
                profit_multiplier *= 1.2
                trail_multiplier *= 1.2
                self.logger.debug(f"High volume detected at trade entry - using enhanced ATR multipliers")
            
            # Calculate base levels
            if side == 'long':
                base_stop = current_price - (current_atr * stop_multiplier)
                base_profit = current_price + (current_atr * profit_multiplier)
                base_quick = current_price + (current_atr * (profit_multiplier * 0.6))
            else:  # short
                base_stop = current_price + (current_atr * stop_multiplier)
                base_profit = current_price - (current_atr * profit_multiplier)
                base_quick = current_price - (current_atr * (profit_multiplier * 0.6))
            
            # Apply hunting zone avoidance
            adjusted_stop = self._avoid_hunting_zones(current_price, base_stop, side, current_atr)
            adjusted_profit = self._avoid_hunting_zones(current_price, base_profit, side, current_atr)
            adjusted_quick = self._avoid_hunting_zones(current_price, base_quick, side, current_atr)
            
            return {
                'stop_loss': float(adjusted_stop),
                'take_profit': float(adjusted_profit),
                'quick_profit': float(adjusted_quick),
                'trailing_distance': float(current_atr * trail_multiplier),
                'atr_value': float(current_atr),
                'is_high_volume': bool(is_high_volume)
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating ATR exits: {e}")
            return self._calculate_dollar_exits(trade)  # Fallback to dollar-based
    
    def _calculate_dollar_exits(self, trade: Dict) -> Dict:
        """Calculate traditional dollar-based exit levels (fallback method)"""
        entry_price = trade['entry_price']
        position_size = trade['position_size']
        side = trade['side']
        
        if side == 'long':
            stop_loss = entry_price - (self.stop_loss_usd / position_size)
            take_profit = entry_price + (self.take_profit_usd / position_size)
            quick_profit = entry_price + (self.quick_profit_usd / position_size)
        else:  # short
            stop_loss = entry_price + (self.stop_loss_usd / position_size)
            take_profit = entry_price - (self.take_profit_usd / position_size)
            quick_profit = entry_price - (self.quick_profit_usd / position_size)
        
        return {
            'stop_loss': float(stop_loss),
            'take_profit': float(take_profit),
            'quick_profit': float(quick_profit),
            'trailing_distance': float(self.trailing_stop_usd / position_size),
            'atr_value': 0.0,
            'is_high_volume': False
        }
    
    def run_backtest(self, historical_data: Dict, progress_callback=None) -> Dict:
        """
        Run backtest using historical data and current strategy settings
        
        Args:
            historical_data: Historical data from fetch_historical_data
            progress_callback: Function to call with progress updates
            
        Returns:
            Dictionary containing backtest results and metrics
        """
        try:
            self.logger.info("=== STARTING DASHBOARD-STYLE BACKTEST ===")
            self.reset_backtest()
            
            df_1m = historical_data['1m']  # Critical for precise exit timing!
            df_1h = historical_data['1h']
            df_3m = historical_data['3m']
            
            self.logger.info(f"Data loaded: 1M = {len(df_1m)} candles, 3M = {len(df_3m)} candles, 1H = {len(df_1h)} candles")
            
            if df_1m.empty or df_1h.empty or df_3m.empty:
                raise ValueError("Historical data is empty")
            
            # Quick test of Dashboard methods
            try:
                self.logger.info("Testing Dashboard methods...")
                test_3m = df_3m.head(200).copy()
                test_1h = df_1h.head(min(150, len(df_1h))).copy()  # Use available data, reduced requirement
                self.logger.info(f"Testing with {len(test_1h)} 1H candles and {len(test_3m)} 3M candles")
                
                test_3m_indicators = self.strategy.calculate_dashboard_technical_indicators(test_3m)
                self.logger.info("✓ Dashboard 3M indicators working")
                
                test_1h_indicators = self.strategy.calculate_dashboard_higher_timeframe_indicators(test_1h, 3, -3)
                self.logger.info("✓ Dashboard 1H indicators working")
                
                test_signal = self.strategy.generate_dashboard_signals(test_3m_indicators, test_1h_indicators, self.config)
                self.logger.info(f"✓ Dashboard signal generation working: {test_signal.get('signal', 'NO_SIGNAL')}")
                
            except Exception as test_error:
                self.logger.error(f"❌ Dashboard methods test FAILED: {test_error}")
                import traceback
                self.logger.error(f"Traceback: {traceback.format_exc()}")
                return self._get_empty_results()
            
            # Set strategy data
            self.strategy.trend_data_1h = df_1h
            self.strategy.signal_data_3m = df_3m
            
            if progress_callback:
                progress_callback({"step": "Analyzing trends", "progress": 10})
            
            # Calculate indicators for entire period
            self._calculate_indicators(df_1h, df_3m)
            
            if progress_callback:
                progress_callback({"step": "Generating signals", "progress": 30})
            
            # Process candles chronologically - generate signals AND manage positions
            self._process_backtest_candles(df_3m, df_1h, df_1m, progress_callback)
            
            # Close any remaining positions
            self._close_all_positions(df_3m.iloc[-1])
            
            if progress_callback:
                progress_callback({"step": "Calculating metrics", "progress": 95})
            
            # Calculate performance metrics
            results = self._calculate_performance_metrics()
            
            if progress_callback:
                progress_callback({"step": "Backtest complete", "progress": 100})
            
            self.logger.info(f"Backtest completed: {len(self.trades_history)} trades executed")
            return results
            
        except Exception as e:
            self.logger.error(f"Error running backtest: {e}")
            import traceback
            self.logger.error(f"Full traceback: {traceback.format_exc()}")
            raise e
    
    def _calculate_indicators(self, df_1h: pd.DataFrame, df_3m: pd.DataFrame):
        """Calculate technical indicators for historical data"""
        # The strategy will calculate indicators when we call analyze methods
        # No need to call internal methods that don't exist
        pass
    
    def _process_backtest_candles(self, df_3m: pd.DataFrame, df_1h: pd.DataFrame, df_1m: pd.DataFrame, progress_callback=None):
        """Process candles chronologically, checking signals and exits at each step"""
        total_candles = len(df_3m)
        processed_candles = 0
        signals_found = 0
        
        # Cache for 1H analysis - only update when 1H candle closes
        cached_1h_analysis = None
        last_1h_candle_time = None
        
        self.logger.info(f"Starting candle-by-candle processing: {total_candles} 3M candles")
        
        # Performance tracking
        start_time = time.time()
        last_progress_time = start_time
        
        for i in range(100, len(df_3m)):  # Start from 100 for enough data
            processed_candles += 1
            current_candle = df_3m.iloc[i]
            current_timestamp = df_3m.index[i]
            
            # Progress reporting every 100 candles
            if processed_candles % 100 == 0:
                current_time = time.time()
                elapsed_since_start = current_time - start_time
                elapsed_since_last = current_time - last_progress_time
                
                processing_rate = 100 / elapsed_since_last if elapsed_since_last > 0 else 0
                overall_rate = processed_candles / elapsed_since_start if elapsed_since_start > 0 else 0
                
                progress_percent = (processed_candles / (total_candles - 100)) * 100
                estimated_remaining = (total_candles - 100 - processed_candles) / overall_rate if overall_rate > 0 else 0
                
                self.logger.info(f"PROGRESS: {processed_candles}/{total_candles-100} candles ({progress_percent:.1f}%) | "
                               f"Rate: {processing_rate:.1f} candles/sec | "
                               f"Positions: {len(self.current_positions)} | "
                               f"ETA: {estimated_remaining/60:.1f} min")
                
                # Send progress update to UI
                if progress_callback:
                    progress_callback({
                        "step": f"Processing candles: {processed_candles}/{total_candles-100}",
                        "progress": min(90, 10 + (progress_percent * 0.8)),  # 10-90% range
                        "details": f"{processing_rate:.1f} candles/sec, {len(self.current_positions)} positions",
                        "eta_minutes": estimated_remaining/60 if estimated_remaining > 0 else None
                    })
                
                last_progress_time = current_time
            
            # STEP 1: Check exits for ALL open positions FIRST (with smart skipping)
            if self._should_check_exits(current_candle, df_3m, i):
                # Use precise 1M data for exit timing (CRITICAL FOR PROFITABILITY!)
                current_1m_data = df_1m[df_1m.index <= current_timestamp]
                self._check_and_process_exits_with_1m_precision(current_candle, df_3m, current_1m_data, i)
            
            # STEP 2: Check for new signals on EVERY 3M candle close (every 3 minutes)
            # NEVER skip candles - every signal confirmation matters in trading!
            # Get current 3M data slice (use view for performance)
            current_3m_data = df_3m.iloc[:i+1]
            
            # Find 1H candles up to current timestamp (use view for performance)
            current_1h_data = df_1h[df_1h.index <= current_timestamp]
            
            if len(current_1h_data) < 20:  # Need minimum 1H data
                continue
            
            # Check if we have a new 1H candle
            current_1h_candle_time = current_1h_data.index[-1] if len(current_1h_data) > 0 else None
            
            if cached_1h_analysis is None or current_1h_candle_time != last_1h_candle_time:
                # Check if 1H data already has pre-calculated indicators
                if len(current_1h_data) >= 30 and 'Trend_Direction' in current_1h_data.columns:
                    # Use pre-calculated 1H indicators (ultra-fast)
                    cached_1h_analysis = current_1h_data
                    last_1h_candle_time = current_1h_candle_time
                    self.logger.debug(f"Using pre-calculated 1H indicators (ultra-fast)")
                elif len(current_1h_data) >= 30:
                    # Fallback: Calculate 1H indicators on-demand (slower)
                    try:
                        cached_1h_analysis = self.strategy.calculate_dashboard_higher_timeframe_indicators(
                            current_1h_data.copy(),  # Strategy needs copy for safety
                            self.config.get('futures_strategy', {}).get('trend_bullish_threshold', 3),
                            self.config.get('futures_strategy', {}).get('trend_bearish_threshold', -3)
                        )
                        last_1h_candle_time = current_1h_candle_time
                        self.logger.debug(f"Calculated 1H indicators on-demand (slower)")
                    except Exception as e:
                        self.logger.debug(f"Insufficient 1H data for indicators at candle {i}: {len(current_1h_data)} candles")
                        continue  # Skip this candle and continue
                else:
                    continue  # Not enough 1H data yet, skip this iteration
            
            # Use pre-calculated indicators (ultra-fast performance)
            # Check if current 3M data already has indicators pre-calculated
            if 'Total_Score' in current_3m_data.columns:
                # Data already has indicators - no calculation needed!
                current_3m_with_indicators = current_3m_data
                indicators_time = 0  # No calculation time
            else:
                # Fallback: Calculate indicators on-demand (slower path)
                indicators_start = time.time()
                current_3m_with_indicators = self.strategy.calculate_dashboard_technical_indicators(current_3m_data.copy())
                indicators_time = time.time() - indicators_start
            
            # Generate signal only if we have 1H analysis
            if cached_1h_analysis is None:
                continue  # Skip if no 1H analysis available yet
            
            signal_start = time.time()
            signal_data = self.strategy.generate_dashboard_signals(
                current_3m_with_indicators,
                cached_1h_analysis,
                self.config
            )
            signal_time = time.time() - signal_start
            
            # Log performance bottlenecks every 1000 candles
            if processed_candles % 1000 == 0:
                if indicators_time > 0:
                    self.logger.debug(f"Performance: Indicators={indicators_time*1000:.1f}ms, Signals={signal_time*1000:.1f}ms per candle")
                else:
                    self.logger.debug(f"Performance: Pre-calculated indicators (0ms), Signals={signal_time*1000:.1f}ms per candle")
            
            # Process signal if valid
            if signal_data and signal_data.get('signal', 0) != 0:
                signals_found += 1
                signal_info = {
                    'timestamp': current_timestamp,
                    'price': current_candle['Close'],
                    'signal': signal_data['signal'],
                    'strength': signal_data['strength'],
                    'type': signal_data['type'],
                    'reason': signal_data.get('decision_reasoning', ''),
                    'index': i
                }
                
                # Log signal generation
                self.logger.info(f"🔔 SIGNAL FOUND #{signals_found}: {signal_info['type']} at {current_timestamp} | "
                               f"Price: ${signal_info['price']:.2f} | Strength: {signal_info['strength']} | "
                               f"Reason: {signal_info['reason']}")
                
                # Try to open position
                if self._can_open_position(signal_info):
                    position_size = self._calculate_position_size(signal_info['price'])
                    trade = self._execute_simulated_trade(signal_info, position_size, df_3m)
                    
                    if trade:
                        self.trades_history.append(trade)
                        self.current_positions[trade['id']] = trade
                        self.last_trade_time = signal_info['timestamp']
                        self.current_capital -= trade['margin_used']
                        
                        self.logger.info(f"Trade EXECUTED #{len(self.trades_history)}: {signal_info['type']} at {signal_info['timestamp']} price={signal_info['price']}")
            
            # Update progress
            if progress_callback and processed_candles % 100 == 0:
                progress = 30 + (processed_candles / total_candles) * 60
                progress_callback({"step": f"Processing candle {processed_candles}/{total_candles}", "progress": int(progress)})
        
        self.logger.info(f"Candle processing complete: {processed_candles} candles, {signals_found} signals, {len(self.trades_history)} trades")
    
    def _should_check_exits(self, current_candle, df_3m, current_idx):
        """Smart exit checking - skip when no positions or minimal price movement"""
        # Skip if no positions exist
        if not self.current_positions:
            return False
        
        # Always check on first few candles for safety
        if current_idx < 105:
            return True
            
        # Skip if price hasn't moved significantly since last check
        if hasattr(self, '_last_exit_check_price') and hasattr(self, '_last_exit_check_idx'):
            current_price = float(current_candle['Close'])
            last_price = float(self._last_exit_check_price)
            
            price_change_pct = abs((current_price - last_price) / last_price)
            candles_since_last = current_idx - self._last_exit_check_idx
            
            # Skip if price moved <0.1% and less than 10 candles since last check
            if price_change_pct < 0.001 and candles_since_last < 10:
                return False
        
        # Update tracking variables
        self._last_exit_check_price = float(current_candle['Close'])
        self._last_exit_check_idx = current_idx
        
        return True
    
    def _should_check_for_signal(self, timestamp):
        """Determine if we should check for signals at this timestamp"""
        # Check signals every 3 minutes (on candle close)
        return True  # For now, check every candle
    
    def _check_and_process_exits_with_1m_precision(self, current_3m_candle, df_3m, df_1m, current_idx):
        """Check and process exits using 1M precision for accurate timing - CRITICAL FOR PROFITABILITY!"""
        if not self.current_positions or df_1m.empty:
            return
        
        # Get all 1M candles between the last 3M candle and current 3M candle for precise exit timing
        current_3m_time = current_3m_candle.name
        prev_3m_time = df_3m.index[current_idx - 1] if current_idx > 0 else current_3m_time
        
        # Get 1M candles in the current 3M period for precise exit checking
        relevant_1m_candles = df_1m[(df_1m.index > prev_3m_time) & (df_1m.index <= current_3m_time)]
        
        if relevant_1m_candles.empty:
            # Fallback to 3M candle if no 1M data available
            self._check_and_process_exits(current_3m_candle, df_3m, current_idx)
            return
        
        positions_to_close = []
        
        # Check each 1M candle for precise exit timing
        for minute_timestamp, minute_candle in relevant_1m_candles.iterrows():
            if not self.current_positions:  # No more positions to check
                break
                
            current_price = minute_candle['Close']
            high_price = minute_candle['High']
            low_price = minute_candle['Low']
            
            for trade_id, trade in list(self.current_positions.items()):
                if trade['status'] != 'open':
                    continue
                
                exit_reason = None
                exit_price = current_price
                
                # First, update trailing stops based on current HIGH price (most favorable move)
                original_stop = trade['stop_loss']
                
                if trade['side'] == 'long':
                    # Update trailing stop if new high achieved
                    if high_price > trade.get('high_water_mark', trade['entry_price']):
                        trade['high_water_mark'] = high_price
                        # Calculate new trailing stop
                        if trade.get('exit_method') == 'ATR' and trade.get('trailing_distance'):
                            new_trailing_stop = high_price - trade['trailing_distance']
                        else:
                            new_trailing_stop = high_price - (self.trailing_stop_usd / trade['position_size'])
                        # Move stop up only (never down)
                        trade['stop_loss'] = max(trade['stop_loss'], new_trailing_stop)
                        
                        if trade['stop_loss'] > original_stop:
                            self.logger.debug(f"🔄 Trailing stop updated for LONG #{trade['id']}: ${original_stop:.2f} → ${trade['stop_loss']:.2f} (High: ${high_price:.2f})")
                
                else:  # short position  
                    # Update trailing stop if new low achieved
                    if low_price < trade.get('low_water_mark', trade['entry_price']):
                        trade['low_water_mark'] = low_price
                        # Calculate new trailing stop
                        if trade.get('exit_method') == 'ATR' and trade.get('trailing_distance'):
                            new_trailing_stop = low_price + trade['trailing_distance']
                        else:
                            new_trailing_stop = low_price + (self.trailing_stop_usd / trade['position_size'])
                        # Move stop down only (never up)
                        trade['stop_loss'] = min(trade['stop_loss'], new_trailing_stop)
                        
                        if trade['stop_loss'] < original_stop:
                            self.logger.debug(f"🔄 Trailing stop updated for SHORT #{trade['id']}: ${original_stop:.2f} → ${trade['stop_loss']:.2f} (Low: ${low_price:.2f})")

                # Now check exit conditions using HIGH/LOW for maximum accuracy
                if trade['side'] == 'long':
                    # For long positions, check if LOW hit stop loss or HIGH hit take profit
                    if low_price <= trade['stop_loss']:
                        if trade['stop_loss'] > original_stop:
                            exit_reason = "Trailing Stop"  # Distinguish trailing from regular stop
                        else:
                            exit_reason = "Stop Loss"
                        exit_price = trade['stop_loss']  # Assume exact stop hit
                    elif high_price >= trade['take_profit']:
                        exit_reason = "Take Profit" 
                        exit_price = trade['take_profit']  # Assume exact target hit
                    elif high_price >= trade.get('quick_profit', float('inf')):
                        exit_reason = "Quick Profit"
                        exit_price = trade['quick_profit']
                else:  # short position
                    # For short positions, check if HIGH hit stop loss or LOW hit take profit
                    if high_price >= trade['stop_loss']:
                        if trade['stop_loss'] < original_stop:
                            exit_reason = "Trailing Stop"  # Distinguish trailing from regular stop
                        else:
                            exit_reason = "Stop Loss"
                        exit_price = trade['stop_loss']
                    elif low_price <= trade['take_profit']:
                        exit_reason = "Take Profit"
                        exit_price = trade['take_profit']
                    elif low_price <= trade.get('quick_profit', float('-inf')):
                        exit_reason = "Quick Profit"
                        exit_price = trade['quick_profit']
                
                # Execute exit with precise timing using existing method
                if exit_reason:
                    # Close the position using the existing method
                    self._close_position(trade, exit_price, minute_timestamp, exit_reason)
                    
                    # Remove from current positions (safe deletion)  
                    if trade_id in self.current_positions:
                        del self.current_positions[trade_id]
                    
                    self.logger.info(f"🎯 PRECISE EXIT: {trade['side']} position #{trade['id']} at {minute_timestamp} | "
                                   f"Price: ${exit_price:.2f} | Reason: {exit_reason} | "
                                   f"P&L: ${trade.get('pnl', 0):.2f} (1M precision)")
        
        # Log performance benefit
        if relevant_1m_candles is not None and len(relevant_1m_candles) > 1:
            self.logger.debug(f"Used {len(relevant_1m_candles)} 1M candles for precise exit timing (vs 1 3M candle)")
    
    def _close_position(self, trade, exit_price, exit_time, exit_reason):
        """Close a position and update trade records"""
        try:
            # Calculate final P&L
            if trade['side'] == 'long':
                pnl = (exit_price - trade['entry_price']) * trade['position_size'] - trade['fees']
            else:  # short
                pnl = (trade['entry_price'] - exit_price) * trade['position_size'] - trade['fees']
            
            # Update trade record
            trade['exit_price'] = exit_price
            trade['exit_time'] = exit_time
            trade['exit_reason'] = exit_reason
            trade['pnl'] = pnl
            trade['status'] = 'closed'
            trade['duration'] = (exit_time - trade['entry_time']).total_seconds() / 60  # Duration in minutes
            
            # Update capital (add back margin + P&L)
            self.current_capital += trade['margin_used'] + pnl
            
            # Add to closed positions list
            self.closed_positions.append(trade.copy())
            
            self.logger.debug(f"Position closed: {trade['side']} #{trade['id']} | "
                            f"Entry: ${trade['entry_price']:.2f} | Exit: ${exit_price:.2f} | "
                            f"P&L: ${pnl:.2f} | Reason: {exit_reason}")
            
        except Exception as e:
            self.logger.error(f"Error closing position {trade.get('id', 'unknown')}: {e}")
    
    def _check_and_process_exits(self, current_candle, df_3m, current_idx):
        """Check and process exits for all open positions"""
        if not self.current_positions:
            return
            
        current_price = current_candle['Close']
        current_time = current_candle.name
        positions_to_close = []
        
        for trade_id, trade in list(self.current_positions.items()):
            if trade['status'] != 'open':
                continue
            
            exit_reason = None
            exit_price = current_price
            
            # Calculate current P&L
            if trade['side'] == 'long':
                current_pnl = (current_price - trade['entry_price']) * trade['position_size'] - trade['fees']
            else:  # short
                current_pnl = (trade['entry_price'] - current_price) * trade['position_size'] - trade['fees']
            
            # Check exit conditions in priority order
            
            # Priority 1: Max Risk Hit
            if current_pnl <= -self.max_risk_usd:
                exit_reason = "Max Risk Hit"
            
            # Priority 2: Stop Loss
            elif trade['side'] == 'long' and current_price <= trade['stop_loss']:
                exit_reason = "Stop Loss"
            elif trade['side'] == 'short' and current_price >= trade['stop_loss']:
                exit_reason = "Stop Loss"
            
            # Priority 3: Take Profit
            elif trade['side'] == 'long' and current_price >= trade['take_profit']:
                exit_reason = "Take Profit"
            elif trade['side'] == 'short' and current_price <= trade['take_profit']:
                exit_reason = "Take Profit"
            
            # Priority 4: Quick Profit
            elif trade['side'] == 'long' and current_price >= trade['quick_profit']:
                exit_reason = "Quick Profit"
            elif trade['side'] == 'short' and current_price <= trade['quick_profit']:
                exit_reason = "Quick Profit"
            
            # Update trailing stop
            if exit_reason is None:
                if trade['side'] == 'long':
                    if current_price > trade['high_water_mark']:
                        trade['high_water_mark'] = current_price
                        # Update trailing stop
                        if trade.get('exit_method') == 'ATR' and trade.get('trailing_distance'):
                            new_trailing_stop = current_price - trade['trailing_distance']
                        else:
                            new_trailing_stop = current_price - (self.trailing_stop_usd / trade['position_size'])
                        trade['stop_loss'] = max(trade['stop_loss'], new_trailing_stop)
                else:  # short
                    if current_price < trade['low_water_mark']:
                        trade['low_water_mark'] = current_price
                        # Update trailing stop
                        if trade.get('exit_method') == 'ATR' and trade.get('trailing_distance'):
                            new_trailing_stop = current_price + trade['trailing_distance']
                        else:
                            new_trailing_stop = current_price + (self.trailing_stop_usd / trade['position_size'])
                        trade['stop_loss'] = min(trade['stop_loss'], new_trailing_stop)
            
            # Mark for closure if exit condition met
            if exit_reason:
                positions_to_close.append((trade_id, trade, exit_price, current_time, exit_reason))
        
        # Close positions
        for trade_id, trade, exit_price, exit_time, reason in positions_to_close:
            self._close_position(trade, exit_price, exit_time, reason)
            # Remove from current positions (safe deletion)
            if trade_id in self.current_positions:
                del self.current_positions[trade_id]
            # Capital already updated in _close_position, don't double-add
            
            self.logger.info(f"Position CLOSED #{trade_id}: {reason} at {exit_time} price={exit_price} P&L=${trade['pnl']:.2f}")
    
    
    def _generate_all_signals(self, df_3m: pd.DataFrame, df_1h: pd.DataFrame) -> List[Dict]:
        """Generate trading signals using Dashboard logic for entire historical period"""
        signals = []
        
        # Cache for 1H analysis - only update when 1H candle closes
        cached_1h_analysis = None
        last_1h_candle_time = None
        
        # Iterate through 3m candles to generate signals
        total_candles = len(df_3m)
        processed_candles = 0
        signals_checked = 0
        
        self.logger.info(f"Starting signal generation loop: {total_candles} 3M candles, {len(df_1h)} 1H candles")
        
        for i in range(len(df_3m)):
            if i < 100:  # Need minimum data for Dashboard indicators
                continue
                
            # Get current 3M data slice (from start to current candle)
            current_3m_data = df_3m.iloc[:i+1].copy()
            
            # For 1H data, we need to find the corresponding timeframe
            # Get timestamp of current 3M candle
            current_timestamp = df_3m.index[i]
            
            # Find 1H candles up to current timestamp
            current_1h_data = df_1h[df_1h.index <= current_timestamp].copy()
            
            # Ensure we have enough 1H data for higher timeframe analysis  
            min_1h_required = 20  # Need at least 20 1H candles for trend analysis
            if len(current_1h_data) < min_1h_required:
                continue
            
            processed_candles += 1
            
            # Log first few iterations to debug
            if processed_candles <= 5:
                self.logger.info(f"Processing candle {i}: Current time={current_timestamp}, 3M data len={len(current_3m_data)}, 1H data len={len(current_1h_data)}")
            
            # Check and process exits for existing positions BEFORE looking for new signals
            current_candle = df_3m.iloc[i]
            current_signal = None  # Will be updated after signal generation
                
            # Update strategy with current data  
            try:
                # Debug current slice
                if i % 100 == 0:  # Every 100 candles
                    self.logger.info(f"Processing candle {i}/{len(df_3m)}, 3M data: {len(current_3m_data)}, 1H data: {len(current_1h_data)}")
                
                # Calculate Dashboard indicators for 3M data
                current_3m_with_indicators = self.strategy.calculate_dashboard_technical_indicators(current_3m_data.copy())
                
                # Check if we have a new 1H candle (only recalculate 1H indicators when 1H candle closes)
                current_1h_candle_time = current_1h_data.index[-1] if len(current_1h_data) > 0 else None
                
                if cached_1h_analysis is None or current_1h_candle_time != last_1h_candle_time:
                    # New 1H candle detected or first calculation - recalculate 1H indicators
                    cached_1h_analysis = self.strategy.calculate_dashboard_higher_timeframe_indicators(
                        current_1h_data.copy(),
                        self.config.get('futures_strategy', {}).get('trend_bullish_threshold', 3),
                        self.config.get('futures_strategy', {}).get('trend_bearish_threshold', -3)
                    )
                    last_1h_candle_time = current_1h_candle_time
                    
                    if signals_checked <= 10:  # Log first few 1H updates
                        self.logger.info(f"1H candle update at 3M candle {i}: New 1H time={current_1h_candle_time}")
                
                # Use cached 1H analysis
                current_1h_with_indicators = cached_1h_analysis
                
                # Generate Dashboard-style signal
                signals_checked += 1
                signal_data = self.strategy.generate_dashboard_signals(
                    current_3m_with_indicators,
                    current_1h_with_indicators,
                    self.config
                )
                
                # Log first few signal generations
                if signals_checked <= 5:
                    self.logger.info(f"Signal check {signals_checked}: Generated signal_data keys: {list(signal_data.keys()) if signal_data else 'None'}")
                
                # Debug signal generation - more detailed
                if i % 200 == 0 or signal_data.get('signal', 0) != 0:  # Every 200 candles OR when signal is found
                    latest_3m = current_3m_with_indicators.iloc[-1]
                    latest_1h = current_1h_with_indicators.iloc[-1]
                    total_score = latest_3m.get('Total_Score', 0)
                    trend_direction = latest_1h.get('Trend_Direction', 'UNKNOWN')
                    signal_val = signal_data.get('signal', 0)
                    trade_decision = signal_data.get('trade_decision', 'NO_TRADE')
                    
                    self.logger.info(f"Debug at candle {i}: 3M Score={total_score}, 1H Trend={trend_direction}, "
                                   f"Signal={signal_val}, Decision={trade_decision}")
                    
                    if signal_val != 0:
                        self.logger.info(f"VALID SIGNAL FOUND: {signal_data.get('decision_reasoning', '')}")
                
                
                # Dashboard already did all validation - if signal != 0, it's valid
                if signal_data and signal_data.get('signal', 0) != 0:
                    signal_info = {
                        'timestamp': df_3m.index[i],
                        'price': df_3m.iloc[i]['Close'],
                        'signal': signal_data['signal'],
                        'strength': signal_data['strength'],
                        'type': signal_data['type'],
                        'reason': signal_data.get('decision_reasoning', ''),
                        'trade_decision': signal_data.get('trade_decision', ''),
                        '1h_trend': signal_data.get('1h_trend', ''),
                        '3m_score': signal_data.get('3m_total_score', 0),
                        'scoring_breakdown': signal_data.get('3m_scoring_breakdown', {}),
                        'index': i
                    }
                    signals.append(signal_info)
                    current_signal = signal_info  # Store for signal reversal check
                    
                    # Log Dashboard-style signal for debugging
                    self.logger.info(f"Dashboard Signal Generated: {signal_data['type']} at {df_3m.index[i]} | "
                                   f"3M Score: {signal_data.get('3m_total_score', 0)} | "
                                   f"1H Trend: {signal_data.get('1h_trend', 'Unknown')} | "
                                   f"Decision: {signal_data.get('decision_reasoning', '')}")
                
                # Process exits with signal reversal check (after signal generation)
                self._process_position_exits(current_candle, current_signal)
                    
            except Exception as e:
                self.logger.error(f"Error generating signal at index {i}: {e}")
                self._process_position_exits(current_candle, None)  # Still process exits even if signal generation fails
                continue
        
        self.logger.info(f"Signal generation complete: Total candles={total_candles}, Processed={processed_candles}, Signals checked={signals_checked}, Valid signals found={len(signals)}")
        return signals
    
    def _process_signal(self, signal: Dict, df_3m: pd.DataFrame):
        """Process a trading signal and simulate trade execution"""
        try:
            # Check if we can take a new position
            if not self._can_open_position(signal):
                # Log first 10 rejections to understand the pattern
                if len(self.trades_history) < 50:  # Only log early in backtest
                    self.logger.info(f"Signal REJECTED at {signal['timestamp']}: {signal['type']} strength={signal['strength']}")
                return
            
            # Calculate position size
            position_size = self._calculate_position_size(signal['price'])
            
            # Simulate trade execution
            trade = self._execute_simulated_trade(signal, position_size, df_3m)
            
            if trade:
                self.trades_history.append(trade)
                self.current_positions[trade['id']] = trade
                self.last_trade_time = signal['timestamp']
                
                # Update capital
                self.current_capital -= trade['margin_used']
                
                # Log trade execution
                self.logger.info(f"Trade EXECUTED #{len(self.trades_history)}: {signal['type']} at {signal['timestamp']} price={signal['price']} strength={signal['strength']}")
            else:
                self.logger.warning(f"Failed to execute trade for signal at {signal['timestamp']}")
                
        except Exception as e:
            self.logger.error(f"Error processing signal: {e}")
    
    def _process_position_exits(self, current_candle, current_signal=None):
        """Check and process exits for all open positions including signal reversal"""
        try:
            current_price = current_candle['Close']
            current_time = current_candle.name
            
            # Check each open position for exit conditions
            positions_to_close = []
            
            for trade_id, trade in self.current_positions.items():
                if trade['status'] != 'open':
                    continue
                    
                exit_reason = None
                exit_price = current_price
                
                # Calculate current P&L for max risk check
                if trade['side'] == 'long':
                    current_pnl = (current_price - trade['entry_price']) * trade['position_size'] - trade['fees']
                else:  # short
                    current_pnl = (trade['entry_price'] - current_price) * trade['position_size'] - trade['fees']
                
                # Priority 1: Check Max Risk (immediate exit if loss too large)
                if current_pnl <= -self.max_risk_usd:
                    exit_reason = "Max Risk Hit"
                
                # Priority 2: Check Quick Profit (take small profits quickly)
                elif trade['side'] == 'long' and current_price >= trade['quick_profit']:
                    exit_reason = "Quick Profit"
                elif trade['side'] == 'short' and current_price <= trade['quick_profit']:
                    exit_reason = "Quick Profit"
                
                # Priority 3: Check Signal Reversal (close position if opposite signal appears)
                elif current_signal and self._is_signal_reversal(trade, current_signal):
                    exit_reason = "Signal Reversal"
                
                # Priority 4: Update trailing stops and check regular exits
                else:
                    # Update high/low water marks for trailing stop
                    if trade['side'] == 'long':
                        if current_price > trade['high_water_mark']:
                            trade['high_water_mark'] = current_price
                            
                            # Use ATR-based or dollar-based trailing
                            if trade.get('exit_method') == 'ATR' and trade.get('trailing_distance'):
                                new_trailing_stop = current_price - trade['trailing_distance']
                            else:
                                new_trailing_stop = current_price - (self.trailing_stop_usd / trade['position_size'])
                            
                            trade['stop_loss'] = max(trade['stop_loss'], new_trailing_stop)
                            
                        # Check exit conditions for LONG
                        if current_price <= trade['stop_loss']:
                            exit_reason = "Stop Loss"
                        elif current_price >= trade['take_profit']:
                            exit_reason = "Take Profit"
                            
                    else:  # short
                        if current_price < trade['low_water_mark']:
                            trade['low_water_mark'] = current_price
                            
                            # Use ATR-based or dollar-based trailing
                            if trade.get('exit_method') == 'ATR' and trade.get('trailing_distance'):
                                new_trailing_stop = current_price + trade['trailing_distance']
                            else:
                                new_trailing_stop = current_price + (self.trailing_stop_usd / trade['position_size'])
                            
                            trade['stop_loss'] = min(trade['stop_loss'], new_trailing_stop)
                            
                        # Check exit conditions for SHORT
                        if current_price >= trade['stop_loss']:
                            exit_reason = "Stop Loss"
                        elif current_price <= trade['take_profit']:
                            exit_reason = "Take Profit"
                
                # Mark position for closure if exit condition met
                if exit_reason:
                    positions_to_close.append((trade_id, trade, exit_price, current_time, exit_reason))
            
            # Close positions that hit exit conditions
            for trade_id, trade, exit_price, exit_time, reason in positions_to_close:
                self._close_position(trade, exit_price, exit_time, reason)
                del self.current_positions[trade_id]
                
                # Return margin to capital
                self.current_capital += trade['margin_used']
                
                self.logger.info(f"Position CLOSED #{trade_id}: {reason} at {exit_time} price={exit_price} P&L=${trade['pnl']:.2f}")
                        
        except Exception as e:
            self.logger.error(f"Error processing position exits: {e}")
    
    def _is_signal_reversal(self, trade: Dict, current_signal: Dict) -> bool:
        """Check if current signal is opposite to the trade direction"""
        if not current_signal:
            return False
            
        trade_side = trade['side']
        signal_type = current_signal.get('type', '').upper()
        
        # Long position gets closed by SHORT signal
        if trade_side == 'long' and signal_type == 'SHORT':
            return True
        # Short position gets closed by LONG signal  
        elif trade_side == 'short' and signal_type == 'LONG':
            return True
            
        return False
    
    def _can_open_position(self, signal: Dict) -> bool:
        """Check if we can open a new position based on risk rules"""
        # Max positions check
        if len(self.current_positions) >= self.max_positions:
            self.logger.debug(f"Signal rejected: Max positions ({self.max_positions}) reached")
            return False
        
        # Time between trades check
        if (self.last_trade_time and 
            (signal['timestamp'] - self.last_trade_time).total_seconds() < self.min_time_between_trades):
            self.logger.debug(f"Signal rejected: Time between trades too short")
            return False
        
        # Daily loss check
        if self.daily_pnl <= -self.max_daily_loss:
            self.logger.debug(f"Signal rejected: Daily loss limit reached")
            return False
        
        # Signal strength check - basic minimum
        if abs(signal['strength']) < self.min_signal_strength:
            self.logger.debug(f"Signal rejected: Strength {signal['strength']} below minimum {self.min_signal_strength}")
            return False
        
        # Dashboard already validated signal thresholds - no need to re-validate
        # The signal wouldn't exist if it didn't meet Dashboard criteria
        
        return True
    
    def _calculate_position_size(self, price: float) -> float:
        """Calculate position size based on current settings"""
        # Use configured position size in USD
        position_value = min(self.position_size_usd, self.current_capital * 0.1)  # Max 10% of capital
        
        # Apply leverage
        quantity = (position_value * self.leverage) / price
        
        return max(quantity, 0.001)  # Minimum position size
    
    def _execute_simulated_trade(self, signal: Dict, position_size: float, df_3m: pd.DataFrame) -> Optional[Dict]:
        """Simulate trade execution and management"""
        try:
            entry_price = signal['price']
            entry_time = signal['timestamp']
            trade_id = f"bt_{len(self.trades_history)}"
            
            side = 'long' if signal['signal'] > 0 else 'short'
            
            # Create basic trade info for exit calculation
            basic_trade = {
                'entry_price': entry_price,
                'side': side,
                'position_size': position_size
            }
            
            # Find the current index in the 3M dataframe
            current_idx = signal.get('index', len(df_3m) - 1)  # Use signal index if available
            
            # Calculate ATR-based or dollar-based exits
            exit_levels = self._calculate_anti_hunt_exits(basic_trade, df_3m, current_idx)
            
            stop_loss = exit_levels['stop_loss']
            take_profit = exit_levels['take_profit']
            quick_profit = exit_levels['quick_profit']
            
            # Calculate margin used (simplified)
            margin_used = (position_size * entry_price) / self.leverage
            
            trade = {
                'id': trade_id,
                'side': side,
                'entry_time': entry_time,
                'entry_price': entry_price,
                'position_size': position_size,
                'stop_loss': stop_loss,
                'take_profit': take_profit,
                'quick_profit': quick_profit,
                'margin_used': margin_used,
                'signal_strength': signal['strength'],
                'signal_type': signal['type'],
                'status': 'open',
                'high_water_mark': entry_price if side == 'long' else entry_price,
                'low_water_mark': entry_price if side == 'short' else entry_price,
                'exit_price': None,
                'exit_time': None,
                'exit_reason': None,
                'pnl': 0.0,
                'fees': margin_used * 0.001,  # 0.1% fee
                'trailing_distance': exit_levels['trailing_distance'],
                'atr_value': exit_levels['atr_value'],
                'is_high_volume_entry': exit_levels['is_high_volume'],
                'exit_method': 'ATR' if self.atr_exits_enabled else 'Dollar'
            }
            
            return trade
            
        except Exception as e:
            self.logger.error(f"Error executing simulated trade: {e}")
            return None
    
    def _close_all_positions(self, final_candle):
        """Close all remaining positions at the end of backtest"""
        for trade_id, trade in list(self.current_positions.items()):
            self._close_position(trade, final_candle['Close'], final_candle.name, "Backtest end")
    
    def _close_position(self, trade: Dict, exit_price: float, exit_time, reason: str):
        """Close a position and calculate P&L"""
        try:
            trade['exit_price'] = exit_price
            trade['exit_time'] = exit_time
            trade['status'] = 'closed'
            trade['exit_reason'] = reason
            
            # Calculate P&L
            if trade['side'] == 'long':
                pnl = (exit_price - trade['entry_price']) * trade['position_size']
            else:  # short
                pnl = (trade['entry_price'] - exit_price) * trade['position_size']
            
            pnl -= trade['fees']  # Subtract fees
            trade['pnl'] = pnl
            
            # Update capital
            self.current_capital += trade['margin_used'] + pnl
            self.daily_pnl += pnl
            
            # Move to closed positions
            self.closed_positions.append(trade)
            del self.current_positions[trade['id']]
            
            # Update equity curve
            self.equity_curve.append({
                'timestamp': exit_time,
                'capital': self.current_capital,
                'pnl': pnl
            })
            
            # Update drawdown
            if self.current_capital > self.peak_capital:
                self.peak_capital = self.current_capital
            
            current_drawdown = (self.peak_capital - self.current_capital) / self.peak_capital * 100
            if current_drawdown > self.max_drawdown:
                self.max_drawdown = current_drawdown
                
        except Exception as e:
            self.logger.error(f"Error closing position: {e}")
    
    def _calculate_performance_metrics(self) -> Dict:
        """Calculate comprehensive performance metrics"""
        if not self.closed_positions:
            return self._get_empty_results()
        
        trades = self.closed_positions
        total_trades = len(trades)
        
        if total_trades == 0:
            return self._get_empty_results()
        
        # Basic metrics
        total_pnl = sum(trade['pnl'] for trade in trades)
        winning_trades = [trade for trade in trades if trade['pnl'] > 0]
        losing_trades = [trade for trade in trades if trade['pnl'] < 0]
        
        win_count = len(winning_trades)
        loss_count = len(losing_trades)
        win_rate = (win_count / total_trades) * 100 if total_trades > 0 else 0
        
        # P&L metrics
        avg_win = sum(trade['pnl'] for trade in winning_trades) / win_count if win_count > 0 else 0
        avg_loss = sum(trade['pnl'] for trade in losing_trades) / loss_count if loss_count > 0 else 0
        avg_trade = total_pnl / total_trades
        
        largest_win = max(trade['pnl'] for trade in winning_trades) if winning_trades else 0
        largest_loss = min(trade['pnl'] for trade in losing_trades) if losing_trades else 0
        
        # Return metrics
        final_capital = self.initial_capital + total_pnl
        total_return_pct = (total_pnl / self.initial_capital) * 100
        
        # Risk metrics
        profit_factor = abs(avg_win * win_count / (avg_loss * loss_count)) if avg_loss != 0 and loss_count > 0 else 0
        
        # Time-based metrics
        if trades:
            first_trade = min(trades, key=lambda x: x['entry_time'])
            last_trade = max(trades, key=lambda x: x['exit_time'] or x['entry_time'])
            backtest_duration = (last_trade['exit_time'] - first_trade['entry_time']).days
            
            # Annualized return
            years = max(backtest_duration / 365.25, 1/365.25)  # Minimum 1 day
            annualized_return = ((final_capital / self.initial_capital) ** (1/years) - 1) * 100
        else:
            backtest_duration = 0
            annualized_return = 0
        
        # Trade duration metrics
        durations = []
        for trade in trades:
            if trade['exit_time'] and trade['entry_time']:
                duration_hours = (trade['exit_time'] - trade['entry_time']).total_seconds() / 3600
                durations.append(duration_hours)
        
        avg_duration_hours = sum(durations) / len(durations) if durations else 0
        
        # Consecutive metrics
        consecutive_wins = 0
        consecutive_losses = 0
        max_consecutive_wins = 0
        max_consecutive_losses = 0
        
        for trade in trades:
            if trade['pnl'] > 0:
                consecutive_wins += 1
                consecutive_losses = 0
                max_consecutive_wins = max(max_consecutive_wins, consecutive_wins)
            else:
                consecutive_losses += 1
                consecutive_wins = 0
                max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
        
        # Exit reason statistics
        exit_reasons = {}
        for trade in trades:
            reason = trade.get('exit_reason', 'Unknown')
            exit_reasons[reason] = exit_reasons.get(reason, 0) + 1
        
        return {
            'summary': {
                'initial_capital': self.initial_capital,
                'final_capital': final_capital,
                'total_pnl': total_pnl,
                'total_return_pct': total_return_pct,
                'annualized_return_pct': annualized_return,
                'max_drawdown_pct': self.max_drawdown,
                'backtest_duration_days': backtest_duration
            },
            'trade_stats': {
                'total_trades': total_trades,
                'winning_trades': win_count,
                'losing_trades': loss_count,
                'win_rate_pct': win_rate,
                'avg_trade_pnl': avg_trade,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'largest_win': largest_win,
                'largest_loss': largest_loss,
                'profit_factor': profit_factor,
                'avg_duration_hours': avg_duration_hours,
                'max_consecutive_wins': max_consecutive_wins,
                'max_consecutive_losses': max_consecutive_losses,
                'exit_reasons': exit_reasons
            },
            'risk_metrics': {
                'max_drawdown_pct': self.max_drawdown,
                'sharpe_ratio': self._calculate_sharpe_ratio(),
                'calmar_ratio': annualized_return / max(self.max_drawdown, 1),
                'win_loss_ratio': abs(avg_win / avg_loss) if avg_loss != 0 else 0
            },
            'trades': [
                {
                    'id': trade['id'],
                    'side': trade['side'],
                    'entry_time': trade['entry_time'].isoformat(),
                    'exit_time': trade['exit_time'].isoformat() if trade['exit_time'] else None,
                    'entry_price': trade['entry_price'],
                    'exit_price': trade['exit_price'],
                    'position_size': trade['position_size'],
                    'pnl': trade['pnl'],
                    'signal_type': trade['signal_type'],
                    'signal_strength': trade['signal_strength'],
                    'exit_reason': trade.get('exit_reason', 'Unknown'),
                    'exit_method': trade.get('exit_method', 'Dollar'),
                    'atr_value': float(trade.get('atr_value', 0)),
                    'high_volume_entry': bool(trade.get('is_high_volume_entry', False))
                }
                for trade in trades
            ],
            'equity_curve': self.equity_curve,
            'settings_used': {
                'initial_capital': self.initial_capital,
                'position_size_usd': self.position_size_usd,
                'max_positions': self.max_positions,
                'leverage': self.leverage,
                'stop_loss_usd': self.stop_loss_usd,
                'take_profit_usd': self.take_profit_usd,
                'long_signal_threshold': self.long_signal_threshold,
                'short_signal_threshold': self.short_signal_threshold,
                'min_signal_strength': self.min_signal_strength,
                'atr_exits_enabled': self.atr_exits_enabled,
                'atr_period': self.atr_period,
                'stop_loss_atr_multiplier': self.stop_loss_atr_multiplier,
                'take_profit_atr_multiplier': self.take_profit_atr_multiplier,
                'trailing_atr_multiplier': self.trailing_atr_multiplier
            }
        }
    
    def _calculate_sharpe_ratio(self) -> float:
        """Calculate Sharpe ratio from trade returns"""
        if not self.closed_positions:
            return 0.0
        
        returns = [trade['pnl'] / self.initial_capital for trade in self.closed_positions]
        if len(returns) < 2:
            return 0.0
        
        avg_return = np.mean(returns)
        std_return = np.std(returns)
        
        if std_return == 0:
            return 0.0
        
        # Assuming risk-free rate of 0 for simplicity
        sharpe = avg_return / std_return
        return sharpe * np.sqrt(252)  # Annualized (assuming daily returns)
    
    def _get_empty_results(self) -> Dict:
        """Return empty results structure when no trades"""
        return {
            'summary': {
                'initial_capital': self.initial_capital,
                'final_capital': self.initial_capital,
                'total_pnl': 0.0,
                'total_return_pct': 0.0,
                'annualized_return_pct': 0.0,
                'max_drawdown_pct': 0.0,
                'backtest_duration_days': 0
            },
            'trade_stats': {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'win_rate_pct': 0.0,
                'avg_trade_pnl': 0.0,
                'avg_win': 0.0,
                'avg_loss': 0.0,
                'largest_win': 0.0,
                'largest_loss': 0.0,
                'profit_factor': 0.0,
                'avg_duration_hours': 0.0,
                'max_consecutive_wins': 0,
                'max_consecutive_losses': 0
            },
            'risk_metrics': {
                'max_drawdown_pct': 0.0,
                'sharpe_ratio': 0.0,
                'calmar_ratio': 0.0,
                'win_loss_ratio': 0.0
            },
            'trades': [],
            'equity_curve': [],
            'settings_used': {
                'initial_capital': self.initial_capital,
                'position_size_usd': self.position_size_usd,
                'max_positions': self.max_positions,
                'leverage': self.leverage,
                'stop_loss_usd': self.stop_loss_usd,
                'take_profit_usd': self.take_profit_usd,
                'long_signal_threshold': self.long_signal_threshold,
                'short_signal_threshold': self.short_signal_threshold,
                'min_signal_strength': self.min_signal_strength
            }
        }