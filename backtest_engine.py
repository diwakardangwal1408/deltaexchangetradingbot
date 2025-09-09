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
        self.max_risk_usd = float(dollar_risk.get('max_risk_usd', 150.0))
        
        # Setup logging first
        self.logger = logging.getLogger(__name__)
        
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
            
            # For backtesting, we'll fetch a large amount of historical data
            # The current API doesn't support date range filtering, so we'll fetch recent data
            
            # Calculate how many candles we need based on date range
            days_diff = (datetime.strptime(to_date, '%Y-%m-%d') - datetime.strptime(from_date, '%Y-%m-%d')).days
            hours_needed = days_diff * 24
            candles_3m_needed = hours_needed * 20  # 20 3-minute candles per hour
            
            # Fetch 1h data for trend analysis (limit to reasonable amount)
            data_1h = self.delta_client.get_historical_candles(
                symbol='BTCUSD',
                resolution='1h',
                count=min(hours_needed, 500)  # API limit consideration
            )
            
            if progress_callback:
                progress_callback({"step": "Fetching 3m data", "progress": 50})
            
            # Fetch 3m data for signal generation (limit to reasonable amount) 
            data_3m = self.delta_client.get_historical_candles(
                symbol='BTCUSD',
                resolution='3m',
                count=min(candles_3m_needed, 1000)  # API limit consideration
            )
            
            if progress_callback:
                progress_callback({"step": "Processing data", "progress": 80})
            
            # Process data into DataFrames
            df_1h = self._process_candle_data(data_1h)
            df_3m = self._process_candle_data(data_3m)
            
            self.logger.info(f"Fetched historical data: {len(df_1h)} 1h candles, {len(df_3m)} 3m candles")
            
            if progress_callback:
                progress_callback({"step": "Data ready", "progress": 100})
            
            return {
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
            self.reset_backtest()
            
            df_1h = historical_data['1h']
            df_3m = historical_data['3m']
            
            if df_1h.empty or df_3m.empty:
                raise ValueError("Historical data is empty")
            
            # Set strategy data
            self.strategy.trend_data_1h = df_1h
            self.strategy.signal_data_3m = df_3m
            
            if progress_callback:
                progress_callback({"step": "Analyzing trends", "progress": 10})
            
            # Calculate indicators for entire period
            self._calculate_indicators(df_1h, df_3m)
            
            if progress_callback:
                progress_callback({"step": "Generating signals", "progress": 30})
            
            # Generate all signals
            signals = self._generate_all_signals(df_3m)
            
            if progress_callback:
                progress_callback({"step": "Simulating trades", "progress": 50})
            
            # Simulate trading
            total_signals = len(signals)
            for i, signal in enumerate(signals):
                self._process_signal(signal, df_3m)
                
                if progress_callback and i % 10 == 0:  # Update every 10 signals
                    progress = 50 + (i / total_signals) * 40
                    progress_callback({"step": f"Processing signal {i+1}/{total_signals}", "progress": int(progress)})
            
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
            raise e
    
    def _calculate_indicators(self, df_1h: pd.DataFrame, df_3m: pd.DataFrame):
        """Calculate technical indicators for historical data"""
        # The strategy will calculate indicators when we call analyze methods
        # No need to call internal methods that don't exist
        pass
    
    def _generate_all_signals(self, df_3m: pd.DataFrame) -> List[Dict]:
        """Generate trading signals for entire historical period"""
        signals = []
        
        # Iterate through 3m candles to generate signals
        for i in range(len(df_3m)):
            if i < 50:  # Need minimum data for indicators
                continue
                
            # Feed current data slice to strategy
            current_3m_data = df_3m.iloc[:i+1].copy()
            
            # Update strategy with current data
            try:
                # Set the 3m data in strategy
                self.strategy.signal_data_3m = current_3m_data
                
                # Calculate indicators for current data slice
                self.strategy.calculate_3m_indicators()
                
                # Also update 1h trend for context (use simple logic for backtesting)
                if len(current_3m_data) >= 20:  # Need sufficient data
                    recent_closes = current_3m_data['Close'].tail(20)
                    price_trend = recent_closes.pct_change().sum()
                    
                    if price_trend > 0.02:  # 2% upward trend
                        self.strategy.current_trend = 'BULLISH'
                    elif price_trend < -0.02:  # 2% downward trend  
                        self.strategy.current_trend = 'BEARISH'
                    else:
                        self.strategy.current_trend = 'NEUTRAL'
                
                # Generate signal with updated data
                signal_data = self.strategy.generate_3m_signals()
                
                
                if signal_data and signal_data.get('signal', 0) != 0:
                    signal_info = {
                        'timestamp': df_3m.index[i],
                        'price': df_3m.iloc[i]['Close'],
                        'signal': signal_data['signal'],
                        'strength': signal_data['strength'],
                        'type': signal_data['type'],
                        'reason': signal_data.get('reason', ''),
                        'index': i
                    }
                    signals.append(signal_info)
                    
            except Exception as e:
                self.logger.error(f"Error generating signal at index {i}: {e}")
                continue
        
        return signals
    
    def _process_signal(self, signal: Dict, df_3m: pd.DataFrame):
        """Process a trading signal and simulate trade execution"""
        try:
            # Check if we can take a new position
            if not self._can_open_position(signal):
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
                
        except Exception as e:
            self.logger.error(f"Error processing signal: {e}")
    
    def _can_open_position(self, signal: Dict) -> bool:
        """Check if we can open a new position based on risk rules"""
        # Max positions check
        if len(self.current_positions) >= self.max_positions:
            return False
        
        # Time between trades check
        if (self.last_trade_time and 
            (signal['timestamp'] - self.last_trade_time).total_seconds() < self.min_time_between_trades):
            return False
        
        # Daily loss check
        if self.daily_pnl <= -self.max_daily_loss:
            return False
        
        # Signal strength check
        if abs(signal['strength']) < self.min_signal_strength:
            return False
        
        # Signal threshold check
        if signal['signal'] > 0 and signal['strength'] < self.long_signal_threshold:
            return False
        if signal['signal'] < 0 and signal['strength'] > self.short_signal_threshold:
            return False
        
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
            trade_id = f"bt_{len(self.trades_history)}_{int(entry_time.timestamp())}"
            
            # Calculate stop loss and take profit levels
            if signal['signal'] > 0:  # Long position
                stop_loss = entry_price - (self.stop_loss_usd / position_size)
                take_profit = entry_price + (self.take_profit_usd / position_size)
                side = 'long'
            else:  # Short position
                stop_loss = entry_price + (self.stop_loss_usd / position_size)
                take_profit = entry_price - (self.take_profit_usd / position_size)
                side = 'short'
            
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
                'margin_used': margin_used,
                'signal_strength': signal['strength'],
                'signal_type': signal['type'],
                'status': 'open',
                'high_water_mark': entry_price if side == 'long' else entry_price,
                'low_water_mark': entry_price if side == 'short' else entry_price,
                'exit_price': None,
                'exit_time': None,
                'pnl': 0.0,
                'fees': margin_used * 0.001  # 0.1% fee
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
                'max_consecutive_losses': max_consecutive_losses
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
                    'exit_reason': trade.get('exit_reason', 'Unknown')
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
                'min_signal_strength': self.min_signal_strength
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