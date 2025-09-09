import pandas as pd
import numpy as np
from datetime import datetime
import logging
from delta_exchange_client import DeltaExchangeClient
from logger_config import get_logger

class BTCMultiTimeframeStrategy:
    """
    Multi-timeframe BTC Options Trading Strategy
    - 1H timeframe: Trend identification using MA crossover (stronger signals)
    - 3m timeframe: Entry/exit signals with existing technical analysis
    - Trailing stop loss implementation
    - Only trade in direction of 1H trend
    """
    
    def __init__(self, api_key, api_secret, paper_trading=True):
        self.client = DeltaExchangeClient(api_key, api_secret, paper_trading)
        
        # Strategy parameters for 1-hour timeframe (Conservative for stability)
        self.trend_fast_ma = 50    # Fast SMA for trend (50 periods on 1h = ~2 days) 
        self.trend_slow_ma = 200   # Slow SMA for trend (200 periods on 1h = ~8 days)
        self.dow_periods = 30      # For higher highs/lower lows analysis (30 hours = 1.25 days)
        self.ha_smooth = 10        # Heikin Ashi smoothing periods (10 hours)
        
        # Trend persistence settings
        self.trend_confirmation_threshold = 3  # Require 3+ confirmations to change trend
        self.min_trend_strength_change = 2.0   # Minimum strength change to update trend
        
        # 3-minute signal parameters (adjusted for shorter timeframe)
        self.signal_bb_period = 40        # 40 periods on 3m = ~2 hours
        self.signal_atr_period = 28       # 28 periods on 3m = ~1.4 hours  
        self.signal_rsi_period = 14       # Standard RSI
        self.signal_volume_period = 40    # Volume SMA
        
        # Risk management
        self.max_risk_per_trade = 0.005   # 0.5% max risk
        self.trailing_stop_pct = 0.15     # 15% trailing stop
        self.min_signal_strength = 4      # Minimum signal confirmation
        
        # Trend state (1-hour timeframe)
        self.current_trend = 'NEUTRAL'    # BULLISH, BEARISH, NEUTRAL
        self.trend_strength = 0           # 0-10 scale
        self.ma_cross_signal = 'NONE'     # SMA crossover signal  
        self.dow_theory_trend = 'NONE'    # Dow theory trend
        self.heikin_ashi_trend = 'NONE'   # Heikin Ashi trend
        self.trend_timestamp = None       # When trend was last updated
        
        # Trend persistence tracking
        self.trend_cache = {
            'last_analysis_hour': None,    # Hour of last analysis
            'trend_confirmations': 0,      # How many confirmations current trend has
            'previous_trend': 'NEUTRAL',   # Previous trend for comparison
            'trend_start_time': None,      # When current trend started
            'last_candle_time': None       # Time of last processed 1H candle
        }
        
        # Setup logging using centralized configuration
        self.logger = get_logger(__name__, 'INFO', 'delta_btc_trading.log')
        
    def fetch_multi_timeframe_data(self):
        """Fetch data for both timeframes"""
        try:
            data = self.client.get_multi_timeframe_data()
            if data:
                self.trend_data_1h = self.process_candle_data(data['1h'])
                self.signal_data_3m = self.process_candle_data(data['3m'])
                return True
            return False
        except Exception as e:
            self.logger.error(f"Error fetching multi-timeframe data: {e}")
            return False
    
    def process_candle_data(self, candles):
        """Convert candle data to pandas DataFrame"""
        try:
            df = pd.DataFrame(candles)
            # Ensure we have the right column names (adjust based on Delta Exchange response format)
            if 'open' in df.columns:
                df = df.rename(columns={'open': 'Open', 'high': 'High', 'low': 'Low', 'close': 'Close', 'volume': 'Volume'})
            
            # Convert to numeric
            for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Sort by timestamp
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df = df.sort_values('timestamp').reset_index(drop=True)
            
            return df
        except Exception as e:
            self.logger.error(f"Error processing candle data: {e}")
            return pd.DataFrame()
    
    def calculate_sma(self, data, period):
        """Calculate Simple Moving Average for more stable trends"""
        return data.rolling(window=period).mean()
    
    def is_new_hour_candle(self):
        """Check if we have a new 1H candle since last analysis"""
        try:
            if self.trend_data_1h.empty:
                return False
            
            # Get the timestamp of the latest candle
            latest_candle = self.trend_data_1h.iloc[-1]
            if 'timestamp' not in self.trend_data_1h.columns:
                return True  # If no timestamp, assume new data
            
            current_candle_time = latest_candle['timestamp']
            
            # Check if this is a different hour than our last analysis
            if self.trend_cache['last_candle_time'] is None:
                self.trend_cache['last_candle_time'] = current_candle_time
                return True
            
            last_hour = self.trend_cache['last_candle_time'].hour
            current_hour = current_candle_time.hour
            
            if current_hour != last_hour:
                self.trend_cache['last_candle_time'] = current_candle_time
                return True
            
            return False  # Same hour, no new candle
            
        except Exception as e:
            self.logger.warning(f"Error checking for new candle: {e}")
            return True  # Default to allowing update on error
    
    def apply_trend_persistence(self, proposed_trend, proposed_strength):
        """Apply trend persistence logic - require strong confirmation to change trends"""
        try:
            previous_trend = self.current_trend
            
            # If proposed trend is same as current, just update strength if significant change
            if proposed_trend == self.current_trend:
                if abs(proposed_strength - self.trend_strength) >= self.min_trend_strength_change:
                    self.trend_strength = proposed_strength
                    self.trend_cache['trend_confirmations'] += 1
                    self.logger.debug(f"Trend strength updated: {self.trend_strength:.1f}")
                return False  # No trend change
            
            # If we're in NEUTRAL and getting a directional signal, be more receptive
            if self.current_trend == 'NEUTRAL' and proposed_trend in ['BULLISH', 'BEARISH']:
                if proposed_strength >= 6:  # Require strong signal to start new trend
                    self.current_trend = proposed_trend
                    self.trend_strength = proposed_strength
                    self.trend_cache['trend_confirmations'] = 1
                    self.trend_cache['trend_start_time'] = datetime.now().isoformat()
                    self.logger.info(f"New trend established: {self.current_trend} (strength: {self.trend_strength:.1f})")
                    return True
                else:
                    self.logger.debug(f"Insufficient strength to establish new trend: {proposed_strength:.1f} < 6.0")
                    return False
            
            # If we're changing from one directional trend to another, require very strong confirmation
            if (self.current_trend in ['BULLISH', 'BEARISH'] and 
                proposed_trend in ['BULLISH', 'BEARISH'] and 
                proposed_trend != self.current_trend):
                
                # Require multiple confirmations and strong signal for trend reversal
                if (proposed_strength >= 8 and 
                    self.trend_cache['trend_confirmations'] >= self.trend_confirmation_threshold):
                    
                    self.logger.info(
                        f"TREND REVERSAL: {self.current_trend} -> {proposed_trend} | "
                        f"Strength: {proposed_strength:.1f} | Confirmations: {self.trend_cache['trend_confirmations']}"
                    )
                    
                    self.current_trend = proposed_trend
                    self.trend_strength = proposed_strength
                    self.trend_cache['trend_confirmations'] = 1  # Reset confirmations
                    self.trend_cache['trend_start_time'] = datetime.now().isoformat()
                    return True
                else:
                    # Trend reversal rejected - increment confirmations for next time
                    self.trend_cache['trend_confirmations'] += 1
                    self.logger.info(
                        f"Trend reversal rejected: {self.current_trend} -> {proposed_trend} | "
                        f"Need strength ≥8 (got {proposed_strength:.1f}) and {self.trend_confirmation_threshold} confirmations "
                        f"(got {self.trend_cache['trend_confirmations']})"
                    )
                    return False
            
            # If moving from directional trend to NEUTRAL, require confirmation
            if (self.current_trend in ['BULLISH', 'BEARISH'] and proposed_trend == 'NEUTRAL'):
                if self.trend_cache['trend_confirmations'] >= 2:  # Less strict for going to neutral
                    self.logger.info(f"Trend neutralized: {self.current_trend} -> NEUTRAL")
                    self.current_trend = 'NEUTRAL'
                    self.trend_strength = proposed_strength
                    self.trend_cache['trend_confirmations'] = 0
                    return True
                else:
                    self.trend_cache['trend_confirmations'] += 1
                    self.logger.debug(f"Trend neutralization pending: {self.trend_cache['trend_confirmations']}/2 confirmations")
                    return False
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error in trend persistence logic: {e}")
            # Fallback to simple update on error
            self.current_trend = proposed_trend
            self.trend_strength = proposed_strength
            return True
    
    def calculate_heikin_ashi(self, df):
        """Calculate Heikin Ashi candles"""
        ha_df = df.copy()
        
        # HA Close = (O + H + L + C) / 4
        ha_df['HA_Close'] = (df['Open'] + df['High'] + df['Low'] + df['Close']) / 4
        
        # HA Open = (previous HA Open + previous HA Close) / 2
        ha_df['HA_Open'] = 0.0
        ha_df.iloc[0, ha_df.columns.get_loc('HA_Open')] = df.iloc[0]['Open']
        
        for i in range(1, len(ha_df)):
            ha_df.iloc[i, ha_df.columns.get_loc('HA_Open')] = (
                ha_df.iloc[i-1]['HA_Open'] + ha_df.iloc[i-1]['HA_Close']
            ) / 2
        
        # HA High = max(High, HA Open, HA Close)
        ha_df['HA_High'] = ha_df[['High', 'HA_Open', 'HA_Close']].max(axis=1)
        
        # HA Low = min(Low, HA Open, HA Close)
        ha_df['HA_Low'] = ha_df[['Low', 'HA_Open', 'HA_Close']].min(axis=1)
        
        return ha_df
    
    def analyze_dow_theory(self, df):
        """Analyze trend using Dow Theory - Higher Highs/Lower Lows"""
        try:
            if len(df) < self.dow_periods * 2:
                return 'INSUFFICIENT_DATA'
            
            # Find recent highs and lows
            recent_data = df.tail(self.dow_periods)
            
            # Calculate rolling max/min for pivot identification
            window = 3
            recent_data = recent_data.copy()
            recent_data['High_Roll'] = recent_data['High'].rolling(window=window, center=True).max()
            recent_data['Low_Roll'] = recent_data['Low'].rolling(window=window, center=True).min()
            
            # Identify higher highs and higher lows (bullish)
            highs = recent_data[recent_data['High'] == recent_data['High_Roll']]['High'].dropna()
            lows = recent_data[recent_data['Low'] == recent_data['Low_Roll']]['Low'].dropna()
            
            if len(highs) >= 2 and len(lows) >= 2:
                # Check for higher highs and higher lows
                higher_highs = all(highs.iloc[i] > highs.iloc[i-1] for i in range(1, min(3, len(highs))))
                higher_lows = all(lows.iloc[i] > lows.iloc[i-1] for i in range(1, min(3, len(lows))))
                
                # Check for lower highs and lower lows  
                lower_highs = all(highs.iloc[i] < highs.iloc[i-1] for i in range(1, min(3, len(highs))))
                lower_lows = all(lows.iloc[i] < lows.iloc[i-1] for i in range(1, min(3, len(lows))))
                
                if higher_highs and higher_lows:
                    return 'BULLISH'
                elif lower_highs and lower_lows:
                    return 'BEARISH'
                else:
                    return 'SIDEWAYS'
            
            return 'NEUTRAL'
            
        except Exception as e:
            self.logger.warning(f"Error in Dow Theory analysis: {e}")
            return 'ERROR'
    
    def analyze_heikin_ashi_trend(self, ha_df):
        """Analyze trend using Heikin Ashi candles"""
        try:
            if len(ha_df) < self.ha_smooth:
                return 'INSUFFICIENT_DATA'
            
            # Look at recent candles
            recent_candles = ha_df.tail(self.ha_smooth)
            
            bullish_candles = 0
            bearish_candles = 0
            
            for _, candle in recent_candles.iterrows():
                if candle['HA_Close'] > candle['HA_Open']:
                    bullish_candles += 1
                elif candle['HA_Close'] < candle['HA_Open']:
                    bearish_candles += 1
            
            # Determine trend based on candle dominance
            if bullish_candles >= (self.ha_smooth * 0.7):  # 70% or more bullish
                return 'BULLISH'
            elif bearish_candles >= (self.ha_smooth * 0.7):  # 70% or more bearish
                return 'BEARISH'
            else:
                return 'SIDEWAYS'
                
        except Exception as e:
            self.logger.warning(f"Error in Heikin Ashi analysis: {e}")
            return 'ERROR'
    
    def analyze_trend_1h(self):
        """Analyze trend using 1-hour timeframe with conservative parameters and trend persistence"""
        try:
            if self.trend_data_1h.empty:
                return
            
            # Only update trend analysis on new 1H candle close
            if not self.is_new_hour_candle():
                self.logger.debug("No new 1H candle, skipping trend analysis")
                return
            
            self.logger.info("New 1H candle detected - performing trend analysis")
            
            # Calculate SMAs for more stable trend identification
            self.trend_data_1h['SMA_Fast'] = self.calculate_sma(self.trend_data_1h['Close'], self.trend_fast_ma)
            self.trend_data_1h['SMA_Slow'] = self.calculate_sma(self.trend_data_1h['Close'], self.trend_slow_ma)
            
            # Calculate Heikin Ashi candles
            ha_data = self.calculate_heikin_ashi(self.trend_data_1h)
            
            # Get latest values
            latest = self.trend_data_1h.iloc[-1]
            prev = self.trend_data_1h.iloc[-2] if len(self.trend_data_1h) > 1 else latest
            
            fast_sma = latest['SMA_Fast']
            slow_sma = latest['SMA_Slow']
            prev_fast_sma = prev['SMA_Fast']
            prev_slow_sma = prev['SMA_Slow']
            
            # Update timestamp
            self.trend_timestamp = datetime.now().isoformat()
            
            # 1. SMA Crossover Analysis (More stable than EMA)
            if pd.isna(fast_sma) or pd.isna(slow_sma):
                self.ma_cross_signal = 'INSUFFICIENT_DATA'
                base_strength = 0
            elif fast_sma > slow_sma:
                if prev_fast_sma <= prev_slow_sma:  # Fresh bullish crossover
                    self.ma_cross_signal = 'BULLISH_CROSSOVER'
                    base_strength = 9  # Slightly lower than EMA for conservative approach
                    self.logger.info("STRONG BULLISH SMA CROSSOVER detected on 1H!")
                else:
                    self.ma_cross_signal = 'BULLISH'
                    # More conservative strength calculation
                    separation_pct = ((fast_sma - slow_sma) / slow_sma) * 100
                    base_strength = min(8, max(4, separation_pct * 50))  # Reduced multiplier
            elif fast_sma < slow_sma:
                if prev_fast_sma >= prev_slow_sma:  # Fresh bearish crossover
                    self.ma_cross_signal = 'BEARISH_CROSSOVER'
                    base_strength = 9
                    self.logger.info("STRONG BEARISH SMA CROSSOVER detected on 1H!")
                else:
                    self.ma_cross_signal = 'BEARISH'
                    separation_pct = ((slow_sma - fast_sma) / slow_sma) * 100
                    base_strength = min(8, max(4, separation_pct * 50))
            else:
                self.ma_cross_signal = 'NEUTRAL'
                base_strength = 0
            
            # 2. Dow Theory Analysis
            self.dow_theory_trend = self.analyze_dow_theory(self.trend_data_1h)
            
            # 3. Heikin Ashi Analysis
            self.heikin_ashi_trend = self.analyze_heikin_ashi_trend(ha_data)
            
            # 4. Combine all signals for final trend determination
            bullish_signals = 0
            bearish_signals = 0
            
            # SMA vote
            if 'BULLISH' in self.ma_cross_signal:
                bullish_signals += 2 if 'CROSSOVER' in self.ma_cross_signal else 1
            elif 'BEARISH' in self.ma_cross_signal:
                bearish_signals += 2 if 'CROSSOVER' in self.ma_cross_signal else 1
            
            # Dow Theory vote
            if self.dow_theory_trend == 'BULLISH':
                bullish_signals += 1
            elif self.dow_theory_trend == 'BEARISH':
                bearish_signals += 1
            
            # Heikin Ashi vote
            if self.heikin_ashi_trend == 'BULLISH':
                bullish_signals += 1
            elif self.heikin_ashi_trend == 'BEARISH':
                bearish_signals += 1
            
            # Calculate proposed trend based on signals
            proposed_trend = 'NEUTRAL'
            proposed_strength = base_strength / 2
            
            if bullish_signals > bearish_signals:
                proposed_trend = 'BULLISH'
                proposed_strength = min(10, base_strength + (bullish_signals - bearish_signals))
            elif bearish_signals > bullish_signals:
                proposed_trend = 'BEARISH'
                proposed_strength = min(10, base_strength + (bearish_signals - bullish_signals))
            
            # Apply trend persistence logic
            trend_changed = self.apply_trend_persistence(proposed_trend, proposed_strength)
            
            # Price position confirmation (only if trend wasn't rejected by persistence logic)
            if trend_changed or self.current_trend != 'NEUTRAL':
                current_price = latest['Close']
                if current_price > fast_sma > slow_sma and self.current_trend == 'BULLISH':
                    self.trend_strength = min(10, self.trend_strength + 0.5)  # Smaller boost
                elif current_price < fast_sma < slow_sma and self.current_trend == 'BEARISH':
                    self.trend_strength = min(10, self.trend_strength + 0.5)
            
            # Log comprehensive trend analysis
            status = "CHANGED" if trend_changed else "MAINTAINED"
            confirmations = self.trend_cache['trend_confirmations']
            
            self.logger.info(
                f"1H Trend Analysis - {status}: {self.current_trend} ({self.trend_strength:.1f}/10) | "
                f"Confirmations: {confirmations} | SMA: {self.ma_cross_signal} | "
                f"Dow: {self.dow_theory_trend} | HA: {self.heikin_ashi_trend} | "
                f"SMA Fast: {fast_sma:.2f} | SMA Slow: {slow_sma:.2f}"
            )
            
        except Exception as e:
            self.logger.error(f"Error analyzing 1H trend: {e}")
            self.current_trend = 'NEUTRAL'
            self.trend_strength = 0
            self.ma_cross_signal = 'ERROR'
            self.dow_theory_trend = 'ERROR'
            self.heikin_ashi_trend = 'ERROR'
    
    def calculate_3m_indicators(self):
        """Calculate technical indicators on 3-minute data"""
        try:
            if self.signal_data_3m.empty:
                return
            
            # Bollinger Bands
            self.signal_data_3m['BB_Middle'] = self.signal_data_3m['Close'].rolling(window=self.signal_bb_period).mean()
            rolling_std = self.signal_data_3m['Close'].rolling(window=self.signal_bb_period).std()
            self.signal_data_3m['BB_Upper'] = self.signal_data_3m['BB_Middle'] + (rolling_std * 2.0)
            self.signal_data_3m['BB_Lower'] = self.signal_data_3m['BB_Middle'] - (rolling_std * 2.0)
            self.signal_data_3m['BB_Position'] = (self.signal_data_3m['Close'] - self.signal_data_3m['BB_Lower']) / (self.signal_data_3m['BB_Upper'] - self.signal_data_3m['BB_Lower'])
            self.signal_data_3m['BB_Width'] = (self.signal_data_3m['BB_Upper'] - self.signal_data_3m['BB_Lower']) / self.signal_data_3m['BB_Middle']
            
            # ATR
            high = self.signal_data_3m['High']
            low = self.signal_data_3m['Low'] 
            close = self.signal_data_3m['Close']
            
            tr1 = high - low
            tr2 = abs(high - close.shift(1))
            tr3 = abs(low - close.shift(1))
            
            true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            self.signal_data_3m['ATR'] = true_range.rolling(window=self.signal_atr_period).mean()
            self.signal_data_3m['ATR_Pct'] = self.signal_data_3m['ATR'] / self.signal_data_3m['Close']
            
            # RSI
            delta = self.signal_data_3m['Close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=self.signal_rsi_period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=self.signal_rsi_period).mean()
            rs = gain / loss
            self.signal_data_3m['RSI'] = 100 - (100 / (1 + rs))
            
            # Volume indicators
            self.signal_data_3m['Volume_SMA'] = self.signal_data_3m['Volume'].rolling(window=self.signal_volume_period).mean()
            self.signal_data_3m['Volume_Ratio'] = self.signal_data_3m['Volume'] / self.signal_data_3m['Volume_SMA']
            
            
        except Exception as e:
            self.logger.error(f"Error calculating 3m indicators: {e}")
    
    def generate_3m_signals(self):
        """Generate entry signals on 3-minute timeframe (only in 1H trend direction)"""
        try:
            if self.signal_data_3m.empty:
                return {'signal': 0, 'strength': 0, 'type': 'NONE', 'reason': 'No data'}
            
            latest = self.signal_data_3m.iloc[-1]
            
            # Generate signals in 1H trend direction
            if self.current_trend == 'BULLISH':
                return self._generate_bullish_signal(latest)
            elif self.current_trend == 'BEARISH':
                return self._generate_bearish_signal(latest)
            else:  # NEUTRAL - evaluate both signals
                bullish = self._generate_bullish_signal(latest)
                bearish = self._generate_bearish_signal(latest)
                # Return the stronger signal
                if abs(bullish.get('strength', 0)) >= abs(bearish.get('strength', 0)):
                    return bullish
                else:
                    return bearish
            
            return {'signal': 0, 'strength': 0, 'type': 'NONE', 'reason': 'No signals generated'}
            
        except Exception as e:
            self.logger.error(f"Error generating 3m signals: {e}")
            return {'signal': 0, 'strength': 0, 'type': 'NONE', 'reason': 'Error'}
    
    def _generate_bullish_signal(self, latest):
        """Generate CALL signal when 1H trend is bullish"""
        signal_strength = 0
        reasons = []
        
        # Only look for CALL opportunities in 1H bullish trend
        
        # 1. Oversold on BB (pullback in uptrend)
        if latest['BB_Position'] < 0.3:
            signal_strength += 2
            reasons.append("Oversold pullback")
        elif latest['BB_Position'] < 0.4:
            signal_strength += 1
            reasons.append("Mild oversold")
        
        # 2. RSI oversold but turning up (bullish divergence)
        if latest['RSI'] < 35:
            signal_strength += 2
            reasons.append("RSI oversold")
        elif latest['RSI'] < 45:
            signal_strength += 1
            reasons.append("RSI below midline")
        
        # 3. High volume (confirmation)
        if latest['Volume_Ratio'] > 1.5:
            signal_strength += 2
            reasons.append("High volume")
        elif latest['Volume_Ratio'] > 1.2:
            signal_strength += 1
            reasons.append("Above avg volume")
        
        # 4. ATR expansion (volatility increase)
        if latest['ATR_Pct'] > 0.02:  # 2% ATR
            signal_strength += 1
            reasons.append("ATR expansion")
        
        # 5. Trend strength bonus
        trend_bonus = int(self.trend_strength / 3)  # 0-3 bonus points
        signal_strength += trend_bonus
        if trend_bonus > 0:
            reasons.append(f"Strong trend (+{trend_bonus})")
        
        return {
            'signal': 1 if signal_strength >= self.min_signal_strength else 0,
            'strength': signal_strength,
            'type': 'CALL',
            'reasons': reasons,
            'bb_position': latest['BB_Position'],
            'rsi': latest['RSI'],
            'volume_ratio': latest['Volume_Ratio'],
            'atr_pct': latest['ATR_Pct']
        }
    
    def _generate_bearish_signal(self, latest):
        """Generate PUT signal when 1H trend is bearish"""
        signal_strength = 0
        reasons = []
        
        # Only look for PUT opportunities in 1H bearish trend
        
        # 1. Overbought on BB (pullback in downtrend)
        if latest['BB_Position'] > 0.7:
            signal_strength += 2
            reasons.append("Overbought pullback")
        elif latest['BB_Position'] > 0.6:
            signal_strength += 1
            reasons.append("Mild overbought")
        
        # 2. RSI overbought but turning down
        if latest['RSI'] > 65:
            signal_strength += 2
            reasons.append("RSI overbought")
        elif latest['RSI'] > 55:
            signal_strength += 1
            reasons.append("RSI above midline")
        
        # 3. High volume (confirmation)
        if latest['Volume_Ratio'] > 1.5:
            signal_strength += 2
            reasons.append("High volume")
        elif latest['Volume_Ratio'] > 1.2:
            signal_strength += 1
            reasons.append("Above avg volume")
        
        # 4. ATR expansion
        if latest['ATR_Pct'] > 0.02:
            signal_strength += 1
            reasons.append("ATR expansion")
        
        # 5. Trend strength bonus
        trend_bonus = int(self.trend_strength / 3)
        signal_strength += trend_bonus
        if trend_bonus > 0:
            reasons.append(f"Strong trend (+{trend_bonus})")
        
        return {
            'signal': -1 if signal_strength >= self.min_signal_strength else 0,
            'strength': signal_strength,
            'type': 'PUT',
            'reasons': reasons,
            'bb_position': latest['BB_Position'],
            'rsi': latest['RSI'],
            'volume_ratio': latest['Volume_Ratio'],
            'atr_pct': latest['ATR_Pct']
        }
    
    def calculate_trailing_stop(self, entry_price, current_price, signal_type, highest_price=None, lowest_price=None):
        """Calculate trailing stop loss"""
        try:
            if signal_type == 'CALL':
                # For CALL options, trail from the highest price achieved
                if highest_price is None:
                    highest_price = max(entry_price, current_price)
                else:
                    highest_price = max(highest_price, current_price)
                
                # Trailing stop is X% below highest price
                trailing_stop = highest_price * (1 - self.trailing_stop_pct)
                
                return {
                    'stop_price': trailing_stop,
                    'highest_price': highest_price,
                    'should_exit': current_price <= trailing_stop
                }
            
            elif signal_type == 'PUT':
                # For PUT options, trail from the lowest price achieved  
                if lowest_price is None:
                    lowest_price = min(entry_price, current_price)
                else:
                    lowest_price = min(lowest_price, current_price)
                
                # Trailing stop is X% above lowest price
                trailing_stop = lowest_price * (1 + self.trailing_stop_pct)
                
                return {
                    'stop_price': trailing_stop,
                    'lowest_price': lowest_price,
                    'should_exit': current_price >= trailing_stop
                }
                
        except Exception as e:
            self.logger.error(f"Error calculating trailing stop: {e}")
            return None
    
    def run_analysis(self):
        """Main analysis function - run this periodically"""
        try:
            self.logger.info("=== Running Multi-Timeframe Analysis ===")
            
            # 1. Fetch data for both timeframes
            if not self.fetch_multi_timeframe_data():
                self.logger.error("Failed to fetch multi-timeframe data")
                return None
            
            # 2. Analyze 1H trend (stronger trend signals)
            self.analyze_trend_1h()
            
            # 3. Calculate 3m indicators
            self.calculate_3m_indicators()
            
            # 4. Generate 3m signals (only in trend direction)
            signal_result = self.generate_3m_signals()
            
            # 5. Return complete analysis
            analysis = {
                'timestamp': datetime.now().isoformat(),
                'trend_1h': {
                    'direction': self.current_trend,
                    'strength': self.trend_strength,
                    'sma_cross': self.ma_cross_signal,
                    'dow_theory': self.dow_theory_trend,
                    'heikin_ashi': self.heikin_ashi_trend,
                    'updated_at': self.trend_timestamp,
                    'confirmations': self.trend_cache['trend_confirmations'],
                    'trend_start_time': self.trend_cache['trend_start_time'],
                    'last_candle_time': str(self.trend_cache['last_candle_time']) if self.trend_cache['last_candle_time'] else None
                },
                'signal_3m': signal_result,
                'current_price': self.signal_data_3m.iloc[-1]['Close'] if not self.signal_data_3m.empty else 0
            }
            
            self.logger.info(f"Multi-timeframe analysis complete: 1H Trend={self.current_trend} ({self.trend_strength:.1f}/10), 3m Signal={signal_result['type']} ({signal_result['strength']}/10)")
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error in multi-timeframe analysis: {e}")
            return None

# Example usage
if __name__ == "__main__":
    # This would use your actual API credentials
    strategy = BTCMultiTimeframeStrategy("your_api_key", "your_api_secret", paper_trading=True)
    
    # Run analysis
    result = strategy.run_analysis()
    if result:
        print("Multi-timeframe Analysis Result:")
        print(f"1H Trend: {result['trend_1h']['direction']} (Strength: {result['trend_1h']['strength']}/10)")
        print(f"3m Signal: {result['signal_3m']['type']} (Strength: {result['signal_3m']['strength']}/10)")
        if result['signal_3m']['reasons']:
            print(f"Reasons: {', '.join(result['signal_3m']['reasons'])}")