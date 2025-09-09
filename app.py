from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
import json
import os
import subprocess
import threading
import time
from datetime import datetime, timezone, timedelta
import asyncio
import logging
import pandas as pd
import numpy as np
from delta_exchange_client import DeltaExchangeClient
from delta_btc_strategy import DeltaBTCOptionsTrader
from btc_multi_timeframe_strategy import BTCMultiTimeframeStrategy
from config_manager import config_manager
from logger_config import get_logger
from candle_timing import is_candle_closed, get_last_candle_close_time, seconds_until_next_candle_close, format_candle_times_display

# Global indicator caches based on proper candle timing
_indicator_cache = {
    '3m': {
        'data': None,
        'last_candle_close': None,
        'calculated_at': None
    },
    '1h': {
        'data': None,
        'analysis_result': None,
        'last_candle_close': None,
        'calculated_at': None
    }
}

app = Flask(__name__)
app.secret_key = 'your-secret-key-change-this'  # Change this to a random secret key

# Initialize logger for Flask app
try:
    config = config_manager.get_all_config()
    logging_config = config.get('logging', {})
    console_level = logging_config.get('console_level', 'INFO')
    log_file = logging_config.get('log_file', 'delta_btc_trading.log')
    app_logger = get_logger('flask_app', console_level, log_file)
except Exception as e:
    app_logger = get_logger('flask_app', 'INFO', 'delta_btc_trading.log')
    app_logger.warning(f"Could not load logging config, using defaults: {e}")

# Global variables for bot management
trading_bot_process = None
bot_running = False

# Global client cache for reuse
_cached_delta_client = None
_cached_strategy = None

def get_real_time_market_timestamp():
    """Get current market timestamp from Delta Exchange ticker API"""
    try:
        delta_client = get_delta_client()
        response = delta_client._make_public_request('/v2/tickers/BTCUSD')
        ticker = response.get('result')
        if ticker and ticker.get('timestamp'):
            # Convert microsecond timestamp to datetime
            ticker_ts_sec = ticker['timestamp'] / 1000000
            return datetime.fromtimestamp(ticker_ts_sec).isoformat()
    except Exception as e:
        logging.warning(f"Failed to get real-time market timestamp: {e}")
    
    # Fallback to current system time
    return datetime.now().isoformat()

def get_delta_client():
    """Get Delta Exchange client instance (cached for reuse)"""
    global _cached_delta_client
    
    if _cached_delta_client is None:
        config = config_manager.get_all_config()
        _cached_delta_client = DeltaExchangeClient(
            api_key=config['api_key'],
            api_secret=config['api_secret'],
            paper_trading=config.get('paper_trading', False)
        )
    
    return _cached_delta_client

def get_strategy():
    """Get BTCMultiTimeframeStrategy instance (cached for reuse)"""
    global _cached_strategy
    
    if _cached_strategy is None:
        config = config_manager.get_all_config()
        _cached_strategy = BTCMultiTimeframeStrategy(
            api_key=config['api_key'],
            api_secret=config['api_secret'],
            paper_trading=config.get('paper_trading', False)
        )
    
    return _cached_strategy

def get_cached_indicators(timeframe: str):
    """Get cached indicators, only recalculate when candle actually closes"""
    global _indicator_cache
    
    try:
        interval_minutes = 3 if timeframe == '3m' else 60
        cache_key = timeframe
        
        # Check if candle has closed since last calculation
        last_candle_close = get_last_candle_close_time(interval_minutes)
        
        cache = _indicator_cache[cache_key]
        need_recalculation = (
            cache['last_candle_close'] != last_candle_close or
            cache['data'] is None
        )
        
        if need_recalculation:
            logging.info(f"{timeframe.upper()} Cache UPDATE: New candle closed at {last_candle_close}")
            
            # Get fresh data
            df_raw = get_btc_data(timeframe=timeframe)
            if df_raw is None or len(df_raw) == 0:
                return None
            
            if timeframe == '3m':
                # For 3M, we store the raw data with technical indicators already calculated
                cache['data'] = df_raw
                cache['last_candle_close'] = last_candle_close
                cache['calculated_at'] = datetime.now()
                return df_raw
            
            elif timeframe == '1h':
                # For 1H, we need to calculate higher timeframe analysis
                config = load_config()
                futures_config = config.get('futures_strategy', {})
                trend_bullish_threshold = futures_config.get('trend_bullish_threshold', 3)
                trend_bearish_threshold = futures_config.get('trend_bearish_threshold', -3)
                
                # Calculate higher timeframe indicators
                df_1h_analyzed = calculate_higher_timeframe_indicators(
                    df_raw, 
                    trend_bullish_threshold, 
                    trend_bearish_threshold
                )
                
                # Update cache
                cache['data'] = df_raw
                cache['analysis_result'] = df_1h_analyzed
                cache['last_candle_close'] = last_candle_close
                cache['calculated_at'] = datetime.now()
                
                return df_1h_analyzed
        else:
            logging.info(f"{timeframe.upper()} Cache HIT: Using cached data from {cache['calculated_at']}")
            if timeframe == '3m':
                return cache['data']
            else:
                return cache['analysis_result']
                
    except Exception as e:
        logging.error(f"Error in get_cached_indicators({timeframe}): {e}")
        return None

def get_btc_data(timeframe='5m'):
    """Get BTC data and calculate indicators"""
    try:
        delta_client = get_delta_client()
        
        # Get historical data directly from Delta Exchange
        if timeframe == '3m':
            interval = '3m'
        elif timeframe == '5m':
            interval = '5m'
        elif timeframe == '1h':
            interval = '1h'
        else:
            interval = '5m'
        
        # Get candle data - more for 1h timeframe
        count = 200 if interval == '1h' else 100
        candles = delta_client.get_historical_candles('BTCUSD', interval, count)
        
        if candles and len(candles) > 0:
            # Convert to DataFrame - candles is a list of dictionaries
            df = pd.DataFrame(candles)
            
            # Debug the DataFrame structure
            logging.info(f"Candle data columns: {df.columns.tolist()}")
            logging.info(f"Sample candle: {candles[0] if candles else 'None'}")
            
            # Ensure proper column names and data types
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            for col in required_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                else:
                    logging.error(f"Missing required column: {col}")
                    return None
            
            # Handle timestamp
            if 'time' in df.columns:
                df['timestamp'] = pd.to_datetime(df['time'], unit='s')
            elif 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
            else:
                # Create a basic timestamp index
                freq_map = {'3m': '3T', '5m': '5T', '1h': '1H'}
                df['timestamp'] = pd.date_range(end=pd.Timestamp.now(), periods=len(df), freq=freq_map.get(timeframe, '5T'))
            
            # Ensure we have valid data
            if df.empty or df[required_columns].isnull().all().all():
                logging.error("DataFrame is empty or contains no valid data")
                return None
            
            # Calculate technical indicators
            df = calculate_technical_indicators(df)
            logging.info(f"After technical indicators: {type(df)}")
            
            if timeframe == '1h':
                df = calculate_higher_timeframe_indicators(df)
                logging.info(f"After higher timeframe indicators: {type(df)}")
            
            return df
        
        return None
    except Exception as e:
        logging.error(f"Error getting BTC data: {e}")
        return None

def calculate_technical_indicators(df):
    """Calculate technical indicators: VWAP, Parabolic SAR, ATR, and Price Action"""
    try:
        # Ensure we have enough data
        if len(df) < 50:
            raise Exception("Insufficient data for technical indicators calculation")
        
        # Calculate VWAP (Volume Weighted Average Price)
        df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
        df['vwap_numerator'] = (df['typical_price'] * df['volume']).cumsum()
        df['volume_cumsum'] = df['volume'].cumsum()
        df['VWAP'] = df['vwap_numerator'] / df['volume_cumsum']
        
        # Reset VWAP daily (approximate using 480 3-minute periods = 24 hours)
        reset_period = 288
        for i in range(reset_period, len(df), reset_period):
            df.loc[i:, 'vwap_numerator'] = (df.loc[i:, 'typical_price'] * df.loc[i:, 'volume']).cumsum()
            df.loc[i:, 'volume_cumsum'] = df.loc[i:, 'volume'].cumsum()
            df.loc[i:, 'VWAP'] = df.loc[i:, 'vwap_numerator'] / df.loc[i:, 'volume_cumsum']
        
        # Calculate VWAP deviation percentage
        df['VWAP_Dev'] = ((df['close'] - df['VWAP']) / df['VWAP']) * 100
        
        # Calculate Parabolic SAR
        def parabolic_sar(high, low, close, af_start=0.02, af_increment=0.02, af_max=0.2):
            sar = np.zeros(len(high))
            trend = np.zeros(len(high))
            af = np.zeros(len(high))
            ep = np.zeros(len(high))
            
            # Initialize first values
            sar[0] = low[0]
            trend[0] = 1  # 1 for uptrend, -1 for downtrend
            af[0] = af_start
            ep[0] = high[0]
            
            for i in range(1, len(high)):
                if trend[i-1] == 1:  # Uptrend
                    sar[i] = sar[i-1] + af[i-1] * (ep[i-1] - sar[i-1])
                    
                    if high[i] > ep[i-1]:
                        ep[i] = high[i]
                        af[i] = min(af[i-1] + af_increment, af_max)
                    else:
                        ep[i] = ep[i-1]
                        af[i] = af[i-1]
                    
                    if sar[i] > low[i]:
                        trend[i] = -1
                        sar[i] = ep[i-1]
                        af[i] = af_start
                        ep[i] = low[i]
                    else:
                        trend[i] = 1
                        
                else:  # Downtrend
                    sar[i] = sar[i-1] + af[i-1] * (ep[i-1] - sar[i-1])
                    
                    if low[i] < ep[i-1]:
                        ep[i] = low[i]
                        af[i] = min(af[i-1] + af_increment, af_max)
                    else:
                        ep[i] = ep[i-1]
                        af[i] = af[i-1]
                    
                    if sar[i] < high[i]:
                        trend[i] = 1
                        sar[i] = ep[i-1]
                        af[i] = af_start
                        ep[i] = high[i]
                    else:
                        trend[i] = -1
            
            return sar, trend
        
        sar_values, sar_trend = parabolic_sar(df['high'].values, df['low'].values, df['close'].values)
        df['SAR'] = sar_values
        df['SAR_Trend'] = sar_trend
        
        # Calculate ATR (Average True Range)
        df['TR'] = np.maximum(
            df['high'] - df['low'],
            np.maximum(
                abs(df['high'] - df['close'].shift(1)),
                abs(df['low'] - df['close'].shift(1))
            )
        )
        df['ATR'] = df['TR'].rolling(window=14).mean()
        df['ATR_Pct'] = (df['ATR'] / df['close']) * 100
        
        # Price Action Analysis
        # Higher Highs and Higher Lows (HH/HL) for uptrend
        # Lower Highs and Lower Lows (LH/LL) for downtrend
        lookback = 5
        df['Local_High'] = df['high'].rolling(window=lookback*2+1, center=True).max()
        df['Local_Low'] = df['low'].rolling(window=lookback*2+1, center=True).min()
        df['Is_High'] = (df['high'] == df['Local_High'])
        df['Is_Low'] = (df['low'] == df['Local_Low'])
        
        # Calculate swing highs and lows
        swing_highs = df[df['Is_High']]['high']
        swing_lows = df[df['Is_Low']]['low']
        
        # Determine price action trend
        df['PA_Trend'] = 0  # 0: neutral, 1: uptrend, -1: downtrend
        
        if len(swing_highs) >= 2 and len(swing_lows) >= 2:
            recent_highs = swing_highs.tail(2).values
            recent_lows = swing_lows.tail(2).values
            
            higher_highs = recent_highs[-1] > recent_highs[-2] if len(recent_highs) >= 2 else False
            higher_lows = recent_lows[-1] > recent_lows[-2] if len(recent_lows) >= 2 else False
            lower_highs = recent_highs[-1] < recent_highs[-2] if len(recent_highs) >= 2 else False
            lower_lows = recent_lows[-1] < recent_lows[-2] if len(recent_lows) >= 2 else False
            
            if higher_highs and higher_lows:
                df.loc[df.index[-20:], 'PA_Trend'] = 1  # Uptrend
            elif lower_highs and lower_lows:
                df.loc[df.index[-20:], 'PA_Trend'] = -1  # Downtrend
        
        # Generate trading signals using 5-point scoring system for each indicator
        # Total possible points: 20 (5 points each from VWAP, SAR, ATR, Price Action)
        # Signal threshold: configurable via settings
        
        df['VWAP_Score'] = 0
        df['SAR_Score'] = 0
        df['ATR_Score'] = 0
        df['PA_Score'] = 0
        df['Total_Score'] = 0
        df['Signal'] = 0
        df['Signal_Strength'] = 0
        df['Entry_Type'] = 'None'
        df['Confidence'] = 0.0
        
        # VWAP Scoring (5 points max)
        vwap_dev = ((df['close'] - df['VWAP']) / df['VWAP']) * 100  # VWAP deviation percentage
        df.loc[vwap_dev > 1.0, 'VWAP_Score'] = 5      # Strong bullish (>1% above VWAP)
        df.loc[(vwap_dev > 0.5) & (vwap_dev <= 1.0), 'VWAP_Score'] = 3   # Moderate bullish
        df.loc[(vwap_dev > 0) & (vwap_dev <= 0.5), 'VWAP_Score'] = 1     # Weak bullish
        df.loc[(vwap_dev < 0) & (vwap_dev >= -0.5), 'VWAP_Score'] = -1   # Weak bearish
        df.loc[(vwap_dev < -0.5) & (vwap_dev >= -1.0), 'VWAP_Score'] = -3 # Moderate bearish
        df.loc[vwap_dev < -1.0, 'VWAP_Score'] = -5     # Strong bearish (<-1% below VWAP)
        
        # SAR Scoring (5 points max)
        sar_strength = abs((df['close'] - df['SAR']) / df['close']) * 100  # Distance from SAR
        df.loc[(df['SAR_Trend'] == 1) & (sar_strength > 0.5), 'SAR_Score'] = 5   # Strong uptrend
        df.loc[(df['SAR_Trend'] == 1) & (sar_strength > 0.2), 'SAR_Score'] = 3   # Moderate uptrend
        df.loc[(df['SAR_Trend'] == 1) & (sar_strength <= 0.2), 'SAR_Score'] = 1  # Weak uptrend
        df.loc[(df['SAR_Trend'] == -1) & (sar_strength <= 0.2), 'SAR_Score'] = -1 # Weak downtrend
        df.loc[(df['SAR_Trend'] == -1) & (sar_strength > 0.2), 'SAR_Score'] = -3  # Moderate downtrend
        df.loc[(df['SAR_Trend'] == -1) & (sar_strength > 0.5), 'SAR_Score'] = -5  # Strong downtrend
        
        # ATR Scoring (5 points max) - Higher volatility gets more points
        df.loc[df['ATR_Pct'] > 1.0, 'ATR_Score'] = 5      # Very high volatility
        df.loc[(df['ATR_Pct'] > 0.5) & (df['ATR_Pct'] <= 1.0), 'ATR_Score'] = 4   # High volatility
        df.loc[(df['ATR_Pct'] > 0.2) & (df['ATR_Pct'] <= 0.5), 'ATR_Score'] = 3   # Moderate volatility
        df.loc[(df['ATR_Pct'] > 0.1) & (df['ATR_Pct'] <= 0.2), 'ATR_Score'] = 2   # Low-moderate volatility
        df.loc[(df['ATR_Pct'] > 0.05) & (df['ATR_Pct'] <= 0.1), 'ATR_Score'] = 1  # Low volatility
        df.loc[df['ATR_Pct'] <= 0.05, 'ATR_Score'] = 0    # Very low volatility (no points)
        
        # Price Action Scoring (5 points max)
        df.loc[df['PA_Trend'] == 1, 'PA_Score'] = 5       # Strong bullish structure
        df.loc[df['PA_Trend'] == 0, 'PA_Score'] = 0       # Neutral/sideways
        df.loc[df['PA_Trend'] == -1, 'PA_Score'] = -5     # Strong bearish structure
        
        # Calculate total score
        df['Total_Score'] = df['VWAP_Score'] + df['SAR_Score'] + df['ATR_Score'] + df['PA_Score']
        
        # Generate signals based on configurable thresholds
        config = {}
        try:
            config = config_manager.get_all_config()
        except:
            pass
        
        futures_config = config.get('futures_strategy', {})
        bullish_threshold = futures_config.get('long_signal_threshold', 5)
        bearish_threshold = futures_config.get('short_signal_threshold', -7)
        
        strong_bull = df['Total_Score'] >= bullish_threshold
        strong_bear = df['Total_Score'] <= bearish_threshold
        
        # Assign signals only when score exceeds threshold
        df.loc[strong_bull, 'Signal'] = 1
        df.loc[strong_bull, 'Signal_Strength'] = df.loc[strong_bull, 'Total_Score']
        df.loc[strong_bull, 'Entry_Type'] = 'CALL'
        df.loc[strong_bull, 'Confidence'] = (df.loc[strong_bull, 'Total_Score'] / 20.0).clip(0, 1)
        
        df.loc[strong_bear, 'Signal'] = -1
        df.loc[strong_bear, 'Signal_Strength'] = abs(df.loc[strong_bear, 'Total_Score'])
        df.loc[strong_bear, 'Entry_Type'] = 'PUT'
        df.loc[strong_bear, 'Confidence'] = (abs(df.loc[strong_bear, 'Total_Score']) / 20.0).clip(0, 1)
        
        return df
        
    except Exception as e:
        raise Exception(f"Technical indicators calculation failed: {str(e)}")

def calculate_higher_timeframe_indicators(df, bullish_threshold=3, bearish_threshold=-3):
    """Calculate higher timeframe trend indicators with configurable thresholds"""
    try:
        if len(df) < 100:
            raise Exception("Insufficient data for higher timeframe analysis")
        
        # Fisher Transform calculation
        def fisher_transform(data, period=10):
            high_low = (data['high'] + data['low']) / 2
            min_low = high_low.rolling(window=period).min()
            max_high = high_low.rolling(window=period).max()
            
            raw_value = 2 * ((high_low - min_low) / (max_high - min_low) - 0.5)
            raw_value = raw_value.clip(-0.999, 0.999)  # Prevent overflow
            
            fisher = 0.5 * np.log((1 + raw_value) / (1 - raw_value))
            fisher_signal = fisher.shift(1)
            
            return fisher, fisher_signal
        
        # True Strength Index calculation
        def true_strength_index(data, r=25, s=13):
            close_diff = data['close'].diff()
            abs_close_diff = close_diff.abs()
            
            double_smoothed_pc = close_diff.ewm(span=r).mean().ewm(span=s).mean()
            double_smoothed_apc = abs_close_diff.ewm(span=r).mean().ewm(span=s).mean()
            
            tsi = 100 * (double_smoothed_pc / double_smoothed_apc)
            return tsi
        
        # Pivot Points calculation
        def calculate_pivot_points(data):
            # Daily pivot points using previous day's data
            prev_high = data['high'].shift(1)
            prev_low = data['low'].shift(1)
            prev_close = data['close'].shift(1)
            
            pivot = (prev_high + prev_low + prev_close) / 3
            r1 = 2 * pivot - prev_low
            s1 = 2 * pivot - prev_high
            r2 = pivot + (prev_high - prev_low)
            s2 = pivot - (prev_high - prev_low)
            
            return pivot, r1, s1, r2, s2
        
        # Dow Theory analysis
        def dow_theory_analysis(data, short_period=20, long_period=50):
            sma_short = data['close'].rolling(window=short_period).mean()
            sma_long = data['close'].rolling(window=long_period).mean()
            
            # Higher highs and higher lows for uptrend
            highs = data['high'].rolling(window=5).max()
            lows = data['low'].rolling(window=5).min()
            
            higher_highs = highs > highs.shift(5)
            higher_lows = lows > lows.shift(5)
            lower_highs = highs < highs.shift(5)
            lower_lows = lows < lows.shift(5)
            
            uptrend = (sma_short > sma_long) & higher_highs & higher_lows
            downtrend = (sma_short < sma_long) & lower_highs & lower_lows
            
            return uptrend, downtrend
        
        # Williams Alligator calculation
        def williams_alligator(data):
            jaw = data['close'].rolling(window=13).mean().shift(8)  # Blue line
            teeth = data['close'].rolling(window=8).mean().shift(5)  # Red line
            lips = data['close'].rolling(window=5).mean().shift(3)  # Green line
            
            # Alligator is sleeping when lines are intertwined
            # Alligator is eating when price is above/below all lines
            eating_up = (data['close'] > jaw) & (data['close'] > teeth) & (data['close'] > lips)
            eating_down = (data['close'] < jaw) & (data['close'] < teeth) & (data['close'] < lips)
            sleeping = ~(eating_up | eating_down)
            
            return jaw, teeth, lips, eating_up, eating_down, sleeping
        
        # Calculate all indicators
        fisher, fisher_signal = fisher_transform(df)
        tsi = true_strength_index(df)
        pivot, r1, s1, r2, s2 = calculate_pivot_points(df)
        uptrend, downtrend = dow_theory_analysis(df)
        jaw, teeth, lips, eating_up, eating_down, sleeping = williams_alligator(df)
        
        # Get latest values for scoring
        latest = df.iloc[-1]
        latest_idx = len(df) - 1
        
        # Scoring system (3 points each, max 15 total)
        scores = {
            'fisher': 0,
            'tsi': 0, 
            'pivot': 0,
            'dow': 0,
            'alligator': 0
        }
        
        meanings = {
            'fisher': 'Neutral',
            'tsi': 'Neutral',
            'pivot': 'At Pivot',
            'dow': 'Sideways',
            'alligator': 'Sleeping'
        }
        
        # Fisher Transform scoring
        if not pd.isna(fisher.iloc[-1]) and not pd.isna(fisher_signal.iloc[-1]):
            fisher_val = fisher.iloc[-1]
            fisher_sig = fisher_signal.iloc[-1]
            if fisher_val > fisher_sig and fisher_val > 0:
                scores['fisher'] = 3
                meanings['fisher'] = 'Strong Bull'
            elif fisher_val > fisher_sig and fisher_val > -0.5:
                scores['fisher'] = 2  
                meanings['fisher'] = 'Moderate Bull'
            elif fisher_val > fisher_sig:
                scores['fisher'] = 1
                meanings['fisher'] = 'Weak Bull'
            elif fisher_val < fisher_sig and fisher_val < 0:
                scores['fisher'] = -3
                meanings['fisher'] = 'Strong Bear'
            elif fisher_val < fisher_sig and fisher_val < 0.5:
                scores['fisher'] = -2
                meanings['fisher'] = 'Moderate Bear'
            elif fisher_val < fisher_sig:
                scores['fisher'] = -1
                meanings['fisher'] = 'Weak Bear'
        
        # TSI scoring
        if not pd.isna(tsi.iloc[-1]):
            tsi_val = tsi.iloc[-1]
            if tsi_val > 25:
                scores['tsi'] = 3
                meanings['tsi'] = 'Strong Bull'
            elif tsi_val > 10:
                scores['tsi'] = 2
                meanings['tsi'] = 'Moderate Bull'
            elif tsi_val > 0:
                scores['tsi'] = 1
                meanings['tsi'] = 'Weak Bull'
            elif tsi_val < -25:
                scores['tsi'] = -3
                meanings['tsi'] = 'Strong Bear'
            elif tsi_val < -10:
                scores['tsi'] = -2
                meanings['tsi'] = 'Moderate Bear'
            elif tsi_val < 0:
                scores['tsi'] = -1
                meanings['tsi'] = 'Weak Bear'
        
        # Pivot Points scoring
        if not pd.isna(pivot.iloc[-1]):
            current_price = latest['close']
            pivot_val = pivot.iloc[-1]
            r1_val = r1.iloc[-1]
            s1_val = s1.iloc[-1]
            
            if current_price > r1_val:
                scores['pivot'] = 3
                meanings['pivot'] = 'Above R1'
            elif current_price > pivot_val:
                scores['pivot'] = 1
                meanings['pivot'] = 'Above Pivot'
            elif current_price < s1_val:
                scores['pivot'] = -3
                meanings['pivot'] = 'Below S1'
            elif current_price < pivot_val:
                scores['pivot'] = -1
                meanings['pivot'] = 'Below Pivot'
        
        # Dow Theory scoring
        if latest_idx < len(uptrend) and latest_idx < len(downtrend):
            if uptrend.iloc[-1]:
                scores['dow'] = 3
                meanings['dow'] = 'Clear Uptrend'
            elif uptrend.iloc[-5:].sum() >= 3:  # Recent bullish signals
                scores['dow'] = 1
                meanings['dow'] = 'Emerging Uptrend'
            elif downtrend.iloc[-1]:
                scores['dow'] = -3
                meanings['dow'] = 'Clear Downtrend'
            elif downtrend.iloc[-5:].sum() >= 3:  # Recent bearish signals
                scores['dow'] = -1
                meanings['dow'] = 'Emerging Downtrend'
        
        # Williams Alligator scoring
        if latest_idx < len(eating_up) and latest_idx < len(eating_down):
            if eating_up.iloc[-1]:
                scores['alligator'] = 3
                meanings['alligator'] = 'Eating Up'
            elif eating_up.iloc[-3:].sum() >= 2:  # Recently eating up
                scores['alligator'] = 1
                meanings['alligator'] = 'Waking Up'
            elif eating_down.iloc[-1]:
                scores['alligator'] = -3
                meanings['alligator'] = 'Eating Down' 
            elif eating_down.iloc[-3:].sum() >= 2:  # Recently eating down
                scores['alligator'] = -1
                meanings['alligator'] = 'Waking Down'
        
        # Calculate total score and trend determination
        total_score = sum(scores.values())
        
        if total_score >= bullish_threshold:
            overall_trend = 'Bullish'
            trend_strength = 'Strong'
        elif total_score <= bearish_threshold:
            overall_trend = 'Bearish'
            trend_strength = 'Strong'
        else:
            overall_trend = 'Neutral'
            trend_strength = 'Neutral'
        
        # Compile results
        result = {
            'overall_trend': overall_trend,
            'trend_strength': trend_strength,
            'total_score': total_score,
            'max_score': 15,
            'candle_time': df.iloc[-1]['timestamp'],
            'indicators': {
                'fisher': {
                    'value': fisher.iloc[-1] if not pd.isna(fisher.iloc[-1]) else None,
                    'score': scores['fisher'],
                    'meaning': meanings['fisher']
                },
                'tsi': {
                    'value': tsi.iloc[-1] if not pd.isna(tsi.iloc[-1]) else None,
                    'score': scores['tsi'], 
                    'meaning': meanings['tsi']
                },
                'pivot': {
                    'position': 'Above Pivot' if latest['close'] > pivot.iloc[-1] else 'Below Pivot',
                    'score': scores['pivot'],
                    'meaning': meanings['pivot']
                },
                'dow': {
                    'signal': meanings['dow'],
                    'score': scores['dow'],
                    'meaning': meanings['dow']
                },
                'alligator': {
                    'state': meanings['alligator'],
                    'score': scores['alligator'],
                    'meaning': meanings['alligator']
                }
            }
        }
        
        # Add trend indicators to DataFrame
        df['Trend_Direction'] = overall_trend
        df['Trend_Strength'] = total_score
        
        return df
        
    except Exception as e:
        raise Exception(f"Higher timeframe indicators calculation failed: {str(e)}")

# Disable template caching for development
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
if app.debug:
    app.jinja_env.auto_reload = True
    app.config['TEMPLATES_AUTO_RELOAD'] = True

# Global variables for trading bot management
trading_bot = None
trading_thread = None
bot_running = False
trading_status = {
    'status': 'Stopped',
    'last_update': None,
    'current_positions': 0,
    'daily_pnl': 0.0,
    'total_trades': 0
}

def load_config():
    """Load configuration from application.config file"""
    try:
        return config_manager.get_all_config()
    except Exception as e:
        app_logger.error(f"Error loading config: {e}")
        return {}

def save_config(config):
    """Save configuration to application.config file"""
    try:
        return config_manager.save_config(config)
    except Exception as e:
        app_logger.error(f"Error saving config: {e}")
        return False

def test_api_connection(api_key, api_secret, paper_trading=False):
    """Test API connection with detailed error reporting - NO FALLBACKS"""
    try:
        app_logger.info(f"Creating client with paper_trading={paper_trading}")
        client = DeltaExchangeClient(api_key, api_secret, paper_trading)
        
        app_logger.info("Testing BTC price retrieval...")
        btc_price = client.get_current_btc_price()
        
        app_logger.info(f"BTC price test successful: ${btc_price:,.2f}")
        return True, btc_price
            
    except Exception as e:
        app_logger.error(f"API connection test failed: {e}")
        return False, str(e)

@app.route('/')
def dashboard():
    """Main dashboard - NO FALLBACKS"""
    config = load_config()
    
    # Test API connection if configured
    api_status = "Not Connected"
    
    if config.get('api_key') and config.get('api_secret'):
        try:
            client = DeltaExchangeClient(
                config['api_key'], 
                config['api_secret'], 
                config.get('paper_trading', False)
            )
            
            # Test connection by getting BTC price
            btc_price = client.get_current_btc_price()
            api_status = "Connected"
            
        except Exception as e:
            api_status = f"Error: {str(e)[:50]}..."
    
    return render_template('dashboard.html', 
                         config=config,
                         api_status=api_status,
                         trading_status=trading_status,
                         bot_running=bot_running)

@app.route('/settings')
def settings():
    """Settings page"""
    config = config_manager.get_all_config()
    print("CONSOLE: SETTINGS GET - Loading settings page")  # This will show in server console
    app_logger.info(f"SETTINGS GET - Loading settings page")
    app_logger.debug(f"Settings page config source - max_positions: {config.get('max_positions')}")
    app_logger.debug(f"Config keys: {list(config.keys())}")
    app_logger.info(f"SETTINGS GET - Current trend_bullish_threshold: {config.get('futures_strategy', {}).get('trend_bullish_threshold')}")
    return render_template('settings.html', config=config)

@app.route('/save_settings', methods=['POST'])
def save_settings():
    """Save settings"""
    print("CONSOLE: SAVE_SETTINGS - Route called!")  # This will show in server console
    app_logger.info(f"SAVE_SETTINGS - Route called! Form data received.")
    app_logger.info(f"SAVE_SETTINGS - Form keys: {list(request.form.keys())}")
    try:
        config = {
            'api_key': request.form.get('api_key', '').strip(),
            'api_secret': request.form.get('api_secret', '').strip(),
            'paper_trading': request.form.get('paper_trading') == 'on',
            'portfolio_size': float(request.form.get('portfolio_size', 100000)),
            'position_size_usd': float(request.form.get('position_size_usd', 500)),
            'max_daily_loss': float(request.form.get('max_daily_loss', 2000)),
            'max_positions': int(request.form.get('max_positions', 2)),
            'futures_strategy': {
                'enabled': request.form.get('futures_enabled') == 'on',
                'long_signal_threshold': int(request.form.get('futures_long_threshold', 5)),
                'short_signal_threshold': int(request.form.get('futures_short_threshold', -7)),
                'leverage': int(request.form.get('futures_leverage', 10)),
                'position_size_usd': int(request.form.get('futures_position_size_usd', 500)),
                'min_signal_strength': int(request.form.get('futures_min_signal_strength', 4)),
                'require_trend_alignment': request.form.get('futures_require_trend_alignment') == 'on',
                'min_trend_strength': int(request.form.get('futures_min_trend_strength', 5)),
                'min_time_between_trades': int(request.form.get('futures_min_time_between_trades', 3600)),
                'trend_bullish_threshold': int(request.form.get('trend_bullish_threshold', 3)),
                'trend_bearish_threshold': int(request.form.get('trend_bearish_threshold', -3))
            },
            'neutral_strategy': {
                'enabled': request.form.get('neutral_enabled') == 'on',
                'lot_size': int(request.form.get('neutral_lot_size', 1)),
                'leverage_percentage': float(request.form.get('neutral_leverage', 50)),
                'strike_distance': int(request.form.get('strike_distance', 8)),
                'expiry_days': 1,
                'trailing_stop_loss_pct': 20.0,
                'profit_target_pct': float(request.form.get('neutral_profit_target', 30)),
                'stop_loss_pct': float(request.form.get('neutral_stop_loss', 50)),
                'min_time_between_neutral_trades': 7200
            },
            'dollar_based_risk': {
                'enabled': request.form.get('exit_mode', 'dollar') == 'dollar',
                'stop_loss_usd': float(request.form.get('dollar_stop_loss', 100)),
                'take_profit_usd': float(request.form.get('dollar_take_profit', 200)),
                'trailing_stop_usd': float(request.form.get('dollar_trailing_stop', 50)),
                'quick_profit_usd': float(request.form.get('dollar_quick_profit', 60)),
                'max_risk_usd': float(request.form.get('dollar_max_risk', 150)),
                'daily_loss_limit_usd': float(request.form.get('dollar_daily_loss_limit', 500))
            },
            'trading_timing': {
                'trading_start_time': request.form.get('trading_start_time', '17:30').strip(),
                'timezone': request.form.get('timezone', 'Asia/Kolkata').strip()
            },
            'atr_exits': {
                'enabled': request.form.get('exit_mode', 'dollar') == 'atr',
                'atr_period': int(request.form.get('atr_period', 14)),
                'stop_loss_atr_multiplier': float(request.form.get('stop_loss_atr_multiplier', 2.0)),
                'take_profit_atr_multiplier': float(request.form.get('take_profit_atr_multiplier', 3.0)),
                'trailing_atr_multiplier': float(request.form.get('trailing_atr_multiplier', 1.5)),
                'buffer_zone_atr_multiplier': float(request.form.get('buffer_zone_atr_multiplier', 0.3)),
                'volume_threshold_percentile': float(request.form.get('volume_threshold_percentile', 70)),
                'hunting_zone_offset': float(request.form.get('hunting_zone_offset', 5))
            }
        }
        
        app_logger.debug(f"DEBUG: Attempting to save config...")
        app_logger.debug(f"DEBUG: Sample values - API Key: {config.get('api_key', '')[:10]}..., Portfolio Size: {config.get('portfolio_size')}")
        app_logger.debug(f"DEBUG: Max positions value being saved: {config.get('max_positions')}")
        
        # Debug the specific issue
        app_logger.info(f"FORM DATA - trend_bullish_threshold: '{request.form.get('trend_bullish_threshold')}' (type: {type(request.form.get('trend_bullish_threshold'))})")
        app_logger.info(f"PROCESSED - trend_bullish_threshold: {config.get('futures_strategy', {}).get('trend_bullish_threshold')} (type: {type(config.get('futures_strategy', {}).get('trend_bullish_threshold'))})")
        
        if config_manager.save_config(config):
            # Reload config after successful save
            config_manager.load_config()
            
            # Verify what actually got saved
            saved_config = config_manager.get_all_config()
            saved_bullish = saved_config.get('futures_strategy', {}).get('trend_bullish_threshold', 'NOT_FOUND')
            app_logger.info(f"VERIFIED SAVE - trend_bullish_threshold in file: {saved_bullish}")
            
            app_logger.debug(f"DEBUG: Configuration saved successfully!")
            flash('Settings saved successfully!', 'success')
        else:
            app_logger.debug(f"DEBUG: Configuration save failed!")
            flash('Error saving settings!', 'error')
            
    except Exception as e:
        flash(f'Error saving settings: {str(e)}', 'error')
    
    return redirect(url_for('settings'))

@app.route('/test_api', methods=['POST'])
def test_api():
    """Test API connection"""
    try:
        api_key = request.form.get('api_key', '').strip()
        api_secret = request.form.get('api_secret', '').strip()
        paper_trading = request.form.get('paper_trading') == 'on'
        
        if not api_key or not api_secret:
            return jsonify({'success': False, 'message': 'API Key and Secret are required'})
        
        # Test connection with detailed error handling
        print(f"Testing API connection - Paper Trading: {paper_trading}")
        
        success, result = test_api_connection(api_key, api_secret, paper_trading)
        
        if success:
            # Additional tests for paper trading mode
            if paper_trading:
                try:
                    # Test simulated functions
                    client = DeltaExchangeClient(api_key, api_secret, paper_trading)
                    balance = client.get_account_balance()
                    portfolio = client.get_portfolio_summary()
                    
                    return jsonify({
                        'success': True, 
                        'message': f'Paper Trading Mode - Connection successful! BTC Price: ${result:,.2f}. Simulated balance: $10,000 USDT',
                        'btc_price': result,
                        'mode': 'paper',
                        'balance': balance,
                        'portfolio': portfolio
                    })
                except Exception as e:
                    return jsonify({
                        'success': True, 
                        'message': f'BTC Price working (${result:,.2f}) but simulation error: {str(e)}',
                        'btc_price': result,
                        'mode': 'paper'
                    })
            else:
                return jsonify({
                    'success': True, 
                    'message': f'Live Trading Mode - Connection successful! BTC Price: ${result:,.2f}',
                    'btc_price': result,
                    'mode': 'live'
                })
        else:
            return jsonify({'success': False, 'message': f'Connection failed: {result}'})
            
    except Exception as e:
        print(f"API test error: {e}")
        return jsonify({'success': False, 'message': f'Test error: {str(e)}'})

@app.route('/trades')
def trades():
    """Trade history page"""
    return render_template('trades.html')

@app.route('/api/trades')
def api_trades():
    """Get trade history from Delta Exchange using official Trade History API"""
    try:
        config = config_manager.get_all_config()
        
        # Use existing DeltaExchangeClient
        delta_client = get_delta_client()
        
        # Get USD to INR conversion rate
        usd_to_inr = float(config.get('USD', 85))
        
        # Get trade history using Delta Exchange Trade History API
        fills_response = delta_client.get_fills_history(page_size=50)
        
        if not isinstance(fills_response, dict) or 'result' not in fills_response:
            raise Exception("Invalid response format from Delta Exchange API")
        
        all_trades = []
        fills = fills_response['result']
        
        for fill in fills:
            if not isinstance(fill, dict):
                continue
                
            # Extract fill data
            fill_id = fill.get('id', 'unknown')
            product_symbol = fill.get('product_symbol', 'Unknown')
            side = fill.get('side', 'unknown').upper()
            size = float(fill.get('size', 0))
            price = float(fill.get('price', 0))
            commission = float(fill.get('commission', 0))
            created_at = fill.get('created_at', '')
            role = fill.get('role', 'unknown')  # maker/taker
            order_id = fill.get('order_id', '')
            
            # Calculate values in USD and INR
            trade_value_usd = size * price
            trade_value_inr = trade_value_usd * usd_to_inr
            commission_inr = commission * usd_to_inr
            price_inr = price * usd_to_inr
            
            formatted_trade = {
                'trade_id': f"fill_{fill_id}",
                'order_id': order_id,
                'symbol': product_symbol,
                'side': side,
                'size': size,
                'price': price,
                'price_inr': price_inr,
                'value': trade_value_usd,
                'value_inr': trade_value_inr,
                'fee': commission,
                'fee_inr': commission_inr,
                'timestamp': created_at,
                'role': role.title(),
                'strategy': 'Live Trading',
                'status': 'Executed',
                'source': 'Delta Exchange',
                'entry_date': created_at.split('T')[0] if 'T' in created_at else created_at[:10] if len(created_at) >= 10 else '',
                'entry_time': created_at.split('T')[1].split('.')[0] if 'T' in created_at else created_at[11:19] if len(created_at) >= 19 else ''
            }
            all_trades.append(formatted_trade)
        
        # Sort by timestamp (most recent first)
        all_trades.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
        
        return jsonify({
            'success': True,
            'trades': all_trades,
            'count': len(all_trades),
            'usd_to_inr_rate': usd_to_inr,
            'timestamp': datetime.now().isoformat(),
            'source': 'Delta Exchange'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Failed to fetch trades from Delta Exchange: {str(e)}',
            'trades': [],
            'count': 0,
            'timestamp': datetime.now().isoformat()
        })

@app.route('/api/status')
def api_status():
    """Get trading bot status"""
    global trading_bot_process, bot_running, trading_status
    
    # Check if bot process is still alive
    if bot_running and trading_bot_process:
        if trading_bot_process.poll() is not None:  # Process has terminated
            bot_running = False
            trading_bot_process = None
    
    return jsonify({
        'success': True,
        'bot_running': bot_running,
        'status': trading_status.get('status', 'Stopped'),
        'current_positions': trading_status.get('current_positions', 0),
        'daily_pnl': trading_status.get('daily_pnl', 0.0),
        'total_trades': trading_status.get('total_trades', 0),
        'pid': trading_bot_process.pid if trading_bot_process else None,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/btc_price')
def api_btc_price():
    """Get current BTC price from Delta Exchange"""
    try:
        config = config_manager.get_all_config()
        delta_client = DeltaExchangeClient(config['api_key'], config['api_secret'])
        price = delta_client.get_current_btc_price()
        return jsonify({
            'success': True,
            'price': price,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'price': 0,
            'timestamp': datetime.now().isoformat()
        })

@app.route('/api/wallet')
def api_wallet():
    """Get wallet balance from Delta Exchange"""
    try:
        config = config_manager.get_all_config()
        
        # Use existing DeltaExchangeClient
        delta_client = get_delta_client()
        
        # Get account balance using our existing method
        balance = delta_client.get_account_balance()
        
        # Get USD to INR conversion rate from config
        usd_to_inr = float(config.get('USD', 85))  # Default to 85 if not found
        
        # Format balances for UI
        balances = []
        if isinstance(balance, dict):
            for currency, info in balance.items():
                if isinstance(info, dict):
                    usd_balance = float(info.get('balance', 0))
                    available_usd = float(info.get('available', 0))
                    
                    balances.append({
                        'currency': currency,
                        'balance_usd': usd_balance,
                        'balance_inr': usd_balance * usd_to_inr,
                        'available_usd': available_usd,
                        'available_inr': available_usd * usd_to_inr,
                        'conversion_rate': usd_to_inr
                    })
            
        return jsonify({
            'success': True,
            'balances': balances,
            'usd_to_inr_rate': usd_to_inr,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'balances': [],
            'timestamp': datetime.now().isoformat()
        })

@app.route('/api/positions')
def api_positions():
    """Get current positions from Delta Exchange"""
    try:
        from delta_rest_client import DeltaRestClient
        config = config_manager.get_all_config()
        
        delta_client = DeltaRestClient(
            base_url='https://api.india.delta.exchange',
            api_key=config['api_key'],
            api_secret=config['api_secret']
        )
        
        # Get all positions by iterating through products
        active_positions = []
        try:
            # Try to get positions for BTCUSD first
            btc_position = delta_client.get_position(27)  # BTCUSD product ID
            if btc_position and btc_position.get('result'):
                pos_data = btc_position['result']
                if float(pos_data.get('size', 0)) != 0:
                    active_positions.append(pos_data)
        except:
            pass
        
        return jsonify({
            'success': True,
            'positions': active_positions,
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e),
            'positions': [],
            'timestamp': datetime.now().isoformat()
        })

@app.route('/api/signal_data')
def api_signal_data():
    """Get current trading signals and technical indicators for 3-minute timeframe"""
    try:
        # Get cached 3-minute data (only updates when 3M candle closes)
        df = get_cached_indicators('3m')
        if df is not None and len(df) > 0:
            latest = df.iloc[-1]
            
            # Extract all the technical indicator values
            indicators = {
                'current_price': float(latest.get('close', 0)),
                'candle_close_price': float(latest.get('close', 0)),
                'vwap': float(latest.get('VWAP', 0)) if not pd.isna(latest.get('VWAP', 0)) else None,
                'vwap_deviation': float(latest.get('VWAP_Dev', 0)) if not pd.isna(latest.get('VWAP_Dev', 0)) else None,
                'parabolic_sar': float(latest.get('SAR', 0)) if not pd.isna(latest.get('SAR', 0)) else None,
                'sar_trend': int(latest.get('SAR_Trend', 0)) if not pd.isna(latest.get('SAR_Trend', 0)) else None,
                'atr_pct': float(latest.get('ATR_Pct', 0)) if not pd.isna(latest.get('ATR_Pct', 0)) else None,
                'price_action_trend': int(latest.get('PA_Trend', 0)) if not pd.isna(latest.get('PA_Trend', 0)) else 0
            }
            
            # Extract scoring data
            scoring = {
                'vwap_score': int(latest.get('VWAP_Score', 0)),
                'sar_score': int(latest.get('SAR_Score', 0)),
                'atr_score': int(latest.get('ATR_Score', 0)),
                'pa_score': int(latest.get('PA_Score', 0)),
                'total_score': int(latest.get('Total_Score', 0)),
                'bullish_threshold': 5,  # Default values
                'bearish_threshold': -7
            }
            
            # Get thresholds from config
            try:
                config = load_config()
                futures_config = config.get('futures_strategy', {})
                scoring['bullish_threshold'] = futures_config.get('long_signal_threshold', 5)
                scoring['bearish_threshold'] = futures_config.get('short_signal_threshold', -7)
            except:
                pass
            
            # Get 1H trend analysis for trade decision
            trade_analysis = None
            try:
                # Get cached 1H trend analysis (only recalculates when 1H candle closes)
                df_1h_analyzed = get_cached_indicators('1h')
                if df_1h_analyzed is not None and len(df_1h_analyzed) > 0:
                    latest_1h = df_1h_analyzed.iloc[-1]
                    
                    # Extract 1H trend data
                    trend_direction = latest_1h.get('Trend_Direction', 'NEUTRAL')
                    trend_score = int(latest_1h.get('Trend_Strength', 0))
                    
                    # Get 3M signal scores
                    signals_3m_score = scoring['total_score']
                    long_threshold = scoring['bullish_threshold']
                    short_threshold = scoring['bearish_threshold']
                    
                    # Decision logic: Signal only triggers when 1H trend aligns with 3M signal direction
                    trade_decision = 'NO_TRADE'
                    decision_reasoning = 'No qualifying signal'
                    
                    # Check for LONG signal
                    if signals_3m_score >= long_threshold:
                        if trend_direction == 'Bullish':
                            trade_decision = 'LONG_FUTURES'
                            decision_reasoning = f'3M score ({signals_3m_score}) ≥ Long threshold ({long_threshold}) AND 1H trend is Bullish. Trend alignment confirmed.'
                        elif trend_direction == 'Neutral':
                            trade_decision = 'LONG_FUTURES'
                            decision_reasoning = f'3M score ({signals_3m_score}) ≥ Long threshold ({long_threshold}) AND 1H trend is Neutral. Weak trend allows signal.'
                        else:  # Bearish
                            trade_decision = 'NO_TRADE'
                            decision_reasoning = f'3M score ({signals_3m_score}) suggests LONG but 1H trend is Bearish. No trend alignment - signal blocked.'
                    
                    # Check for SHORT signal
                    elif signals_3m_score <= short_threshold:
                        if trend_direction == 'Bearish':
                            trade_decision = 'SHORT_FUTURES'
                            decision_reasoning = f'3M score ({signals_3m_score}) ≤ Short threshold ({short_threshold}) AND 1H trend is Bearish. Trend alignment confirmed.'
                        elif trend_direction == 'Neutral':
                            trade_decision = 'SHORT_FUTURES'
                            decision_reasoning = f'3M score ({signals_3m_score}) ≤ Short threshold ({short_threshold}) AND 1H trend is Neutral. Weak trend allows signal.'
                        else:  # Bullish
                            trade_decision = 'NO_TRADE'
                            decision_reasoning = f'3M score ({signals_3m_score}) suggests SHORT but 1H trend is Bullish. No trend alignment - signal blocked.'
                    
                    # No directional signal from 3M
                    else:
                        if trend_direction == 'Neutral' and abs(signals_3m_score) < 3:
                            trade_decision = 'NEUTRAL_STRATEGY'
                            decision_reasoning = f'3M score ({signals_3m_score}) between thresholds and 1H trend is Neutral. Consider neutral strategy.'
                        else:
                            trade_decision = 'NO_TRADE'
                            decision_reasoning = f'3M score ({signals_3m_score}) between thresholds ({short_threshold} to {long_threshold}). No directional signal.'
                    
                    # Create trade analysis object
                    trade_analysis = {
                        'trade_decision': trade_decision,
                        '1h_trend': trend_direction,
                        '1h_trend_score': trend_score,
                        '3m_total_score': signals_3m_score,
                        '3m_long_threshold': long_threshold,
                        '3m_short_threshold': short_threshold,
                        'decision_reasoning': decision_reasoning,
                        'trend_alignment': {
                            'required': True,
                            'status': 'ALIGNED' if trade_decision in ['LONG_FUTURES', 'SHORT_FUTURES'] else 'NOT_ALIGNED' if trade_decision == 'NO_TRADE' else 'NEUTRAL'
                        }
                    }
                
            except Exception as trend_error:
                # Fallback if 1H trend analysis fails
                trade_analysis = {
                    'trade_decision': 'NO_TRADE',
                    '1h_trend': 'ERROR',
                    '1h_trend_score': 0,
                    '3m_total_score': scoring['total_score'],
                    '3m_long_threshold': scoring['bullish_threshold'],
                    '3m_short_threshold': scoring['bearish_threshold'],
                    'decision_reasoning': f'Unable to analyze 1H trend: {str(trend_error)}',
                    'trend_alignment': {
                        'required': True,
                        'status': 'ERROR'
                    }
                }
            
            # Get current market time from real-time ticker instead of historical candle time
            candle_close_time = None
            try:
                delta_client = get_delta_client()
                response = delta_client._make_public_request('/v2/tickers/BTCUSD')
                ticker = response.get('result')
                if ticker and ticker.get('timestamp'):
                    # Convert microsecond timestamp to datetime
                    ticker_ts_sec = ticker['timestamp'] / 1000000
                    candle_close_time = datetime.fromtimestamp(ticker_ts_sec).isoformat()
            except Exception as e:
                logging.warning(f"Failed to get real-time market timestamp: {e}")
                # Fallback to estimated time based on candle data
                if 'timestamp' in df.columns and not pd.isna(latest.get('timestamp')):
                    candle_open_time = pd.to_datetime(latest['timestamp'])
                    candle_close_time = (candle_open_time + pd.Timedelta(minutes=3)).isoformat()
            
            return jsonify({
                'success': True,
                'signal': int(latest.get('Signal', 0)),
                'strength': float(latest.get('Signal_Strength', 0)),
                'entry_type': latest.get('Entry_Type', 'NEUTRAL'),
                'indicators': indicators,
                'scoring': scoring,
                'trade_analysis': trade_analysis,  # NEW: Comprehensive trade analysis
                'candle_close_time': candle_close_time,
                'timestamp': datetime.now().isoformat()
            })
        else:
            return jsonify({
                'success': False,
                'message': 'No 3-minute data available from Delta Exchange',
                'signal': 0,
                'strength': 0,
                'entry_type': 'NEUTRAL',
                'indicators': {
                    'current_price': None,
                    'candle_close_price': None,
                    'vwap': None,
                    'vwap_deviation': None,
                    'parabolic_sar': None,
                    'sar_trend': None,
                    'atr_pct': None,
                    'price_action_trend': 0
                },
                'scoring': {
                    'vwap_score': 0,
                    'sar_score': 0,
                    'atr_score': 0,
                    'pa_score': 0,
                    'total_score': 0,
                    'bullish_threshold': 5,
                    'bearish_threshold': -7
                },
                'trade_analysis': {
                    'trade_decision': 'NO_TRADE',
                    '1h_trend': 'NO_DATA',
                    '1h_trend_score': 0,
                    '3m_total_score': 0,
                    '3m_long_threshold': 5,
                    '3m_short_threshold': -7,
                    'decision_reasoning': 'No 3-minute data available',
                    'trend_alignment': {
                        'required': True,
                        'status': 'NO_DATA'
                    }
                },
                'candle_close_time': None,
                'timestamp': datetime.now().isoformat()
            })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Failed to fetch signal data: {str(e)}',
            'error': str(e),
            'signal': 0,
            'strength': 0,
            'entry_type': 'NEUTRAL',
            'indicators': {
                'current_price': None,
                'candle_close_price': None,
                'vwap': None,
                'vwap_deviation': None,
                'parabolic_sar': None,
                'sar_trend': None,
                'atr_pct': None,
                'price_action_trend': 0
            },
            'scoring': {
                'vwap_score': 0,
                'sar_score': 0,
                'atr_score': 0,
                'pa_score': 0,
                'total_score': 0,
                'bullish_threshold': 5,
                'bearish_threshold': -7
            },
            'trade_analysis': {
                'trade_decision': 'NO_TRADE',
                '1h_trend': 'ERROR',
                '1h_trend_score': 0,
                '3m_total_score': 0,
                '3m_long_threshold': 5,
                '3m_short_threshold': -7,
                'decision_reasoning': f'API error: {str(e)}',
                'trend_alignment': {
                    'required': True,
                    'status': 'ERROR'
                }
            },
            'candle_close_time': None,
            'timestamp': datetime.now().isoformat()
        })

@app.route('/api/higher_timeframe_trend')
def api_higher_timeframe_trend():
    """Get higher timeframe trend data with full indicator analysis"""
    # Calculate the last 1h candle close time based on Delta Exchange timing
    # Delta Exchange 1h candles close at :30 past each hour (5:30, 6:30, 7:30, etc.)
    # This aligns with their contract start time of 5:30 PM IST
    current_time = datetime.now()
    last_hour_close = current_time.replace(minute=30, second=0, microsecond=0)
    
    # If we haven't reached :30 past the hour yet, go back to previous hour's :30
    if current_time.minute < 30:
        last_hour_close = last_hour_close - timedelta(hours=1)
    
    try:
        # Get cached 1-hour analysis (only recalculates when 1H candle closes)
        df_1h_analyzed = get_cached_indicators('1h')
        if df_1h_analyzed is not None and len(df_1h_analyzed) > 0:
            # Extract the result from cached analysis
            latest = df_1h_analyzed.iloc[-1]
            
            # Use cached analysis results directly (no need to recalculate)
            try:
                # Extract trend data from cached analysis
                trend_direction = latest.get('Trend_Direction', 'NEUTRAL')
                trend_strength_score = float(latest.get('Trend_Strength', 0))
                
                # Get raw 1H data for detailed indicator calculation (for display purposes)
                df_1h_raw = _indicator_cache['1h']['data'] if _indicator_cache['1h']['data'] is not None else get_btc_data(timeframe='1h')
                
                # Get threshold values from config
                config = load_config()
                futures_config = config.get('futures_strategy', {})
                bullish_threshold = futures_config.get('trend_bullish_threshold', 3)
                bearish_threshold = futures_config.get('trend_bearish_threshold', -3)
                
                # Extract detailed results from the analysis function
                # Since calculate_higher_timeframe_indicators returns the result dict, we need to call it properly
                
                # Fisher Transform calculation
                def fisher_transform(data, period=10):
                    high_low = (data['high'] + data['low']) / 2
                    min_low = high_low.rolling(window=period).min()
                    max_high = high_low.rolling(window=period).max()
                    
                    raw_value = 2 * ((high_low - min_low) / (max_high - min_low) - 0.5)
                    raw_value = raw_value.clip(-0.999, 0.999)
                    
                    fisher = 0.5 * np.log((1 + raw_value) / (1 - raw_value))
                    fisher_signal = fisher.shift(1)
                    
                    return fisher, fisher_signal
                
                # True Strength Index calculation
                def true_strength_index(data, r=25, s=13):
                    close_diff = data['close'].diff()
                    abs_close_diff = close_diff.abs()
                    
                    double_smoothed_pc = close_diff.ewm(span=r).mean().ewm(span=s).mean()
                    double_smoothed_apc = abs_close_diff.ewm(span=r).mean().ewm(span=s).mean()
                    
                    tsi = 100 * (double_smoothed_pc / double_smoothed_apc)
                    return tsi
                
                # Calculate pivot points
                def calculate_pivot_points(data):
                    prev_high = data['high'].shift(1)
                    prev_low = data['low'].shift(1)
                    prev_close = data['close'].shift(1)
                    
                    pivot = (prev_high + prev_low + prev_close) / 3
                    r1 = 2 * pivot - prev_low
                    s1 = 2 * pivot - prev_high
                    
                    return pivot, r1, s1
                
                # Dow Theory analysis
                def dow_theory_analysis(data, short_period=20, long_period=50):
                    sma_short = data['close'].rolling(window=short_period).mean()
                    sma_long = data['close'].rolling(window=long_period).mean()
                    
                    highs = data['high'].rolling(window=5).max()
                    lows = data['low'].rolling(window=5).min()
                    
                    higher_highs = highs > highs.shift(5)
                    higher_lows = lows > lows.shift(5)
                    lower_highs = highs < highs.shift(5)
                    lower_lows = lows < lows.shift(5)
                    
                    uptrend = (sma_short > sma_long) & higher_highs & higher_lows
                    downtrend = (sma_short < sma_long) & lower_highs & lower_lows
                    
                    return uptrend, downtrend
                
                # Williams Alligator calculation
                def williams_alligator(data):
                    jaw = data['close'].rolling(window=13).mean().shift(8)
                    teeth = data['close'].rolling(window=8).mean().shift(5)
                    lips = data['close'].rolling(window=5).mean().shift(3)
                    
                    eating_up = (data['close'] > jaw) & (data['close'] > teeth) & (data['close'] > lips)
                    eating_down = (data['close'] < jaw) & (data['close'] < teeth) & (data['close'] < lips)
                    
                    return jaw, teeth, lips, eating_up, eating_down
                
                # Calculate all indicators using raw data for display
                fisher, fisher_signal = fisher_transform(df_1h_raw)
                tsi = true_strength_index(df_1h_raw)
                pivot, r1, s1 = calculate_pivot_points(df_1h_raw)
                uptrend, downtrend = dow_theory_analysis(df_1h_raw)
                jaw, teeth, lips, eating_up, eating_down = williams_alligator(df_1h_raw)
                
                # Get latest values and calculate scores
                latest_candle = df_1h_raw.iloc[-1]
                latest_idx = len(df_1h_raw) - 1
                
                # Scoring system
                scores = {'fisher': 0, 'tsi': 0, 'pivot': 0, 'dow': 0, 'alligator': 0}
                meanings = {'fisher': 'Neutral', 'tsi': 'Neutral', 'pivot': 'At Pivot', 'dow': 'Sideways', 'alligator': 'Sleeping'}
                
                # Fisher Transform scoring
                if not pd.isna(fisher.iloc[-1]) and not pd.isna(fisher_signal.iloc[-1]):
                    fisher_val = fisher.iloc[-1]
                    fisher_sig = fisher_signal.iloc[-1]
                    if fisher_val > fisher_sig and fisher_val > 0:
                        scores['fisher'] = 3
                        meanings['fisher'] = 'Strong Bull'
                    elif fisher_val > fisher_sig and fisher_val > -0.5:
                        scores['fisher'] = 2
                        meanings['fisher'] = 'Moderate Bull'
                    elif fisher_val > fisher_sig:
                        scores['fisher'] = 1
                        meanings['fisher'] = 'Weak Bull'
                    elif fisher_val < fisher_sig and fisher_val < 0:
                        scores['fisher'] = -3
                        meanings['fisher'] = 'Strong Bear'
                    elif fisher_val < fisher_sig and fisher_val < 0.5:
                        scores['fisher'] = -2
                        meanings['fisher'] = 'Moderate Bear'
                    elif fisher_val < fisher_sig:
                        scores['fisher'] = -1
                        meanings['fisher'] = 'Weak Bear'
                
                # TSI scoring
                if not pd.isna(tsi.iloc[-1]):
                    tsi_val = tsi.iloc[-1]
                    if tsi_val > 25:
                        scores['tsi'] = 3
                        meanings['tsi'] = 'Strong Bull'
                    elif tsi_val > 10:
                        scores['tsi'] = 2
                        meanings['tsi'] = 'Moderate Bull'
                    elif tsi_val > 0:
                        scores['tsi'] = 1
                        meanings['tsi'] = 'Weak Bull'
                    elif tsi_val < -25:
                        scores['tsi'] = -3
                        meanings['tsi'] = 'Strong Bear'
                    elif tsi_val < -10:
                        scores['tsi'] = -2
                        meanings['tsi'] = 'Moderate Bear'
                    elif tsi_val < 0:
                        scores['tsi'] = -1
                        meanings['tsi'] = 'Weak Bear'
                
                # Pivot Points scoring
                if not pd.isna(pivot.iloc[-1]):
                    current_price = latest_candle['close']
                    pivot_val = pivot.iloc[-1]
                    r1_val = r1.iloc[-1]
                    s1_val = s1.iloc[-1]
                    
                    if current_price > r1_val:
                        scores['pivot'] = 3
                        meanings['pivot'] = 'Above R1'
                    elif current_price > pivot_val:
                        scores['pivot'] = 1
                        meanings['pivot'] = 'Above Pivot'
                    elif current_price < s1_val:
                        scores['pivot'] = -3
                        meanings['pivot'] = 'Below S1'
                    elif current_price < pivot_val:
                        scores['pivot'] = -1
                        meanings['pivot'] = 'Below Pivot'
                
                # Dow Theory scoring
                if latest_idx < len(uptrend) and latest_idx < len(downtrend):
                    if uptrend.iloc[-1]:
                        scores['dow'] = 3
                        meanings['dow'] = 'Clear Uptrend'
                    elif uptrend.iloc[-5:].sum() >= 3:
                        scores['dow'] = 1
                        meanings['dow'] = 'Emerging Uptrend'
                    elif downtrend.iloc[-1]:
                        scores['dow'] = -3
                        meanings['dow'] = 'Clear Downtrend'
                    elif downtrend.iloc[-5:].sum() >= 3:
                        scores['dow'] = -1
                        meanings['dow'] = 'Emerging Downtrend'
                
                # Williams Alligator scoring
                if latest_idx < len(eating_up) and latest_idx < len(eating_down):
                    if eating_up.iloc[-1]:
                        scores['alligator'] = 3
                        meanings['alligator'] = 'Eating Up'
                    elif eating_up.iloc[-3:].sum() >= 2:
                        scores['alligator'] = 1
                        meanings['alligator'] = 'Waking Up'
                    elif eating_down.iloc[-1]:
                        scores['alligator'] = -3
                        meanings['alligator'] = 'Eating Down'
                    elif eating_down.iloc[-3:].sum() >= 2:
                        scores['alligator'] = -1
                        meanings['alligator'] = 'Waking Down'
                
                # Calculate total score
                total_score = sum(scores.values())
                
                # Determine overall trend
                if total_score >= bullish_threshold:
                    overall_trend = 'Bullish'
                    trend_strength = 'Strong'
                elif total_score <= bearish_threshold:
                    overall_trend = 'Bearish'
                    trend_strength = 'Strong'
                else:
                    overall_trend = 'Neutral'
                    trend_strength = 'Weak'
                
                return jsonify({
                    'success': True,
                    'overall_trend': overall_trend,
                    'trend_strength': trend_strength,
                    'total_score': total_score,
                    'max_score': 15,
                    'candle_close_time': last_hour_close.isoformat(),
                    'indicators': {
                        'fisher': {
                            'value': fisher.iloc[-1] if not pd.isna(fisher.iloc[-1]) else None,
                            'score': scores['fisher'],
                            'meaning': meanings['fisher']
                        },
                        'tsi': {
                            'value': tsi.iloc[-1] if not pd.isna(tsi.iloc[-1]) else None,
                            'score': scores['tsi'],
                            'meaning': meanings['tsi']
                        },
                        'pivot': {
                            'position': 'Above Pivot' if latest_candle['close'] > pivot.iloc[-1] else 'Below Pivot',
                            'score': scores['pivot'],
                            'meaning': meanings['pivot']
                        },
                        'dow': {
                            'signal': meanings['dow'],
                            'score': scores['dow'],
                            'meaning': meanings['dow']
                        },
                        'alligator': {
                            'state': meanings['alligator'],
                            'score': scores['alligator'],
                            'meaning': meanings['alligator']
                        }
                    },
                    'timestamp': datetime.now().isoformat()
                })
                
            except Exception as analysis_error:
                # Fallback to basic response if detailed analysis fails
                return jsonify({
                    'success': False,
                    'message': f'Analysis failed: {str(analysis_error)}',
                    'overall_trend': 'NEUTRAL',
                    'trend_strength': 'Unknown',
                    'total_score': 0,
                    'max_score': 15,
                    'candle_close_time': last_hour_close.isoformat(),
                    'indicators': {
                        'fisher': {'value': None, 'score': 0, 'meaning': 'Data unavailable'},
                        'tsi': {'value': None, 'score': 0, 'meaning': 'Data unavailable'},
                        'pivot': {'position': '--', 'score': 0, 'meaning': 'Data unavailable'},
                        'dow': {'signal': '--', 'score': 0, 'meaning': 'Data unavailable'},
                        'alligator': {'state': '--', 'score': 0, 'meaning': 'Data unavailable'}
                    },
                    'timestamp': datetime.now().isoformat()
                })
        else:
            return jsonify({
                'success': False,
                'message': 'No 1h data available from Delta Exchange',
                'overall_trend': 'NEUTRAL',
                'trend_strength': 'Unknown',
                'total_score': 0,
                'max_score': 15,
                'candle_close_time': last_hour_close.isoformat(),
                'indicators': {
                    'fisher': {'value': None, 'score': 0, 'meaning': 'No data'},
                    'tsi': {'value': None, 'score': 0, 'meaning': 'No data'},
                    'pivot': {'position': '--', 'score': 0, 'meaning': 'No data'},
                    'dow': {'signal': '--', 'score': 0, 'meaning': 'No data'},
                    'alligator': {'state': '--', 'score': 0, 'meaning': 'No data'}
                },
                'timestamp': datetime.now().isoformat()
            })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Failed to fetch higher timeframe data: {str(e)}',
            'overall_trend': 'NEUTRAL',
            'trend_strength': 'Error',
            'total_score': 0,
            'max_score': 15,
            'candle_close_time': last_hour_close.isoformat(),
            'indicators': {
                'fisher': {'value': None, 'score': 0, 'meaning': 'Connection error'},
                'tsi': {'value': None, 'score': 0, 'meaning': 'Connection error'},
                'pivot': {'position': '--', 'score': 0, 'meaning': 'Connection error'},
                'dow': {'signal': '--', 'score': 0, 'meaning': 'Connection error'},
                'alligator': {'state': '--', 'score': 0, 'meaning': 'Connection error'}
            },
            'timestamp': datetime.now().isoformat()
        })

@app.route('/start_bot', methods=['POST'])
def start_bot():
    """Start the trading bot"""
    global trading_bot_process, bot_running
    
    try:
        if bot_running:
            return jsonify({
                'success': False,
                'message': 'Bot is already running',
                'timestamp': datetime.now().isoformat()
            })
        
        import subprocess
        import sys
        
        # Start the bot as a separate process
        trading_bot_process = subprocess.Popen([
            sys.executable, 'delta_btc_strategy.py'
        ], cwd=os.getcwd())
        
        bot_running = True
        
        return jsonify({
            'success': True,
            'message': 'Trading bot started successfully',
            'pid': trading_bot_process.pid,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'Failed to start bot: {str(e)}',
            'timestamp': datetime.now().isoformat()
        })

@app.route('/stop_bot', methods=['POST'])
def stop_bot():
    """Stop the trading bot"""
    global trading_bot_process, bot_running
    
    try:
        if not bot_running or trading_bot_process is None:
            return jsonify({
                'success': False,
                'message': 'Bot is not running',
                'timestamp': datetime.now().isoformat()
            })
        
        # Terminate the bot process
        trading_bot_process.terminate()
        try:
            trading_bot_process.wait(timeout=10)  # Wait up to 10 seconds
        except subprocess.TimeoutExpired:
            trading_bot_process.kill()  # Force kill if it doesn't terminate gracefully
        
        bot_running = False
        trading_bot_process = None
        
        return jsonify({
            'success': True,
            'message': 'Trading bot stopped successfully',
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        bot_running = False
        trading_bot_process = None
        return jsonify({
            'success': False,
            'message': f'Error stopping bot: {str(e)}',
            'timestamp': datetime.now().isoformat()
        })

# All Excel formatting functions removed - using Delta Exchange only

@app.route('/backtesting')
def backtesting():
    """Backtesting page"""
    return render_template('backtesting.html')

@app.route('/api/backtest/simple', methods=['POST'])
def api_backtest_simple():
    """Simple non-streaming backtest endpoint"""
    import json
    from backtest_engine import BacktestEngine
    
    try:
        data = request.get_json()
        from_date = data.get('from_date', '2024-01-01')
        to_date = data.get('to_date', '2024-01-31')
        
        # Run backtest with existing cached clients
        delta_client = get_delta_client()
        strategy = get_strategy()
        engine = BacktestEngine(delta_client=delta_client, strategy=strategy)
        historical_data = engine.fetch_historical_data(from_date, to_date)
        results = engine.run_backtest(historical_data)
        
        # Return simple results
        simple_results = {
            'total_trades': results.get('trade_stats', {}).get('total_trades', 0),
            'total_pnl': results.get('summary', {}).get('total_pnl', 0.0),
            'win_rate': results.get('trade_stats', {}).get('win_rate_pct', 0.0)
        }
        
        return json.dumps({'success': True, 'results': simple_results})
        
    except Exception as e:
        return json.dumps({'success': False, 'error': str(e)})

@app.route('/api/backtest/test', methods=['POST'])
def api_backtest_test():
    """Simple test endpoint"""
    import json
    from flask import Response
    
    def generate_test():
        yield f"data: {json.dumps({'step': 'Test step 1', 'progress': 25})}\n\n"
        yield f"data: {json.dumps({'step': 'Test step 2', 'progress': 50})}\n\n"
        yield f"data: {json.dumps({'step': 'Test step 3', 'progress': 75})}\n\n"
        yield f"data: {json.dumps({'step': 'Test completed', 'progress': 100, 'results': {{'test': 'success'}}})}\n\n"
    
    response = Response(generate_test(), mimetype='text/event-stream')
    response.headers['Cache-Control'] = 'no-cache'
    response.headers['Connection'] = 'keep-alive'
    response.headers['X-Accel-Buffering'] = 'no'
    return response

@app.route('/api/backtest/run', methods=['POST'])
def api_backtest_run():
    """Run backtest with streaming progress updates"""
    import json
    from flask import Response
    from backtest_engine import BacktestEngine
    import time
    
    app_logger.info("DEBUG: Backtest request received")
    
    # Extract request data outside of generator to avoid context issues
    try:
        data = request.get_json()
        from_date = data.get('from_date')
        to_date = data.get('to_date')
        
        if not from_date or not to_date:
            return Response(
                f"data: {json.dumps({'error': 'Missing date parameters'})}\n\n",
                mimetype='text/plain'
            )
    except Exception as e:
        return Response(
            f"data: {json.dumps({'error': f'Request parsing failed: {str(e)}'})}\n\n",
            mimetype='text/plain'
        )
    
    def generate_backtest(from_date, to_date):
        try:
            
            # Initialize backtest engine with existing cached clients
            delta_client = get_delta_client()
            strategy = get_strategy()
            engine = BacktestEngine(delta_client=delta_client, strategy=strategy)
            
            # Progress callback for streaming updates
            def progress_callback(progress_data):
                yield f"data: {json.dumps(progress_data)}\n\n"
            
            # Fetch historical data with progress updates
            yield f"data: {json.dumps({'step': 'Initializing backtest', 'progress': 0})}\n\n"
            time.sleep(0.1)  # Small delay to ensure message is sent
            
            try:
                # Fetch historical data
                def data_progress(progress_info):
                    pass  # We'll handle progress inline
                
                yield f"data: {json.dumps({'step': 'Fetching historical data', 'progress': 10})}\n\n"
                historical_data = engine.fetch_historical_data(from_date, to_date)
                
                yield f"data: {json.dumps({'step': 'Data fetched successfully', 'progress': 30})}\n\n"
                
                # Run backtest with progress updates
                def backtest_progress(progress_info):
                    pass  # We'll handle progress inline
                
                yield f"data: {json.dumps({'step': 'Analyzing historical candles', 'progress': 40})}\n\n"
                yield f"data: {json.dumps({'step': 'Generating trading signals', 'progress': 50})}\n\n"
                yield f"data: {json.dumps({'step': 'Simulating trade execution', 'progress': 70})}\n\n"
                results = engine.run_backtest(historical_data)
                
                yield f"data: {json.dumps({'step': 'Calculating performance metrics', 'progress': 90})}\n\n"
                
                # Debug: Check if results is valid
                app_logger.info(f"DEBUG: Results type: {type(results)}")
                app_logger.info(f"DEBUG: Results keys: {results.keys() if isinstance(results, dict) else 'Not a dict'}")
                
                # Check if results contain large DataFrames
                if isinstance(results, dict):
                    for key, value in results.items():
                        if hasattr(value, 'shape'):  # DataFrame or numpy array
                            app_logger.info(f"DEBUG: {key} shape: {value.shape}")
                        elif isinstance(value, (list, tuple)):
                            app_logger.info(f"DEBUG: {key} length: {len(value)}")
                        else:
                            app_logger.info(f"DEBUG: {key} type: {type(value)}")
                
                app_logger.info("DEBUG: Reached make_json_safe section - about to define function")
                
                # Send final results with comprehensive error handling
                def make_json_safe(obj):
                    """Recursively convert objects to JSON-safe types"""
                    if isinstance(obj, dict):
                        return {str(k): make_json_safe(v) for k, v in obj.items()}
                    elif isinstance(obj, (list, tuple)):
                        return [make_json_safe(item) for item in obj]
                    elif hasattr(obj, 'isoformat'):  # datetime objects
                        return obj.isoformat()
                    elif hasattr(obj, 'item'):  # numpy scalars
                        return obj.item()
                    elif hasattr(obj, 'tolist'):  # numpy arrays
                        return obj.tolist()
                    elif obj is None or isinstance(obj, (str, int, float, bool)):
                        return obj
                    else:
                        return str(obj)
                
                try:
                    app_logger.info("DEBUG: Entered try block for JSON processing")
                    # Convert results to JSON-safe format with size limits
                    app_logger.info("DEBUG: About to call make_json_safe")
                    safe_results = make_json_safe(results)
                    app_logger.info("DEBUG: make_json_safe completed")
                    
                    
                    # Debug: Check safe_results content
                    app_logger.info(f"DEBUG: safe_results type: {type(safe_results)}")
                    if isinstance(safe_results, dict):
                        app_logger.info(f"DEBUG: safe_results keys: {list(safe_results.keys())}")
                        for key, value in safe_results.items():
                            if isinstance(value, (list, tuple)):
                                app_logger.info(f"DEBUG: safe_results[{key}] length: {len(value)}")
                            else:
                                app_logger.info(f"DEBUG: safe_results[{key}] type: {type(value)}")
                    
                    # Check JSON size before sending
                    final_payload = {'step': 'Backtest completed', 'progress': 100, 'results': safe_results}
                    json_str = json.dumps(final_payload)
                    json_size_mb = len(json_str.encode('utf-8')) / (1024 * 1024)
                    app_logger.info(f"Sending backtest results - JSON size: {json_size_mb:.2f} MB")
                    app_logger.info(f"DEBUG: JSON string length: {len(json_str)} characters")
                    app_logger.info(f"DEBUG: First 200 chars of JSON: {json_str[:200]}")
                    
                    if json_size_mb > 10:  # If larger than 10MB, send summary only
                        app_logger.warning(f"Results too large ({json_size_mb:.2f} MB), sending summary only")
                        summary_only = {
                            'summary': safe_results.get('summary', {}),
                            'trade_stats': safe_results.get('trade_stats', {}),
                            'risk_metrics': safe_results.get('risk_metrics', {}),
                            'settings_used': safe_results.get('settings_used', {}),
                            'size_warning': f'Full results too large ({json_size_mb:.2f} MB) - showing summary only'
                        }
                        final_response = f"data: {json.dumps({'step': 'Backtest completed', 'progress': 100, 'results': summary_only})}\n\n"
                        app_logger.info(f"DEBUG: About to yield summary response, length: {len(final_response)}")
                        yield final_response
                        app_logger.info("DEBUG: Summary response yielded successfully")
                    else:
                        final_response = f"data: {json.dumps({'step': 'Backtest completed', 'progress': 100, 'results': safe_results})}\n\n"
                        app_logger.info(f"DEBUG: About to yield full response, length: {len(final_response)}")
                        yield final_response
                        app_logger.info("DEBUG: Full response yielded successfully")
                except Exception as json_error:
                    app_logger.error(f"DEBUG: JSON serialization error: {json_error}")
                    # Send minimal summary if all else fails
                    minimal_summary = {
                        'summary': {
                            'total_trades': 0,
                            'total_pnl': 0.0,
                            'total_return_pct': 0.0,
                            'error': 'Results processing failed'
                        }
                    }
                    try:
                        # Try to extract basic info safely
                        if isinstance(results, dict):
                            trade_stats = results.get('trade_stats', {})
                            summary = results.get('summary', {})
                            minimal_summary['summary'].update({
                                'total_trades': int(trade_stats.get('total_trades', 0)) if trade_stats.get('total_trades') is not None else 0,
                                'total_pnl': float(summary.get('total_pnl', 0.0)) if summary.get('total_pnl') is not None else 0.0,
                                'total_return_pct': float(summary.get('total_return_pct', 0.0)) if summary.get('total_return_pct') is not None else 0.0
                            })
                    except:
                        pass
                    yield f"data: {json.dumps({'step': 'Backtest completed', 'progress': 100, 'results': minimal_summary})}\n\n"
                
            except Exception as e:
                logging.error(f"Backtest error: {e}")
                yield f"data: {json.dumps({'error': f'Backtest failed: {str(e)}'})}\n\n"
                
        except Exception as e:
            logging.error(f"Backtest setup error: {e}")
            yield f"data: {json.dumps({'error': f'Setup failed: {str(e)}'})}\n\n"
    
    # Return Server-Sent Events response
    response = Response(
        generate_backtest(from_date, to_date),
        mimetype='text/event-stream'
    )
    response.headers['Cache-Control'] = 'no-cache'
    response.headers['X-Accel-Buffering'] = 'no'
    response.headers['Access-Control-Allow-Origin'] = '*'
    return response

@app.route('/api/backtest/debug', methods=['POST'])
def api_backtest_debug():
    """Debug non-streaming backtest for troubleshooting"""
    try:
        data = request.get_json()
        from_date = data.get('from_date')
        to_date = data.get('to_date')
        
        if not from_date or not to_date:
            return {'success': False, 'error': 'Missing date parameters'}
        
        # Get clients
        delta_client = get_delta_client()
        strategy = get_strategy()
        
        # Run backtest
        from backtest_engine import BacktestEngine
        engine = BacktestEngine(delta_client=delta_client, strategy=strategy)
        historical_data = engine.fetch_historical_data(from_date, to_date)
        results = engine.run_backtest(historical_data)
        
        app_logger.info(f"Debug backtest completed, results keys: {list(results.keys()) if isinstance(results, dict) else 'not dict'}")
        
        return {'success': True, 'results': results}
        
    except Exception as e:
        app_logger.error(f"Debug backtest error: {e}")
        return {'success': False, 'error': str(e)}

@app.route('/api/config')
def api_config():
    """Get current configuration for backtesting"""
    try:
        config = config_manager.get_all_config()
        return jsonify(config)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/candle_times/<trading_start_time>')
def api_candle_times(trading_start_time):
    """Get formatted candle times for settings page display"""
    try:
        times_3m, times_1h = format_candle_times_display(trading_start_time)
        return jsonify({
            'status': 'success',
            '3m_times': times_3m,
            '1h_times': times_1h
        })
    except Exception as e:
        return jsonify({
            'status': 'error', 
            'message': str(e),
            '3m_times': '17:33, 17:36, 17:39...',
            '1h_times': '18:30, 19:30, 20:30...'
        })

if __name__ == '__main__':
    # Ensure templates and static directories exist
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static/css', exist_ok=True)
    os.makedirs('static/js', exist_ok=True)
    
    app.run(debug=True, host='0.0.0.0', port=5000)