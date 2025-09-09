import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class BTCHighWinRateStrategy:
    """
    Optimized BTC Options Trading Strategy targeting 80%+ win rate
    Based on comprehensive backtesting and parameter optimization
    """
    
    def __init__(self, portfolio_size=100000):
        self.portfolio_size = portfolio_size
        
        # Optimized parameters for high win rate - adjusted for 15m timeframe
        self.bb_period = 80  # 20 hours on 15m timeframe
        self.bb_std_dev = 2.5
        self.atr_period = 56  # 14 hours on 15m timeframe  
        self.volume_threshold = 2.0
        self.min_signal_strength = 4
        
        # Conservative risk management for high win rate
        self.max_risk_per_trade = 0.005  # 0.5% max risk per trade
        self.max_daily_loss = 0.015      # 1.5% max daily loss
        self.stop_loss_atr_multiplier = 1.5
        self.take_profit_atr_multiplier = 3.0
        
        # Trading controls
        self.max_positions = 2
        self.min_time_between_trades = 60  # 1 hour minimum
        
        self.data = None
        self.trades = []
        
    def fetch_btc_data(self, period="60d"):
        """Fetch BTC 15-minute data for intraday trading"""
        print("Fetching BTC/USD 15-minute data...")
        ticker = yf.Ticker("BTC-USD")
        self.data = ticker.history(period=period, interval="15m")
        print(f"Fetched {len(self.data)} 15-minute bars of data")
        return self.data
    
    def calculate_technical_indicators(self):
        """Calculate all technical indicators"""
        # Bollinger Bands
        self.data['BB_Middle'] = self.data['Close'].rolling(window=self.bb_period).mean()
        rolling_std = self.data['Close'].rolling(window=self.bb_period).std()
        self.data['BB_Upper'] = self.data['BB_Middle'] + (rolling_std * self.bb_std_dev)
        self.data['BB_Lower'] = self.data['BB_Middle'] - (rolling_std * self.bb_std_dev)
        self.data['BB_Position'] = (self.data['Close'] - self.data['BB_Lower']) / (self.data['BB_Upper'] - self.data['BB_Lower'])
        self.data['BB_Width'] = (self.data['BB_Upper'] - self.data['BB_Lower']) / self.data['BB_Middle']
        
        # ATR
        high = self.data['High']
        low = self.data['Low']
        close = self.data['Close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        self.data['ATR'] = true_range.rolling(window=self.atr_period).mean()
        self.data['ATR_Pct'] = self.data['ATR'] / self.data['Close']
        
        # Volume indicators
        self.data['Volume_SMA'] = self.data['Volume'].rolling(window=20).mean()
        self.data['Volume_Ratio'] = self.data['Volume'] / self.data['Volume_SMA']
        
        # On Balance Volume
        price_change = np.sign(self.data['Close'].diff())
        self.data['OBV'] = (price_change * self.data['Volume']).fillna(0).cumsum()
        self.data['OBV_SMA'] = self.data['OBV'].rolling(window=10).mean()
        
        # Price action patterns
        body_size = abs(self.data['Close'] - self.data['Open'])
        candle_range = self.data['High'] - self.data['Low']
        
        # Hammer pattern
        lower_shadow = np.minimum(self.data['Open'], self.data['Close']) - self.data['Low']
        upper_shadow = self.data['High'] - np.maximum(self.data['Open'], self.data['Close'])
        self.data['Hammer'] = (lower_shadow > 2 * body_size) & (upper_shadow < body_size) & (candle_range > 0)
        
        # Engulfing patterns
        bullish_engulfing = ((self.data['Close'] > self.data['Open']) & 
                           (self.data['Close'].shift(1) < self.data['Open'].shift(1)) &
                           (self.data['Open'] < self.data['Close'].shift(1)) &
                           (self.data['Close'] > self.data['Open'].shift(1)))
        
        bearish_engulfing = ((self.data['Close'] < self.data['Open']) & 
                           (self.data['Close'].shift(1) > self.data['Open'].shift(1)) &
                           (self.data['Open'] > self.data['Close'].shift(1)) &
                           (self.data['Close'] < self.data['Open'].shift(1)))
        
        self.data['Bullish_Engulfing'] = bullish_engulfing.fillna(False)
        self.data['Bearish_Engulfing'] = bearish_engulfing.fillna(False)
        
        # Additional indicators for confirmation
        self.data['RSI'] = self.calculate_rsi()
        self.data['MACD'], self.data['MACD_Signal'] = self.calculate_macd()
        
    def calculate_rsi(self, period=14):
        """Calculate RSI"""
        delta = self.data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def calculate_macd(self, fast=12, slow=26, signal=9):
        """Calculate MACD"""
        exp1 = self.data['Close'].ewm(span=fast).mean()
        exp2 = self.data['Close'].ewm(span=slow).mean()
        macd = exp1 - exp2
        macd_signal = macd.ewm(span=signal).mean()
        return macd, macd_signal
    
    def generate_high_probability_signals(self):
        """Generate very selective high-probability signals"""
        self.data['Signal'] = 0
        self.data['Signal_Strength'] = 0
        self.data['Entry_Type'] = ''
        self.data['Confidence'] = 0.0
        
        for i in range(30, len(self.data)):  # Skip first 30 for indicator stability
            long_score = 0
            short_score = 0
            
            current_row = self.data.iloc[i]
            prev_rows = self.data.iloc[i-10:i]
            
            # LONG SIGNAL CONDITIONS (Very restrictive for high win rate)
            
            # 1. Extreme oversold on Bollinger Bands
            if current_row['BB_Position'] < 0.1:  # Very oversold
                long_score += 2
            elif current_row['BB_Position'] < 0.2:  # Moderately oversold
                long_score += 1
            
            # 2. High volume confirmation (very important)
            if current_row['Volume_Ratio'] > self.volume_threshold:
                long_score += 2
            elif current_row['Volume_Ratio'] > 1.5:
                long_score += 1
            
            # 3. Price action reversal patterns
            if current_row['Hammer'] or current_row['Bullish_Engulfing']:
                long_score += 2
            
            # 4. Volatility expansion
            if current_row['ATR_Pct'] > prev_rows['ATR_Pct'].mean() * 1.2:
                long_score += 1
            
            # 5. Volume flow confirmation
            if current_row['OBV'] > current_row['OBV_SMA']:
                long_score += 1
            
            # 6. RSI oversold but turning up
            if current_row['RSI'] < 35 and current_row['RSI'] > self.data['RSI'].iloc[i-1]:
                long_score += 1
            
            # 7. MACD bullish divergence
            if (current_row['MACD'] > current_row['MACD_Signal'] and 
                self.data['MACD'].iloc[i-1] <= self.data['MACD_Signal'].iloc[i-1]):
                long_score += 1
            
            # 8. Price bouncing from support (BB lower band)
            if (current_row['Close'] > current_row['BB_Lower'] and 
                self.data['Low'].iloc[i-1] <= self.data['BB_Lower'].iloc[i-1] and
                current_row['Close'] > self.data['Close'].iloc[i-1]):
                long_score += 2
            
            # SHORT SIGNAL CONDITIONS (Very restrictive)
            
            # 1. Extreme overbought on Bollinger Bands
            if current_row['BB_Position'] > 0.9:  # Very overbought
                short_score += 2
            elif current_row['BB_Position'] > 0.8:  # Moderately overbought
                short_score += 1
            
            # 2. High volume confirmation
            if current_row['Volume_Ratio'] > self.volume_threshold:
                short_score += 2
            elif current_row['Volume_Ratio'] > 1.5:
                short_score += 1
            
            # 3. Bearish reversal patterns
            if current_row['Bearish_Engulfing']:
                short_score += 2
            
            # 4. Volatility expansion
            if current_row['ATR_Pct'] > prev_rows['ATR_Pct'].mean() * 1.2:
                short_score += 1
            
            # 5. Volume flow confirmation
            if current_row['OBV'] < current_row['OBV_SMA']:
                short_score += 1
            
            # 6. RSI overbought and turning down
            if current_row['RSI'] > 65 and current_row['RSI'] < self.data['RSI'].iloc[i-1]:
                short_score += 1
            
            # 7. MACD bearish divergence
            if (current_row['MACD'] < current_row['MACD_Signal'] and 
                self.data['MACD'].iloc[i-1] >= self.data['MACD_Signal'].iloc[i-1]):
                short_score += 1
            
            # 8. Price rejecting resistance (BB upper band)
            if (current_row['Close'] < current_row['BB_Upper'] and 
                self.data['High'].iloc[i-1] >= self.data['BB_Upper'].iloc[i-1] and
                current_row['Close'] < self.data['Close'].iloc[i-1]):
                short_score += 2
            
            # Generate signals only for very high confidence setups
            if long_score >= self.min_signal_strength:
                self.data.loc[self.data.index[i], 'Signal'] = 1
                self.data.loc[self.data.index[i], 'Signal_Strength'] = long_score
                self.data.loc[self.data.index[i], 'Entry_Type'] = 'CALL'
                self.data.loc[self.data.index[i], 'Confidence'] = min(long_score / 10.0, 1.0)
                
            elif short_score >= self.min_signal_strength:
                self.data.loc[self.data.index[i], 'Signal'] = -1
                self.data.loc[self.data.index[i], 'Signal_Strength'] = short_score
                self.data.loc[self.data.index[i], 'Entry_Type'] = 'PUT'
                self.data.loc[self.data.index[i], 'Confidence'] = min(short_score / 10.0, 1.0)
    
    def calculate_position_sizing(self):
        """Calculate conservative position sizing"""
        self.data['Stop_Loss'] = np.where(
            self.data['Signal'] == 1,
            self.data['Close'] - (self.data['ATR'] * self.stop_loss_atr_multiplier),
            np.where(
                self.data['Signal'] == -1,
                self.data['Close'] + (self.data['ATR'] * self.stop_loss_atr_multiplier),
                np.nan
            )
        )
        
        self.data['Take_Profit'] = np.where(
            self.data['Signal'] == 1,
            self.data['Close'] + (self.data['ATR'] * self.take_profit_atr_multiplier),
            np.where(
                self.data['Signal'] == -1,
                self.data['Close'] - (self.data['ATR'] * self.take_profit_atr_multiplier),
                np.nan
            )
        )
        
        # Very conservative position sizing
        max_risk_amount = self.portfolio_size * self.max_risk_per_trade
        self.data['Risk_Per_Share'] = abs(self.data['Close'] - self.data['Stop_Loss'])
        
        self.data['Position_Size'] = np.where(
            (self.data['Signal'] != 0) & (self.data['Risk_Per_Share'] > 0),
            max_risk_amount / self.data['Risk_Per_Share'],
            0
        )
        
        # For options - calculate contracts (minimum 1, maximum 5 for safety)
        self.data['Option_Contracts'] = np.where(
            self.data['Signal'] != 0,
            np.clip(self.data['Position_Size'] / 100, 1, 5),
            0
        )
    
    def backtest_strategy(self):
        """Backtest the high win rate strategy"""
        trades = []
        current_position = None
        
        for i, (date, row) in enumerate(self.data.iterrows()):
            if row['Signal'] != 0 and current_position is None:
                # Enter position
                current_position = {
                    'entry_date': date,
                    'entry_price': row['Close'],
                    'signal': row['Signal'],
                    'stop_loss': row['Stop_Loss'],
                    'take_profit': row['Take_Profit'],
                    'position_size': row['Position_Size'],
                    'entry_type': row['Entry_Type'],
                    'confidence': row['Confidence'],
                    'signal_strength': row['Signal_Strength']
                }
                
            elif current_position is not None:
                # Check exit conditions
                exit_trade = False
                exit_reason = ''
                exit_price = row['Close']
                
                if current_position['signal'] == 1:  # Long position
                    if row['Low'] <= current_position['stop_loss']:
                        exit_trade = True
                        exit_reason = 'Stop Loss'
                        exit_price = current_position['stop_loss']
                    elif row['High'] >= current_position['take_profit']:
                        exit_trade = True
                        exit_reason = 'Take Profit'
                        exit_price = current_position['take_profit']
                else:  # Short position
                    if row['High'] >= current_position['stop_loss']:
                        exit_trade = True
                        exit_reason = 'Stop Loss'
                        exit_price = current_position['stop_loss']
                    elif row['Low'] <= current_position['take_profit']:
                        exit_trade = True
                        exit_reason = 'Take Profit'
                        exit_price = current_position['take_profit']
                
                # Force exit after 5 days (for daily options simulation)
                days_in_trade = (date - current_position['entry_date']).days
                if days_in_trade >= 5:
                    exit_trade = True
                    exit_reason = 'Time Exit'
                    exit_price = row['Close']
                
                if exit_trade:
                    # Calculate P&L
                    if current_position['signal'] == 1:
                        pnl = (exit_price - current_position['entry_price']) * current_position['position_size']
                        pnl_pct = (exit_price - current_position['entry_price']) / current_position['entry_price']
                    else:
                        pnl = (current_position['entry_price'] - exit_price) * current_position['position_size']
                        pnl_pct = (current_position['entry_price'] - exit_price) / current_position['entry_price']
                    
                    trades.append({
                        'entry_date': current_position['entry_date'],
                        'exit_date': date,
                        'entry_price': current_position['entry_price'],
                        'exit_price': exit_price,
                        'signal': current_position['signal'],
                        'pnl': pnl,
                        'pnl_pct': pnl_pct * 100,
                        'exit_reason': exit_reason,
                        'entry_type': current_position['entry_type'],
                        'confidence': current_position['confidence'],
                        'signal_strength': current_position['signal_strength'],
                        'days_in_trade': days_in_trade
                    })
                    
                    current_position = None
        
        self.trades = pd.DataFrame(trades)
        return self.trades
    
    def analyze_performance(self):
        """Comprehensive performance analysis"""
        if self.trades.empty:
            return {}
        
        winning_trades = self.trades[self.trades['pnl'] > 0]
        losing_trades = self.trades[self.trades['pnl'] <= 0]
        
        total_trades = len(self.trades)
        win_rate = len(winning_trades) / total_trades * 100 if total_trades > 0 else 0
        
        avg_win = winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0
        avg_loss = losing_trades['pnl'].mean() if len(losing_trades) > 0 else 0
        
        avg_win_pct = winning_trades['pnl_pct'].mean() if len(winning_trades) > 0 else 0
        avg_loss_pct = losing_trades['pnl_pct'].mean() if len(losing_trades) > 0 else 0
        
        profit_factor = abs(avg_win * len(winning_trades) / (avg_loss * len(losing_trades))) if avg_loss != 0 else float('inf')
        
        total_pnl = self.trades['pnl'].sum()
        max_win = winning_trades['pnl'].max() if len(winning_trades) > 0 else 0
        max_loss = losing_trades['pnl'].min() if len(losing_trades) > 0 else 0
        
        # Calculate max drawdown
        cumulative_pnl = self.trades['pnl'].cumsum()
        rolling_max = cumulative_pnl.expanding().max()
        drawdown = (cumulative_pnl - rolling_max)
        max_drawdown = drawdown.min()
        max_drawdown_pct = (max_drawdown / self.portfolio_size) * 100 if max_drawdown != 0 else 0
        
        # Average days in trade
        avg_days_in_trade = self.trades['days_in_trade'].mean()
        
        performance = {
            'Total Trades': total_trades,
            'Win Rate (%)': win_rate,
            'Winning Trades': len(winning_trades),
            'Losing Trades': len(losing_trades),
            'Average Win ($)': avg_win,
            'Average Loss ($)': avg_loss,
            'Average Win (%)': avg_win_pct,
            'Average Loss (%)': avg_loss_pct,
            'Profit Factor': profit_factor,
            'Total P&L ($)': total_pnl,
            'Max Win ($)': max_win,
            'Max Loss ($)': max_loss,
            'Max Drawdown ($)': max_drawdown,
            'Max Drawdown (%)': max_drawdown_pct,
            'Average Days in Trade': avg_days_in_trade,
            'Total Return (%)': (total_pnl / self.portfolio_size) * 100
        }
        
        return performance
    
    def run_complete_analysis(self):
        """Run complete strategy analysis"""
        print("=" * 80)
        print("BTC HIGH WIN-RATE OPTIONS STRATEGY - OPTIMIZED VERSION")
        print("=" * 80)
        
        # Fetch data and calculate indicators
        self.fetch_btc_data("2y")
        self.calculate_technical_indicators()
        self.generate_high_probability_signals()
        self.calculate_position_sizing()
        
        # Backtest
        trades = self.backtest_strategy()
        performance = self.analyze_performance()
        
        print(f"\nSTRATEGY PERFORMANCE SUMMARY")
        print("-" * 50)
        for key, value in performance.items():
            if isinstance(value, float):
                if '$' in key or 'P&L' in key or 'Win' in key or 'Loss' in key:
                    print(f"{key:<25}: ${value:>10.2f}")
                elif '%' in key or 'Rate' in key or 'Return' in key:
                    print(f"{key:<25}: {value:>10.1f}%")
                else:
                    print(f"{key:<25}: {value:>10.2f}")
            else:
                print(f"{key:<25}: {value:>10}")
        
        # Signal analysis
        signals = self.data[self.data['Signal'] != 0]
        print(f"\nSIGNAL ANALYSIS")
        print("-" * 50)
        print(f"Total Signals Generated: {len(signals)}")
        print(f"Signals with Strength >= 4: {len(signals[signals['Signal_Strength'] >= 4])}")
        print(f"Signals with Strength >= 6: {len(signals[signals['Signal_Strength'] >= 6])}")
        print(f"Average Signal Strength: {signals['Signal_Strength'].mean():.1f}")
        print(f"Average Confidence: {signals['Confidence'].mean():.2f}")
        
        # Strategy assessment
        win_rate = performance['Win Rate (%)']
        print(f"\nSTRATEGY ASSESSMENT")
        print("-" * 50)
        if win_rate >= 80:
            print(f"TARGET ACHIEVED: {win_rate:.1f}% win rate exceeds 80% target!")
            print("Strategy is ready for deployment with high confidence.")
        elif win_rate >= 70:
            print(f"GOOD PERFORMANCE: {win_rate:.1f}% win rate is strong but below 80% target")
            print("Consider further optimization or smaller position sizes.")
        else:
            print(f"NEEDS IMPROVEMENT: {win_rate:.1f}% win rate below target")
            print("Strategy needs further refinement before deployment.")
        
        print(f"\nOPTIMIZED PARAMETERS")
        print("-" * 50)
        print(f"Bollinger Bands: {self.bb_period} period, {self.bb_std_dev} std dev")
        print(f"ATR Period: {self.atr_period}")
        print(f"Volume Threshold: {self.volume_threshold}x")
        print(f"Minimum Signal Strength: {self.min_signal_strength}")
        print(f"Risk per Trade: {self.max_risk_per_trade*100}%")
        print(f"Stop Loss: {self.stop_loss_atr_multiplier}x ATR")
        print(f"Take Profit: {self.take_profit_atr_multiplier}x ATR")
        
        return performance, trades

def main():
    """Run the optimized high win-rate strategy"""
    strategy = BTCHighWinRateStrategy(portfolio_size=100000)
    performance, trades = strategy.run_complete_analysis()
    
    print(f"\nDEPLOYMENT RECOMMENDATIONS")
    print("-" * 50)
    print("• Use daily expiry BTC options on established exchanges")
    print("• Trade only during high volume periods (US/Asian market overlap)")
    print("• Maximum 2 concurrent positions")
    print("• Strict adherence to signal strength requirements")
    print("• Daily portfolio review and risk assessment")
    print("• Consider paper trading first to validate live performance")

if __name__ == "__main__":
    main()