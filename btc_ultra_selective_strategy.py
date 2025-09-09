import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class BTCUltraSelectiveStrategy:
    """
    Ultra-selective BTC Options Trading Strategy targeting 80%+ win rate
    Uses extremely restrictive entry criteria and enhanced confirmation signals
    """
    
    def __init__(self, portfolio_size=100000):
        self.portfolio_size = portfolio_size
        
        # Ultra-conservative parameters for maximum win rate
        self.bb_period = 20
        self.bb_std_dev = 2.8  # Wider bands for clearer extremes
        self.atr_period = 14
        self.volume_threshold = 3.0  # Very high volume requirement
        self.min_signal_strength = 6  # Extremely high confirmation requirement
        
        # Ultra-conservative risk management
        self.max_risk_per_trade = 0.003  # 0.3% max risk per trade
        self.max_daily_loss = 0.01       # 1% max daily loss
        self.stop_loss_atr_multiplier = 1.2  # Very tight stops
        self.take_profit_atr_multiplier = 2.5  # Conservative targets
        
        # Strict trading controls
        self.max_positions = 1  # Only one position at a time
        self.min_time_between_trades = 120  # 2 hours minimum
        
        self.data = None
        self.trades = []
        
    def fetch_btc_data(self, period="3y"):
        """Fetch extended BTC data for more robust analysis"""
        print("Fetching BTC/USD data...")
        ticker = yf.Ticker("BTC-USD")
        self.data = ticker.history(period=period, interval="1d")
        print(f"Fetched {len(self.data)} days of data")
        return self.data
    
    def calculate_advanced_indicators(self):
        """Calculate comprehensive technical indicators with additional confirmations"""
        # Enhanced Bollinger Bands
        self.data['BB_Middle'] = self.data['Close'].rolling(window=self.bb_period).mean()
        rolling_std = self.data['Close'].rolling(window=self.bb_period).std()
        self.data['BB_Upper'] = self.data['BB_Middle'] + (rolling_std * self.bb_std_dev)
        self.data['BB_Lower'] = self.data['BB_Middle'] - (rolling_std * self.bb_std_dev)
        self.data['BB_Position'] = (self.data['Close'] - self.data['BB_Lower']) / (self.data['BB_Upper'] - self.data['BB_Lower'])
        self.data['BB_Width'] = (self.data['BB_Upper'] - self.data['BB_Lower']) / self.data['BB_Middle']
        
        # BB squeeze detection
        self.data['BB_Squeeze'] = self.data['BB_Width'] < self.data['BB_Width'].rolling(20).quantile(0.2)
        
        # Enhanced ATR with percentile ranking
        high = self.data['High']
        low = self.data['Low']
        close = self.data['Close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        self.data['ATR'] = true_range.rolling(window=self.atr_period).mean()
        self.data['ATR_Pct'] = self.data['ATR'] / self.data['Close']
        self.data['ATR_Percentile'] = self.data['ATR_Pct'].rolling(100).rank(pct=True)
        
        # Advanced volume analysis
        self.data['Volume_SMA_10'] = self.data['Volume'].rolling(window=10).mean()
        self.data['Volume_SMA_30'] = self.data['Volume'].rolling(window=30).mean()
        self.data['Volume_Ratio'] = self.data['Volume'] / self.data['Volume_SMA_30']
        self.data['Volume_Spike'] = self.data['Volume'] > (self.data['Volume_SMA_30'] * 2.5)
        
        # Enhanced OBV with trend analysis
        price_change = np.sign(self.data['Close'].diff())
        self.data['OBV'] = (price_change * self.data['Volume']).fillna(0).cumsum()
        self.data['OBV_SMA'] = self.data['OBV'].rolling(window=20).mean()
        self.data['OBV_Trend'] = self.data['OBV'] > self.data['OBV'].shift(5)
        
        # Multiple RSI timeframes
        self.data['RSI_14'] = self.calculate_rsi(14)
        self.data['RSI_21'] = self.calculate_rsi(21)
        self.data['RSI_Oversold'] = (self.data['RSI_14'] < 25) & (self.data['RSI_21'] < 30)
        self.data['RSI_Overbought'] = (self.data['RSI_14'] > 75) & (self.data['RSI_21'] > 70)
        
        # Enhanced MACD
        self.data['MACD'], self.data['MACD_Signal'] = self.calculate_macd()
        self.data['MACD_Histogram'] = self.data['MACD'] - self.data['MACD_Signal']
        self.data['MACD_Bullish'] = (self.data['MACD'] > self.data['MACD_Signal']) & (self.data['MACD_Histogram'] > self.data['MACD_Histogram'].shift(1))
        self.data['MACD_Bearish'] = (self.data['MACD'] < self.data['MACD_Signal']) & (self.data['MACD_Histogram'] < self.data['MACD_Histogram'].shift(1))
        
        # Price action patterns (enhanced)
        body_size = abs(self.data['Close'] - self.data['Open'])
        candle_range = self.data['High'] - self.data['Low']
        lower_shadow = np.minimum(self.data['Open'], self.data['Close']) - self.data['Low']
        upper_shadow = self.data['High'] - np.maximum(self.data['Open'], self.data['Close'])
        
        # Enhanced reversal patterns
        self.data['Strong_Hammer'] = (lower_shadow > 3 * body_size) & (upper_shadow < body_size * 0.5) & (candle_range > 0)
        self.data['Doji_Reversal'] = (body_size / candle_range < 0.05) & (candle_range > self.data['ATR'] * 0.5)
        
        # Enhanced engulfing patterns
        prev_body = abs(self.data['Close'].shift(1) - self.data['Open'].shift(1))
        self.data['Strong_Bullish_Engulfing'] = (
            (self.data['Close'] > self.data['Open']) & 
            (self.data['Close'].shift(1) < self.data['Open'].shift(1)) &
            (self.data['Open'] < self.data['Close'].shift(1)) &
            (self.data['Close'] > self.data['Open'].shift(1)) &
            (body_size > prev_body * 1.5) &  # Significantly larger body
            (self.data['Volume'] > self.data['Volume_SMA_10'])
        )
        
        self.data['Strong_Bearish_Engulfing'] = (
            (self.data['Close'] < self.data['Open']) & 
            (self.data['Close'].shift(1) > self.data['Open'].shift(1)) &
            (self.data['Open'] > self.data['Close'].shift(1)) &
            (self.data['Close'] < self.data['Open'].shift(1)) &
            (body_size > prev_body * 1.5) &  # Significantly larger body
            (self.data['Volume'] > self.data['Volume_SMA_10'])
        )
        
        # Trend confirmation
        self.data['SMA_50'] = self.data['Close'].rolling(window=50).mean()
        self.data['SMA_200'] = self.data['Close'].rolling(window=200).mean()
        self.data['Uptrend'] = self.data['SMA_50'] > self.data['SMA_200']
        self.data['Downtrend'] = self.data['SMA_50'] < self.data['SMA_200']
        
        # Market structure analysis
        self.data['Higher_High'] = (self.data['High'] > self.data['High'].shift(1)) & (self.data['High'].shift(1) > self.data['High'].shift(2))
        self.data['Lower_Low'] = (self.data['Low'] < self.data['Low'].shift(1)) & (self.data['Low'].shift(1) < self.data['Low'].shift(2))
        
    def calculate_rsi(self, period=14):
        """Enhanced RSI calculation"""
        delta = self.data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def calculate_macd(self, fast=12, slow=26, signal=9):
        """Enhanced MACD calculation"""
        exp1 = self.data['Close'].ewm(span=fast).mean()
        exp2 = self.data['Close'].ewm(span=slow).mean()
        macd = exp1 - exp2
        macd_signal = macd.ewm(span=signal).mean()
        return macd, macd_signal
    
    def generate_ultra_selective_signals(self):
        """Generate extremely selective signals with multiple confirmations"""
        self.data['Signal'] = 0
        self.data['Signal_Strength'] = 0
        self.data['Entry_Type'] = ''
        self.data['Confidence'] = 0.0
        
        for i in range(200, len(self.data)):  # Skip first 200 for indicator stability
            long_score = 0
            short_score = 0
            
            current_row = self.data.iloc[i]
            prev_5 = self.data.iloc[i-5:i]
            prev_20 = self.data.iloc[i-20:i]
            
            # ULTRA-SELECTIVE LONG SIGNAL CONDITIONS
            
            # 1. Extreme BB oversold with bounce confirmation
            if current_row['BB_Position'] < 0.05:  # Extremely oversold
                long_score += 3
            elif current_row['BB_Position'] < 0.1 and current_row['Close'] > current_row['BB_Lower']:
                long_score += 2
            
            # 2. Massive volume spike (critical for high win rate)
            if current_row['Volume_Ratio'] > self.volume_threshold:
                long_score += 3
                if current_row['Volume_Spike']:  # Additional bonus for extreme volume
                    long_score += 1
            
            # 3. Strong reversal patterns with volume confirmation
            if (current_row['Strong_Hammer'] or current_row['Strong_Bullish_Engulfing']) and current_row['Volume_Ratio'] > 1.5:
                long_score += 3
            elif current_row['Doji_Reversal'] and current_row['BB_Position'] < 0.2:
                long_score += 2
            
            # 4. Volatility expansion from squeeze
            if current_row['ATR_Percentile'] > 0.7 and prev_5['BB_Squeeze'].any():
                long_score += 2
            
            # 5. Multiple RSI confirmations
            if current_row['RSI_Oversold']:
                long_score += 2
                # RSI turning up from oversold
                if current_row['RSI_14'] > self.data['RSI_14'].iloc[i-1]:
                    long_score += 1
            
            # 6. MACD bullish momentum
            if current_row['MACD_Bullish']:
                long_score += 2
            
            # 7. OBV trend confirmation
            if current_row['OBV_Trend'] and current_row['OBV'] > current_row['OBV_SMA']:
                long_score += 2
            
            # 8. Market structure support (only in uptrend for highest probability)
            if current_row['Uptrend'] and not current_row['Lower_Low']:
                long_score += 1
            
            # 9. Multi-timeframe price action confirmation
            if (current_row['Close'] > prev_5['Close'].mean() and 
                current_row['High'] > prev_5['High'].max() * 0.98):
                long_score += 1
            
            # 10. Momentum divergence (price vs volume)
            if (current_row['Close'] > self.data['Close'].iloc[i-5] and
                current_row['Volume'] > prev_5['Volume'].mean() * 1.5):
                long_score += 1
            
            # ULTRA-SELECTIVE SHORT SIGNAL CONDITIONS
            
            # 1. Extreme BB overbought with rejection
            if current_row['BB_Position'] > 0.95:  # Extremely overbought
                short_score += 3
            elif current_row['BB_Position'] > 0.9 and current_row['Close'] < current_row['BB_Upper']:
                short_score += 2
            
            # 2. Massive volume spike
            if current_row['Volume_Ratio'] > self.volume_threshold:
                short_score += 3
                if current_row['Volume_Spike']:
                    short_score += 1
            
            # 3. Strong bearish reversal patterns
            if current_row['Strong_Bearish_Engulfing'] and current_row['Volume_Ratio'] > 1.5:
                short_score += 3
            elif current_row['Doji_Reversal'] and current_row['BB_Position'] > 0.8:
                short_score += 2
            
            # 4. Volatility expansion from squeeze
            if current_row['ATR_Percentile'] > 0.7 and prev_5['BB_Squeeze'].any():
                short_score += 2
            
            # 5. Multiple RSI confirmations
            if current_row['RSI_Overbought']:
                short_score += 2
                # RSI turning down from overbought
                if current_row['RSI_14'] < self.data['RSI_14'].iloc[i-1]:
                    short_score += 1
            
            # 6. MACD bearish momentum
            if current_row['MACD_Bearish']:
                short_score += 2
            
            # 7. OBV trend divergence
            if not current_row['OBV_Trend'] and current_row['OBV'] < current_row['OBV_SMA']:
                short_score += 2
            
            # 8. Market structure resistance (only in downtrend for highest probability)
            if current_row['Downtrend'] and not current_row['Higher_High']:
                short_score += 1
            
            # 9. Multi-timeframe price action confirmation
            if (current_row['Close'] < prev_5['Close'].mean() and 
                current_row['Low'] < prev_5['Low'].min() * 1.02):
                short_score += 1
            
            # 10. Momentum divergence (price vs volume)
            if (current_row['Close'] < self.data['Close'].iloc[i-5] and
                current_row['Volume'] > prev_5['Volume'].mean() * 1.5):
                short_score += 1
            
            # Generate signals only for ULTRA-HIGH confidence setups
            if long_score >= self.min_signal_strength:
                # Additional final filter: must be near recent low
                recent_low = prev_20['Low'].min()
                if current_row['Close'] <= recent_low * 1.05:  # Within 5% of recent low
                    self.data.loc[self.data.index[i], 'Signal'] = 1
                    self.data.loc[self.data.index[i], 'Signal_Strength'] = long_score
                    self.data.loc[self.data.index[i], 'Entry_Type'] = 'CALL'
                    self.data.loc[self.data.index[i], 'Confidence'] = min(long_score / 15.0, 1.0)
                
            elif short_score >= self.min_signal_strength:
                # Additional final filter: must be near recent high
                recent_high = prev_20['High'].max()
                if current_row['Close'] >= recent_high * 0.95:  # Within 5% of recent high
                    self.data.loc[self.data.index[i], 'Signal'] = -1
                    self.data.loc[self.data.index[i], 'Signal_Strength'] = short_score
                    self.data.loc[self.data.index[i], 'Entry_Type'] = 'PUT'
                    self.data.loc[self.data.index[i], 'Confidence'] = min(short_score / 15.0, 1.0)
    
    def calculate_position_sizing(self):
        """Ultra-conservative position sizing"""
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
        
        # Ultra-conservative position sizing
        max_risk_amount = self.portfolio_size * self.max_risk_per_trade
        self.data['Risk_Per_Share'] = abs(self.data['Close'] - self.data['Stop_Loss'])
        
        self.data['Position_Size'] = np.where(
            (self.data['Signal'] != 0) & (self.data['Risk_Per_Share'] > 0),
            max_risk_amount / self.data['Risk_Per_Share'],
            0
        )
        
        # For options - ultra-conservative contract sizing
        self.data['Option_Contracts'] = np.where(
            self.data['Signal'] != 0,
            np.clip(self.data['Position_Size'] / 100, 1, 3),  # Maximum 3 contracts
            0
        )
    
    def backtest_strategy(self):
        """Backtest with enhanced exit logic"""
        trades = []
        current_position = None
        
        for i, (date, row) in enumerate(self.data.iterrows()):
            if row['Signal'] != 0 and current_position is None:
                # Enter position with additional validation
                current_position = {
                    'entry_date': date,
                    'entry_price': row['Close'],
                    'signal': row['Signal'],
                    'stop_loss': row['Stop_Loss'],
                    'take_profit': row['Take_Profit'],
                    'position_size': row['Position_Size'],
                    'entry_type': row['Entry_Type'],
                    'confidence': row['Confidence'],
                    'signal_strength': row['Signal_Strength'],
                    'entry_atr': row['ATR']
                }
                
            elif current_position is not None:
                # Enhanced exit conditions
                exit_trade = False
                exit_reason = ''
                exit_price = row['Close']
                
                days_in_trade = (date - current_position['entry_date']).days
                
                if current_position['signal'] == 1:  # Long position
                    # Stop loss
                    if row['Low'] <= current_position['stop_loss']:
                        exit_trade = True
                        exit_reason = 'Stop Loss'
                        exit_price = current_position['stop_loss']
                    
                    # Take profit
                    elif row['High'] >= current_position['take_profit']:
                        exit_trade = True
                        exit_reason = 'Take Profit'
                        exit_price = current_position['take_profit']
                    
                    # Trailing stop (move stop to breakeven after 50% profit)
                    elif row['Close'] >= current_position['entry_price'] * 1.02:  # 2% profit
                        profit_target = current_position['entry_price'] + (current_position['entry_atr'] * 0.5)
                        if row['Low'] <= profit_target:
                            exit_trade = True
                            exit_reason = 'Trailing Stop'
                            exit_price = profit_target
                    
                else:  # Short position
                    # Stop loss
                    if row['High'] >= current_position['stop_loss']:
                        exit_trade = True
                        exit_reason = 'Stop Loss'
                        exit_price = current_position['stop_loss']
                    
                    # Take profit
                    elif row['Low'] <= current_position['take_profit']:
                        exit_trade = True
                        exit_reason = 'Take Profit'
                        exit_price = current_position['take_profit']
                    
                    # Trailing stop
                    elif row['Close'] <= current_position['entry_price'] * 0.98:  # 2% profit
                        profit_target = current_position['entry_price'] - (current_position['entry_atr'] * 0.5)
                        if row['High'] >= profit_target:
                            exit_trade = True
                            exit_reason = 'Trailing Stop'
                            exit_price = profit_target
                
                # Force exit after 3 days (for daily options simulation)
                if days_in_trade >= 3:
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
        """Enhanced performance analysis"""
        if self.trades.empty:
            return {'Total Trades': 0, 'Win Rate (%)': 0}
        
        winning_trades = self.trades[self.trades['pnl'] > 0]
        losing_trades = self.trades[self.trades['pnl'] <= 0]
        
        total_trades = len(self.trades)
        win_rate = len(winning_trades) / total_trades * 100 if total_trades > 0 else 0
        
        performance = {
            'Total Trades': total_trades,
            'Win Rate (%)': win_rate,
            'Winning Trades': len(winning_trades),
            'Losing Trades': len(losing_trades),
            'Average Win ($)': winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0,
            'Average Loss ($)': losing_trades['pnl'].mean() if len(losing_trades) > 0 else 0,
            'Average Win (%)': winning_trades['pnl_pct'].mean() if len(winning_trades) > 0 else 0,
            'Average Loss (%)': losing_trades['pnl_pct'].mean() if len(losing_trades) > 0 else 0,
            'Profit Factor': abs(winning_trades['pnl'].sum() / losing_trades['pnl'].sum()) if losing_trades['pnl'].sum() != 0 else float('inf'),
            'Total P&L ($)': self.trades['pnl'].sum(),
            'Max Win ($)': winning_trades['pnl'].max() if len(winning_trades) > 0 else 0,
            'Max Loss ($)': losing_trades['pnl'].min() if len(losing_trades) > 0 else 0,
            'Average Days in Trade': self.trades['days_in_trade'].mean(),
            'Total Return (%)': (self.trades['pnl'].sum() / self.portfolio_size) * 100,
            'Average Confidence': self.trades['confidence'].mean(),
            'Average Signal Strength': self.trades['signal_strength'].mean()
        }
        
        return performance
    
    def run_complete_analysis(self):
        """Run ultra-selective strategy analysis"""
        print("=" * 80)
        print("BTC ULTRA-SELECTIVE OPTIONS STRATEGY - 80%+ WIN RATE TARGET")
        print("=" * 80)
        
        # Fetch data and calculate advanced indicators
        self.fetch_btc_data("3y")
        self.calculate_advanced_indicators()
        self.generate_ultra_selective_signals()
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
        print(f"Signals with Strength >= 6: {len(signals[signals['Signal_Strength'] >= 6])}")
        print(f"Signals with Strength >= 8: {len(signals[signals['Signal_Strength'] >= 8])}")
        print(f"Average Signal Strength: {signals['Signal_Strength'].mean():.1f}")
        print(f"Average Confidence: {signals['Confidence'].mean():.2f}")
        
        # Strategy assessment
        win_rate = performance['Win Rate (%)']
        print(f"\nSTRATEGY ASSESSMENT")
        print("-" * 50)
        if win_rate >= 80:
            print(f"TARGET ACHIEVED: {win_rate:.1f}% win rate EXCEEDS 80% target!")
            print("Strategy is ready for deployment with HIGH confidence.")
        elif win_rate >= 75:
            print(f"EXCELLENT: {win_rate:.1f}% win rate is very close to 80% target")
            print("Strategy shows great promise with minor adjustments needed.")
        elif win_rate >= 70:
            print(f"GOOD: {win_rate:.1f}% win rate is strong but needs refinement")
            print("Consider even more selective criteria or parameter adjustment.")
        else:
            print(f"NEEDS WORK: {win_rate:.1f}% win rate requires significant improvement")
            print("Strategy needs major refinement before deployment.")
        
        print(f"\nULTRA-SELECTIVE PARAMETERS")
        print("-" * 50)
        print(f"Bollinger Bands: {self.bb_period} period, {self.bb_std_dev} std dev")
        print(f"ATR Period: {self.atr_period}")
        print(f"Volume Threshold: {self.volume_threshold}x (EXTREME)")
        print(f"Minimum Signal Strength: {self.min_signal_strength} (ULTRA-HIGH)")
        print(f"Risk per Trade: {self.max_risk_per_trade*100}% (ULTRA-CONSERVATIVE)")
        print(f"Stop Loss: {self.stop_loss_atr_multiplier}x ATR (VERY TIGHT)")
        print(f"Take Profit: {self.take_profit_atr_multiplier}x ATR")
        print(f"Max Positions: {self.max_positions} (ONE AT A TIME)")
        
        # Trade breakdown by exit reason
        if not trades.empty:
            print(f"\nTRADE EXIT ANALYSIS")
            print("-" * 50)
            exit_reasons = trades['exit_reason'].value_counts()
            for reason, count in exit_reasons.items():
                win_rate_by_reason = (trades[trades['exit_reason'] == reason]['pnl'] > 0).mean() * 100
                print(f"{reason:<15}: {count:>3} trades ({win_rate_by_reason:>5.1f}% win rate)")
        
        return performance, trades

def main():
    """Run the ultra-selective strategy"""
    strategy = BTCUltraSelectiveStrategy(portfolio_size=100000)
    performance, trades = strategy.run_complete_analysis()
    
    print(f"\nDEPLOYMENT STRATEGY")
    print("-" * 50)
    print("• ONLY trade signals with strength >= 6")
    print("• Require 3x average volume for entry")
    print("• Use ONLY daily expiry options")
    print("• Maximum 1 position at any time")
    print("• 2+ hour minimum between trades")
    print("• Paper trade for 1 month before going live")
    print("• Start with smallest position sizes")
    print("• Monitor performance weekly and adjust if needed")

if __name__ == "__main__":
    main()