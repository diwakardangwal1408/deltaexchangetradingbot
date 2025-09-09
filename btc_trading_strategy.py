import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class BTCTradingStrategy:
    def __init__(self):
        self.data = None
        self.signals = None
        self.position_size_pct = 0.02  # 2% of portfolio per trade
        self.max_risk_per_trade = 0.01  # 1% max risk per trade
        self.stop_loss_atr_multiplier = 2.0
        self.take_profit_atr_multiplier = 4.0
        
    def fetch_btc_data(self, period="60d"):
        """Fetch BTC/USD historical data - 15min timeframe for intraday trading"""
        print("Fetching BTC/USD 15-minute data...")
        ticker = yf.Ticker("BTC-USD")
        self.data = ticker.history(period=period, interval="15m")
        print(f"Fetched {len(self.data)} 15-minute bars of data")
        return self.data
    
    def calculate_bollinger_bands(self, period=80, std_dev=2):
        """Calculate Bollinger Bands - 80 periods = 20 hours on 15m timeframe"""
        self.data['BB_Middle'] = self.data['Close'].rolling(window=period).mean()
        rolling_std = self.data['Close'].rolling(window=period).std()
        self.data['BB_Upper'] = self.data['BB_Middle'] + (rolling_std * std_dev)
        self.data['BB_Lower'] = self.data['BB_Middle'] - (rolling_std * std_dev)
        self.data['BB_Width'] = (self.data['BB_Upper'] - self.data['BB_Lower']) / self.data['BB_Middle']
        self.data['BB_Position'] = (self.data['Close'] - self.data['BB_Lower']) / (self.data['BB_Upper'] - self.data['BB_Lower'])
    
    def calculate_atr(self, period=56):
        """Calculate Average True Range - 56 periods = 14 hours on 15m timeframe"""
        high = self.data['High']
        low = self.data['Low']
        close = self.data['Close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        self.data['ATR'] = true_range.rolling(window=period).mean()
        self.data['ATR_Pct'] = self.data['ATR'] / self.data['Close']
    
    def calculate_volume_indicators(self):
        """Calculate volume-based indicators"""
        self.data['Volume_SMA'] = self.data['Volume'].rolling(window=80).mean()  # 20 hours on 15m
        self.data['Volume_Ratio'] = self.data['Volume'] / self.data['Volume_SMA']
        
        # On Balance Volume
        self.data['OBV'] = (np.sign(self.data['Close'].diff()) * self.data['Volume']).fillna(0).cumsum()
        self.data['OBV_SMA'] = self.data['OBV'].rolling(window=40).mean()  # 10 hours on 15m
    
    def identify_price_action_patterns(self):
        """Identify key price action patterns"""
        # Doji patterns
        body_size = abs(self.data['Close'] - self.data['Open'])
        candle_range = self.data['High'] - self.data['Low']
        self.data['Doji'] = (body_size / candle_range) < 0.1
        
        # Hammer patterns
        lower_shadow = np.minimum(self.data['Open'], self.data['Close']) - self.data['Low']
        upper_shadow = self.data['High'] - np.maximum(self.data['Open'], self.data['Close'])
        self.data['Hammer'] = (lower_shadow > 2 * body_size) & (upper_shadow < body_size)
        
        # Engulfing patterns
        bullish_engulfing = ((self.data['Close'] > self.data['Open']) & 
                           (self.data['Close'].shift(1) < self.data['Open'].shift(1)) &
                           (self.data['Open'] < self.data['Close'].shift(1)) &
                           (self.data['Close'] > self.data['Open'].shift(1)))
        
        bearish_engulfing = ((self.data['Close'] < self.data['Open']) & 
                           (self.data['Close'].shift(1) > self.data['Open'].shift(1)) &
                           (self.data['Open'] > self.data['Close'].shift(1)) &
                           (self.data['Close'] < self.data['Open'].shift(1)))
        
        self.data['Bullish_Engulfing'] = bullish_engulfing
        self.data['Bearish_Engulfing'] = bearish_engulfing
    
    def generate_trading_signals(self):
        """Generate high-probability trading signals"""
        # Initialize signal columns
        self.data['Signal'] = 0
        self.data['Signal_Strength'] = 0
        self.data['Entry_Type'] = ''
        
        # Conditions for LONG signals
        long_conditions = [
            # BB bounce from lower band with high volume
            (self.data['BB_Position'] < 0.2) & (self.data['Volume_Ratio'] > 1.5),
            
            # Price action + volume confirmation
            (self.data['Hammer'] | self.data['Bullish_Engulfing']) & (self.data['Volume_Ratio'] > 1.2),
            
            # ATR expansion with BB squeeze release
            (self.data['ATR_Pct'] > self.data['ATR_Pct'].rolling(10).mean()) & 
            (self.data['BB_Width'] > self.data['BB_Width'].rolling(20).mean()),
            
            # OBV confirmation
            self.data['OBV'] > self.data['OBV_SMA']
        ]
        
        # Conditions for SHORT signals
        short_conditions = [
            # BB rejection from upper band with high volume
            (self.data['BB_Position'] > 0.8) & (self.data['Volume_Ratio'] > 1.5),
            
            # Bearish price action + volume
            self.data['Bearish_Engulfing'] & (self.data['Volume_Ratio'] > 1.2),
            
            # ATR expansion with BB squeeze release (bearish)
            (self.data['ATR_Pct'] > self.data['ATR_Pct'].rolling(10).mean()) & 
            (self.data['BB_Width'] > self.data['BB_Width'].rolling(20).mean()) &
            (self.data['Close'] < self.data['BB_Middle']),
            
            # OBV divergence
            self.data['OBV'] < self.data['OBV_SMA']
        ]
        
        # Calculate signal strength and generate signals
        for i in range(len(self.data)):
            long_score = sum([condition.iloc[i] if i < len(condition) else False for condition in long_conditions])
            short_score = sum([condition.iloc[i] if i < len(condition) else False for condition in short_conditions])
            
            # High probability signals require multiple confirmations
            if long_score >= 3:
                self.data.loc[self.data.index[i], 'Signal'] = 1
                self.data.loc[self.data.index[i], 'Signal_Strength'] = long_score
                self.data.loc[self.data.index[i], 'Entry_Type'] = 'CALL'
            elif short_score >= 3:
                self.data.loc[self.data.index[i], 'Signal'] = -1
                self.data.loc[self.data.index[i], 'Signal_Strength'] = short_score
                self.data.loc[self.data.index[i], 'Entry_Type'] = 'PUT'
    
    def calculate_position_sizing(self):
        """Calculate position size based on ATR and risk management"""
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
        
        # Risk per share
        self.data['Risk_Per_Share'] = abs(self.data['Close'] - self.data['Stop_Loss'])
        
        # Position size calculation (for futures)
        portfolio_value = 100000  # Assume $100k portfolio
        max_risk_amount = portfolio_value * self.max_risk_per_trade
        
        self.data['Position_Size'] = np.where(
            self.data['Signal'] != 0,
            max_risk_amount / self.data['Risk_Per_Share'],
            0
        )
        
        # For options - calculate contracts needed
        self.data['Option_Contracts'] = np.where(
            self.data['Signal'] != 0,
            np.maximum(1, self.data['Position_Size'] / 100),  # Minimum 1 contract
            0
        )
    
    def backtest_strategy(self):
        """Backtest the trading strategy"""
        trades = []
        current_position = None
        
        for i, row in self.data.iterrows():
            if row['Signal'] != 0 and current_position is None:
                # Enter position
                current_position = {
                    'entry_date': i,
                    'entry_price': row['Close'],
                    'signal': row['Signal'],
                    'stop_loss': row['Stop_Loss'],
                    'take_profit': row['Take_Profit'],
                    'position_size': row['Position_Size'],
                    'entry_type': row['Entry_Type']
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
                
                if exit_trade:
                    # Calculate P&L
                    if current_position['signal'] == 1:
                        pnl = (exit_price - current_position['entry_price']) * current_position['position_size']
                    else:
                        pnl = (current_position['entry_price'] - exit_price) * current_position['position_size']
                    
                    trades.append({
                        'entry_date': current_position['entry_date'],
                        'exit_date': i,
                        'entry_price': current_position['entry_price'],
                        'exit_price': exit_price,
                        'signal': current_position['signal'],
                        'pnl': pnl,
                        'exit_reason': exit_reason,
                        'entry_type': current_position['entry_type'],
                        'position_size': current_position['position_size']
                    })
                    
                    current_position = None
        
        self.trades_df = pd.DataFrame(trades)
        return self.trades_df
    
    def analyze_performance(self):
        """Analyze strategy performance"""
        if self.trades_df.empty:
            return {}
        
        winning_trades = self.trades_df[self.trades_df['pnl'] > 0]
        losing_trades = self.trades_df[self.trades_df['pnl'] <= 0]
        
        total_trades = len(self.trades_df)
        win_rate = len(winning_trades) / total_trades * 100
        
        avg_win = winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0
        avg_loss = losing_trades['pnl'].mean() if len(losing_trades) > 0 else 0
        
        profit_factor = abs(avg_win * len(winning_trades) / (avg_loss * len(losing_trades))) if avg_loss != 0 else float('inf')
        
        total_pnl = self.trades_df['pnl'].sum()
        max_drawdown = self.calculate_max_drawdown()
        
        performance = {
            'Total Trades': total_trades,
            'Win Rate (%)': win_rate,
            'Winning Trades': len(winning_trades),
            'Losing Trades': len(losing_trades),
            'Average Win': avg_win,
            'Average Loss': avg_loss,
            'Profit Factor': profit_factor,
            'Total P&L': total_pnl,
            'Max Drawdown (%)': max_drawdown,
            'Sharpe Ratio': self.calculate_sharpe_ratio()
        }
        
        return performance
    
    def calculate_max_drawdown(self):
        """Calculate maximum drawdown"""
        cumulative_pnl = self.trades_df['pnl'].cumsum()
        rolling_max = cumulative_pnl.expanding().max()
        drawdown = (cumulative_pnl - rolling_max) / rolling_max * 100
        return drawdown.min()
    
    def calculate_sharpe_ratio(self):
        """Calculate Sharpe ratio"""
        returns = self.trades_df['pnl'] / 100000  # Normalize by portfolio value
        if returns.std() == 0:
            return 0
        return (returns.mean() / returns.std()) * np.sqrt(252)  # Annualized
    
    def run_complete_analysis(self):
        """Run the complete trading strategy analysis"""
        print("=== BTC Trading Strategy Analysis ===")
        
        # Fetch data and calculate indicators
        self.fetch_btc_data("3y")
        self.calculate_bollinger_bands()
        self.calculate_atr()
        self.calculate_volume_indicators()
        self.identify_price_action_patterns()
        
        # Generate signals and backtest
        self.generate_trading_signals()
        self.calculate_position_sizing()
        trades = self.backtest_strategy()
        
        # Analyze performance
        performance = self.analyze_performance()
        
        print("\n=== Strategy Performance ===")
        for key, value in performance.items():
            if isinstance(value, float):
                print(f"{key}: {value:.2f}")
            else:
                print(f"{key}: {value}")
        
        print(f"\n=== Recent Signals ===")
        recent_signals = self.data[self.data['Signal'] != 0].tail(10)
        for date, row in recent_signals.iterrows():
            print(f"{date.strftime('%Y-%m-%d')}: {row['Entry_Type']} - Strength: {row['Signal_Strength']}")
        
        return performance, trades

if __name__ == "__main__":
    strategy = BTCTradingStrategy()
    performance, trades = strategy.run_complete_analysis()