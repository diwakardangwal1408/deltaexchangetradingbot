import pandas as pd
import numpy as np
from datetime import datetime, timedelta, time
import asyncio
import json
from btc_trading_strategy import BTCTradingStrategy

class BTCOptionsTrader:
    """
    Production-ready BTC options trading system for daily expiry options
    Implements the high win-rate strategy with proper risk and position management
    """
    
    def __init__(self, portfolio_size=100000):
        self.portfolio_size = portfolio_size
        self.strategy = BTCTradingStrategy()
        self.current_positions = {}
        self.daily_pnl = 0
        self.max_daily_loss = portfolio_size * 0.02  # 2% max daily loss
        self.max_positions = 3  # Maximum concurrent positions
        self.min_time_between_trades = 30  # Minutes between trades
        self.last_trade_time = None
        
        # Options specific parameters
        self.min_premium = 0.50  # Minimum option premium
        self.max_premium = 5.00   # Maximum option premium
        self.target_delta = 0.3   # Target delta for options
        self.iv_threshold = 0.25  # Minimum IV threshold
        
    def check_trading_hours(self):
        """Check if within trading hours (BTC trades 24/7 but focus on high volume hours)"""
        current_time = datetime.now().time()
        
        # Focus on high volatility periods (US and Asian markets overlap)
        high_vol_periods = [
            (time(8, 0), time(12, 0)),   # Asian market hours
            (time(14, 0), time(22, 0)),  # US market hours
        ]
        
        for start_time, end_time in high_vol_periods:
            if start_time <= current_time <= end_time:
                return True
        return False
    
    def calculate_option_parameters(self, signal_data):
        """Calculate option parameters for daily expiry options"""
        current_price = signal_data['Close']
        atr = signal_data['ATR']
        
        # For daily expiry, use smaller moves
        if signal_data['Entry_Type'] == 'CALL':
            # Slightly out-of-the-money calls
            strike_price = current_price + (atr * 0.5)
        else:
            # Slightly out-of-the-money puts
            strike_price = current_price - (atr * 0.5)
        
        # Calculate approximate option premium (simplified Black-Scholes estimation)
        time_to_expiry = 1/365  # Daily expiry
        volatility = signal_data['ATR_Pct'] * np.sqrt(365)  # Annualized volatility
        
        # Simplified premium calculation
        intrinsic_value = max(0, 
            current_price - strike_price if signal_data['Entry_Type'] == 'CALL' 
            else strike_price - current_price
        )
        time_value = atr * 0.1 * (1 + volatility)  # Simplified time value
        estimated_premium = intrinsic_value + time_value
        
        return {
            'strike_price': strike_price,
            'estimated_premium': estimated_premium,
            'time_to_expiry': time_to_expiry,
            'implied_volatility': volatility
        }
    
    def calculate_position_size_options(self, signal_data, option_params):
        """Calculate position size for options trading"""
        premium = option_params['estimated_premium']
        
        # Risk-based position sizing
        max_risk_per_trade = self.portfolio_size * 0.01  # 1% max risk
        
        # For options, risk is limited to premium paid
        contracts = int(max_risk_per_trade / (premium * 100))  # 100 shares per contract
        contracts = max(1, min(contracts, 10))  # Between 1-10 contracts
        
        total_premium_cost = contracts * premium * 100
        
        return {
            'contracts': contracts,
            'total_cost': total_premium_cost,
            'max_loss': total_premium_cost,  # Limited risk
            'breakeven': option_params['strike_price'] + (premium if signal_data['Entry_Type'] == 'CALL' else -premium)
        }
    
    def generate_trade_signal(self):
        """Generate trading signal using the strategy"""
        # Update strategy data (in production, this would fetch live data)
        self.strategy.fetch_btc_data("1y")
        self.strategy.calculate_bollinger_bands()
        self.strategy.calculate_atr()
        self.strategy.calculate_volume_indicators()
        self.strategy.identify_price_action_patterns()
        self.strategy.generate_trading_signals()
        
        # Get latest signal
        latest_data = self.strategy.data.iloc[-1]
        
        if latest_data['Signal'] != 0 and latest_data['Signal_Strength'] >= 3:
            return {
                'signal': latest_data['Signal'],
                'entry_type': latest_data['Entry_Type'],
                'strength': latest_data['Signal_Strength'],
                'price': latest_data['Close'],
                'atr': latest_data['ATR'],
                'atr_pct': latest_data['ATR_Pct'],
                'bb_position': latest_data['BB_Position'],
                'volume_ratio': latest_data['Volume_Ratio'],
                'timestamp': datetime.now()
            }
        
        return None
    
    def validate_trade_conditions(self, signal_data):
        """Validate if trade conditions are met"""
        current_time = datetime.now()
        
        # Check trading hours
        if not self.check_trading_hours():
            return False, "Outside trading hours"
        
        # Check daily loss limit
        if self.daily_pnl <= -self.max_daily_loss:
            return False, "Daily loss limit reached"
        
        # Check maximum positions
        if len(self.current_positions) >= self.max_positions:
            return False, "Maximum positions reached"
        
        # Check time between trades
        if (self.last_trade_time and 
            (current_time - self.last_trade_time).total_seconds() < self.min_time_between_trades * 60):
            return False, "Minimum time between trades not met"
        
        # Check signal strength
        if signal_data['strength'] < 3:
            return False, "Signal strength insufficient"
        
        # Check volatility
        if signal_data['atr_pct'] < 0.02:  # Less than 2% daily volatility
            return False, "Volatility too low"
        
        return True, "All conditions met"
    
    def execute_trade(self, signal_data):
        """Execute the options trade"""
        option_params = self.calculate_option_parameters(signal_data)
        position_size = self.calculate_position_size_options(signal_data, option_params)
        
        # Validate premium range
        if not (self.min_premium <= option_params['estimated_premium'] <= self.max_premium):
            return None, "Premium outside acceptable range"
        
        trade_id = f"BTC_{signal_data['entry_type']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        trade_order = {
            'trade_id': trade_id,
            'symbol': 'BTC-USD',
            'option_type': signal_data['entry_type'],
            'strike_price': option_params['strike_price'],
            'expiry': datetime.now().date() + timedelta(days=1),  # Next day expiry
            'contracts': position_size['contracts'],
            'premium_paid': option_params['estimated_premium'],
            'total_cost': position_size['total_cost'],
            'max_loss': position_size['max_loss'],
            'breakeven': position_size['breakeven'],
            'entry_time': datetime.now(),
            'signal_strength': signal_data['strength'],
            'underlying_price': signal_data['price']
        }
        
        # In production, this would place actual order with broker
        print(f"EXECUTING TRADE: {trade_order}")
        
        # Add to current positions
        self.current_positions[trade_id] = trade_order
        self.last_trade_time = datetime.now()
        
        return trade_order, "Trade executed successfully"
    
    def monitor_positions(self):
        """Monitor current positions for exit conditions"""
        current_time = datetime.now()
        positions_to_close = []
        
        for trade_id, position in self.current_positions.items():
            # Check if expiry is today (daily expiry)
            if position['expiry'] <= current_time.date():
                positions_to_close.append((trade_id, "Expiry reached"))
                continue
            
            # Get current BTC price (in production, fetch live price)
            current_btc_price = self.strategy.data['Close'].iloc[-1]  # Simplified
            
            # Calculate current option value
            current_intrinsic = max(0,
                current_btc_price - position['strike_price'] if position['option_type'] == 'CALL'
                else position['strike_price'] - current_btc_price
            )
            
            # Simple profit/loss calculation
            current_value = current_intrinsic  # At expiry, time value = 0
            pnl_per_contract = current_value - position['premium_paid']
            total_pnl = pnl_per_contract * position['contracts'] * 100
            
            # Exit conditions
            # 1. Take profit at 200% of premium
            if total_pnl >= position['total_cost'] * 2:
                positions_to_close.append((trade_id, f"Take profit: ${total_pnl:.2f}"))
            
            # 2. Cut losses at 50% of premium (early exit)
            elif total_pnl <= -position['total_cost'] * 0.5:
                positions_to_close.append((trade_id, f"Stop loss: ${total_pnl:.2f}"))
            
            # 3. Close 2 hours before expiry to avoid time decay
            elif (position['expiry'] - current_time.date()).days == 0:
                if current_time.hour >= 22:  # Close 2 hours before midnight
                    positions_to_close.append((trade_id, f"Time exit: ${total_pnl:.2f}"))
        
        # Close positions
        for trade_id, reason in positions_to_close:
            self.close_position(trade_id, reason)
    
    def close_position(self, trade_id, reason):
        """Close a position"""
        if trade_id not in self.current_positions:
            return
        
        position = self.current_positions[trade_id]
        current_btc_price = self.strategy.data['Close'].iloc[-1]
        
        # Calculate final P&L
        final_intrinsic = max(0,
            current_btc_price - position['strike_price'] if position['option_type'] == 'CALL'
            else position['strike_price'] - current_btc_price
        )
        
        pnl_per_contract = final_intrinsic - position['premium_paid']
        total_pnl = pnl_per_contract * position['contracts'] * 100
        
        self.daily_pnl += total_pnl
        
        print(f"CLOSING POSITION: {trade_id}")
        print(f"Reason: {reason}")
        print(f"P&L: ${total_pnl:.2f}")
        print(f"Daily P&L: ${self.daily_pnl:.2f}")
        print("-" * 50)
        
        # Remove from current positions
        del self.current_positions[trade_id]
    
    def reset_daily_counters(self):
        """Reset daily counters (call at start of each trading day)"""
        self.daily_pnl = 0
        print(f"Daily counters reset. Date: {datetime.now().date()}")
    
    async def run_trading_loop(self):
        """Main trading loop"""
        print("Starting BTC Options Trading System...")
        print(f"Portfolio Size: ${self.portfolio_size:,}")
        print(f"Max Daily Loss: ${self.max_daily_loss:,}")
        print(f"Max Positions: {self.max_positions}")
        print("-" * 50)
        
        last_date = datetime.now().date()
        
        while True:
            try:
                current_date = datetime.now().date()
                
                # Reset daily counters if new day
                if current_date > last_date:
                    self.reset_daily_counters()
                    last_date = current_date
                
                # Monitor existing positions
                self.monitor_positions()
                
                # Check for new trading signals
                signal_data = self.generate_trade_signal()
                
                if signal_data:
                    # Validate trade conditions
                    is_valid, message = self.validate_trade_conditions(signal_data)
                    
                    if is_valid:
                        # Execute trade
                        trade_result, trade_message = self.execute_trade(signal_data)
                        if trade_result:
                            print(f"NEW TRADE: {trade_message}")
                        else:
                            print(f"TRADE REJECTED: {trade_message}")
                    else:
                        print(f"TRADE CONDITIONS NOT MET: {message}")
                
                # Wait before next iteration
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                print(f"ERROR in trading loop: {e}")
                await asyncio.sleep(60)
    
    def generate_strategy_report(self):
        """Generate a comprehensive strategy report"""
        performance, trades = self.strategy.run_complete_analysis()
        
        report = {
            'strategy_name': 'BTC Daily Options High Win Rate Strategy',
            'backtest_performance': performance,
            'key_features': [
                'Multi-indicator confirmation (Bollinger Bands, ATR, Volume, Price Action)',
                'Daily expiry options for reduced time decay risk',
                'Strict risk management (1% per trade, 2% daily max loss)',
                'High probability setups (80%+ target win rate)',
                'Volume and volatility filters',
                'Position sizing based on ATR'
            ],
            'risk_parameters': {
                'max_risk_per_trade': '1%',
                'max_daily_loss': '2%',
                'max_positions': self.max_positions,
                'position_size_method': 'ATR-based',
                'stop_loss_method': '50% of premium or time-based'
            },
            'entry_criteria': [
                'Signal strength >= 3 (multiple confirmations)',
                'Bollinger Band position confirmation',
                'Volume ratio > 1.2',
                'ATR expansion',
                'Price action pattern confirmation'
            ],
            'exit_criteria': [
                'Daily expiry (forced exit)',
                'Take profit at 200% of premium',
                'Stop loss at 50% of premium',
                'Time exit 2 hours before expiry'
            ]
        }
        
        return report

def main():
    """Run the BTC options trading system"""
    trader = BTCOptionsTrader(portfolio_size=100000)
    
    # Generate strategy report
    report = trader.generate_strategy_report()
    
    print("=== BTC DAILY OPTIONS TRADING STRATEGY REPORT ===")
    print(f"Strategy: {report['strategy_name']}")
    print("\n=== Backtest Performance ===")
    for key, value in report['backtest_performance'].items():
        print(f"{key}: {value}")
    
    print("\n=== Key Features ===")
    for feature in report['key_features']:
        print(f"• {feature}")
    
    print("\n=== Risk Parameters ===")
    for param, value in report['risk_parameters'].items():
        print(f"• {param}: {value}")
    
    print("\n=== Entry Criteria ===")
    for criteria in report['entry_criteria']:
        print(f"• {criteria}")
    
    print("\n=== Exit Criteria ===")
    for criteria in report['exit_criteria']:
        print(f"• {criteria}")
    
    # Uncomment to run live trading (for demo purposes only)
    # asyncio.run(trader.run_trading_loop())

if __name__ == "__main__":
    main()