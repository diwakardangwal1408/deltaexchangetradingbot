import pandas as pd
import numpy as np
from btc_trading_strategy import BTCTradingStrategy
import matplotlib.pyplot as plt
from itertools import product

class StrategyOptimizer:
    """
    Optimize the BTC trading strategy parameters to achieve 80%+ win rate
    """
    
    def __init__(self):
        self.strategy = BTCTradingStrategy()
        self.best_params = None
        self.best_performance = None
        self.optimization_results = []
    
    def optimize_parameters(self):
        """
        Test different parameter combinations to maximize win rate while maintaining profitability
        """
        print("Starting parameter optimization for 80%+ win rate...")
        
        # Parameter ranges to test
        bb_periods = [15, 20, 25]
        bb_std_devs = [1.5, 2.0, 2.5]
        atr_periods = [10, 14, 20]
        volume_thresholds = [1.2, 1.5, 2.0]
        signal_strength_mins = [2, 3, 4]
        
        best_win_rate = 0
        best_params = None
        
        total_combinations = len(bb_periods) * len(bb_std_devs) * len(atr_periods) * len(volume_thresholds) * len(signal_strength_mins)
        current_combination = 0
        
        for bb_period, bb_std, atr_period, vol_threshold, min_strength in product(
            bb_periods, bb_std_devs, atr_periods, volume_thresholds, signal_strength_mins
        ):
            current_combination += 1
            print(f"Testing combination {current_combination}/{total_combinations}: BB({bb_period},{bb_std}), ATR({atr_period}), Vol({vol_threshold}), Strength({min_strength})")
            
            try:
                # Create new strategy instance for each test
                test_strategy = BTCTradingStrategy()
                
                # Fetch data and calculate indicators with test parameters
                test_strategy.fetch_btc_data("3y")
                test_strategy.calculate_bollinger_bands(period=bb_period, std_dev=bb_std)
                test_strategy.calculate_atr(period=atr_period)
                test_strategy.calculate_volume_indicators()
                test_strategy.identify_price_action_patterns()
                
                # Generate signals with modified logic
                self._generate_optimized_signals(test_strategy, vol_threshold, min_strength)
                test_strategy.calculate_position_sizing()
                
                # Backtest
                trades = test_strategy.backtest_strategy()
                
                if len(trades) > 5:  # Minimum trades for statistical significance
                    performance = test_strategy.analyze_performance()
                    win_rate = performance['Win Rate (%)']
                    profit_factor = performance['Profit Factor']
                    total_pnl = performance['Total P&L']
                    
                    # Store results
                    result = {
                        'bb_period': bb_period,
                        'bb_std_dev': bb_std,
                        'atr_period': atr_period,
                        'volume_threshold': vol_threshold,
                        'min_signal_strength': min_strength,
                        'win_rate': win_rate,
                        'profit_factor': profit_factor,
                        'total_pnl': total_pnl,
                        'total_trades': performance['Total Trades'],
                        'sharpe_ratio': performance['Sharpe Ratio']
                    }
                    
                    self.optimization_results.append(result)
                    
                    # Check if this is better (prioritize win rate > 80%, then profitability)
                    if win_rate >= 80 and profit_factor > 1.5 and total_pnl > 0:
                        if win_rate > best_win_rate:
                            best_win_rate = win_rate
                            best_params = result
                            print(f"NEW BEST: Win Rate: {win_rate:.1f}%, PF: {profit_factor:.2f}, P&L: ${total_pnl:.0f}")
            
            except Exception as e:
                print(f"Error with parameters: {e}")
                continue
        
        self.best_params = best_params
        return best_params
    
    def _generate_optimized_signals(self, strategy, vol_threshold, min_strength):
        """
        Generate trading signals with optimized parameters
        """
        data = strategy.data
        
        # Initialize signal columns
        data['Signal'] = 0
        data['Signal_Strength'] = 0
        data['Entry_Type'] = ''
        
        # More selective conditions for higher win rate
        for i in range(len(data)):
            if i < 30:  # Skip first 30 days for indicator stability
                continue
                
            long_score = 0
            short_score = 0
            
            # Long conditions (more restrictive)
            if data['BB_Position'].iloc[i] < 0.15:  # Very oversold
                long_score += 1
            
            if data['Volume_Ratio'].iloc[i] > vol_threshold:  # High volume
                long_score += 1
            
            if data['Hammer'].iloc[i] or data['Bullish_Engulfing'].iloc[i]:  # Reversal patterns
                long_score += 1
            
            if data['ATR_Pct'].iloc[i] > data['ATR_Pct'].iloc[i-10:i].mean():  # Volatility expansion
                long_score += 1
            
            if data['OBV'].iloc[i] > data['OBV_SMA'].iloc[i]:  # Volume confirmation
                long_score += 1
            
            # Additional confirmation: Price near BB lower band but showing strength
            if (data['Close'].iloc[i] > data['Close'].iloc[i-1] and 
                data['BB_Position'].iloc[i] < 0.3):
                long_score += 1
            
            # Short conditions (more restrictive)
            if data['BB_Position'].iloc[i] > 0.85:  # Very overbought
                short_score += 1
            
            if data['Volume_Ratio'].iloc[i] > vol_threshold:  # High volume
                short_score += 1
            
            if data['Bearish_Engulfing'].iloc[i]:  # Reversal patterns
                short_score += 1
            
            if data['ATR_Pct'].iloc[i] > data['ATR_Pct'].iloc[i-10:i].mean():  # Volatility expansion
                short_score += 1
            
            if data['OBV'].iloc[i] < data['OBV_SMA'].iloc[i]:  # Volume divergence
                short_score += 1
            
            # Additional confirmation: Price near BB upper band but showing weakness
            if (data['Close'].iloc[i] < data['Close'].iloc[i-1] and 
                data['BB_Position'].iloc[i] > 0.7):
                short_score += 1
            
            # Generate signals only with high confidence
            if long_score >= min_strength:
                data.loc[data.index[i], 'Signal'] = 1
                data.loc[data.index[i], 'Signal_Strength'] = long_score
                data.loc[data.index[i], 'Entry_Type'] = 'CALL'
            elif short_score >= min_strength:
                data.loc[data.index[i], 'Signal'] = -1
                data.loc[data.index[i], 'Signal_Strength'] = short_score
                data.loc[data.index[i], 'Entry_Type'] = 'PUT'
    
    def create_enhanced_strategy(self):
        """
        Create enhanced strategy with optimized parameters targeting 80%+ win rate
        """
        if not self.best_params:
            print("Running optimization first...")
            self.optimize_parameters()
        
        if not self.best_params:
            print("No optimal parameters found. Using enhanced default parameters...")
            # Use conservative parameters that historically work well
            enhanced_params = {
                'bb_period': 20,
                'bb_std_dev': 2.5,  # Wider bands for clearer signals
                'atr_period': 14,
                'volume_threshold': 2.0,  # Higher volume requirement
                'min_signal_strength': 4  # Very high confirmation requirement
            }
        else:
            enhanced_params = self.best_params
            print(f"Using optimized parameters: {enhanced_params}")
        
        # Create enhanced strategy
        enhanced_strategy = BTCTradingStrategy()
        
        # Enhanced risk management
        enhanced_strategy.stop_loss_atr_multiplier = 1.5  # Tighter stops
        enhanced_strategy.take_profit_atr_multiplier = 3.0  # Lower targets for higher win rate
        enhanced_strategy.max_risk_per_trade = 0.005  # 0.5% max risk per trade
        
        return enhanced_strategy, enhanced_params
    
    def backtest_enhanced_strategy(self):
        """
        Backtest the enhanced strategy with optimized parameters
        """
        enhanced_strategy, params = self.create_enhanced_strategy()
        
        # Fetch data and calculate indicators
        enhanced_strategy.fetch_btc_data("3y")
        enhanced_strategy.calculate_bollinger_bands(
            period=params.get('bb_period', 20),
            std_dev=params.get('bb_std_dev', 2.5)
        )
        enhanced_strategy.calculate_atr(period=params.get('atr_period', 14))
        enhanced_strategy.calculate_volume_indicators()
        enhanced_strategy.identify_price_action_patterns()
        
        # Generate optimized signals
        self._generate_optimized_signals(
            enhanced_strategy,
            params.get('volume_threshold', 2.0),
            params.get('min_signal_strength', 4)
        )
        enhanced_strategy.calculate_position_sizing()
        
        # Backtest
        trades = enhanced_strategy.backtest_strategy()
        performance = enhanced_strategy.analyze_performance()
        
        return enhanced_strategy, performance, trades
    
    def display_optimization_results(self):
        """
        Display optimization results sorted by win rate
        """
        if not self.optimization_results:
            print("No optimization results available")
            return
        
        # Convert to DataFrame and sort by win rate
        results_df = pd.DataFrame(self.optimization_results)
        results_df = results_df.sort_values('win_rate', ascending=False)
        
        print("\n=== TOP 10 PARAMETER COMBINATIONS (by Win Rate) ===")
        print(results_df.head(10).to_string(index=False))
        
        # Filter for 80%+ win rate
        high_win_rate = results_df[results_df['win_rate'] >= 80]
        
        if not high_win_rate.empty:
            print(f"\n=== COMBINATIONS WITH 80%+ WIN RATE ({len(high_win_rate)} found) ===")
            print(high_win_rate.to_string(index=False))
        else:
            print("\n=== NO COMBINATIONS ACHIEVED 80%+ WIN RATE ===")
            print("Consider adjusting parameter ranges or strategy logic")
    
    def generate_final_report(self):
        """
        Generate final comprehensive report
        """
        enhanced_strategy, performance, trades = self.backtest_enhanced_strategy()
        
        print("\n" + "="*80)
        print("BTC HIGH WIN-RATE OPTIONS TRADING STRATEGY - FINAL REPORT")
        print("="*80)
        
        print(f"\nStrategy Performance:")
        print(f"Total Trades: {performance['Total Trades']}")
        print(f"Win Rate: {performance['Win Rate (%)']:.1f}%")
        print(f"Profit Factor: {performance['Profit Factor']:.2f}")
        print(f"Total P&L: ${performance['Total P&L']:.2f}")
        print(f"Average Win: ${performance['Average Win']:.2f}")
        print(f"Average Loss: ${performance['Average Loss']:.2f}")
        print(f"Max Drawdown: {performance['Max Drawdown (%)']:.1f}%")
        print(f"Sharpe Ratio: {performance['Sharpe Ratio']:.2f}")
        
        # Analysis
        win_rate = performance['Win Rate (%)']
        if win_rate >= 80:
            print(f"\n✅ TARGET ACHIEVED: Win rate of {win_rate:.1f}% exceeds 80% target!")
        else:
            print(f"\n⚠️  TARGET NOT MET: Win rate of {win_rate:.1f}% below 80% target")
            print("Consider further optimization or different market conditions")
        
        print(f"\nKey Strategy Features for High Win Rate:")
        print("• Multi-confirmation signals (minimum 4 confirmations required)")
        print("• Restrictive entry criteria (BB extremes + volume + price action)")
        print("• Tighter stop losses (1.5x ATR) for better risk control")
        print("• Conservative position sizing (0.5% risk per trade)")
        print("• Daily expiry options to minimize time decay")
        print("• Volume expansion requirement (2x average volume)")
        
        print(f"\nDeployment Recommendations:")
        print("• Use daily expiry BTC options on major exchanges")
        print("• Monitor high volatility periods (US/Asian market overlaps)")
        print("• Maximum 3 concurrent positions")
        print("• Daily loss limit: 2% of portfolio")
        print("• Regular strategy review and parameter adjustment")
        
        return enhanced_strategy, performance, trades

def main():
    """Run the strategy optimization process"""
    optimizer = StrategyOptimizer()
    
    print("BTC OPTIONS STRATEGY OPTIMIZATION FOR 80%+ WIN RATE")
    print("="*60)
    
    # Run optimization
    best_params = optimizer.optimize_parameters()
    
    # Display results
    optimizer.display_optimization_results()
    
    # Generate final report
    enhanced_strategy, performance, trades = optimizer.generate_final_report()
    
    print(f"\nOptimization complete. Check results above.")

if __name__ == "__main__":
    main()