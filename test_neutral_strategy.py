#!/usr/bin/env python3
"""
Test script for the neutral strangle strategy
"""

import asyncio
import json
from delta_exchange_client import DeltaExchangeClient
from delta_btc_strategy import DeltaBTCOptionsTrader
from config_manager import config_manager

async def test_neutral_strategy():
    """Test the neutral strategy functionality"""
    try:
        print("Testing Neutral Strangle Strategy")
        print("=" * 50)
        
        # Load config
        config = config_manager.get_all_config()
        
        print(f"Paper Trading Mode: {config.get('paper_trading', True)}")
        print(f"Neutral Strategy Enabled: {config.get('neutral_strategy', {}).get('enabled', True)}")
        
        # Initialize trader
        trader = DeltaBTCOptionsTrader()
        
        # Test API connection
        print("\n1. Testing API connection...")
        btc_price = trader.delta_client.get_current_btc_price()
        print(f"   Current BTC Price: ${btc_price:,.2f}")
        
        # Test options discovery
        print("\n2. Testing options discovery...")
        options = trader.delta_client.get_daily_expiry_options()
        print(f"   Found {len(options)} daily expiry options")
        
        if options:
            print("   Sample options:")
            for i, opt in enumerate(options[:5]):  # Show first 5
                print(f"     {i+1}. {opt['symbol']} - Strike: {opt['strike_price']} - Type: {opt['option_type']}")
        
        # Test higher timeframe analysis
        print("\n3. Testing higher timeframe analysis...")
        try:
            from app import calculate_higher_timeframe_indicators
            import pandas as pd
            
            # Get 1H candle data
            live_data = trader.delta_client.get_historical_candles('BTCUSD', '1h', 100)
            if live_data:
                df = pd.DataFrame(live_data)
                df['timestamp'] = pd.to_datetime(df['time'], unit='s', utc=True)
                df = df.sort_values('timestamp')
                
                trend_result = calculate_higher_timeframe_indicators(df)
                print(f"   Overall Trend: {trend_result['overall_trend']}")
                print(f"   Trend Score: {trend_result['total_score']}/{trend_result['max_score']}")
                
                # Test neutral strategy trigger
                if trend_result['overall_trend'] == 'Neutral':
                    print("\n4. NEUTRAL TREND DETECTED - Testing strangle logic...")
                    
                    # Test strike selection
                    from datetime import datetime, timedelta
                    target_expiry = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
                    
                    strangle_info = trader.get_strangle_strikes(btc_price, target_expiry)
                    if strangle_info:
                        print(f"   Call Strike: {strangle_info['call_strike']}")
                        print(f"   Put Strike: {strangle_info['put_strike']}")
                        print(f"   Distance from ATM: {abs(strangle_info['call_strike'] - btc_price)/btc_price*100:.1f}%")
                        print("   [SUCCESS] Strangle strategy ready for execution")
                    else:
                        print("   [ERROR] Could not find suitable strangle strikes")
                else:
                    print(f"\n4. Trend is {trend_result['overall_trend']} - neutral strategy not triggered")
            else:
                print("   [ERROR] Could not fetch 1H candle data")
                
        except Exception as e:
            print(f"   [ERROR] Error in higher timeframe analysis: {e}")
        
        # Test configuration
        print("\n5. Configuration Summary:")
        neutral_config = config.get('neutral_strategy', {})
        print(f"   Lot Size: {neutral_config.get('lot_size', 1)}")
        print(f"   Leverage: {neutral_config.get('leverage_percentage', 50)}%")
        print(f"   Strike Distance: {neutral_config.get('strike_distance', 8)}")
        print(f"   Stop Loss: {neutral_config.get('stop_loss_pct', 50)}%")
        print(f"   Profit Target: {neutral_config.get('profit_target_pct', 30)}%")
        
        print("\n[SUCCESS] All tests completed successfully!")
        print("\nTo enable the neutral strategy:")
        print("1. Ensure 'enabled': true in neutral_strategy config")
        print("2. Start the trading bot from the web interface")
        print("3. Monitor for neutral trend conditions")
        
    except Exception as e:
        print(f"\n[ERROR] Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_neutral_strategy())