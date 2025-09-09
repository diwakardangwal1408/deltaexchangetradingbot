#!/usr/bin/env python3
"""
Test script to verify lot size calculations and premium/margin calculations
"""

import asyncio
from delta_exchange_client import DeltaExchangeClient
from config_manager import config_manager

async def test_lot_sizes():
    """Test lot size and calculation functionality"""
    try:
        print("Testing Delta Exchange Lot Size Calculations")
        print("=" * 60)
        
        # Load configuration
        config = config_manager.get_all_config()
        
        if not config.get('api_key') or not config.get('api_secret'):
            print("ERROR: API credentials not configured")
            return
        
        # Initialize client
        client = DeltaExchangeClient(
            api_key=config['api_key'],
            api_secret=config['api_secret'],
            paper_trading=config.get('paper_trading', True)
        )
        
        print(f"Paper Trading Mode: {config.get('paper_trading', True)}")
        
        # Test 1: Get BTC price
        print("\n1. Testing BTC price fetch...")
        btc_price = client.get_current_btc_price()
        print(f"   Current BTC Price: ${btc_price:,.2f}")
        
        # Test 2: Fetch product specifications
        print("\n2. Fetching BTC options specifications...")
        specs = client.get_btc_options_specifications()
        print(f"   Found specifications for {len(specs)} BTC options")
        
        # Show sample specifications
        sample_symbols = list(specs.keys())[:3]
        for symbol in sample_symbols:
            spec = specs[symbol]
            print(f"   {symbol}:")
            print(f"     Contract Value: {spec.get('contract_value')} BTC per lot")
            print(f"     Strike Price: {spec.get('strike_price')}")
            print(f"     Option Type: {spec.get('option_type')}")
        
        # Test 3: Get available options
        print("\n3. Fetching available options...")
        options = client.get_daily_expiry_options()
        print(f"   Found {len(options)} available options")
        
        if options:
            # Test with first available option
            test_option = options[0]
            symbol = test_option['symbol']
            mark_price = test_option.get('mark_price', 0.001)
            
            print(f"\n4. Testing calculations with {symbol}")
            print(f"   Mark Price: {mark_price:.6f}")
            
            # Test lot size retrieval
            lot_size = client.get_lot_size_for_symbol(symbol)
            print(f"   Lot Size: {lot_size} BTC per lot")
            
            # Test premium calculation for different lot quantities
            for lots in [1, 2, 5]:
                premium = client.calculate_premium_in_usdt(symbol, mark_price, lots)
                print(f"   Premium for {lots} lots: ${premium:.2f}")
            
            # Test margin calculation for selling
            for lots in [1, 2, 5]:
                margin = client.calculate_margin_requirement(symbol, mark_price, lots, 'sell')
                print(f"   Margin for selling {lots} lots: ${margin:.2f}")
            
            print(f"\n5. Detailed breakdown for 1 lot:")
            print(f"   Symbol: {symbol}")
            print(f"   Option Price: {mark_price:.6f}")
            print(f"   Lot Size: {lot_size} BTC")
            print(f"   BTC Price: ${btc_price:.2f}")
            print(f"   Premium = {mark_price:.6f} * {lot_size} * ${btc_price:.2f} * 1 lot")
            print(f"   Premium = ${mark_price * lot_size * btc_price:.2f}")
            
            # Compare with our calculation
            our_premium = client.calculate_premium_in_usdt(symbol, mark_price, 1)
            print(f"   Our calculation: ${our_premium:.2f}")
            
        print(f"\n" + "=" * 60)
        print("Lot size calculation test completed!")
        
    except Exception as e:
        print(f"ERROR: Test failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_lot_sizes())