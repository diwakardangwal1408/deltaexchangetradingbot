#!/usr/bin/env python3
"""
Test script to demonstrate lot size calculation logic without requiring API access
"""

def test_lot_calculations():
    """Test lot size calculation logic"""
    print("Delta Exchange India BTC Options - Lot Size Calculation Test")
    print("=" * 65)
    
    # Delta Exchange India specifications
    lot_size_btc = 0.001  # 1 lot = 0.001 BTC
    btc_price = 111000  # Example BTC price
    
    print(f"Delta Exchange India Specifications:")
    print(f"  Lot Size: {lot_size_btc} BTC per lot")
    print(f"  Current BTC Price: ${btc_price:,.2f}")
    print()
    
    # Example option prices and calculations
    test_cases = [
        {"symbol": "C-BTC-111500-080925", "option_price": 0.002, "option_type": "Call"},
        {"symbol": "P-BTC-110500-080925", "option_price": 0.0015, "option_type": "Put"},
        {"symbol": "C-BTC-112000-080925", "option_price": 0.001, "option_type": "Call"}
    ]
    
    print("Premium Calculations for Different Lot Quantities:")
    print("-" * 65)
    
    for case in test_cases:
        symbol = case["symbol"]
        option_price = case["option_price"]
        option_type = case["option_type"]
        
        print(f"\n{symbol} ({option_type})")
        print(f"Option Price: {option_price:.6f}")
        
        for lots in [1, 2, 5, 10]:
            # Premium calculation: option_price * lot_size_btc * btc_price * num_lots
            premium_usdt = option_price * lot_size_btc * btc_price * lots
            
            print(f"  {lots:2d} lots: ${premium_usdt:8.2f} "
                  f"({option_price:.6f} × {lot_size_btc} BTC × ${btc_price:,.0f} × {lots})")
    
    print(f"\n" + "=" * 65)
    
    # Margin calculations for selling options
    print("Margin Requirements for Selling Options (Simplified):")
    print("-" * 65)
    
    initial_margin_rate = 0.10  # 10% initial margin
    
    for case in test_cases:
        symbol = case["symbol"]
        option_price = case["option_price"]
        option_type = case["option_type"]
        
        print(f"\n{symbol} ({option_type}) - SELLING")
        
        for lots in [1, 2, 5]:
            # For selling: margin based on underlying value
            underlying_value = lot_size_btc * btc_price * lots
            margin_required = underlying_value * initial_margin_rate
            
            # Premium received
            premium_received = option_price * lot_size_btc * btc_price * lots
            
            # Net margin (margin - premium received, but minimum 10% of premium)
            net_margin = max(margin_required - premium_received, premium_received * 0.1)
            
            print(f"  {lots:2d} lots:")
            print(f"    Underlying Value: ${underlying_value:8.2f}")
            print(f"    Margin Required:  ${margin_required:8.2f} ({initial_margin_rate:.1%} of underlying)")
            print(f"    Premium Received: ${premium_received:8.2f}")
            print(f"    Net Margin:       ${net_margin:8.2f}")
    
    print(f"\n" + "=" * 65)
    print("Key Benefits of Proper Lot Size Handling:")
    print("  1. Accurate premium calculations using 0.001 BTC lot size")
    print("  2. Correct margin requirements for option selling")
    print("  3. Proper risk management based on actual position sizes")
    print("  4. Compliance with Delta Exchange India contract specifications")
    print("=" * 65)

if __name__ == "__main__":
    test_lot_calculations()