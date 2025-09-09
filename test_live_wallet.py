#!/usr/bin/env python3
"""
Test script specifically for testing live Delta Exchange API wallet balance
This script temporarily switches to live mode to test real API connectivity
"""

import os
import sys
from delta_exchange_client import DeltaExchangeClient
from config_manager import config_manager

def test_live_wallet_balance():
    """Test live wallet balance fetching with Delta Exchange API"""
    
    print("Testing LIVE Delta Exchange API Wallet Balance")
    print("=" * 60)
    print("WARNING: This test will attempt to connect to your LIVE Delta Exchange account")
    print("Make sure your API credentials have the appropriate permissions:")
    print("- Read access to wallet/balance")  
    print("- Read access to positions")
    print("- Read access to orders")
    print("=" * 60)
    
    try:
        # Load configuration
        config = config_manager.get_all_config()
        
        if not config.get('api_key') or not config.get('api_secret'):
            print("ERROR: API credentials not configured")
            return False
        
        print(f"API Key: {config['api_key'][:10]}...{config['api_key'][-4:]}")
        print(f"Currently in paper trading: {config.get('paper_trading', True)}")
        
        # Create client in LIVE mode (override paper trading)
        print("\nCreating client in LIVE TRADING mode...")
        client = DeltaExchangeClient(
            config['api_key'],
            config['api_secret'], 
            paper_trading=False  # Force live mode for this test
        )
        
        # Test 1: BTC Price (public endpoint - should work regardless)
        print("\n1. Testing BTC Price (Public API)")
        try:
            btc_price = client.get_current_btc_price()
            print(f"   [SUCCESS] BTC Price: ${btc_price:,.2f}")
        except Exception as e:
            print(f"   [FAILED] BTC Price failed: {e}")
            print("   [ERROR] If public API fails, check network connectivity")
            return False
        
        # Test 2: Live Account Balance (private endpoint)  
        print("\n2. Testing LIVE Account Balance (Private API)")
        print("   Using endpoint: /v2/wallet")
        try:
            balance = client.get_account_balance()
            print(f"   [SUCCESS] Live balance fetched successfully!")
            
            print("   [LIVE] Real Delta Exchange account balances:")
            total_usd_value = 0
            
            for currency, info in balance.items():
                balance_amt = float(info.get('balance', 0))
                available_amt = float(info.get('available', 0))
                
                print(f"     {currency}:")
                print(f"       Total Balance: {balance_amt:,.6f}")
                print(f"       Available: {available_amt:,.6f}")
                print(f"       Asset ID: {info.get('asset_id', 'N/A')}")
                
                # Try to calculate USD value for major assets
                if currency in ['USD', 'USDT', 'USDC']:
                    total_usd_value += balance_amt
                elif currency == 'BTC':
                    total_usd_value += balance_amt * btc_price
                    
            print(f"\n   [ESTIMATE] Approximate Total USD Value: ${total_usd_value:,.2f}")
            print(f"   [NOTE] This is an estimate based on BTC price and USD stablecoins only")
                    
        except Exception as e:
            print(f"   [FAILED] Live balance fetch failed: {e}")
            print(f"   [DEBUG] Full error details:")
            print(f"   {str(e)}")
            
            # Provide troubleshooting guidance
            print(f"\n   [TROUBLESHOOTING] Possible issues:")
            print(f"   1. API Key Permissions:")
            print(f"      - Ensure 'Read' permission is enabled for Wallet")
            print(f"      - Check if API key is active and not expired")
            print(f"   2. API Endpoint Issues:")
            print(f"      - Delta Exchange may have updated endpoints")
            print(f"      - Rate limiting might be in effect")
            print(f"   3. Network/Connectivity:")
            print(f"      - Firewall blocking HTTPS requests")
            print(f"      - DNS resolution issues with api.delta.exchange")
            print(f"   4. Account Status:")
            print(f"      - Account might be restricted or under review")
            
            return False
            
        # Test 3: Live Portfolio Summary
        print("\n3. Testing LIVE Portfolio Summary")
        try:
            portfolio = client.get_portfolio_summary()
            print(f"   [SUCCESS] Live portfolio summary fetched!")
            print(f"     Total Balance: ${portfolio.get('total_balance', 0):,.2f}")
            print(f"     Available Balance: ${portfolio.get('available_balance', 0):,.2f}")
            print(f"     Unrealized PnL: ${portfolio.get('unrealized_pnl', 0):,.2f}")
            print(f"     Margin Used: ${portfolio.get('margin_used', 0):,.2f}")
            print(f"     Active Positions: {portfolio.get('positions', 0)}")
            
        except Exception as e:
            print(f"   [FAILED] Live portfolio summary failed: {e}")
        
        # Test 4: Positions (if any)
        print("\n4. Testing LIVE Positions")
        try:
            positions = client.get_positions()
            print(f"   [SUCCESS] Positions fetched successfully!")
            print(f"   [INFO] Active Positions: {len(positions)}")
            
            if positions:
                print("   [POSITIONS] Current active positions:")
                for i, pos in enumerate(positions[:5]):  # Show max 5 positions
                    symbol = pos.get('product_symbol', 'Unknown')
                    size = pos.get('size', 0)
                    side = pos.get('side', 'Unknown')
                    pnl = pos.get('unrealized_pnl', 0)
                    print(f"     {i+1}. {symbol} | Side: {side} | Size: {size} | PnL: ${pnl}")
            else:
                print("   [INFO] No active positions found")
                
        except Exception as e:
            print(f"   [FAILED] Positions fetch failed: {e}")
        
        print(f"\n" + "=" * 60)
        print("LIVE API TEST COMPLETED!")
        print("If all tests passed, your Delta Exchange API integration is working correctly.")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"ERROR: Live test setup failed: {e}")
        return False

def main():
    """Main function with user confirmation"""
    
    print("LIVE TRADING API TEST")
    print("This will test your actual Delta Exchange account via API")
    print("Make sure you understand what this test does before proceeding.")
    print()
    
    response = input("Do you want to proceed with live API testing? (yes/no): ").lower().strip()
    
    if response in ['yes', 'y']:
        success = test_live_wallet_balance()
        sys.exit(0 if success else 1)
    else:
        print("Test cancelled by user.")
        sys.exit(0)

if __name__ == "__main__":
    main()