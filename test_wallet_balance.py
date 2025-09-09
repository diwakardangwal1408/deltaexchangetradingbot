#!/usr/bin/env python3
"""
Test script to verify Delta Exchange API wallet balance fetching
"""

import os
import sys
from delta_exchange_client import DeltaExchangeClient
from config_manager import config_manager

def test_wallet_balance():
    """Test wallet balance fetching with real Delta Exchange API"""
    
    print("Testing Delta Exchange API Wallet Balance Fetching")
    print("=" * 60)
    
    try:
        # Load configuration
        config = config_manager.get_all_config()
        
        if not config.get('api_key') or not config.get('api_secret'):
            print("ERROR: API credentials not configured")
            print("Please configure your API credentials in the web interface")
            return False
        
        # Create client
        print(f"Creating client with paper_trading: {config.get('paper_trading', True)}")
        client = DeltaExchangeClient(
            config['api_key'],
            config['api_secret'], 
            config.get('paper_trading', True)
        )
        
        # Test 1: BTC Price (public endpoint)
        print("\n1. Testing BTC Price (Public API)")
        try:
            btc_price = client.get_current_btc_price()
            print(f"   [SUCCESS] BTC Price: ${btc_price:,.2f}")
        except Exception as e:
            print(f"   [FAILED] BTC Price failed: {e}")
        
        # Test 2: Account Balance (private endpoint)  
        print("\n2. Testing Account Balance (Private API)")
        try:
            balance = client.get_account_balance()
            print(f"   [SUCCESS] Balance fetched successfully!")
            
            if config.get('paper_trading', True):
                print("   [PAPER] Paper Trading Mode - Using simulated balances:")
                for currency, info in balance.items():
                    print(f"     {currency}: Balance={info.get('balance')} | Available={info.get('available')}")
            else:
                print("   [LIVE] Live Trading Mode - Real account balances:")
                for currency, info in balance.items():
                    print(f"     {currency}: Balance={info.get('balance')} | Available={info.get('available')}")
                    
        except Exception as e:
            print(f"   [FAILED] Balance fetch failed: {e}")
            print(f"   [INFO] Error details: This might be due to:")
            print(f"      - Invalid API credentials")
            print(f"      - API endpoint changes")
            print(f"      - Network connectivity issues")
            print(f"      - Delta Exchange API rate limiting")
            
        # Test 3: Portfolio Summary
        print("\n3. Testing Portfolio Summary")
        try:
            portfolio = client.get_portfolio_summary()
            print(f"   [SUCCESS] Portfolio summary fetched successfully!")
            print(f"     Total Balance: ${portfolio.get('total_balance', 0):,.2f}")
            print(f"     Available Balance: ${portfolio.get('available_balance', 0):,.2f}")
            print(f"     Unrealized PnL: ${portfolio.get('unrealized_pnl', 0):,.2f}")
            print(f"     Margin Used: ${portfolio.get('margin_used', 0):,.2f}")
            print(f"     Active Positions: {portfolio.get('positions', 0)}")
            
        except Exception as e:
            print(f"   [FAILED] Portfolio summary failed: {e}")
        
        print(f"\n" + "=" * 60)
        print("Test completed!")
        
        return True
        
    except Exception as e:
        print(f"ERROR: Test setup failed: {e}")
        return False

if __name__ == "__main__":
    success = test_wallet_balance()
    sys.exit(0 if success else 1)