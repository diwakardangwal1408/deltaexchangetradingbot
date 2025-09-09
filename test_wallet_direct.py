#!/usr/bin/env python3
"""
Direct test for wallet balance with live API
"""

import os
import sys
from delta_exchange_client import DeltaExchangeClient
from config_manager import config_manager

def test_direct():
    """Test wallet balance directly with live mode"""
    
    print("Direct Wallet Balance Test - LIVE MODE")
    print("=" * 50)
    
    try:
        # Load configuration
        config = config_manager.get_all_config()
        
        print(f"Paper trading setting: {config.get('paper_trading')}")
        print(f"API Key: {config['api_key'][:10]}...")
        
        # Create client with live mode
        client = DeltaExchangeClient(
            config['api_key'],
            config['api_secret'], 
            paper_trading=False  # Force live mode
        )
        
        print("\n1. Testing BTC Price...")
        try:
            btc_price = client.get_current_btc_price()
            print(f"   [SUCCESS] BTC Price: ${btc_price:,.2f}")
        except Exception as e:
            print(f"   [FAILED] BTC Price: {e}")
            return
        
        print("\n2. Testing Account Balance...")
        print("   Base URL: https://api.india.delta.exchange")
        print("   Endpoint: /v2/wallet/balances")
        try:
            balance = client.get_account_balance()
            print(f"   [SUCCESS] Balance fetched!")
            
            for currency, info in balance.items():
                print(f"     {currency}: {info.get('balance')} (Available: {info.get('available')})")
                
        except Exception as e:
            print(f"   [FAILED] Balance fetch: {e}")
            
            # Print detailed error information
            print(f"\nDetailed Error Analysis:")
            error_str = str(e)
            
            if "401" in error_str:
                print("   - 401 Unauthorized: API credentials are invalid or expired")
                print("   - Check if API key and secret are correct")
                print("   - Verify API key permissions include 'Read' for wallet")
                print("   - Ensure API key is active and not suspended")
            elif "403" in error_str:
                print("   - 403 Forbidden: API key doesn't have required permissions")
                print("   - Enable 'Read' permission for wallet/balance in Delta Exchange settings")
            elif "404" in error_str:
                print("   - 404 Not Found: API endpoint might be incorrect")
                print("   - Verify if Delta Exchange updated their API endpoints")
            elif "wallet/balances" in error_str:
                print("   - ERROR: Code is still using old endpoint!")
                print("   - Expected: /v2/wallet")
                print("   - Actual: /v2/wallet/balances")
                print("   - Need to restart server or clear Python cache")
            else:
                print(f"   - Unknown error: {error_str}")
        
        print("\n3. Testing Authentication Signature...")
        try:
            # Try a simple authenticated request
            signature, timestamp = client._generate_signature('GET', '/v2/wallet')
            print(f"   [SUCCESS] Signature generated: {signature[:20]}...")
            print(f"   [SUCCESS] Timestamp: {timestamp}")
        except Exception as e:
            print(f"   [FAILED] Signature generation: {e}")
        
        return True
        
    except Exception as e:
        print(f"ERROR: {e}")
        return False

if __name__ == "__main__":
    test_direct()