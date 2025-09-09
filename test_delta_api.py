#!/usr/bin/env python3
"""
Delta Exchange API Test Script
Helps diagnose API connection issues and find working endpoints
"""

import json
from delta_exchange_client import DeltaExchangeClient
from config_manager import config_manager

def test_delta_api():
    """Test Delta Exchange API connection with detailed diagnostics"""
    
    # Load config
    try:
        config = config_manager.get_all_config()
    except Exception as e:
        print(f"❌ Error loading config: {e}")
        return
    
    api_key = config.get('api_key')
    api_secret = config.get('api_secret')
    paper_trading = config.get('paper_trading', True)
    
    if not api_key or not api_secret:
        print("ERROR: API credentials not found in config")
        return
    
    print("Testing Delta Exchange API Connection")
    print("=" * 50)
    print(f"API Key: {api_key[:10]}..." if api_key else "Not set")
    print(f"Paper Trading: {paper_trading}")
    print(f"Base URL: {'https://testnet-api.delta.exchange' if paper_trading else 'https://api.delta.exchange'}")
    print()
    
    # Initialize client
    client = DeltaExchangeClient(api_key, api_secret, paper_trading)
    
    # Test 1: Basic API connection
    print("Test 1: Basic API Connection")
    try:
        # Try a simple endpoint
        response = client._make_request('GET', '/v2/products')
        if response.get('success'):
            print("SUCCESS: Basic API connection successful")
            products = response.get('result', [])
            print(f"   Found {len(products)} products")
            
            # Show sample products
            btc_products = [p for p in products[:5] if 'BTC' in p.get('symbol', '')]
            for product in btc_products:
                print(f"   - {product.get('symbol')} ({product.get('product_type')})")
        else:
            print(f"FAILED: Basic API connection failed: {response}")
    except Exception as e:
        print(f"ERROR: Basic API connection error: {e}")
    
    print()
    
    # Test 2: BTC Price
    print("Test 2: BTC Price Retrieval")
    try:
        btc_price = client.get_current_btc_price()
        if btc_price:
            print(f"SUCCESS: BTC Price: ${btc_price:,.2f}")
        else:
            print("FAILED: Failed to get BTC price")
    except Exception as e:
        print(f"ERROR: BTC price error: {e}")
    
    print()
    
    # Test 3: Account Balance
    print("Test 3: Account Balance")
    try:
        balance = client.get_account_balance()
        if balance:
            print("SUCCESS: Account balance retrieved:")
            if isinstance(balance, dict):
                for currency, amount in balance.items():
                    if isinstance(amount, dict):
                        print(f"   {currency}: {amount}")
                    else:
                        print(f"   {currency}: {amount}")
            else:
                print(f"   Balance data: {balance}")
        else:
            print("WARNING: No balance data (might be normal for testnet)")
    except Exception as e:
        print(f"ERROR: Balance error: {e}")
    
    print()
    
    # Test 4: Portfolio Summary
    print("Test 4: Portfolio Summary")
    try:
        portfolio = client.get_portfolio_summary()
        if portfolio:
            print("SUCCESS: Portfolio summary retrieved:")
            if isinstance(portfolio, dict):
                for key, value in portfolio.items():
                    print(f"   {key}: {value}")
            else:
                print(f"   Portfolio data: {portfolio}")
        else:
            print("WARNING: No portfolio data (might be normal for testnet)")
    except Exception as e:
        print(f"ERROR: Portfolio error: {e}")
    
    print()
    
    # Test 5: BTC Options
    print("Test 5: BTC Options")
    try:
        options = client.get_btc_options()
        if options:
            print(f"SUCCESS: Found {len(options)} BTC options")
            
            # Show daily expiry options
            daily_options = client.get_daily_expiry_options()
            print(f"   Daily expiry options: {len(daily_options)}")
            
            for option in daily_options[:3]:  # Show first 3
                print(f"   - {option.get('symbol')} | Strike: ${option.get('strike_price')} | Type: {option.get('option_type')}")
        else:
            print("FAILED: No BTC options found")
    except Exception as e:
        print(f"ERROR: BTC options error: {e}")
    
    print()
    print("API Test Complete")
    print("=" * 50)
    
    # Summary
    print("\nSUMMARY:")
    if btc_price:
        print("SUCCESS: API Connection: Working")
    else:
        print("FAILED: API Connection: Issues detected")
    
    if balance or portfolio:
        print("SUCCESS: Account Access: Working")
    else:
        print("WARNING: Account Access: Limited (normal for testnet)")
    
    print("\nRECOMMENDATIONS:")
    if paper_trading:
        print("- You're using testnet - perfect for testing!")
        print("- Balance/portfolio data may be limited on testnet")
        print("- Focus on getting BTC price and options data working")
    else:
        print("- You're using live API - ensure you have proper permissions")
        print("- Check your API key has trading permissions")
        print("- Verify your account is funded if needed")
    
    print("\nTROUBLESHOOTING:")
    print("- If BTC price fails: Check Delta Exchange service status")
    print("- If balance fails: Normal for testnet, check live account funding") 
    print("- If options fail: Verify BTC options are available for today/tomorrow")

if __name__ == "__main__":
    test_delta_api()