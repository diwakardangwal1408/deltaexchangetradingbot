#!/usr/bin/env python3
"""
Setup script for Delta Exchange BTC Options Trading
Helps configure API credentials and validate setup
"""

import json
import os
import getpass
from delta_exchange_client import DeltaExchangeClient

def setup_api_credentials():
    """Setup API credentials securely"""
    print("=== Delta Exchange API Setup ===")
    print()
    print("You need to obtain API credentials from Delta Exchange:")
    print("1. Log into your Delta Exchange account")
    print("2. Go to Settings > API Management")
    print("3. Create a new API key with trading permissions")
    print("4. Copy your API Key and Secret")
    print()
    
    api_key = input("Enter your Delta Exchange API Key: ").strip()
    if not api_key:
        print("❌ API Key cannot be empty")
        return None, None
    
    api_secret = getpass.getpass("Enter your Delta Exchange API Secret: ").strip()
    if not api_secret:
        print("❌ API Secret cannot be empty")
        return None, None
    
    return api_key, api_secret

def test_api_connection(api_key, api_secret, paper_trading=True):
    """Test API connection"""
    print("\n=== Testing API Connection ===")
    
    try:
        client = DeltaExchangeClient(api_key, api_secret, paper_trading)
        
        print("✓ Creating client...")
        
        # Test basic connection
        btc_price = client.get_current_btc_price()
        if btc_price:
            print(f"✓ API connection successful! Current BTC price: ${btc_price:,.2f}")
        else:
            print("❌ Failed to get BTC price - check API credentials")
            return False
        
        # Test options data
        options = client.get_daily_expiry_options()
        print(f"✓ Found {len(options)} daily expiry BTC options")
        
        if options:
            sample_option = options[0]
            print(f"✓ Sample option: {sample_option['symbol']} (Strike: ${sample_option.get('strike_price', 'N/A')})")
        
        # Test account access
        try:
            balance = client.get_account_balance()
            if balance:
                print("✓ Account balance access successful")
            else:
                print("⚠️ Could not fetch account balance (this might be normal for testnet)")
        except Exception as e:
            print(f"⚠️ Account balance error: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ API connection failed: {e}")
        return False

def update_config_file(api_key, api_secret):
    """Update configuration file with API credentials"""
    config_file = "delta_config.json"
    
    try:
        # Load existing config
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        # Update credentials
        config['api_key'] = api_key
        config['api_secret'] = api_secret
        
        # Save updated config
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"✓ Configuration saved to {config_file}")
        return True
        
    except Exception as e:
        print(f"❌ Failed to update config file: {e}")
        return False

def setup_paper_trading_preferences():
    """Setup paper trading preferences"""
    print("\n=== Trading Mode Setup ===")
    
    mode = input("Do you want to use paper trading mode? (Y/n): ").strip().lower()
    paper_trading = mode != 'n'
    
    if paper_trading:
        print("✓ Paper trading mode selected (RECOMMENDED for testing)")
        print("  - Uses Delta Exchange testnet")
        print("  - No real money at risk")
        print("  - Perfect for strategy validation")
    else:
        print("⚠️ LIVE trading mode selected")
        print("  - Uses real money")
        print("  - Only use after thorough testing")
        
        confirm = input("Are you sure you want live trading? (type 'yes'): ").strip().lower()
        if confirm != 'yes':
            print("Switching back to paper trading mode for safety")
            paper_trading = True
    
    return paper_trading

def setup_risk_parameters():
    """Setup risk management parameters"""
    print("\n=== Risk Management Setup ===")
    
    try:
        portfolio_size = float(input("Enter your portfolio size in USD (default 100000): ") or "100000")
        position_size = float(input("Enter position size in USD per trade (default 500): ") or "500")
        max_daily_loss = float(input("Enter maximum daily loss in USD (default 2000): ") or "2000")
        max_positions = int(input("Enter maximum concurrent positions (default 2): ") or "2")
        
        # Validate parameters
        if position_size > portfolio_size * 0.1:
            print("⚠️ Position size seems large (>10% of portfolio)")
            confirm = input("Continue with this position size? (y/N): ").strip().lower()
            if confirm != 'y':
                position_size = portfolio_size * 0.05  # 5% default
                print(f"Adjusted position size to ${position_size}")
        
        return {
            'portfolio_size': portfolio_size,
            'position_size_usd': position_size,
            'max_daily_loss': max_daily_loss,
            'max_positions': max_positions
        }
        
    except ValueError:
        print("❌ Invalid input. Using default values.")
        return {
            'portfolio_size': 100000,
            'position_size_usd': 500,
            'max_daily_loss': 2000,
            'max_positions': 2
        }

def create_complete_config(api_key, api_secret, paper_trading, risk_params):
    """Create complete configuration file"""
    config = {
        "api_key": api_key,
        "api_secret": api_secret,
        "paper_trading": paper_trading,
        "portfolio_size": risk_params['portfolio_size'],
        "max_positions": risk_params['max_positions'],
        "position_size_usd": risk_params['position_size_usd'],
        "max_daily_loss": risk_params['max_daily_loss'],
        "min_premium": 0.001,
        "max_premium": 0.01,
        "order_timeout": 300,
        "risk_management": {
            "max_risk_per_trade": 0.005,
            "stop_loss_pct": 50,
            "take_profit_pct": 100,
            "quick_profit_pct": 30,
            "quick_profit_time_hours": 1,
            "time_exit_hours": 20
        },
        "strategy_params": {
            "min_signal_strength": 4,
            "volume_threshold": 2.0,
            "min_time_between_trades": 3600
        },
        "logging": {
            "level": "INFO",
            "file": "delta_btc_trading.log",
            "max_file_size": "10MB",
            "backup_count": 5
        },
        "notifications": {
            "enabled": False,
            "webhook_url": "",
            "discord_webhook": "",
            "telegram_bot_token": "",
            "telegram_chat_id": ""
        }
    }
    
    try:
        with open("delta_config.json", 'w') as f:
            json.dump(config, f, indent=2)
        print("✓ Complete configuration saved to delta_config.json")
        return True
    except Exception as e:
        print(f"❌ Failed to save configuration: {e}")
        return False

def display_next_steps():
    """Display next steps for the user"""
    print("\n" + "="*60)
    print("🎉 SETUP COMPLETE!")
    print("="*60)
    print()
    print("Next steps to start trading:")
    print()
    print("1. INSTALL DEPENDENCIES:")
    print("   pip install -r requirements_btc.txt")
    print("   pip install requests asyncio")
    print()
    print("2. TEST THE SETUP:")
    print("   python delta_btc_strategy.py")
    print()
    print("3. MONITOR THE LOGS:")
    print("   tail -f delta_btc_trading.log")
    print()
    print("4. PAPER TRADE FIRST:")
    print("   - Let it run for at least 1-2 weeks")
    print("   - Monitor performance and adjust parameters")
    print("   - Only switch to live trading after validation")
    print()
    print("5. CONFIGURATION FILES:")
    print("   - delta_config.json: Main configuration")
    print("   - delta_btc_trading.log: Trading logs")
    print("   - delta_trading_state.json: Trading state (auto-created)")
    print()
    print("⚠️  IMPORTANT REMINDERS:")
    print("   - Start with small position sizes")
    print("   - Monitor performance regularly")
    print("   - Cryptocurrency trading involves significant risk")
    print("   - Never trade more than you can afford to lose")
    print()

def main():
    """Main setup function"""
    print("🚀 Delta Exchange BTC Options Trading Setup")
    print("="*50)
    
    # Step 1: API Credentials
    api_key, api_secret = setup_api_credentials()
    if not api_key or not api_secret:
        print("❌ Setup cancelled - API credentials required")
        return
    
    # Step 2: Trading mode
    paper_trading = setup_paper_trading_preferences()
    
    # Step 3: Test connection
    if not test_api_connection(api_key, api_secret, paper_trading):
        print("❌ Setup failed - API connection test failed")
        return
    
    # Step 4: Risk parameters
    risk_params = setup_risk_parameters()
    
    # Step 5: Create configuration
    if create_complete_config(api_key, api_secret, paper_trading, risk_params):
        display_next_steps()
    else:
        print("❌ Setup failed - Could not create configuration")

if __name__ == "__main__":
    main()