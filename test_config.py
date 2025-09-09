#!/usr/bin/env python3
"""
Test script for the new configuration system
"""

from config_manager import config_manager
import json

def test_config_system():
    """Test configuration save and load functionality"""
    print("Testing Configuration System")
    print("=" * 50)
    
    # 1. Test loading current config
    print("1. Loading current configuration...")
    config = config_manager.get_all_config()
    print(f"   SUCCESS: Loaded {len(config)} configuration sections")
    print(f"   Paper Trading: {config.get('paper_trading')}")
    print(f"   Portfolio Size: ${config.get('portfolio_size'):,.0f}")
    
    # 2. Test modifying and saving configuration
    print("\n2. Testing configuration save...")
    
    # Create a test configuration change
    test_config = config.copy()
    test_config['portfolio_size'] = 150000.0  # Change portfolio size
    test_config['paper_trading'] = False      # Change paper trading mode
    test_config['max_positions'] = 5          # Change max positions
    
    # Save the modified config
    success = config_manager.save_config(test_config)
    if success:
        print("   SUCCESS: Configuration saved successfully")
    else:
        print("   ERROR: Failed to save configuration")
        return
    
    # 3. Test reloading to verify save worked
    print("\n3. Verifying saved configuration...")
    reloaded_config = config_manager.get_all_config()
    
    if reloaded_config.get('portfolio_size') == 150000.0:
        print("   SUCCESS: Portfolio size correctly updated")
    else:
        print(f"   ERROR: Portfolio size not updated: {reloaded_config.get('portfolio_size')}")
    
    if reloaded_config.get('paper_trading') == False:
        print("   SUCCESS: Paper trading mode correctly updated")
    else:
        print(f"   ERROR: Paper trading not updated: {reloaded_config.get('paper_trading')}")
    
    if reloaded_config.get('max_positions') == 5:
        print("   SUCCESS: Max positions correctly updated")
    else:
        print(f"   ERROR: Max positions not updated: {reloaded_config.get('max_positions')}")
    
    # 4. Restore original configuration
    print("\n4. Restoring original configuration...")
    restore_success = config_manager.save_config(config)
    if restore_success:
        print("   SUCCESS: Original configuration restored")
    else:
        print("   ERROR: Failed to restore original configuration")
    
    print("\n" + "=" * 50)
    print("Configuration system test completed!")

if __name__ == "__main__":
    test_config_system()