#!/usr/bin/env python3
"""
Test script to execute a single conditional trade on Delta Exchange
This will help debug the exact execution flow and identify where the error occurs
"""

import sys
import os
import time
from datetime import datetime

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from conditional_trades import ConditionalTradeManager
from delta_exchange_client import DeltaExchangeClient
from config_manager import config_manager
from logger_config import get_logger

def test_single_trade_execution():
    """Test executing a single 1-lot BTC trade on Delta Exchange"""
    
    logger = get_logger(__name__)
    logger.info("=" * 50)
    logger.info("STARTING CONDITIONAL TRADE TEST")
    logger.info("=" * 50)
    
    try:
        # Load configuration
        config = config_manager.get_all_config()
        logger.info(f"Loaded config - Paper trading: {config.get('paper_trading', True)}")
        
        # Create Delta Exchange client
        logger.info("Creating Delta Exchange client...")
        delta_client = DeltaExchangeClient(
            api_key=config['api_key'],
            api_secret=config['api_secret'],
            paper_trading=config.get('paper_trading', True)
        )
        
        # Test API connection first
        logger.info("Testing API connection...")
        current_price = delta_client.get_current_btc_price()
        logger.info(f"Current BTC price: ${current_price:,.2f}")
        
        # Create conditional trade manager
        logger.info("Creating conditional trade manager...")
        trade_manager = ConditionalTradeManager(delta_client)
        
        # Create a test conditional trade
        # We'll set the condition to trigger immediately for testing
        test_target_price = current_price - 1000  # Set target $1000 below current price
        
        logger.info(f"Creating test conditional trade:")
        logger.info(f"  - Condition: BTC > ${test_target_price:,.2f} (should trigger immediately)")
        logger.info(f"  - Side: BUY")
        logger.info(f"  - Lots: 1")
        logger.info(f"  - Type: MARKET")
        
        # Create the conditional trade
        result = trade_manager.create_conditional_trade(
            condition_type=">",  # Greater than (should trigger immediately since current > target-1000)
            target_price=test_target_price,
            number_of_lots=1,
            order_side="buy",
            order_type="market"
        )
        
        if not result.get('success'):
            logger.error(f"Failed to create conditional trade: {result.get('error')}")
            return False
        
        trade_id = result['trade_id']
        logger.info(f"✅ Conditional trade created successfully: {trade_id}")
        
        # Wait a moment for the price monitor to pick it up and execute
        logger.info("Waiting for trade execution...")
        
        # Check if condition should be met
        logger.info(f"Checking condition logic:")
        logger.info(f"  - Current price: ${current_price:,.2f}")
        logger.info(f"  - Target price: ${test_target_price:,.2f}")
        logger.info(f"  - Condition: > (current_price >= target_price)")
        logger.info(f"  - Should trigger: {current_price >= test_target_price}")
        
        # Manually trigger the execution check (simulate what price monitor does)
        logger.info("Manually checking for trade execution...")
        execution_results = trade_manager.check_and_execute_trades(current_price)
        
        if execution_results:
            logger.info(f"Found {len(execution_results)} trades to execute")
            for result in execution_results:
                try:
                    execution_result = result['future'].result(timeout=30)
                    logger.info(f"Execution result: {execution_result}")
                    
                    if execution_result.get('success'):
                        logger.info("🎉 CONDITIONAL TRADE EXECUTED SUCCESSFULLY!")
                        logger.info(f"Order ID: {execution_result.get('order_id')}")
                        logger.info(f"Executed Price: ${execution_result.get('executed_price', 0):,.2f}")
                        return True
                    else:
                        logger.error(f"❌ Trade execution failed: {execution_result.get('error')}")
                        return False
                        
                except Exception as e:
                    logger.error(f"❌ Exception during execution: {e}")
                    return False
        else:
            logger.warning("No trades were triggered for execution")
            
            # Check active trades
            active_trades = trade_manager.get_active_trades()
            logger.info(f"Active trades count: {len(active_trades)}")
            for trade in active_trades:
                logger.info(f"Trade {trade['trade_id']}: {trade['condition_type']} {trade['target_price']}")
            
            return False
    
    except Exception as e:
        logger.error(f"❌ Test failed with exception: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False
    
    finally:
        logger.info("=" * 50)
        logger.info("CONDITIONAL TRADE TEST COMPLETED")
        logger.info("=" * 50)

if __name__ == "__main__":
    print("Testing conditional trade execution with 1 lot...")
    print("This will attempt to place a small test trade on Delta Exchange")
    print("Make sure your API credentials are configured correctly")
    print()
    
    success = test_single_trade_execution()
    
    if success:
        print("\nSUCCESS: Conditional trade executed successfully!")
    else:
        print("\nFAILED: Conditional trade execution failed - check logs for details")
    
    print("\nCheck delta_btc_trading.log for detailed execution logs")