#!/usr/bin/env python3
"""
Conditional Trade Manager
Main business logic for conditional trading functionality
"""

import uuid
import threading
from datetime import datetime
from typing import Dict, List, Optional, Any
from concurrent.futures import ThreadPoolExecutor
import logging

from .conditional_trade_models import ConditionalTrade, ConditionType, OrderSide, OrderType, TradeStatus
from delta_exchange_client import DeltaExchangeClient
from config_manager import config_manager
from logger_config import get_logger


class ConditionalTradeManager:
    """
    Manages conditional trades in-memory
    Handles creation, cancellation, and execution of conditional trades
    """
    
    def __init__(self, delta_client: Optional[DeltaExchangeClient] = None):
        self.logger = get_logger(__name__)
        self.config = config_manager.get_all_config()
        
        # Use provided client or create new one
        if delta_client:
            self.delta_client = delta_client
        else:
            self.delta_client = DeltaExchangeClient(
                api_key=self.config['api_key'],
                api_secret=self.config['api_secret'],
                paper_trading=self.config.get('paper_trading', True)
            )
        
        # In-memory storage for active conditional trades
        self._active_trades: Dict[str, ConditionalTrade] = {}
        
        # In-memory storage for executed trades
        self._executed_trades: List[Dict[str, Any]] = []
        self._trade_lock = threading.Lock()
        
        # Thread pool for executing trades
        self._executor = ThreadPoolExecutor(max_workers=3, thread_name_prefix="ConditionalTrade")
        
        self.logger.info("ConditionalTradeManager initialized")
    
    def create_conditional_trade(self,
                               condition_type: str,
                               target_price: float,
                               number_of_lots: int,
                               order_side: str,
                               order_type: str = "market",
                               limit_price: Optional[float] = None,
                               enable_stop_loss: bool = False,
                               stop_loss_price: Optional[float] = None) -> Dict[str, Any]:
        """
        Create a new conditional trade
        
        Returns:
            Dict with success status and trade details or error message
        """
        try:
            # Generate unique trade ID
            trade_id = str(uuid.uuid4())[:8]
            
            # Create conditional trade object
            conditional_trade = ConditionalTrade(
                trade_id=trade_id,
                condition_type=ConditionType(condition_type),
                target_price=target_price,
                number_of_lots=number_of_lots,
                order_side=OrderSide(order_side),
                order_type=OrderType(order_type),
                limit_price=limit_price,
                enable_stop_loss=enable_stop_loss,
                stop_loss_price=stop_loss_price
            )
            
            # Validate trade parameters
            validation_result = self._validate_trade_parameters(conditional_trade)
            if not validation_result["valid"]:
                return {"success": False, "error": validation_result["error"]}
            
            # Add to active trades
            with self._trade_lock:
                self._active_trades[trade_id] = conditional_trade
            
            self.logger.info(f"Created conditional trade {trade_id}: {condition_type} {target_price} -> {order_side} {number_of_lots} lots")
            
            return {
                "success": True,
                "trade_id": trade_id,
                "trade_details": conditional_trade.to_dict()
            }
            
        except ValueError as e:
            self.logger.error(f"Invalid conditional trade parameters: {e}")
            return {"success": False, "error": str(e)}
        except Exception as e:
            self.logger.error(f"Error creating conditional trade: {e}")
            return {"success": False, "error": f"Internal error: {str(e)}"}
    
    def cancel_conditional_trade(self, trade_id: str) -> Dict[str, Any]:
        """Cancel an active conditional trade"""
        try:
            with self._trade_lock:
                if trade_id not in self._active_trades:
                    return {"success": False, "error": "Trade not found"}
                
                trade = self._active_trades[trade_id]
                
                if trade.status != TradeStatus.ACTIVE:
                    return {"success": False, "error": f"Cannot cancel trade in {trade.status.value} status"}
                
                # Mark as cancelled and remove from active trades
                trade.mark_cancelled()
                del self._active_trades[trade_id]
            
            self.logger.info(f"Cancelled conditional trade {trade_id}")
            return {"success": True, "message": "Trade cancelled successfully"}
            
        except Exception as e:
            self.logger.error(f"Error cancelling conditional trade {trade_id}: {e}")
            return {"success": False, "error": f"Internal error: {str(e)}"}
    
    def get_active_trades(self) -> List[Dict[str, Any]]:
        """Get all active conditional trades"""
        with self._trade_lock:
            return [trade.to_dict() for trade in self._active_trades.values()]
    
    def get_trade_preview(self, trade_id: str, current_price: float) -> Optional[Dict[str, Any]]:
        """Get preview information for a specific trade"""
        with self._trade_lock:
            if trade_id in self._active_trades:
                return self._active_trades[trade_id].get_preview_text(current_price)
        return None
    
    def get_all_trade_previews(self, current_price: float) -> List[Dict[str, Any]]:
        """Get preview information for all active trades"""
        previews = []
        with self._trade_lock:
            for trade_id, trade in self._active_trades.items():
                preview = trade.get_preview_text(current_price)
                preview["trade_id"] = trade_id
                preview["created_at"] = trade.created_at.isoformat()
                previews.append(preview)
        return previews
    
    def get_executed_trades(self) -> List[Dict[str, Any]]:
        """Get all executed conditional trades"""
        with self._trade_lock:
            # Return copy of executed trades, most recent first
            return list(reversed(self._executed_trades))
    
    def check_and_execute_trades(self, current_price: float) -> List[Dict[str, Any]]:
        """
        Check all active trades for trigger conditions and execute if met
        
        Returns:
            List of execution results
        """
        triggered_trades = []
        
        with self._trade_lock:
            # Find trades that should be triggered
            for trade_id, trade in list(self._active_trades.items()):
                if trade.is_condition_met(current_price):
                    trade.trigger()
                    triggered_trades.append((trade_id, trade))
                    
                    # Remove from active trades (will be executed asynchronously)
                    del self._active_trades[trade_id]
        
        # Execute triggered trades asynchronously
        execution_results = []
        for trade_id, trade in triggered_trades:
            future = self._executor.submit(self._execute_trade, trade, current_price)
            execution_results.append({
                "trade_id": trade_id,
                "future": future,
                "trade": trade
            })
        
        return execution_results
    
    def _execute_trade(self, trade: ConditionalTrade, trigger_price: float) -> Dict[str, Any]:
        """
        Execute a triggered conditional trade via Delta Exchange using same method as dashboard
        
        Args:
            trade: The conditional trade to execute
            trigger_price: The price at which the trade was triggered
        
        Returns:
            Execution result dictionary
        """
        try:
            self.logger.info(f"Executing conditional trade {trade.trade_id} at trigger price {trigger_price}")
            
            # Get side and order type values safely (handle both enum and string)
            side_value = trade.order_side.value if hasattr(trade.order_side, 'value') else str(trade.order_side)
            order_type_value = trade.order_type.value if hasattr(trade.order_type, 'value') else str(trade.order_type)
            
            self.logger.info(f"Trade details: side={side_value}, type={order_type_value}, lots={trade.number_of_lots}, limit_price={trade.limit_price}")
            
            # Use the same approach as dashboard: find best futures contract using symbol-based approach
            best_contract, message = self._find_best_futures_contract(trade)
            if not best_contract:
                raise Exception(f"Cannot find suitable futures contract: {message}")
            
            # Calculate position size using dashboard's approach
            portfolio_size = self.config.get('position_size_usd', 600)
            quantity = trade.number_of_lots
            
            if order_type_value == 'market':
                self.logger.info(f"Placing MARKET order: symbol={best_contract['symbol']}, side={side_value}, quantity={quantity}")
                order_result = self.delta_client.place_order(
                    symbol=best_contract['symbol'],
                    side=side_value,
                    quantity=quantity,
                    order_type='market'
                )
            else:  # LIMIT order
                # Pass price as float - delta_exchange_client will convert to string
                price_value = float(trade.limit_price)
                self.logger.info(f"Placing LIMIT order: symbol={best_contract['symbol']}, side={side_value}, quantity={quantity}, price={price_value}")
                order_result = self.delta_client.place_order(
                    symbol=best_contract['symbol'],
                    side=side_value,
                    quantity=quantity,
                    order_type='limit',
                    price=price_value
                )
            
            if order_result.get('success'):
                order_id = order_result.get('result', {}).get('id')
                executed_price = float(order_result.get('result', {}).get('average_fill_price', trigger_price))
                
                trade.mark_executed(executed_price, str(order_id))
                
                # Store executed trade details
                executed_trade_details = {
                    "trade_id": trade.trade_id,
                    "order_id": str(order_id),
                    "executed_price": executed_price,
                    "trigger_price": trigger_price,
                    "condition_type": trade.condition_type.value,
                    "target_price": trade.target_price,
                    "order_side": trade.order_side.value,
                    "order_type": trade.order_type.value,
                    "number_of_lots": trade.number_of_lots,
                    "executed_at": datetime.now().isoformat(),
                    "created_at": trade.created_at.isoformat()
                }
                
                with self._trade_lock:
                    self._executed_trades.append(executed_trade_details)
                    # Keep only last 100 executed trades to prevent memory issues
                    if len(self._executed_trades) > 100:
                        self._executed_trades = self._executed_trades[-100:]
                
                self.logger.info(f"Successfully executed conditional trade {trade.trade_id}: Order ID {order_id} at price {executed_price}")
                
                # Stop loss orders disabled per user request
                # No additional stop loss orders will be placed
                
                return {
                    "success": True,
                    "trade_id": trade.trade_id,
                    "order_id": order_id,
                    "executed_price": executed_price,
                    "message": "Trade executed successfully"
                }
            else:
                error_msg = order_result.get('error', 'Unknown execution error')
                trade.mark_failed(error_msg)
                
                self.logger.error(f"Failed to execute conditional trade {trade.trade_id}: {error_msg}")
                return {
                    "success": False,
                    "trade_id": trade.trade_id,
                    "error": error_msg
                }
        
        except Exception as e:
            error_msg = f"Execution error: {str(e)}"
            trade.mark_failed(error_msg)
            self.logger.error(f"Exception executing conditional trade {trade.trade_id}: {e}")
            
            return {
                "success": False,
                "trade_id": trade.trade_id,
                "error": error_msg
            }
    
    def _find_best_futures_contract(self, trade: ConditionalTrade):
        """Find the best futures contract for directional trading - copied from dashboard logic"""
        try:
            # Get available futures from Delta client - same as dashboard
            available_futures = self._get_available_futures()
            
            if not available_futures:
                return None, "No futures contracts available"
            
            current_price = self.delta_client.get_current_btc_price()
            if not current_price:
                return None, "Could not get current BTC price"
            
            # Look for active perpetual futures first (preferred for directional trades)
            best_contract = None
            best_symbol = None
            
            # Prioritize BTCUSD perpetual futures - same priority as dashboard
            priority_symbols = ['BTCUSD', 'BTCUSDT', 'BTCUSD_PERP', 'BTCUSDT_PERP']
            
            for symbol in priority_symbols:
                if symbol in available_futures:
                    contract = available_futures[symbol]
                    if contract.get('trading_status') == 'operational':
                        # Get current orderbook to check liquidity
                        try:
                            orderbook = self.delta_client.get_orderbook(symbol)
                            if orderbook and orderbook.get('buy') and orderbook.get('sell'):
                                best_contract = contract
                                best_symbol = symbol
                                break
                        except Exception as e:
                            self.logger.warning(f"Could not get orderbook for {symbol}: {e}")
                            continue
            
            if not best_contract:
                return None, "No operational futures contracts with sufficient liquidity found"
            
            # Get best prices from orderbook
            try:
                orderbook = self.delta_client.get_orderbook(best_symbol)
                if not orderbook:
                    return None, f"Could not get orderbook for {best_symbol}"
                
                # Determine side and price based on trade direction
                if trade.order_side.value == 'buy':
                    # For buying, use ask price (sell side of orderbook)
                    if not orderbook.get('sell') or len(orderbook['sell']) == 0:
                        return None, f"No ask prices available for {best_symbol}"
                    price = float(orderbook['sell'][0]['price'])
                    side = 'buy'
                else:
                    # For selling, use bid price (buy side of orderbook)  
                    if not orderbook.get('buy') or len(orderbook['buy']) == 0:
                        return None, f"No bid prices available for {best_symbol}"
                    price = float(orderbook['buy'][0]['price'])
                    side = 'sell'
                
                return {
                    'symbol': best_symbol,
                    'price': price,
                    'side': side,
                    'contract_info': best_contract
                }, None
                
            except Exception as e:
                return None, f"Error processing orderbook for {best_symbol}: {e}"
                
        except Exception as e:
            self.logger.error(f"Error finding best futures contract: {e}")
            return None, f"Contract search failed: {e}"
    
    def _get_available_futures(self):
        """Get available BTC futures contracts - same logic as dashboard"""
        try:
            # Get all products from Delta Exchange
            products = self.delta_client.get_products()
            available_futures = {}
            
            for product in products:
                symbol = product.get('symbol', '')
                
                # Look for BTC futures contracts
                if ('BTC' in symbol and 
                    ('USD' in symbol or 'USDT' in symbol) and
                    product.get('contract_type') in ['perpetual_futures', 'futures']):
                    
                    available_futures[symbol] = {
                        'symbol': symbol,
                        'id': product.get('id'),
                        'trading_status': product.get('trading_status', 'unknown'),
                        'contract_type': product.get('contract_type'),
                        'underlying': product.get('underlying_asset', {}),
                        'tick_size': product.get('tick_size'),
                        'lot_size': product.get('contract_unit_currency')
                    }
            
            self.logger.info(f"Found {len(available_futures)} BTC futures contracts")
            return available_futures
            
        except Exception as e:
            self.logger.error(f"Error getting available futures: {e}")
            return {}
    
    
    def _validate_trade_parameters(self, trade: ConditionalTrade) -> Dict[str, Any]:
        """Validate trade parameters against account and market constraints"""
        try:
            # Check if trading is enabled
            if not self.config.get('futures_strategy', {}).get('enabled', False):
                return {"valid": False, "error": "Futures trading is not enabled in configuration"}
            
            # Validate lot size against configuration
            max_position_usd = self.config.get('futures_strategy', {}).get('position_size_usd', 600)
            
            # Get current BTC price for validation
            try:
                ticker = self.delta_client.get_ticker('BTCUSD')
                current_price = float(ticker.get('close', 0))
                
                estimated_position_value = current_price * trade.number_of_lots
                
                if estimated_position_value > max_position_usd * 2:  # Allow some flexibility
                    return {
                        "valid": False, 
                        "error": f"Position size too large. Estimated value: ${estimated_position_value:,.2f}, Max allowed: ${max_position_usd * 2:,.2f}"
                    }
            
            except Exception as e:
                self.logger.warning(f"Could not validate position size against current price: {e}")
            
            # Check stop loss logic
            if trade.enable_stop_loss:
                if trade.order_side == OrderSide.BUY and trade.stop_loss_price >= trade.target_price:
                    return {"valid": False, "error": "Stop loss price should be below target price for BUY orders"}
                elif trade.order_side == OrderSide.SELL and trade.stop_loss_price <= trade.target_price:
                    return {"valid": False, "error": "Stop loss price should be above target price for SELL orders"}
            
            return {"valid": True}
        
        except Exception as e:
            self.logger.error(f"Error validating trade parameters: {e}")
            return {"valid": False, "error": f"Validation error: {str(e)}"}
    
    def get_trade_count(self) -> int:
        """Get count of active conditional trades"""
        with self._trade_lock:
            return len(self._active_trades)
    
    def shutdown(self):
        """Clean shutdown of the trade manager"""
        self.logger.info("Shutting down ConditionalTradeManager")
        self._executor.shutdown(wait=True)
        
        # Cancel all active trades
        with self._trade_lock:
            for trade in self._active_trades.values():
                trade.mark_cancelled()
            self._active_trades.clear()
        
        self.logger.info("ConditionalTradeManager shutdown complete")