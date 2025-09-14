#!/usr/bin/env python3
"""
Conditional Trades Package
Modular conditional trading system for BTC futures
"""

from .conditional_trade_models import ConditionalTrade, ConditionType, OrderSide, OrderType, TradeStatus
from .conditional_trade_manager import ConditionalTradeManager  
from .price_monitor import PriceMonitor
from .conditional_trade_routes import (
    conditional_trade_bp, 
    initialize_conditional_trades, 
    shutdown_conditional_trades,
    register_page_route
)

__version__ = "1.0.0"
__author__ = "BTC Trading Bot"

__all__ = [
    # Models
    'ConditionalTrade',
    'ConditionType', 
    'OrderSide',
    'OrderType',
    'TradeStatus',
    
    # Core classes
    'ConditionalTradeManager',
    'PriceMonitor',
    
    # Flask integration
    'conditional_trade_bp',
    'initialize_conditional_trades',
    'shutdown_conditional_trades', 
    'register_page_route'
]