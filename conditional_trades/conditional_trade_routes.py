#!/usr/bin/env python3
"""
Conditional Trade Routes
Flask blueprint for conditional trading API endpoints
"""

from flask import Blueprint, render_template, request, jsonify
import logging
from typing import Dict, Any
from datetime import datetime

from .conditional_trade_manager import ConditionalTradeManager
from .price_monitor import PriceMonitor
from delta_exchange_client import DeltaExchangeClient
from config_manager import config_manager
from logger_config import get_logger

# Create blueprint
conditional_trade_bp = Blueprint('conditional_trades', __name__, url_prefix='/api/conditional_trades')

# Global instances (will be initialized by main app)
trade_manager: ConditionalTradeManager = None
price_monitor: PriceMonitor = None
logger = get_logger(__name__)


def initialize_conditional_trades(delta_client: DeltaExchangeClient) -> Dict[str, Any]:
    """
    Initialize conditional trade components with existing Delta client
    Called from main Flask app during startup
    
    Args:
        delta_client: Existing DeltaExchangeClient instance
        
    Returns:
        Dict with initialization status and components
    """
    global trade_manager, price_monitor
    
    try:
        # Initialize trade manager with existing client
        trade_manager = ConditionalTradeManager(delta_client)
        
        # Initialize price monitor
        price_monitor = PriceMonitor(
            delta_client=delta_client,
            trade_manager=trade_manager,
            monitor_interval=5  # Check every 5 seconds
        )
        
        # Start price monitor
        price_monitor.start()
        
        logger.info("Conditional trades initialized successfully")
        
        return {
            "success": True,
            "trade_manager": trade_manager,
            "price_monitor": price_monitor
        }
        
    except Exception as e:
        logger.error(f"Failed to initialize conditional trades: {e}")
        return {
            "success": False,
            "error": str(e)
        }


def shutdown_conditional_trades():
    """Graceful shutdown of conditional trade components"""
    global trade_manager, price_monitor
    
    try:
        if price_monitor:
            price_monitor.stop()
            logger.info("Price monitor stopped")
        
        if trade_manager:
            trade_manager.shutdown()
            logger.info("Trade manager shutdown")
            
    except Exception as e:
        logger.error(f"Error during conditional trades shutdown: {e}")


# Routes
@conditional_trade_bp.route('/current_price', methods=['GET'])
def get_current_price():
    """Get current BTC price"""
    try:
        if not price_monitor:
            return jsonify({"success": False, "error": "Price monitor not initialized"})
        
        current_price = price_monitor.get_current_price()
        last_update = price_monitor.get_last_update()
        
        if current_price > 0:
            return jsonify({
                "success": True,
                "price": current_price,
                "timestamp": last_update.isoformat() if last_update else None
            })
        else:
            return jsonify({"success": False, "error": "Price not available"})
            
    except Exception as e:
        logger.error(f"Error getting current price: {e}")
        return jsonify({"success": False, "error": str(e)})


@conditional_trade_bp.route('/create', methods=['POST'])
def create_conditional_trade():
    """Create a new conditional trade"""
    try:
        if not trade_manager:
            return jsonify({"success": False, "error": "Trade manager not initialized"})
        
        data = request.get_json()
        if not data:
            return jsonify({"success": False, "error": "No data provided"})
        
        # Validate required fields
        required_fields = ['condition_type', 'target_price', 'number_of_lots', 'order_side', 'order_type']
        for field in required_fields:
            if field not in data:
                return jsonify({"success": False, "error": f"Missing required field: {field}"})
        
        # Create the conditional trade
        result = trade_manager.create_conditional_trade(
            condition_type=data['condition_type'],
            target_price=float(data['target_price']),
            number_of_lots=int(data['number_of_lots']),
            order_side=data['order_side'],
            order_type=data['order_type'],
            limit_price=float(data['limit_price']) if data.get('limit_price') else None,
            enable_stop_loss=data.get('enable_stop_loss', False),
            stop_loss_price=float(data['stop_loss_price']) if data.get('stop_loss_price') else None
        )
        
        return jsonify(result)
        
    except ValueError as e:
        logger.error(f"Validation error creating conditional trade: {e}")
        return jsonify({"success": False, "error": f"Validation error: {str(e)}"})
    except Exception as e:
        logger.error(f"Error creating conditional trade: {e}")
        return jsonify({"success": False, "error": f"Internal error: {str(e)}"})


@conditional_trade_bp.route('/active', methods=['GET'])
def get_active_trades():
    """Get all active conditional trades"""
    try:
        if not trade_manager:
            return jsonify({"success": False, "error": "Trade manager not initialized"})
        
        active_trades = trade_manager.get_active_trades()
        
        return jsonify({
            "success": True,
            "trades": active_trades,
            "count": len(active_trades)
        })
        
    except Exception as e:
        logger.error(f"Error getting active trades: {e}")
        return jsonify({"success": False, "error": str(e)})


@conditional_trade_bp.route('/cancel/<trade_id>', methods=['POST'])
def cancel_conditional_trade(trade_id):
    """Cancel a specific conditional trade"""
    try:
        if not trade_manager:
            return jsonify({"success": False, "error": "Trade manager not initialized"})
        
        result = trade_manager.cancel_conditional_trade(trade_id)
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error cancelling conditional trade {trade_id}: {e}")
        return jsonify({"success": False, "error": str(e)})


@conditional_trade_bp.route('/preview/<trade_id>', methods=['GET'])
def get_trade_preview(trade_id):
    """Get preview for a specific conditional trade"""
    try:
        if not trade_manager or not price_monitor:
            return jsonify({"success": False, "error": "Services not initialized"})
        
        current_price = price_monitor.get_current_price()
        if current_price <= 0:
            return jsonify({"success": False, "error": "Current price not available"})
        
        preview = trade_manager.get_trade_preview(trade_id, current_price)
        if preview:
            return jsonify({
                "success": True,
                "preview": preview,
                "current_price": current_price
            })
        else:
            return jsonify({"success": False, "error": "Trade not found"})
            
    except Exception as e:
        logger.error(f"Error getting trade preview for {trade_id}: {e}")
        return jsonify({"success": False, "error": str(e)})


@conditional_trade_bp.route('/preview_all', methods=['GET'])
def get_all_trade_previews():
    """Get previews for all active conditional trades"""
    try:
        if not trade_manager or not price_monitor:
            return jsonify({"success": False, "error": "Services not initialized"})
        
        current_price = price_monitor.get_current_price()
        if current_price <= 0:
            return jsonify({"success": False, "error": "Current price not available"})
        
        previews = trade_manager.get_all_trade_previews(current_price)
        
        return jsonify({
            "success": True,
            "previews": previews,
            "current_price": current_price,
            "count": len(previews)
        })
        
    except Exception as e:
        logger.error(f"Error getting all trade previews: {e}")
        return jsonify({"success": False, "error": str(e)})


@conditional_trade_bp.route('/executed', methods=['GET'])
def get_executed_trades():
    """Get all executed conditional trades"""
    try:
        if not trade_manager:
            return jsonify({"success": False, "error": "Trade manager not initialized"})
        
        executed_trades = trade_manager.get_executed_trades()
        
        return jsonify({
            "success": True,
            "executed_trades": executed_trades,
            "count": len(executed_trades)
        })
        
    except Exception as e:
        logger.error(f"Error getting executed trades: {e}")
        return jsonify({"success": False, "error": str(e)})


@conditional_trade_bp.route('/monitor_stats', methods=['GET'])
def get_monitor_stats():
    """Get price monitor statistics"""
    try:
        if not price_monitor:
            return jsonify({"success": False, "error": "Price monitor not initialized"})
        
        stats = price_monitor.get_statistics()
        
        return jsonify({
            "success": True,
            "stats": stats
        })
        
    except Exception as e:
        logger.error(f"Error getting monitor stats: {e}")
        return jsonify({"success": False, "error": str(e)})


@conditional_trade_bp.route('/monitor/start', methods=['POST'])
def start_price_monitor():
    """Start the price monitor (admin endpoint)"""
    try:
        if not price_monitor:
            return jsonify({"success": False, "error": "Price monitor not initialized"})
        
        if price_monitor.is_running():
            return jsonify({"success": False, "error": "Price monitor is already running"})
        
        price_monitor.start()
        
        return jsonify({
            "success": True,
            "message": "Price monitor started successfully"
        })
        
    except Exception as e:
        logger.error(f"Error starting price monitor: {e}")
        return jsonify({"success": False, "error": str(e)})


@conditional_trade_bp.route('/monitor/stop', methods=['POST'])
def stop_price_monitor():
    """Stop the price monitor (admin endpoint)"""
    try:
        if not price_monitor:
            return jsonify({"success": False, "error": "Price monitor not initialized"})
        
        if not price_monitor.is_running():
            return jsonify({"success": False, "error": "Price monitor is not running"})
        
        price_monitor.stop()
        
        return jsonify({
            "success": True,
            "message": "Price monitor stopped successfully"
        })
        
    except Exception as e:
        logger.error(f"Error stopping price monitor: {e}")
        return jsonify({"success": False, "error": str(e)})


@conditional_trade_bp.route('/force_price_update', methods=['POST'])
def force_price_update():
    """Force an immediate price update (testing endpoint)"""
    try:
        if not price_monitor:
            return jsonify({"success": False, "error": "Price monitor not initialized"})
        
        result = price_monitor.force_price_update()
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error forcing price update: {e}")
        return jsonify({"success": False, "error": str(e)})


# Main route for the conditional trade page
def register_page_route(main_app):
    """Register the main page route with the main Flask app"""
    @main_app.route('/conditional_trade')
    def conditional_trade_page():
        """Render the conditional trade page"""
        return render_template('conditional_trade.html')
    
    logger.info("Conditional trade page route registered")


# Export the blueprint and initialization functions
__all__ = [
    'conditional_trade_bp',
    'initialize_conditional_trades', 
    'shutdown_conditional_trades',
    'register_page_route'
]