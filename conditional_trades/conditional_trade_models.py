#!/usr/bin/env python3
"""
Conditional Trade Models
In-memory data structures for conditional trading functionality
"""

from datetime import datetime
from typing import Dict, Any, Optional
from enum import Enum


class ConditionType(Enum):
    GREATER_THAN = ">"
    LESS_THAN = "<"


class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"


class TradeStatus(Enum):
    ACTIVE = "active"
    TRIGGERED = "triggered"
    EXECUTED = "executed"
    CANCELLED = "cancelled"
    FAILED = "failed"


class ConditionalTrade:
    """
    Represents a conditional trade order
    In-memory only - no persistent storage
    """
    
    def __init__(self, 
                 trade_id: str,
                 condition_type: ConditionType,
                 target_price: float,
                 number_of_lots: int,
                 order_side: OrderSide,
                 order_type: OrderType = OrderType.MARKET,
                 limit_price: Optional[float] = None,
                 enable_stop_loss: bool = False,
                 stop_loss_price: Optional[float] = None):
        
        self.trade_id = trade_id
        self.condition_type = condition_type
        self.target_price = target_price
        self.number_of_lots = number_of_lots
        self.order_side = order_side
        self.order_type = order_type
        self.limit_price = limit_price
        self.enable_stop_loss = enable_stop_loss
        self.stop_loss_price = stop_loss_price
        
        # Status tracking
        self.status = TradeStatus.ACTIVE
        self.created_at = datetime.now()
        self.triggered_at: Optional[datetime] = None
        self.executed_at: Optional[datetime] = None
        self.cancelled_at: Optional[datetime] = None
        
        # Execution details (populated when executed)
        self.executed_price: Optional[float] = None
        self.order_id: Optional[str] = None
        self.error_message: Optional[str] = None
        
        # Validate the trade
        self._validate()
    
    def _validate(self):
        """Validate trade parameters"""
        if self.target_price <= 0:
            raise ValueError("Target price must be positive")
        
        if self.number_of_lots <= 0:
            raise ValueError("Number of lots must be positive")
        
        if self.order_type == OrderType.LIMIT and not self.limit_price:
            raise ValueError("Limit price required for limit orders")
        
        if self.enable_stop_loss and not self.stop_loss_price:
            raise ValueError("Stop loss price required when stop loss is enabled")
        
        if self.enable_stop_loss and self.stop_loss_price <= 0:
            raise ValueError("Stop loss price must be positive")
    
    def is_condition_met(self, current_price: float) -> bool:
        """Check if the price condition is met"""
        if self.status != TradeStatus.ACTIVE:
            return False
        
        if self.condition_type == ConditionType.GREATER_THAN:
            return current_price >= self.target_price
        elif self.condition_type == ConditionType.LESS_THAN:
            return current_price <= self.target_price
        
        return False
    
    def trigger(self):
        """Mark trade as triggered"""
        if self.status == TradeStatus.ACTIVE:
            self.status = TradeStatus.TRIGGERED
            self.triggered_at = datetime.now()
    
    def mark_executed(self, executed_price: float, order_id: str):
        """Mark trade as executed with execution details"""
        self.status = TradeStatus.EXECUTED
        self.executed_at = datetime.now()
        self.executed_price = executed_price
        self.order_id = order_id
    
    def mark_cancelled(self):
        """Mark trade as cancelled"""
        self.status = TradeStatus.CANCELLED
        self.cancelled_at = datetime.now()
    
    def mark_failed(self, error_message: str):
        """Mark trade as failed with error message"""
        self.status = TradeStatus.FAILED
        self.error_message = error_message
    
    def get_preview_text(self, current_price: float) -> Dict[str, Any]:
        """Get trade preview information for UI display"""
        condition_text = f"BTC {self.condition_type.value} ${self.target_price:,.2f}"
        
        # Calculate price difference
        if self.condition_type == ConditionType.GREATER_THAN:
            price_diff = self.target_price - current_price
            status_text = "TRIGGERED" if current_price >= self.target_price else f"WAITING (Need ${price_diff:.2f} more)"
        else:  # LESS_THAN
            price_diff = current_price - self.target_price
            status_text = "TRIGGERED" if current_price <= self.target_price else f"WAITING (Need ${price_diff:.2f} drop)"
        
        # Execution preview
        execution_text = f"{self.order_side.value.upper()} {self.number_of_lots} lots"
        if self.order_type == OrderType.MARKET:
            execution_text += " at MARKET price"
            estimated_price = current_price
        else:
            execution_text += f" at LIMIT ${self.limit_price:,.2f}"
            estimated_price = self.limit_price
        
        return {
            "condition": condition_text,
            "status": status_text,
            "execution": execution_text,
            "estimated_price": estimated_price,
            "estimated_total": estimated_price * self.number_of_lots,
            "is_triggered": self.is_condition_met(current_price),
            "price_difference": abs(price_diff)
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert trade to dictionary for JSON serialization"""
        return {
            "trade_id": self.trade_id,
            "condition_type": self.condition_type.value,
            "target_price": self.target_price,
            "number_of_lots": self.number_of_lots,
            "order_side": self.order_side.value,
            "order_type": self.order_type.value,
            "limit_price": self.limit_price,
            "enable_stop_loss": self.enable_stop_loss,
            "stop_loss_price": self.stop_loss_price,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "triggered_at": self.triggered_at.isoformat() if self.triggered_at else None,
            "executed_at": self.executed_at.isoformat() if self.executed_at else None,
            "cancelled_at": self.cancelled_at.isoformat() if self.cancelled_at else None,
            "executed_price": self.executed_price,
            "order_id": self.order_id,
            "error_message": self.error_message
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ConditionalTrade':
        """Create trade from dictionary"""
        trade = cls(
            trade_id=data["trade_id"],
            condition_type=ConditionType(data["condition_type"]),
            target_price=data["target_price"],
            number_of_lots=data["number_of_lots"],
            order_side=OrderSide(data["order_side"]),
            order_type=OrderType(data["order_type"]),
            limit_price=data.get("limit_price"),
            enable_stop_loss=data.get("enable_stop_loss", False),
            stop_loss_price=data.get("stop_loss_price")
        )
        
        # Restore status and timestamps
        trade.status = TradeStatus(data["status"])
        trade.created_at = datetime.fromisoformat(data["created_at"])
        
        if data.get("triggered_at"):
            trade.triggered_at = datetime.fromisoformat(data["triggered_at"])
        if data.get("executed_at"):
            trade.executed_at = datetime.fromisoformat(data["executed_at"])
        if data.get("cancelled_at"):
            trade.cancelled_at = datetime.fromisoformat(data["cancelled_at"])
        
        trade.executed_price = data.get("executed_price")
        trade.order_id = data.get("order_id")
        trade.error_message = data.get("error_message")
        
        return trade