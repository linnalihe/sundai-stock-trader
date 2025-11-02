from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional
from enum import Enum

class TradeAction(str, Enum):
    """Trading action type."""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"

class OrderStatus(str, Enum):
    """Order execution status."""
    PENDING = "PENDING"
    SUBMITTED = "SUBMITTED"
    FILLED = "FILLED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    REJECTED = "REJECTED"
    CANCELLED = "CANCELLED"

class TradingDecision(BaseModel):
    """Trading decision data model."""

    symbol: str = Field(..., description="Stock symbol")
    action: TradeAction = Field(..., description="Trade action")
    quantity: int = Field(..., ge=0, description="Number of shares")
    expected_price: float = Field(..., gt=0, description="Expected price at decision time")
    confidence: str = Field(..., description="Decision confidence (HIGH/MEDIUM/LOW)")
    reasoning: str = Field(..., description="Decision reasoning")
    sentiment_score: float = Field(..., description="Aggregate sentiment score")
    high_impact_count: int = Field(..., ge=0, description="Number of high impact articles")
    analysis_count: int = Field(..., ge=0, description="Total number of analyses")
    decided_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        use_enum_values = True

class TradeExecution(BaseModel):
    """Trade execution data model."""

    id: str = Field(..., description="Unique execution identifier")
    decision_id: str = Field(..., description="Related decision ID")
    order_id: Optional[str] = Field(None, description="Broker order ID")
    symbol: str = Field(..., description="Stock symbol")
    status: OrderStatus = Field(..., description="Order status")
    filled_qty: int = Field(default=0, description="Filled quantity")
    filled_avg_price: float = Field(default=0.0, description="Average fill price")
    submitted_at: Optional[datetime] = None
    filled_at: Optional[datetime] = None
    error_message: Optional[str] = None

    class Config:
        use_enum_values = True
