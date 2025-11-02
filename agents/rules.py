"""Trading rules and thresholds for decision-making."""

from pydantic import BaseModel, Field


class TradingRules(BaseModel):
    """Trading rules and thresholds for decision-making."""

    # Sentiment thresholds
    buy_sentiment_threshold: float = Field(
        default=0.3,
        description="Buy if sentiment score > this value"
    )
    sell_sentiment_threshold: float = Field(
        default=-0.3,
        description="Sell if sentiment score < this value"
    )

    # Impact requirements
    min_impact_for_action: str = Field(
        default="MEDIUM",
        description="Require at least this impact level for actions"
    )

    # Position sizing
    max_position_size: int = Field(
        default=100,
        description="Maximum shares per position"
    )
    base_position_size: int = Field(
        default=10,
        description="Base size for weak signals"
    )

    # Position sizing multipliers
    high_impact_multiplier: float = Field(
        default=2.0,
        description="Multiply position size by this for HIGH impact"
    )
    medium_impact_multiplier: float = Field(
        default=1.5,
        description="Multiply position size by this for MEDIUM impact"
    )
    low_impact_multiplier: float = Field(
        default=1.0,
        description="Multiply position size by this for LOW impact"
    )

    # Confidence thresholds
    high_confidence_min_score: float = Field(
        default=0.6,
        description="Minimum score for HIGH confidence"
    )
    medium_confidence_min_score: float = Field(
        default=0.3,
        description="Minimum score for MEDIUM confidence"
    )

    # Risk management
    max_portfolio_pct_per_position: float = Field(
        default=0.25,
        description="Maximum portfolio percentage per position"
    )
    min_cash_reserve: float = Field(
        default=1000.0,
        description="Minimum cash to keep in reserve"
    )

    # Analysis requirements
    min_analyses_for_decision: int = Field(
        default=3,
        description="Minimum number of recent analyses required"
    )
    high_impact_count_threshold: int = Field(
        default=2,
        description="High impact articles needed for higher confidence"
    )


# Global rules instance
default_rules = TradingRules()


def get_trading_rules() -> TradingRules:
    """Get the current trading rules."""
    return default_rules
