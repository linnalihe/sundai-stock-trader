import pytest
from agents.rules import TradingRules, get_trading_rules


def test_default_rules():
    """Should load default trading rules."""
    rules = get_trading_rules()

    assert rules.buy_sentiment_threshold == 0.3
    assert rules.sell_sentiment_threshold == -0.3
    assert rules.max_position_size == 100
    assert rules.base_position_size == 10


def test_custom_rules():
    """Should create custom rules."""
    rules = TradingRules(
        buy_sentiment_threshold=0.5,
        max_position_size=50
    )

    assert rules.buy_sentiment_threshold == 0.5
    assert rules.max_position_size == 50
    # Defaults still apply
    assert rules.base_position_size == 10


def test_position_size_multipliers():
    """Should have correct multipliers."""
    rules = get_trading_rules()

    assert rules.high_impact_multiplier == 2.0
    assert rules.medium_impact_multiplier == 1.5
    assert rules.low_impact_multiplier == 1.0


def test_confidence_thresholds():
    """Should have correct confidence thresholds."""
    rules = get_trading_rules()

    assert rules.high_confidence_min_score == 0.6
    assert rules.medium_confidence_min_score == 0.3


def test_risk_management_settings():
    """Should have risk management settings."""
    rules = get_trading_rules()

    assert rules.max_portfolio_pct_per_position == 0.25
    assert rules.min_cash_reserve == 1000.0


def test_analysis_requirements():
    """Should have analysis requirements."""
    rules = get_trading_rules()

    assert rules.min_analyses_for_decision == 3
    assert rules.high_impact_count_threshold == 2


def test_rules_are_immutable_after_creation():
    """Rules should be configured at creation."""
    rules1 = TradingRules(buy_sentiment_threshold=0.4)
    rules2 = TradingRules(buy_sentiment_threshold=0.5)

    # Each instance has its own values
    assert rules1.buy_sentiment_threshold == 0.4
    assert rules2.buy_sentiment_threshold == 0.5


def test_all_thresholds_are_valid():
    """All threshold values should be valid."""
    rules = get_trading_rules()

    # Sentiment thresholds should be between -1 and 1
    assert -1.0 <= rules.buy_sentiment_threshold <= 1.0
    assert -1.0 <= rules.sell_sentiment_threshold <= 1.0

    # Position sizes should be positive
    assert rules.max_position_size > 0
    assert rules.base_position_size > 0

    # Multipliers should be positive
    assert rules.high_impact_multiplier > 0
    assert rules.medium_impact_multiplier > 0
    assert rules.low_impact_multiplier > 0

    # Confidence thresholds should be valid
    assert 0 <= rules.high_confidence_min_score <= 1.0
    assert 0 <= rules.medium_confidence_min_score <= 1.0
