import pytest
from config.settings import settings

def test_settings_load_from_env():
    """Settings should load from .env file."""
    assert settings.alpaca_api_key is not None
    assert settings.alpaca_api_secret is not None

def test_settings_paper_trading_default():
    """Paper trading should default to True."""
    assert settings.paper_trading == True

def test_settings_defaults():
    """Settings should have correct defaults."""
    assert settings.max_position_size == 100
    assert settings.risk_percentage == 0.02
    assert settings.max_daily_trades == 5
    assert settings.decision_confidence_threshold == 0.7

def test_settings_llm_config():
    """LLM configuration should have defaults."""
    assert settings.llm_provider in ["openai", "anthropic"]
    assert settings.llm_temperature == 0.3
