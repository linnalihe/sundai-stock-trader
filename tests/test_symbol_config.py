import pytest
from config.symbol_config import symbol_config

def test_load_symbols_config():
    """Should load symbols from YAML file."""
    enabled = symbol_config.get_enabled_symbols("stocks")
    assert "AAPL" in enabled

def test_get_symbol_info():
    """Should get info for specific symbol."""
    info = symbol_config.get_symbol_info("AAPL")
    assert info["name"] == "Apple Inc."
    assert info["enabled"] == True
    assert info["max_position_size"] == 100

def test_disabled_symbols_not_included():
    """Disabled symbols should not be in enabled list."""
    enabled = symbol_config.get_enabled_symbols("stocks")
    assert "GOOGL" not in enabled  # Disabled in config

def test_get_enabled_stocks():
    """Should return only enabled stocks."""
    enabled = symbol_config.get_enabled_symbols("stocks")
    assert isinstance(enabled, list)
    assert len(enabled) >= 1  # At least AAPL

def test_get_crypto_symbols():
    """Should be able to get crypto symbols."""
    crypto = symbol_config.get_enabled_symbols("crypto")
    assert isinstance(crypto, list)
    # BTCUSD is disabled, so should be empty
    assert len(crypto) == 0

def test_symbol_not_found():
    """Should return empty dict for non-existent symbol."""
    info = symbol_config.get_symbol_info("INVALID")
    assert info == {}
