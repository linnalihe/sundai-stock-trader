import pytest
from services.alpaca_market_service import AlpacaMarketService
from config.settings import settings


@pytest.fixture
def market_service():
    """Create market service fixture."""
    return AlpacaMarketService(
        settings.alpaca_api_key,
        settings.alpaca_api_secret,
        paper=True
    )


def test_market_service_init(market_service):
    """Market service should initialize."""
    assert market_service.paper == True
    assert market_service.trading_client is not None
    assert market_service.data_client is not None


def test_get_account(market_service):
    """Should get account information."""
    account = market_service.get_account()

    assert "buying_power" in account
    assert "cash" in account
    assert "portfolio_value" in account
    assert "equity" in account
    assert "status" in account
    assert account["buying_power"] >= 0
    assert account["cash"] >= 0


def test_get_current_price(market_service):
    """Should get current price for AAPL."""
    price = market_service.get_current_price("AAPL")

    assert price > 0
    assert price < 1000  # Sanity check for AAPL price


def test_get_position_may_not_exist(market_service):
    """Should return None for non-existent position or dict if exists."""
    # Use a symbol we likely don't own
    position = market_service.get_position("TSLA")

    # Either None or a dict with qty
    if position:
        assert "symbol" in position
        assert "qty" in position
        assert "avg_entry_price" in position
        assert position["qty"] >= 0
    else:
        assert position is None


def test_get_all_positions(market_service):
    """Should get all positions."""
    positions = market_service.get_all_positions()

    assert isinstance(positions, list)
    # May be empty if no positions
    for position in positions:
        assert "symbol" in position
        assert "qty" in position
        assert "market_value" in position


def test_calculate_max_affordable_qty(market_service):
    """Should calculate affordable quantity."""
    # Test with hypothetical buying power
    qty = market_service.calculate_max_affordable_qty(
        "AAPL",
        buying_power=10000,
        reserve=1000
    )

    assert qty >= 0
    assert qty < 1000  # Sanity check


def test_calculate_max_affordable_qty_insufficient_funds(market_service):
    """Should return 0 if insufficient funds."""
    qty = market_service.calculate_max_affordable_qty(
        "AAPL",
        buying_power=100,
        reserve=100  # All money reserved
    )

    assert qty == 0


def test_calculate_max_affordable_qty_respects_reserve(market_service):
    """Should respect reserve when calculating quantity."""
    buying_power = 10000
    reserve = 1000

    qty = market_service.calculate_max_affordable_qty(
        "AAPL",
        buying_power=buying_power,
        reserve=reserve
    )

    # Get price
    price = market_service.get_current_price("AAPL")

    # Total cost should not exceed buying_power - reserve
    total_cost = qty * price
    assert total_cost <= (buying_power - reserve + price)  # +price for rounding tolerance


def test_get_account_has_valid_status(market_service):
    """Account should have valid status."""
    account = market_service.get_account()

    assert account["status"] in ["ACTIVE", "ACCOUNT_CLOSED", "ACCOUNT_SUSPENDED"]


def test_paper_trading_mode(market_service):
    """Should be in paper trading mode."""
    assert market_service.paper == True


def test_get_position_for_aapl(market_service):
    """Should handle getting position for AAPL."""
    position = market_service.get_position("AAPL")

    # Either has position or doesn't
    if position:
        assert position["symbol"] == "AAPL"
        assert isinstance(position["qty"], int)
        assert isinstance(position["avg_entry_price"], float)
    else:
        assert position is None
