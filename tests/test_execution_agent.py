import pytest
from agents.execution_agent import ExecutionAgent
from services.alpaca_market_service import AlpacaMarketService
from models.trade import TradingDecision, TradeAction
from config.settings import settings
from datetime import datetime


@pytest.fixture
def market_service():
    """Create market service fixture."""
    return AlpacaMarketService(
        settings.alpaca_api_key,
        settings.alpaca_api_secret,
        paper=True
    )


@pytest.fixture
def execution_agent(market_service):
    """Create execution agent fixture."""
    return ExecutionAgent(market_service)


def test_execution_agent_initialization(execution_agent):
    """Execution agent should initialize correctly."""
    assert execution_agent.name == "execution_agent"
    assert execution_agent.market_service is not None


@pytest.mark.asyncio
async def test_execute_hold_action(execution_agent):
    """Should skip execution for HOLD actions."""
    decision = TradingDecision(
        symbol="AAPL",
        action=TradeAction.HOLD,
        quantity=0,
        expected_price=150.0,
        confidence="LOW",
        reasoning="Neutral sentiment",
        sentiment_score=0.0,
        high_impact_count=0,
        analysis_count=3
    )

    execution = await execution_agent.execute(decision)

    # HOLD should return None (no execution)
    assert execution is None


@pytest.mark.asyncio
async def test_execute_buy_order(execution_agent, market_service):
    """Should execute BUY order successfully."""
    # Get current position
    position_before = market_service.get_position("AAPL")
    qty_before = position_before["qty"] if position_before else 0

    decision = TradingDecision(
        symbol="AAPL",
        action=TradeAction.BUY,
        quantity=1,  # Buy just 1 share for testing
        expected_price=150.0,
        confidence="MEDIUM",
        reasoning="Test buy order",
        sentiment_score=0.5,
        high_impact_count=2,
        analysis_count=5
    )

    execution = await execution_agent.execute(decision, timeout=30)

    # Should have execution result
    assert execution is not None
    assert execution.symbol == "AAPL"
    assert execution.order_id is not None

    # Check if filled (may be rejected if market closed)
    if execution.status == "FILLED":
        assert execution.filled_qty == 1
        assert execution.filled_avg_price > 0
        assert execution.filled_at is not None

        # Verify position increased
        position_after = market_service.get_position("AAPL")
        assert position_after is not None
        assert position_after["qty"] == qty_before + 1


@pytest.mark.asyncio
async def test_execute_sell_order_with_position(execution_agent, market_service):
    """Should execute SELL order if we have a position."""
    # First check if we have a position
    position = market_service.get_position("AAPL")

    if not position or position["qty"] == 0:
        pytest.skip("No AAPL position to sell")

    qty_before = position["qty"]

    decision = TradingDecision(
        symbol="AAPL",
        action=TradeAction.SELL,
        quantity=1,  # Sell just 1 share
        expected_price=150.0,
        confidence="MEDIUM",
        reasoning="Test sell order",
        sentiment_score=-0.5,
        high_impact_count=2,
        analysis_count=5
    )

    execution = await execution_agent.execute(decision, timeout=30)

    assert execution is not None
    assert execution.symbol == "AAPL"
    assert execution.order_id is not None

    if execution.status == "FILLED":
        assert execution.filled_qty == 1
        assert execution.filled_avg_price > 0

        # Verify position decreased
        position_after = market_service.get_position("AAPL")
        if position_after:
            assert position_after["qty"] == qty_before - 1


def test_get_recent_executions(execution_agent):
    """Should retrieve recent executions."""
    executions = execution_agent.get_recent_executions("AAPL", limit=5)

    assert isinstance(executions, list)
    assert len(executions) <= 5

    # If we have executions, check structure
    for execution in executions:
        assert hasattr(execution, 'symbol')
        assert hasattr(execution, 'status')
        assert hasattr(execution, 'filled_qty')


@pytest.mark.asyncio
async def test_execution_handles_invalid_symbol(execution_agent):
    """Should handle execution failures gracefully."""
    decision = TradingDecision(
        symbol="INVALID123",  # Invalid symbol
        action=TradeAction.BUY,
        quantity=1,
        expected_price=150.0,
        confidence="MEDIUM",
        reasoning="Test invalid symbol",
        sentiment_score=0.5,
        high_impact_count=1,
        analysis_count=3
    )

    execution = await execution_agent.execute(decision, timeout=10)

    # Should return execution with error
    assert execution is not None
    assert execution.status == "REJECTED"
    assert execution.error_message is not None


@pytest.mark.asyncio
async def test_execution_timeout(execution_agent):
    """Should handle order timeout."""
    # Note: This test may be hard to trigger reliably
    # Just verify timeout parameter is respected
    decision = TradingDecision(
        symbol="AAPL",
        action=TradeAction.BUY,
        quantity=1,
        expected_price=150.0,
        confidence="MEDIUM",
        reasoning="Test timeout",
        sentiment_score=0.5,
        high_impact_count=1,
        analysis_count=3
    )

    # Use very short timeout (unlikely to fill in paper trading instantly)
    import time
    start = time.time()
    execution = await execution_agent.execute(decision, timeout=2)
    elapsed = time.time() - start

    # Should respect timeout (allow some buffer)
    assert elapsed < 5


def test_execution_agent_has_correct_attributes(execution_agent):
    """Execution agent should have all required attributes."""
    assert hasattr(execution_agent, 'market_service')
    assert hasattr(execution_agent, 'execute')
    assert hasattr(execution_agent, 'get_recent_executions')


@pytest.mark.asyncio
async def test_execution_stored_in_database(execution_agent):
    """Execution should be stored in database."""
    decision = TradingDecision(
        symbol="AAPL",
        action=TradeAction.BUY,
        quantity=1,
        expected_price=150.0,
        confidence="MEDIUM",
        reasoning="Test database storage",
        sentiment_score=0.5,
        high_impact_count=1,
        analysis_count=3
    )

    execution = await execution_agent.execute(decision, timeout=30)

    if execution:
        # Should be able to retrieve it
        recent = execution_agent.get_recent_executions("AAPL", limit=1)
        assert len(recent) >= 1

        # Most recent should match
        latest = recent[0]
        assert latest.symbol == "AAPL"


def test_get_recent_executions_returns_list(execution_agent):
    """Should always return a list."""
    executions = execution_agent.get_recent_executions("NONEXISTENT", limit=5)
    assert isinstance(executions, list)
    assert len(executions) == 0


def test_execution_agent_inherits_from_base(execution_agent):
    """Should inherit from BaseAgent."""
    from agents.base import BaseAgent
    assert isinstance(execution_agent, BaseAgent)
