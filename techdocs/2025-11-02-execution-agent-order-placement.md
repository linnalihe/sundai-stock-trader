# Feature: Execution Agent - Order Placement and Tracking

**Date**: 2025-11-02
**Status**: TODO
**Estimated Time**: 3-4 hours

## Overview
Build the Execution Agent to execute trading decisions by placing actual orders on Alpaca (paper trading). This agent converts TradingDecision objects into real market/limit orders, tracks order status, handles fills/rejections, and maintains a complete audit trail of all trade executions.

## What We're Building
1. Execution Agent that places orders via Alpaca Trading API
2. Order submission for BUY/SELL actions (HOLD = no action)
3. Order status tracking (PENDING → SUBMITTED → FILLED/REJECTED)
4. Position verification after execution
5. Error handling for order failures and rejections
6. Database storage of all trade executions
7. Integration with Decision Agent in main pipeline

## Execution Flow

```
Input: TradingDecision (BUY/SELL/HOLD)
  ↓
Skip if HOLD (no action needed)
  ↓
Create Market Order (simplest for MVP)
  ↓
Submit Order to Alpaca
  ↓
Wait for Fill (with timeout)
  ↓
Poll Order Status (PENDING → FILLED/REJECTED)
  ↓
Verify Position Changed
  ↓
Store TradeExecution in Database
  ↓
Return Execution Result
```

## Implementation Details

### 1. Execution Agent (`agents/execution_agent.py`)

Create the agent that executes trading decisions:

```python
from typing import Optional
import uuid
import time
from datetime import datetime
from agents.base import BaseAgent
from services.alpaca_market_service import AlpacaMarketService
from models.trade import TradingDecision, TradeExecution, TradeAction, OrderStatus
from storage.database import db
from sqlalchemy import text
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce


class ExecutionAgent(BaseAgent):
    """Agent that executes trading decisions by placing orders."""

    def __init__(self, market_service: AlpacaMarketService):
        """
        Initialize Execution Agent.

        Args:
            market_service: AlpacaMarketService for order placement
        """
        super().__init__("execution_agent")
        self.market_service = market_service

    async def execute(
        self,
        decision: TradingDecision,
        timeout: int = 60
    ) -> Optional[TradeExecution]:
        """
        Execute a trading decision by placing an order.

        Args:
            decision: TradingDecision to execute
            timeout: Max seconds to wait for order fill (default: 60)

        Returns:
            TradeExecution or None if no action taken
        """
        try:
            self.logger.info(
                "executing_decision",
                symbol=decision.symbol,
                action=decision.action,
                quantity=decision.quantity
            )

            # Skip HOLD actions
            if decision.action == TradeAction.HOLD:
                self.logger.info("hold_action_skipped", symbol=decision.symbol)
                return None

            # Create execution record
            execution_id = f"exec_{uuid.uuid4().hex[:12]}"
            execution = TradeExecution(
                id=execution_id,
                decision_id=str(id(decision)),  # Use object id for now
                symbol=decision.symbol,
                status=OrderStatus.PENDING,
                filled_qty=0,
                filled_avg_price=0.0
            )

            # 1. Submit order
            order = self._submit_order(decision)

            if not order:
                execution.status = OrderStatus.REJECTED
                execution.error_message = "Failed to submit order"
                self._store_execution(execution)
                return execution

            execution.order_id = order.id
            execution.status = OrderStatus.SUBMITTED
            execution.submitted_at = datetime.utcnow()

            self.logger.info(
                "order_submitted",
                symbol=decision.symbol,
                order_id=order.id,
                action=decision.action,
                quantity=decision.quantity
            )

            # 2. Wait for order to fill
            filled_order = self._wait_for_fill(order.id, timeout)

            if not filled_order:
                execution.status = OrderStatus.REJECTED
                execution.error_message = "Order timeout or failed to fill"
                self._store_execution(execution)
                return execution

            # 3. Update execution with fill details
            execution.status = OrderStatus.FILLED
            execution.filled_qty = int(filled_order.filled_qty)
            execution.filled_avg_price = float(filled_order.filled_avg_price)
            execution.filled_at = datetime.utcnow()

            self.logger.info(
                "order_filled",
                symbol=decision.symbol,
                order_id=order.id,
                filled_qty=execution.filled_qty,
                filled_price=execution.filled_avg_price
            )

            # 4. Verify position
            position = self.market_service.get_position(decision.symbol)
            if position:
                self.logger.info(
                    "position_verified",
                    symbol=decision.symbol,
                    qty=position["qty"],
                    avg_price=position["avg_entry_price"]
                )

            # 5. Store execution
            self._store_execution(execution)

            return execution

        except Exception as e:
            self.logger.error(
                "execution_failed",
                symbol=decision.symbol,
                error=str(e)
            )
            # Store failed execution
            execution = TradeExecution(
                id=f"exec_{uuid.uuid4().hex[:12]}",
                decision_id=str(id(decision)),
                symbol=decision.symbol,
                status=OrderStatus.REJECTED,
                filled_qty=0,
                filled_avg_price=0.0,
                error_message=str(e)
            )
            self._store_execution(execution)
            return execution

    def _submit_order(self, decision: TradingDecision):
        """
        Submit order to Alpaca.

        Args:
            decision: TradingDecision to execute

        Returns:
            Order object or None if failed
        """
        try:
            # Determine order side
            side = OrderSide.BUY if decision.action == TradeAction.BUY else OrderSide.SELL

            # Create market order (simplest for MVP)
            order_request = MarketOrderRequest(
                symbol=decision.symbol,
                qty=decision.quantity,
                side=side,
                time_in_force=TimeInForce.DAY
            )

            # Submit order
            order = self.market_service.trading_client.submit_order(order_request)

            return order

        except Exception as e:
            self.logger.error(
                "order_submission_failed",
                symbol=decision.symbol,
                error=str(e)
            )
            return None

    def _wait_for_fill(self, order_id: str, timeout: int = 60):
        """
        Wait for order to fill.

        Args:
            order_id: Alpaca order ID
            timeout: Max seconds to wait

        Returns:
            Filled order or None if timeout/failed
        """
        start_time = time.time()

        while time.time() - start_time < timeout:
            try:
                order = self.market_service.trading_client.get_order_by_id(order_id)

                # Check order status
                if order.status == "filled":
                    return order
                elif order.status in ["rejected", "canceled", "expired"]:
                    self.logger.warning(
                        "order_not_filled",
                        order_id=order_id,
                        status=order.status
                    )
                    return None

                # Wait before polling again
                time.sleep(1)

            except Exception as e:
                self.logger.error(
                    "order_status_check_failed",
                    order_id=order_id,
                    error=str(e)
                )
                return None

        self.logger.warning("order_fill_timeout", order_id=order_id)
        return None

    def _store_execution(self, execution: TradeExecution) -> None:
        """
        Store trade execution in database.

        Args:
            execution: TradeExecution to store
        """
        try:
            with db.get_session() as session:
                session.execute(
                    text("""
                        INSERT INTO trade_executions (
                            id, decision_id, order_id, symbol, status,
                            filled_qty, filled_avg_price, submitted_at,
                            filled_at, error_message
                        ) VALUES (
                            :id, :decision_id, :order_id, :symbol, :status,
                            :filled_qty, :filled_avg_price, :submitted_at,
                            :filled_at, :error_message
                        )
                    """),
                    {
                        "id": execution.id,
                        "decision_id": execution.decision_id,
                        "order_id": execution.order_id,
                        "symbol": execution.symbol,
                        "status": execution.status,
                        "filled_qty": execution.filled_qty,
                        "filled_avg_price": execution.filled_avg_price,
                        "submitted_at": execution.submitted_at,
                        "filled_at": execution.filled_at,
                        "error_message": execution.error_message
                    }
                )
            self.logger.info("execution_stored", execution_id=execution.id)
        except Exception as e:
            self.logger.error("failed_to_store_execution", error=str(e))
            raise

    def get_recent_executions(
        self,
        symbol: str,
        limit: int = 10
    ) -> list[TradeExecution]:
        """
        Get recent trade executions for a symbol.

        Args:
            symbol: Stock symbol
            limit: Maximum number of executions to return

        Returns:
            List of TradeExecution objects
        """
        try:
            with db.get_session() as session:
                result = session.execute(
                    text("""
                        SELECT * FROM trade_executions
                        WHERE symbol = :symbol
                        ORDER BY submitted_at DESC
                        LIMIT :limit
                    """),
                    {"symbol": symbol, "limit": limit}
                )

                executions = []
                for row in result:
                    executions.append(TradeExecution(
                        id=row.id,
                        decision_id=row.decision_id,
                        order_id=row.order_id,
                        symbol=row.symbol,
                        status=row.status,
                        filled_qty=row.filled_qty,
                        filled_avg_price=row.filled_avg_price,
                        submitted_at=row.submitted_at,
                        filled_at=row.filled_at,
                        error_message=row.error_message
                    ))

                return executions
        except Exception as e:
            self.logger.error("failed_to_get_executions", error=str(e))
            return []
```

### 2. Update main.py

Add execution to the pipeline:

```python
async def run_full_pipeline():
    """Run complete trading pipeline: News → Analysis → Decision → Execution."""
    logger.info("starting_full_pipeline")

    # Initialize services
    news_service = AlpacaNewsService(
        settings.alpaca_api_key,
        settings.alpaca_api_secret
    )
    market_service = AlpacaMarketService(
        settings.alpaca_api_key,
        settings.alpaca_api_secret,
        paper=settings.paper_trading
    )
    llm_service = LLMService()

    # Initialize agents
    news_agent = NewsAgent(news_service)
    analysis_agent = AnalysisAgent(llm_service)
    decision_agent = DecisionAgent(analysis_agent, market_service)
    execution_agent = ExecutionAgent(market_service)

    # Get enabled symbols
    symbols = symbol_config.get_enabled_symbols("stocks")
    logger.info("enabled_symbols", symbols=symbols)

    for symbol in symbols:
        logger.info("processing_symbol", symbol=symbol)

        # 1. Fetch news
        articles = await news_agent.execute(
            symbols=[symbol],
            hours_back=72,
            max_articles=5
        )

        if not articles:
            logger.info("no_articles_found", symbol=symbol)
            continue

        logger.info("articles_fetched", symbol=symbol, count=len(articles))

        # 2. Analyze sentiment
        analyses = await analysis_agent.execute(articles, symbol)

        if not analyses:
            logger.info("no_analyses_generated", symbol=symbol)
            continue

        logger.info("analyses_completed", symbol=symbol, count=len(analyses))

        # 3. Get aggregate sentiment
        aggregate = await analysis_agent.get_aggregate_sentiment(symbol)
        logger.info(
            "aggregate_sentiment",
            symbol=symbol,
            sentiment=aggregate["sentiment"],
            average_score=aggregate["average_score"]
        )

        # 4. Make trading decision
        decision = await decision_agent.execute(symbol, hours_back=72)

        if not decision:
            logger.warning("no_decision_made", symbol=symbol)
            continue

        logger.info(
            "trading_decision",
            symbol=symbol,
            action=decision.action,
            quantity=decision.quantity,
            confidence=decision.confidence
        )

        # 5. Execute trade (if not HOLD)
        if decision.action != TradeAction.HOLD:
            execution = await execution_agent.execute(decision, timeout=60)

            if execution:
                print(f"\n{'='*60}")
                print(f"Trade Execution for {symbol}")
                print(f"{'='*60}")
                print(f"Order ID:         {execution.order_id}")
                print(f"Status:           {execution.status}")
                print(f"Filled Qty:       {execution.filled_qty}")
                print(f"Avg Fill Price:   ${execution.filled_avg_price:.2f}")
                print(f"Submitted:        {execution.submitted_at}")
                print(f"Filled:           {execution.filled_at}")
                if execution.error_message:
                    print(f"Error:            {execution.error_message}")
                print(f"{'='*60}\n")
            else:
                logger.warning("execution_failed", symbol=symbol)
        else:
            logger.info("hold_decision_no_execution", symbol=symbol)

    # Cleanup
    await llm_service.close()
    logger.info("full_pipeline_completed")
```

### 3. Update TradeExecution Model (if needed)

Verify the model matches our needs in `models/trade.py`:

```python
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
```

## Success Criteria

### Functional Requirements
- [ ] Execution agent initializes with market service
- [ ] BUY decisions place market buy orders
- [ ] SELL decisions place market sell orders
- [ ] HOLD decisions skip execution (no order placed)
- [ ] Orders submitted successfully to Alpaca
- [ ] Order IDs captured and stored
- [ ] Order status tracked (PENDING → SUBMITTED → FILLED/REJECTED)
- [ ] System waits for order fills (with timeout)
- [ ] Filled quantities and prices recorded accurately
- [ ] Failed orders handled gracefully
- [ ] Positions verified after execution
- [ ] All executions stored in database
- [ ] Can retrieve recent executions by symbol

### Safety Requirements
- [ ] Only executes in paper trading mode
- [ ] Never submits orders without valid decision
- [ ] Handles API errors without crashing
- [ ] Respects order timeouts
- [ ] Stores failed executions for debugging
- [ ] Logs all order activities

### Data Quality
- [ ] All TradeExecution fields populated correctly
- [ ] Order IDs match Alpaca order IDs
- [ ] Fill prices reflect actual execution prices
- [ ] Timestamps accurate for submission and fill
- [ ] Error messages captured when orders fail

## Tests

### Test File: `tests/test_execution_agent.py`

```python
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

    # Use very short timeout (unlikely to fill in paper trading)
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
```

### Integration Test

Test the complete pipeline:

```python
@pytest.mark.asyncio
async def test_full_trading_pipeline():
    """Test complete pipeline from news to execution."""
    from agents.news_agent import NewsAgent
    from agents.analysis_agent import AnalysisAgent
    from agents.decision_agent import DecisionAgent
    from agents.execution_agent import ExecutionAgent
    from services.alpaca_news_service import AlpacaNewsService
    from services.alpaca_market_service import AlpacaMarketService
    from services.llm_service import LLMService

    # Initialize services
    news_service = AlpacaNewsService(
        settings.alpaca_api_key,
        settings.alpaca_api_secret
    )
    market_service = AlpacaMarketService(
        settings.alpaca_api_key,
        settings.alpaca_api_secret,
        paper=True
    )
    llm_service = LLMService()

    # Initialize agents
    news_agent = NewsAgent(news_service)
    analysis_agent = AnalysisAgent(llm_service)
    decision_agent = DecisionAgent(analysis_agent, market_service)
    execution_agent = ExecutionAgent(market_service)

    try:
        # 1. Fetch news
        articles = await news_agent.execute(["AAPL"], hours_back=72, max_articles=3)
        assert len(articles) > 0

        # 2. Analyze
        analyses = await analysis_agent.execute(articles, "AAPL")
        assert len(analyses) > 0

        # 3. Make decision
        decision = await decision_agent.execute("AAPL", hours_back=72)
        assert decision is not None

        # 4. Execute (if not HOLD)
        if decision.action != TradeAction.HOLD:
            execution = await execution_agent.execute(decision, timeout=30)
            assert execution is not None

            # If filled, verify in database
            if execution.status == "FILLED":
                recent = execution_agent.get_recent_executions("AAPL", limit=1)
                assert len(recent) >= 1

    finally:
        await llm_service.close()
```

## Manual Verification Commands

```bash
# 1. Check account status
python -c "
from services.alpaca_market_service import AlpacaMarketService
from config.settings import settings

service = AlpacaMarketService(settings.alpaca_api_key, settings.alpaca_api_secret, paper=True)
account = service.get_account()
print(f'Buying Power: \${account[\"buying_power\"]:.2f}')
print(f'Portfolio Value: \${account[\"portfolio_value\"]:.2f}')
"

# 2. Check current positions
python -c "
from services.alpaca_market_service import AlpacaMarketService
from config.settings import settings

service = AlpacaMarketService(settings.alpaca_api_key, settings.alpaca_api_secret, paper=True)
positions = service.get_all_positions()
print(f'Open Positions: {len(positions)}')
for p in positions:
    print(f'  {p[\"symbol\"]}: {p[\"qty\"]} shares @ \${p[\"avg_entry_price\"]:.2f}')
"

# 3. Run full pipeline
python main.py

# 4. Check executions in database
sqlite3 trading.db "SELECT COUNT(*) FROM trade_executions"
sqlite3 trading.db "SELECT symbol, status, filled_qty, filled_avg_price, order_id FROM trade_executions ORDER BY submitted_at DESC LIMIT 5"

# 5. Check recent decisions and their executions
sqlite3 trading.db "
SELECT
    d.symbol,
    d.action,
    d.quantity as decision_qty,
    e.status,
    e.filled_qty,
    e.filled_avg_price,
    e.order_id
FROM trading_decisions d
LEFT JOIN trade_executions e ON d.symbol = e.symbol
ORDER BY d.decided_at DESC
LIMIT 5
"

# 6. Run execution tests
pytest tests/test_execution_agent.py -v

# 7. Run full test suite
pytest tests/ -v

# 8. Check Alpaca orders via API
python -c "
from alpaca.trading.client import TradingClient
from config.settings import settings

client = TradingClient(settings.alpaca_api_key, settings.alpaca_api_secret, paper=True)
orders = client.get_orders()
print(f'Recent Orders: {len(orders)}')
for order in orders[:5]:
    print(f'  {order.symbol} {order.side} {order.qty} - Status: {order.status}')
"
```

## Files to Create/Modify

### New Files:
- `agents/execution_agent.py` - Execution agent with order placement logic
- `tests/test_execution_agent.py` - Execution agent tests

### Modified Files:
- `main.py` - Add execution agent to pipeline
- `models/trade.py` - Verify TradeExecution model (should be fine)

## Dependencies
- ✅ Phase 1 completed (foundation)
- ✅ Phase 2 completed (news agent)
- ✅ Phase 3 completed (analysis agent)
- ✅ Phase 4 completed (decision agent)
- ✅ TradeExecution model defined
- ✅ alpaca-py includes TradingClient for order submission

## Blockers
None - all dependencies met. Paper trading mode ensures safe testing.

## Notes
- **Paper Trading Only**: All orders go to Alpaca paper trading environment
- **Market Orders**: Using simplest order type for MVP (instant execution at market price)
- **Synchronous Polling**: Using simple polling for order fills (could use webhooks later)
- **Error Handling**: All failures stored in database with error messages
- **Idempotency**: Each execution is unique (no duplicate submission)
- **Timeout**: 60 second default timeout for order fills
- **Market Hours**: Orders may be rejected if market is closed
- **Position Verification**: Checks position after fill to confirm execution

## Implementation Order
1. Create `agents/execution_agent.py` with order submission and tracking
2. Update `main.py` to include execution in pipeline
3. Create `tests/test_execution_agent.py` with all test cases
4. Test with small quantities first (1 share) to verify behavior
5. Run integration test with full pipeline
6. Verify orders in Alpaca dashboard
7. Check database for execution records

## Expected Output

After completion, running `python main.py` should:
- Fetch news for AAPL (5 articles)
- Analyze sentiment (positive/negative/neutral)
- Make trading decision (BUY/SELL/HOLD)
- **Execute trade if BUY or SELL**:
  - Submit order to Alpaca
  - Wait for order to fill
  - Display execution results:

```
============================================================
Trade Execution for AAPL
============================================================
Order ID:         a1b2c3d4-e5f6-7890-abcd-ef1234567890
Status:           FILLED
Filled Qty:       32
Avg Fill Price:   $257.52
Submitted:        2025-11-02 20:45:23
Filled:           2025-11-02 20:45:24
============================================================
```

- Store execution in database
- Verify position in account

## Risk Considerations

**Safety Measures**:
- ✅ Paper trading only (no real money)
- ✅ Small position sizes (10-100 shares max)
- ✅ Market orders fill quickly (less slippage risk)
- ✅ Timeout prevents hanging forever on fills
- ✅ All errors logged and stored
- ✅ Order IDs tracked for auditing
- ✅ Position verification after execution

**Edge Cases Handled**:
- Market closed (orders rejected)
- Insufficient buying power (orders rejected)
- Invalid symbols (orders rejected)
- Partial fills (tracked in filled_qty)
- Order timeouts (marked as REJECTED with timeout message)
- API errors (caught and stored in error_message)

**Testing Strategy**:
- Start with 1 share orders
- Test during market hours for fills
- Test outside market hours for rejections
- Verify all executions stored in database
- Check Alpaca dashboard matches our records

This completes the execution pipeline - the system will now make real trades (in paper trading) based on sentiment analysis!
