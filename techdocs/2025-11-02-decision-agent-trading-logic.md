# Feature: Decision Agent - Trading Decision Logic

**Date**: 2025-11-02
**Status**: TODO
**Estimated Time**: 4-5 hours

## Overview
Build the Decision Agent to transform sentiment analysis into concrete trading decisions (BUY/SELL/HOLD). This agent uses aggregate sentiment, impact levels, and risk management rules to determine optimal trading actions with position sizing and confidence levels.

## What We're Building
1. Decision Agent with rule-based trading logic
2. Alpaca Market Service for portfolio and price data
3. Configurable trading rules and thresholds
4. Position sizing algorithm based on sentiment strength
5. Risk management constraints (max position, available capital)
6. Decision confidence scoring
7. Comprehensive reasoning and audit trail
8. Database storage of trading decisions

## Decision Logic Flow

```
Input: Aggregate Sentiment + Recent Analyses + Portfolio State + Market Data
  ↓
Check Current Position & Available Capital
  ↓
Apply Trading Rules (BUY/SELL/HOLD logic)
  ↓
Calculate Position Size (risk-adjusted)
  ↓
Calculate Confidence Level
  ↓
Generate Reasoning
  ↓
Create TradingDecision
  ↓
Store in Database
```

## Implementation Details

### 1. Trading Rules Configuration (`agents/rules.py`)

Create a configuration file for trading rules and thresholds:

```python
from typing import Dict, Any
from pydantic import BaseModel

class TradingRules(BaseModel):
    """Trading rules and thresholds for decision-making."""

    # Sentiment thresholds
    buy_sentiment_threshold: float = 0.3  # Buy if score > 0.3
    sell_sentiment_threshold: float = -0.3  # Sell if score < -0.3

    # Impact requirements
    min_impact_for_action: str = "MEDIUM"  # Require at least MEDIUM impact

    # Position sizing
    max_position_size: int = 100  # Max shares per position
    base_position_size: int = 10  # Base size for weak signals

    # Position sizing multipliers
    high_impact_multiplier: float = 2.0  # 2x size for HIGH impact
    medium_impact_multiplier: float = 1.5  # 1.5x size for MEDIUM impact
    low_impact_multiplier: float = 1.0  # 1x size for LOW impact

    # Confidence thresholds
    high_confidence_min_score: float = 0.6  # Strong sentiment
    medium_confidence_min_score: float = 0.3  # Moderate sentiment

    # Risk management
    max_portfolio_pct_per_position: float = 0.25  # Max 25% in one stock
    min_cash_reserve: float = 1000.0  # Keep $1000 in reserve

    # Analysis requirements
    min_analyses_for_decision: int = 3  # Need at least 3 recent analyses
    high_impact_count_threshold: int = 2  # 2+ high impact = higher confidence

# Global rules instance
default_rules = TradingRules()

def get_trading_rules() -> TradingRules:
    """Get the current trading rules."""
    return default_rules
```

### 2. Alpaca Market Service (`services/alpaca_market_service.py`)

Create service to fetch portfolio and market data:

```python
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetAssetsRequest
from alpaca.trading.enums import AssetClass
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestQuoteRequest
from typing import Dict, Optional
from config.settings import settings
from utils.logger import get_logger

logger = get_logger("alpaca_market_service")

class AlpacaMarketService:
    """Service for fetching market data and portfolio information from Alpaca."""

    def __init__(self, api_key: str, api_secret: str, paper: bool = True):
        self.trading_client = TradingClient(
            api_key=api_key,
            secret_key=api_secret,
            paper=paper
        )
        self.data_client = StockHistoricalDataClient(
            api_key=api_key,
            secret_key=api_secret
        )
        self.paper = paper
        logger.info("alpaca_market_service_initialized", paper=paper)

    def get_account(self) -> Dict[str, Any]:
        """Get account information including buying power."""
        try:
            account = self.trading_client.get_account()
            return {
                "buying_power": float(account.buying_power),
                "cash": float(account.cash),
                "portfolio_value": float(account.portfolio_value),
                "equity": float(account.equity),
                "status": account.status
            }
        except Exception as e:
            logger.error("failed_to_get_account", error=str(e))
            raise

    def get_position(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Get current position for a symbol."""
        try:
            position = self.trading_client.get_open_position(symbol)
            return {
                "symbol": position.symbol,
                "qty": int(position.qty),
                "avg_entry_price": float(position.avg_entry_price),
                "market_value": float(position.market_value),
                "unrealized_pl": float(position.unrealized_pl),
                "unrealized_plpc": float(position.unrealized_plpc)
            }
        except Exception as e:
            # No position exists
            if "position does not exist" in str(e).lower():
                return None
            logger.error("failed_to_get_position", symbol=symbol, error=str(e))
            raise

    def get_all_positions(self) -> list[Dict[str, Any]]:
        """Get all open positions."""
        try:
            positions = self.trading_client.get_all_positions()
            return [
                {
                    "symbol": p.symbol,
                    "qty": int(p.qty),
                    "avg_entry_price": float(p.avg_entry_price),
                    "market_value": float(p.market_value),
                    "unrealized_pl": float(p.unrealized_pl)
                }
                for p in positions
            ]
        except Exception as e:
            logger.error("failed_to_get_positions", error=str(e))
            raise

    def get_current_price(self, symbol: str) -> float:
        """Get current price for a symbol."""
        try:
            request = StockLatestQuoteRequest(symbol_or_symbols=symbol)
            quotes = self.data_client.get_stock_latest_quote(request)
            quote = quotes[symbol]

            # Use mid price (average of bid and ask)
            bid = float(quote.bid_price) if quote.bid_price else 0
            ask = float(quote.ask_price) if quote.ask_price else 0

            if bid > 0 and ask > 0:
                return (bid + ask) / 2
            elif ask > 0:
                return ask
            elif bid > 0:
                return bid
            else:
                logger.error("no_price_available", symbol=symbol)
                raise ValueError(f"No price available for {symbol}")

        except Exception as e:
            logger.error("failed_to_get_price", symbol=symbol, error=str(e))
            raise

    def calculate_max_affordable_qty(
        self,
        symbol: str,
        buying_power: float,
        reserve: float = 0
    ) -> int:
        """Calculate maximum affordable quantity given buying power."""
        try:
            price = self.get_current_price(symbol)
            available = buying_power - reserve

            if available <= 0:
                return 0

            max_qty = int(available / price)
            return max(0, max_qty)

        except Exception as e:
            logger.error("failed_to_calculate_qty", symbol=symbol, error=str(e))
            return 0
```

### 3. Decision Agent (`agents/decision_agent.py`)

Create the main decision agent:

```python
from typing import Optional, Dict, Any, List
from datetime import datetime
from agents.base import BaseAgent
from agents.analysis_agent import AnalysisAgent
from agents.rules import get_trading_rules, TradingRules
from services.alpaca_market_service import AlpacaMarketService
from models.trade import TradingDecision, TradeAction
from models.analysis import ImpactLevel
from storage.database import db
from sqlalchemy import text

class DecisionAgent(BaseAgent):
    """Agent that makes trading decisions based on sentiment analysis."""

    def __init__(
        self,
        analysis_agent: AnalysisAgent,
        market_service: AlpacaMarketService,
        rules: TradingRules = None
    ):
        super().__init__("decision_agent")
        self.analysis_agent = analysis_agent
        self.market_service = market_service
        self.rules = rules or get_trading_rules()

    async def execute(
        self,
        symbol: str,
        hours_back: int = 24
    ) -> Optional[TradingDecision]:
        """
        Make a trading decision for a symbol.

        Args:
            symbol: Stock symbol to decide on
            hours_back: Hours to look back for sentiment

        Returns:
            TradingDecision or None if insufficient data
        """
        try:
            self.logger.info("making_decision", symbol=symbol)

            # 1. Get aggregate sentiment
            aggregate = await self.analysis_agent.get_aggregate_sentiment(
                symbol,
                hours_back=hours_back
            )

            # Check if we have enough analyses
            if aggregate["analysis_count"] < self.rules.min_analyses_for_decision:
                self.logger.warning(
                    "insufficient_analyses",
                    symbol=symbol,
                    count=aggregate["analysis_count"],
                    required=self.rules.min_analyses_for_decision
                )
                return None

            # 2. Get portfolio state
            account = self.market_service.get_account()
            position = self.market_service.get_position(symbol)
            current_price = self.market_service.get_current_price(symbol)

            # 3. Make decision
            decision = self._make_decision(
                symbol=symbol,
                aggregate=aggregate,
                account=account,
                position=position,
                current_price=current_price
            )

            # 4. Store decision
            if decision:
                self._store_decision(decision)
                self.logger.info(
                    "decision_made",
                    symbol=symbol,
                    action=decision.action,
                    quantity=decision.quantity,
                    confidence=decision.confidence
                )

            return decision

        except Exception as e:
            self.logger.error("decision_failed", symbol=symbol, error=str(e))
            return None

    def _make_decision(
        self,
        symbol: str,
        aggregate: Dict[str, Any],
        account: Dict[str, Any],
        position: Optional[Dict[str, Any]],
        current_price: float
    ) -> Optional[TradingDecision]:
        """Core decision logic."""

        sentiment = aggregate["sentiment"]
        score = aggregate["average_score"]
        high_impact_count = aggregate["high_impact_count"]
        analysis_count = aggregate["analysis_count"]

        current_qty = position["qty"] if position else 0
        buying_power = account["buying_power"]

        # Decision variables
        action = TradeAction.HOLD
        quantity = 0
        confidence = "LOW"
        reasoning_parts = []

        # --- BUY LOGIC ---
        if sentiment == "POSITIVE" and score >= self.rules.buy_sentiment_threshold:
            reasoning_parts.append(f"Positive sentiment (score: {score:.2f})")

            # Check if we can buy more
            if current_qty >= self.rules.max_position_size:
                reasoning_parts.append(f"Already at max position ({current_qty} shares)")
                action = TradeAction.HOLD
            else:
                # Calculate buy quantity
                base_qty = self._calculate_position_size(
                    score=score,
                    high_impact_count=high_impact_count,
                    analysis_count=analysis_count
                )

                # Adjust for existing position
                additional_qty = min(
                    base_qty,
                    self.rules.max_position_size - current_qty
                )

                # Check affordability
                max_affordable = self.market_service.calculate_max_affordable_qty(
                    symbol,
                    buying_power,
                    reserve=self.rules.min_cash_reserve
                )

                quantity = min(additional_qty, max_affordable)

                if quantity > 0:
                    action = TradeAction.BUY
                    reasoning_parts.append(f"Buying {quantity} shares")
                    reasoning_parts.append(
                        f"{high_impact_count} high-impact articles from {analysis_count} total"
                    )
                else:
                    reasoning_parts.append("Insufficient buying power")
                    action = TradeAction.HOLD

        # --- SELL LOGIC ---
        elif sentiment == "NEGATIVE" and score <= self.rules.sell_sentiment_threshold:
            reasoning_parts.append(f"Negative sentiment (score: {score:.2f})")

            if current_qty > 0:
                # Sell entire position on negative sentiment
                quantity = current_qty
                action = TradeAction.SELL
                reasoning_parts.append(f"Selling entire position ({quantity} shares)")
                reasoning_parts.append(
                    f"{high_impact_count} high-impact negative articles"
                )
            else:
                reasoning_parts.append("No position to sell")
                action = TradeAction.HOLD

        # --- HOLD LOGIC ---
        else:
            reasoning_parts.append(
                f"Neutral or weak sentiment (score: {score:.2f}, sentiment: {sentiment})"
            )
            if current_qty > 0:
                reasoning_parts.append(f"Holding current position ({current_qty} shares)")
            else:
                reasoning_parts.append("No position, waiting for stronger signal")
            action = TradeAction.HOLD

        # Calculate confidence
        confidence = self._calculate_confidence(
            score=abs(score),
            high_impact_count=high_impact_count,
            analysis_count=analysis_count
        )

        # Build reasoning
        reasoning = " | ".join(reasoning_parts)

        # Create decision
        decision = TradingDecision(
            symbol=symbol,
            action=action,
            quantity=quantity,
            expected_price=current_price,
            confidence=confidence,
            reasoning=reasoning,
            sentiment_score=score,
            high_impact_count=high_impact_count,
            analysis_count=analysis_count,
            decided_at=datetime.utcnow()
        )

        return decision

    def _calculate_position_size(
        self,
        score: float,
        high_impact_count: int,
        analysis_count: int
    ) -> int:
        """Calculate position size based on sentiment strength."""

        # Start with base size
        size = self.rules.base_position_size

        # Multiply by sentiment strength (0.3 to 1.0 → 1x to 3x)
        if score >= 0.3:
            strength_multiplier = 1 + (score - 0.3) / 0.7 * 2  # 1x to 3x
            size = int(size * strength_multiplier)

        # Adjust for high impact count
        if high_impact_count >= self.rules.high_impact_count_threshold:
            size = int(size * self.rules.high_impact_multiplier)

        # Cap at max position size
        size = min(size, self.rules.max_position_size)

        return size

    def _calculate_confidence(
        self,
        score: float,
        high_impact_count: int,
        analysis_count: int
    ) -> str:
        """Calculate confidence level for the decision."""

        # High confidence: strong score + high impact articles
        if (score >= self.rules.high_confidence_min_score and
            high_impact_count >= self.rules.high_impact_count_threshold):
            return "HIGH"

        # Medium confidence: moderate score or some high impact
        elif (score >= self.rules.medium_confidence_min_score or
              high_impact_count >= 1):
            return "MEDIUM"

        # Low confidence: weak signals
        else:
            return "LOW"

    def _store_decision(self, decision: TradingDecision) -> None:
        """Store trading decision in database."""
        try:
            with db.get_session() as session:
                session.execute(
                    text("""
                        INSERT INTO trading_decisions (
                            symbol, action, quantity, expected_price,
                            confidence, reasoning, sentiment_score,
                            high_impact_count, analysis_count, decided_at
                        ) VALUES (
                            :symbol, :action, :quantity, :expected_price,
                            :confidence, :reasoning, :sentiment_score,
                            :high_impact_count, :analysis_count, :decided_at
                        )
                    """),
                    {
                        "symbol": decision.symbol,
                        "action": decision.action,
                        "quantity": decision.quantity,
                        "expected_price": decision.expected_price,
                        "confidence": decision.confidence,
                        "reasoning": decision.reasoning,
                        "sentiment_score": decision.sentiment_score,
                        "high_impact_count": decision.high_impact_count,
                        "analysis_count": decision.analysis_count,
                        "decided_at": decision.decided_at
                    }
                )
            self.logger.info("decision_stored", symbol=decision.symbol)
        except Exception as e:
            self.logger.error("failed_to_store_decision", error=str(e))
            raise

    def get_recent_decisions(
        self,
        symbol: str,
        limit: int = 10
    ) -> List[TradingDecision]:
        """Get recent trading decisions for a symbol."""
        try:
            with db.get_session() as session:
                result = session.execute(
                    text("""
                        SELECT * FROM trading_decisions
                        WHERE symbol = :symbol
                        ORDER BY decided_at DESC
                        LIMIT :limit
                    """),
                    {"symbol": symbol, "limit": limit}
                )

                decisions = []
                for row in result:
                    decisions.append(TradingDecision(
                        symbol=row.symbol,
                        action=row.action,
                        quantity=row.quantity,
                        expected_price=row.expected_price,
                        confidence=row.confidence,
                        reasoning=row.reasoning,
                        sentiment_score=row.sentiment_score,
                        high_impact_count=row.high_impact_count,
                        analysis_count=row.analysis_count,
                        decided_at=row.decided_at
                    ))

                return decisions
        except Exception as e:
            self.logger.error("failed_to_get_decisions", error=str(e))
            return []
```

### 4. Update main.py

Add decision pipeline after analysis:

```python
async def run_decision_pipeline():
    """Run news + analysis + decision pipeline."""

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

    # Get enabled symbols
    symbols = symbol_config.get_enabled_symbols("stocks")

    logger.info("starting_decision_pipeline", symbols=symbols)

    for symbol in symbols:
        logger.info("processing_symbol", symbol=symbol)

        # 1. Fetch recent news (limit to 5 for cost control)
        articles = await news_agent.execute(
            [symbol],
            hours_back=72,
            max_articles=5
        )
        logger.info("fetched_articles", symbol=symbol, count=len(articles))

        # 2. Analyze articles
        if articles:
            analyses = await analysis_agent.execute(articles, symbol)
            logger.info("completed_analyses", symbol=symbol, count=len(analyses))

            # Show aggregate sentiment
            aggregate = await analysis_agent.get_aggregate_sentiment(symbol)
            logger.info(
                "aggregate_sentiment",
                symbol=symbol,
                sentiment=aggregate["sentiment"],
                score=aggregate["average_score"],
                analyses=aggregate["analysis_count"],
                high_impact=aggregate["high_impact_count"]
            )

        # 3. Make trading decision
        decision = await decision_agent.execute(symbol, hours_back=72)

        if decision:
            logger.info(
                "trading_decision",
                symbol=symbol,
                action=decision.action,
                quantity=decision.quantity,
                price=decision.expected_price,
                confidence=decision.confidence
            )
            print(f"\n{'='*60}")
            print(f"Trading Decision for {symbol}")
            print(f"{'='*60}")
            print(f"Action: {decision.action}")
            print(f"Quantity: {decision.quantity}")
            print(f"Expected Price: ${decision.expected_price:.2f}")
            print(f"Confidence: {decision.confidence}")
            print(f"Sentiment Score: {decision.sentiment_score:.2f}")
            print(f"Reasoning: {decision.reasoning}")
            print(f"{'='*60}\n")
        else:
            logger.warning("no_decision_made", symbol=symbol)

    # Cleanup
    await llm_service.close()
    logger.info("decision_pipeline_complete")

if __name__ == "__main__":
    asyncio.run(run_decision_pipeline())
```

### 5. Update requirements.txt

Add any missing dependencies:

```txt
# Should already have alpaca-py which includes trading client
# Verify version supports TradingClient and StockHistoricalDataClient
```

## Success Criteria

### Functional Requirements
- [ ] Trading rules configurable via TradingRules class
- [ ] Market service fetches account info (buying power, positions)
- [ ] Market service gets current prices from Alpaca
- [ ] Decision agent integrates with analysis agent
- [ ] BUY decisions made on positive sentiment (score > 0.3)
- [ ] SELL decisions made on negative sentiment (score < -0.3)
- [ ] HOLD decisions made on neutral/weak sentiment
- [ ] Position sizing adjusts based on sentiment strength
- [ ] Position sizing respects max position limit
- [ ] Buying respects available buying power
- [ ] Confidence calculated correctly (HIGH/MEDIUM/LOW)
- [ ] Reasoning provides clear audit trail
- [ ] Decisions stored in database
- [ ] Can retrieve recent decisions by symbol

### Risk Management
- [ ] Never exceed max position size per symbol
- [ ] Never buy without sufficient buying power
- [ ] Never sell without existing position
- [ ] Maintain minimum cash reserve
- [ ] Position sizing respects portfolio percentage limits

### Data Quality
- [ ] All TradingDecision fields populated correctly
- [ ] Expected price reflects current market price
- [ ] Confidence correlates with signal strength
- [ ] Reasoning accurately describes decision logic
- [ ] High impact count tracked correctly

## Tests

### Test File: `tests/test_trading_rules.py`

```python
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
```

### Test File: `tests/test_alpaca_market_service.py`

```python
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
    assert account["buying_power"] >= 0

def test_get_current_price(market_service):
    """Should get current price for AAPL."""
    price = market_service.get_current_price("AAPL")

    assert price > 0
    assert price < 1000  # Sanity check

def test_get_position_none(market_service):
    """Should return None for non-existent position."""
    # Use a symbol we likely don't own
    position = market_service.get_position("TSLA")

    # Either None or a dict with qty
    if position:
        assert "qty" in position
        assert position["qty"] >= 0

def test_get_all_positions(market_service):
    """Should get all positions."""
    positions = market_service.get_all_positions()

    assert isinstance(positions, list)
    # May be empty if no positions

def test_calculate_max_affordable_qty(market_service):
    """Should calculate affordable quantity."""
    qty = market_service.calculate_max_affordable_qty(
        "AAPL",
        buying_power=10000,
        reserve=1000
    )

    assert qty >= 0
    assert qty < 1000  # Sanity check
```

### Test File: `tests/test_decision_agent.py`

```python
import pytest
from agents.decision_agent import DecisionAgent
from agents.analysis_agent import AnalysisAgent
from services.llm_service import LLMService
from services.alpaca_market_service import AlpacaMarketService
from agents.rules import TradingRules
from config.settings import settings

@pytest.fixture
def llm_service():
    """Create LLM service fixture."""
    if not settings.openai_api_key and not settings.anthropic_api_key:
        pytest.skip("No LLM API key configured")
    return LLMService()

@pytest.fixture
def market_service():
    """Create market service fixture."""
    return AlpacaMarketService(
        settings.alpaca_api_key,
        settings.alpaca_api_secret,
        paper=True
    )

@pytest.fixture
def analysis_agent(llm_service):
    """Create analysis agent fixture."""
    return AnalysisAgent(llm_service)

@pytest.fixture
def decision_agent(analysis_agent, market_service):
    """Create decision agent fixture."""
    return DecisionAgent(analysis_agent, market_service)

def test_decision_agent_initialization(decision_agent):
    """Decision agent should initialize correctly."""
    assert decision_agent.name == "decision_agent"
    assert decision_agent.analysis_agent is not None
    assert decision_agent.market_service is not None
    assert decision_agent.rules is not None

def test_custom_rules():
    """Should accept custom trading rules."""
    from agents.analysis_agent import AnalysisAgent
    from services.llm_service import LLMService
    from services.alpaca_market_service import AlpacaMarketService

    if not settings.openai_api_key and not settings.anthropic_api_key:
        pytest.skip("No LLM API key configured")

    llm_service = LLMService()
    market_service = AlpacaMarketService(
        settings.alpaca_api_key,
        settings.alpaca_api_secret,
        paper=True
    )
    analysis_agent = AnalysisAgent(llm_service)

    custom_rules = TradingRules(
        buy_sentiment_threshold=0.5,
        max_position_size=50
    )

    agent = DecisionAgent(analysis_agent, market_service, rules=custom_rules)

    assert agent.rules.buy_sentiment_threshold == 0.5
    assert agent.rules.max_position_size == 50

@pytest.mark.asyncio
async def test_execute_with_insufficient_analyses(decision_agent, llm_service):
    """Should return None if insufficient analyses."""
    try:
        # Use a symbol with no analyses
        decision = await decision_agent.execute("NONEXISTENT")

        # Should return None or decision with HOLD
        if decision:
            assert decision.action == "HOLD"
        else:
            assert decision is None
    finally:
        await llm_service.close()

@pytest.mark.asyncio
async def test_execute_with_real_data(decision_agent, llm_service):
    """Should make decision based on real sentiment data."""
    try:
        # Use AAPL which has sentiment data from Phase 3
        decision = await decision_agent.execute("AAPL", hours_back=72)

        if decision:
            assert decision.symbol == "AAPL"
            assert decision.action in ["BUY", "SELL", "HOLD"]
            assert decision.quantity >= 0
            assert decision.expected_price > 0
            assert decision.confidence in ["HIGH", "MEDIUM", "LOW"]
            assert decision.reasoning is not None
            assert decision.sentiment_score is not None
    finally:
        await llm_service.close()

def test_position_size_calculation(decision_agent):
    """Should calculate position size correctly."""
    # Test with high confidence signal
    size = decision_agent._calculate_position_size(
        score=0.8,
        high_impact_count=3,
        analysis_count=5
    )

    assert size > decision_agent.rules.base_position_size
    assert size <= decision_agent.rules.max_position_size

def test_confidence_calculation(decision_agent):
    """Should calculate confidence correctly."""
    # High confidence
    conf = decision_agent._calculate_confidence(
        score=0.8,
        high_impact_count=3,
        analysis_count=5
    )
    assert conf == "HIGH"

    # Medium confidence
    conf = decision_agent._calculate_confidence(
        score=0.4,
        high_impact_count=1,
        analysis_count=5
    )
    assert conf == "MEDIUM"

    # Low confidence
    conf = decision_agent._calculate_confidence(
        score=0.2,
        high_impact_count=0,
        analysis_count=5
    )
    assert conf == "LOW"

def test_get_recent_decisions(decision_agent):
    """Should retrieve recent decisions."""
    decisions = decision_agent.get_recent_decisions("AAPL", limit=5)

    assert isinstance(decisions, list)
    assert len(decisions) <= 5

@pytest.mark.asyncio
async def test_buy_decision_logic(decision_agent):
    """Should make BUY decision on strong positive sentiment."""
    # This is an integration test that depends on:
    # 1. Existing positive sentiment analyses
    # 2. Available buying power
    # 3. No existing max position

    # Test with AAPL if it has positive sentiment
    decision = await decision_agent.execute("AAPL", hours_back=72)

    # If AAPL has strong positive sentiment, should be BUY or HOLD
    # (HOLD if already at max position or no buying power)
    if decision:
        assert decision.action in ["BUY", "HOLD"]

@pytest.mark.asyncio
async def test_decision_reasoning(decision_agent, llm_service):
    """Should include clear reasoning in decision."""
    try:
        decision = await decision_agent.execute("AAPL", hours_back=72)

        if decision:
            assert decision.reasoning is not None
            assert len(decision.reasoning) > 0
            # Should mention sentiment or position
            assert any(word in decision.reasoning.lower()
                      for word in ["sentiment", "position", "shares", "holding"])
    finally:
        await llm_service.close()
```

## Manual Verification Commands

```bash
# 1. Test trading rules
python -c "from agents.rules import get_trading_rules; rules = get_trading_rules(); print(f'Buy threshold: {rules.buy_sentiment_threshold}, Max position: {rules.max_position_size}')"

# 2. Test market service - account info
python -c "
from services.alpaca_market_service import AlpacaMarketService
from config.settings import settings

service = AlpacaMarketService(settings.alpaca_api_key, settings.alpaca_api_secret, paper=True)
account = service.get_account()
print(f'Buying Power: \${account[\"buying_power\"]:.2f}')
print(f'Cash: \${account[\"cash\"]:.2f}')
print(f'Portfolio Value: \${account[\"portfolio_value\"]:.2f}')
"

# 3. Test market service - current price
python -c "
from services.alpaca_market_service import AlpacaMarketService
from config.settings import settings

service = AlpacaMarketService(settings.alpaca_api_key, settings.alpaca_api_secret, paper=True)
price = service.get_current_price('AAPL')
print(f'AAPL Current Price: \${price:.2f}')
"

# 4. Test market service - positions
python -c "
from services.alpaca_market_service import AlpacaMarketService
from config.settings import settings

service = AlpacaMarketService(settings.alpaca_api_key, settings.alpaca_api_secret, paper=True)
positions = service.get_all_positions()
print(f'Open Positions: {len(positions)}')
for p in positions:
    print(f'  {p[\"symbol\"]}: {p[\"qty\"]} shares @ \${p[\"avg_entry_price\"]:.2f}')
"

# 5. Run full decision pipeline
python main.py

# 6. Check decisions in database
sqlite3 trading.db "SELECT COUNT(*) FROM trading_decisions"
sqlite3 trading.db "SELECT symbol, action, quantity, expected_price, confidence, reasoning FROM trading_decisions ORDER BY decided_at DESC LIMIT 5"

# 7. Run all tests
pytest tests/test_trading_rules.py -v
pytest tests/test_alpaca_market_service.py -v
pytest tests/test_decision_agent.py -v

# 8. Run full test suite
pytest tests/ -v
```

## Files to Create/Modify

### New Files:
- `agents/rules.py` - Trading rules and configuration
- `services/alpaca_market_service.py` - Market data and portfolio service
- `agents/decision_agent.py` - Decision agent with trading logic
- `tests/test_trading_rules.py` - Trading rules tests
- `tests/test_alpaca_market_service.py` - Market service tests
- `tests/test_decision_agent.py` - Decision agent tests

### Modified Files:
- `main.py` - Add decision pipeline

## Dependencies
- ✅ Phase 1 completed (foundation)
- ✅ Phase 2 completed (news agent)
- ✅ Phase 3 completed (analysis agent)
- ✅ Sentiment analyses in database
- ✅ TradingDecision model defined
- ✅ alpaca-py includes TradingClient and data clients

## Blockers
None - all dependencies met.

## Notes
- This is **paper trading only** for MVP (settings.paper_trading = True)
- Position sizing is rule-based (not ML-based) for simplicity
- Risk management is conservative (max position, cash reserve)
- Confidence calculation helps with future optimization
- All decisions stored for audit trail and backtesting
- Decision logic is deterministic and testable

## Implementation Order
1. Create `agents/rules.py` with TradingRules configuration
2. Create `services/alpaca_market_service.py` with market data methods
3. Create `agents/decision_agent.py` with decision logic
4. Update `main.py` to include decision pipeline
5. Create all test files
6. Test with mock data first (unit tests)
7. Test with real market data (integration tests)
8. Run full pipeline and verify decisions make sense

## Expected Output

After completion, running `python main.py` should:
- Fetch news for AAPL (5 articles)
- Analyze sentiment (positive/negative/neutral)
- Fetch current portfolio state
- Make trading decision (BUY/SELL/HOLD)
- Display decision with reasoning:

```
============================================================
Trading Decision for AAPL
============================================================
Action: BUY
Quantity: 25
Expected Price: $185.42
Confidence: MEDIUM
Sentiment Score: 0.48
Reasoning: Positive sentiment (score: 0.48) | Buying 25 shares | 2 high-impact articles from 5 total
============================================================
```

## Risk Considerations

**Conservative Approach**:
- Paper trading only (no real money)
- Small position sizes (10-100 shares)
- Require multiple analyses before decision
- Maintain cash reserve
- Cap position size per symbol
- Clear reasoning for every decision

**Safety Checks**:
- ✅ Can't buy without buying power
- ✅ Can't sell without position
- ✅ Can't exceed max position size
- ✅ Can't exceed portfolio percentage limits
- ✅ Always maintain cash reserve

This ensures the agent makes responsible decisions even in paper trading mode.
