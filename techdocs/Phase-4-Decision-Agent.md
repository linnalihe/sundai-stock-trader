# Phase 4: Decision Agent

## Overview
Implement the Decision Agent to make trading decisions (BUY/SELL/HOLD) based on sentiment analysis and risk management rules. This is the brain of the trading system.

## Timeline
**Estimated Duration**: 4-5 days

## Objectives
1. Implement decision logic based on sentiment
2. Add risk management rules
3. Calculate position sizes
4. Add confidence scoring
5. Store trading decisions in database
6. Implement decision validation

## Dependencies
- Phase 1-3 completed
- Analysis results available
- Alpaca account information

## Implementation Tasks

### 1. Risk Manager
**File**: `utils/risk_manager.py`

**Work**:
```python
from typing import Dict, Optional
from config.settings import settings
from utils.logger import get_logger
from alpaca.trading.client import TradingClient

logger = get_logger("risk_manager")

class RiskManager:
    """Manages risk parameters and position sizing."""

    def __init__(self, trading_client: TradingClient):
        self.trading_client = trading_client
        self.max_position_size = settings.max_position_size
        self.max_portfolio_percent = settings.max_portfolio_percent
        self.risk_percentage = settings.risk_percentage
        self.max_daily_trades = settings.max_daily_trades
        self.max_daily_loss_percent = settings.max_daily_loss_percent

        logger.info("risk_manager_initialized")

    def calculate_position_size(
        self,
        symbol: str,
        confidence: float,
        current_price: float
    ) -> int:
        """
        Calculate appropriate position size.

        Args:
            symbol: Stock symbol
            confidence: Decision confidence (0-1)
            current_price: Current stock price

        Returns:
            Number of shares to trade
        """
        # Get account info
        account = self.trading_client.get_account()
        buying_power = float(account.buying_power)
        portfolio_value = float(account.portfolio_value) or buying_power

        # Calculate max position value
        max_position_value = portfolio_value * self.max_portfolio_percent

        # Adjust by confidence
        adjusted_value = max_position_value * confidence

        # Calculate shares
        shares = int(adjusted_value / current_price)

        # Apply hard limit
        shares = min(shares, self.max_position_size)

        logger.info(
            "position_size_calculated",
            symbol=symbol,
            shares=shares,
            confidence=confidence,
            price=current_price
        )

        return shares

    def check_daily_trade_limit(self, symbol: str, today_trades: int) -> bool:
        """Check if daily trade limit reached."""
        if today_trades >= self.max_daily_trades:
            logger.warning(
                "daily_trade_limit_reached",
                symbol=symbol,
                trades_today=today_trades,
                limit=self.max_daily_trades
            )
            return False
        return True

    def check_daily_loss_limit(self, today_pnl: float, portfolio_value: float) -> bool:
        """Check if daily loss limit exceeded."""
        loss_percent = abs(today_pnl) / portfolio_value if portfolio_value > 0 else 0

        if today_pnl < 0 and loss_percent >= self.max_daily_loss_percent:
            logger.warning(
                "daily_loss_limit_exceeded",
                today_pnl=today_pnl,
                loss_percent=loss_percent,
                limit=self.max_daily_loss_percent
            )
            return False
        return True

    def check_existing_position(self, symbol: str) -> Optional[Dict]:
        """
        Check if we already have a position in this symbol.

        Returns:
            Position info dict or None
        """
        try:
            positions = self.trading_client.get_all_positions()

            for position in positions:
                if position.symbol == symbol:
                    return {
                        "symbol": symbol,
                        "qty": int(position.qty),
                        "avg_entry_price": float(position.avg_entry_price),
                        "current_price": float(position.current_price),
                        "market_value": float(position.market_value),
                        "unrealized_pl": float(position.unrealized_pl),
                        "side": position.side
                    }

            return None

        except Exception as e:
            logger.error("position_check_failed", symbol=symbol, error=str(e))
            return None

    def validate_decision(
        self,
        symbol: str,
        action: str,
        quantity: int,
        price: float
    ) -> tuple[bool, str]:
        """
        Validate a trading decision against risk rules.

        Returns:
            (is_valid, reason)
        """
        # Check quantity > 0
        if quantity <= 0:
            return False, "Invalid quantity"

        # Check if we can afford it
        account = self.trading_client.get_account()
        buying_power = float(account.buying_power)
        trade_value = quantity * price

        if action == "BUY" and trade_value > buying_power:
            return False, f"Insufficient buying power: ${buying_power:.2f}"

        # Check existing position
        position = self.check_existing_position(symbol)

        if action == "BUY" and position:
            return False, f"Already have position in {symbol}"

        if action == "SELL" and not position:
            return False, f"No position to sell in {symbol}"

        if action == "SELL" and position and quantity > position["qty"]:
            return False, f"Sell quantity exceeds position size"

        return True, "Valid"
```

**Success Criteria**:
- [ ] Position sizing calculation works
- [ ] Risk limits enforced
- [ ] Position checking works
- [ ] Decision validation functional

---

### 2. Decision Agent Implementation
**File**: `agents/decision_agent.py`

**Work**:
```python
import uuid
from typing import List, Optional, Dict
from datetime import datetime, timedelta
from agents.base import BaseAgent
from models.trade import TradingDecision, TradeAction
from models.analysis import AnalysisResult
from utils.risk_manager import RiskManager
from storage.database import db
from sqlalchemy import text
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
import json

class DecisionAgent(BaseAgent):
    """Agent responsible for making trading decisions."""

    def __init__(
        self,
        trading_client: TradingClient,
        risk_manager: RiskManager
    ):
        super().__init__("decision_agent")
        self.trading_client = trading_client
        self.risk_manager = risk_manager

    async def execute(
        self,
        symbol: str,
        analyses: List[AnalysisResult]
    ) -> Optional[TradingDecision]:
        """
        Make trading decision based on analyses.

        Args:
            symbol: Stock symbol
            analyses: List of recent analyses

        Returns:
            Trading decision or None
        """
        self._log_event(
            "decision_making_started",
            symbol=symbol,
            analysis_count=len(analyses)
        )

        # Calculate aggregate metrics
        metrics = self._calculate_metrics(analyses)

        # Make decision
        decision = await self._make_decision(symbol, metrics, analyses)

        if decision:
            # Validate decision
            is_valid, reason = self._validate_decision(decision)

            if is_valid:
                self._store_decision(decision)
                self._log_event(
                    "decision_made",
                    symbol=symbol,
                    action=decision.action,
                    confidence=decision.confidence
                )
            else:
                self.logger.warning(
                    "decision_invalid",
                    symbol=symbol,
                    action=decision.action,
                    reason=reason
                )
                return None

        return decision

    def _calculate_metrics(self, analyses: List[AnalysisResult]) -> Dict:
        """Calculate aggregate metrics from analyses."""

        if not analyses:
            return {
                "avg_sentiment_score": 0.0,
                "positive_count": 0,
                "negative_count": 0,
                "neutral_count": 0,
                "high_impact_count": 0,
                "total_count": 0
            }

        total_score = sum(a.sentiment_score for a in analyses)
        avg_score = total_score / len(analyses)

        return {
            "avg_sentiment_score": avg_score,
            "positive_count": sum(1 for a in analyses if a.sentiment == "POSITIVE"),
            "negative_count": sum(1 for a in analyses if a.sentiment == "NEGATIVE"),
            "neutral_count": sum(1 for a in analyses if a.sentiment == "NEUTRAL"),
            "high_impact_count": sum(1 for a in analyses if a.impact_level == "HIGH"),
            "total_count": len(analyses)
        }

    async def _make_decision(
        self,
        symbol: str,
        metrics: Dict,
        analyses: List[AnalysisResult]
    ) -> Optional[TradingDecision]:
        """
        Core decision-making logic.

        Decision Rules:
        - BUY: avg sentiment > 0.3 AND high impact news present
        - SELL: avg sentiment < -0.3 AND have position
        - HOLD: otherwise
        """

        avg_score = metrics["avg_sentiment_score"]
        high_impact = metrics["high_impact_count"]

        # Get current position
        position = self.risk_manager.check_existing_position(symbol)

        # Get current price
        try:
            latest_trade = self.trading_client.get_latest_trade(symbol)
            current_price = float(latest_trade.price)
        except Exception as e:
            self.logger.error("price_fetch_failed", symbol=symbol, error=str(e))
            return None

        # Decision logic
        action = TradeAction.HOLD
        confidence = 0.0
        reasoning = ""

        # BUY logic
        if not position and avg_score > 0.3 and high_impact > 0:
            action = TradeAction.BUY
            # Confidence based on sentiment strength and high impact count
            confidence = min(0.5 + (avg_score * 0.3) + (high_impact * 0.1), 1.0)
            reasoning = (
                f"Strong positive sentiment (score: {avg_score:.2f}) "
                f"with {high_impact} high-impact news articles. "
                f"Positive signals: {metrics['positive_count']}/{metrics['total_count']}"
            )

        # SELL logic
        elif position and avg_score < -0.3:
            action = TradeAction.SELL
            confidence = min(0.5 + (abs(avg_score) * 0.3) + (high_impact * 0.1), 1.0)
            reasoning = (
                f"Strong negative sentiment (score: {avg_score:.2f}) "
                f"with existing position. "
                f"Negative signals: {metrics['negative_count']}/{metrics['total_count']}"
            )

        # Conservative BUY on moderate positive sentiment
        elif not position and avg_score > 0.5 and metrics['positive_count'] >= 3:
            action = TradeAction.BUY
            confidence = 0.6 + (avg_score * 0.2)
            reasoning = (
                f"Moderate positive sentiment (score: {avg_score:.2f}) "
                f"with consistent positive news ({metrics['positive_count']} articles)"
            )

        # HOLD logic
        else:
            action = TradeAction.HOLD
            confidence = 0.5
            reasoning = (
                f"Insufficient signal strength. "
                f"Avg sentiment: {avg_score:.2f}, "
                f"Position: {'Yes' if position else 'No'}"
            )

        # Check if confidence meets threshold
        if action != TradeAction.HOLD and confidence < settings.decision_confidence_threshold:
            self.logger.info(
                "confidence_below_threshold",
                symbol=symbol,
                action=action,
                confidence=confidence,
                threshold=settings.decision_confidence_threshold
            )
            action = TradeAction.HOLD
            reasoning += f" [Confidence {confidence:.2f} below threshold {settings.decision_confidence_threshold}]"

        # Calculate quantity
        if action == TradeAction.BUY:
            quantity = self.risk_manager.calculate_position_size(
                symbol, confidence, current_price
            )
        elif action == TradeAction.SELL and position:
            quantity = position["qty"]
        else:
            quantity = 0

        # Create decision
        decision = TradingDecision(
            id=f"decision_{uuid.uuid4().hex[:12]}",
            symbol=symbol,
            action=action,
            quantity=quantity if action != TradeAction.HOLD else 0,
            confidence=confidence,
            reasoning=reasoning,
            price_limit=current_price if action != TradeAction.HOLD else None,
            analysis_ids=[a.id for a in analyses]
        )

        return decision

    def _validate_decision(
        self,
        decision: TradingDecision
    ) -> tuple[bool, str]:
        """Validate decision against risk rules."""

        # HOLD always valid
        if decision.action == TradeAction.HOLD:
            return True, "Hold decision"

        # Check daily trade limit
        today_trades = self._get_today_trade_count(decision.symbol)
        if not self.risk_manager.check_daily_trade_limit(
            decision.symbol, today_trades
        ):
            return False, "Daily trade limit reached"

        # Check daily loss limit
        today_pnl = self._get_today_pnl()
        account = self.trading_client.get_account()
        portfolio_value = float(account.portfolio_value)

        if not self.risk_manager.check_daily_loss_limit(today_pnl, portfolio_value):
            return False, "Daily loss limit exceeded"

        # Validate with risk manager
        return self.risk_manager.validate_decision(
            decision.symbol,
            decision.action,
            decision.quantity,
            decision.price_limit or 0
        )

    def _get_today_trade_count(self, symbol: str) -> int:
        """Get count of trades executed today for symbol."""
        today = datetime.utcnow().date()

        with db.get_session() as session:
            result = session.execute(
                text("""
                    SELECT COUNT(*) FROM trade_executions
                    WHERE symbol = :symbol
                    AND DATE(submitted_at) = :today
                    AND status IN ('FILLED', 'PARTIALLY_FILLED')
                """),
                {"symbol": symbol, "today": today}
            )
            count = result.scalar() or 0
            return count

    def _get_today_pnl(self) -> float:
        """Get today's P&L from closed positions."""
        # For MVP, simplified - just return 0
        # Future: Calculate from closed positions
        return 0.0

    def _store_decision(self, decision: TradingDecision):
        """Store decision in database."""
        with db.get_session() as session:
            try:
                session.execute(
                    text("""
                        INSERT INTO trading_decisions
                        (id, symbol, action, quantity, confidence, reasoning,
                         price_limit, analysis_ids, created_at)
                        VALUES (:id, :symbol, :action, :quantity, :confidence,
                                :reasoning, :price_limit, :analysis_ids, :created_at)
                    """),
                    {
                        "id": decision.id,
                        "symbol": decision.symbol,
                        "action": decision.action,
                        "quantity": decision.quantity,
                        "confidence": decision.confidence,
                        "reasoning": decision.reasoning,
                        "price_limit": decision.price_limit,
                        "analysis_ids": json.dumps(decision.analysis_ids),
                        "created_at": decision.created_at
                    }
                )
                self.logger.info("decision_stored", decision_id=decision.id)

            except Exception as e:
                self.logger.error(
                    "decision_storage_failed",
                    decision_id=decision.id,
                    error=str(e)
                )
                raise

    def get_recent_decisions(
        self,
        symbol: Optional[str] = None,
        limit: int = 10
    ) -> List[TradingDecision]:
        """Retrieve recent decisions from database."""

        query = "SELECT * FROM trading_decisions"
        params = {"limit": limit}

        if symbol:
            query += " WHERE symbol = :symbol"
            params["symbol"] = symbol

        query += " ORDER BY created_at DESC LIMIT :limit"

        with db.get_session() as session:
            result = session.execute(text(query), params)
            rows = result.fetchall()

            decisions = []
            for row in rows:
                decision = TradingDecision(
                    id=row[0],
                    symbol=row[1],
                    action=TradeAction(row[2]),
                    quantity=row[3],
                    confidence=float(row[4]),
                    reasoning=row[5],
                    price_limit=float(row[6]) if row[6] else None,
                    analysis_ids=json.loads(row[7]) if row[7] else [],
                    created_at=row[8]
                )
                decisions.append(decision)

            return decisions
```

**Success Criteria**:
- [ ] Makes BUY decisions on positive sentiment
- [ ] Makes SELL decisions on negative sentiment
- [ ] Respects confidence threshold
- [ ] Validates decisions properly
- [ ] Stores decisions in database
- [ ] Position sizing works correctly
- [ ] Handles edge cases

---

### 3. Update Main Entry Point
**File**: `main.py`

**Work**:
Add decision agent to the pipeline after analysis.

```python
# ... (previous imports)
from agents.decision_agent import DecisionAgent
from utils.risk_manager import RiskManager
from alpaca.trading.client import TradingClient

async def run_full_pipeline():
    """Run news -> analysis -> decision pipeline."""

    # Initialize services
    news_service = NewsService(api_key=settings.news_api_key)
    llm_service = LLMService()
    trading_client = TradingClient(
        settings.alpaca_api_key,
        settings.alpaca_api_secret,
        paper=settings.paper_trading
    )

    # Initialize agents
    news_agent = NewsAgent(news_service)
    analysis_agent = AnalysisAgent(llm_service)
    risk_manager = RiskManager(trading_client)
    decision_agent = DecisionAgent(trading_client, risk_manager)

    # Get symbols
    symbols = symbol_config.get_enabled_symbols("stocks")
    logger.info("pipeline_started", symbols=symbols)

    for symbol in symbols:
        logger.info("processing_symbol", symbol=symbol)

        # 1. Fetch news
        articles = await news_agent.execute([symbol], hours_back=24, max_articles=20)
        if not articles:
            continue

        # 2. Analyze
        analyses = await analysis_agent.execute(articles, symbol)
        if not analyses:
            continue

        # 3. Make decision
        decision = await decision_agent.execute(symbol, analyses)

        if decision:
            logger.info(
                "decision_made",
                symbol=symbol,
                action=decision.action,
                quantity=decision.quantity,
                confidence=decision.confidence,
                reasoning=decision.reasoning
            )

    # Cleanup
    await news_service.close()
    await llm_service.close()

    logger.info("pipeline_completed")
```

**Success Criteria**:
- [ ] Pipeline runs end-to-end
- [ ] Decisions made and logged
- [ ] All components integrated

---

## Testing Checklist

### Unit Tests
**File**: `tests/test_risk_manager.py`
- [ ] Test position sizing
- [ ] Test risk limits
- [ ] Test decision validation

**File**: `tests/test_decision_agent.py`
- [ ] Test BUY logic
- [ ] Test SELL logic
- [ ] Test HOLD logic
- [ ] Test confidence calculation

### Integration Tests
- [ ] Full pipeline with decision
- [ ] Edge cases (no position, existing position)

---

## Phase Completion Criteria

### Must Have
- [ ] Decision logic implemented
- [ ] Risk management rules enforced
- [ ] Position sizing works
- [ ] Decisions validated
- [ ] Decisions stored in database
- [ ] Full pipeline integration
- [ ] Tests passing

### Nice to Have
- [ ] Machine learning-based decisions
- [ ] Multi-factor decision model
- [ ] Backtesting capability

---

## Next Phase
**Phase 5: Execution Agent** - Execute trades via Alpaca API
