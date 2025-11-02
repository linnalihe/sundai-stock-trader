"""Agent that makes trading decisions based on sentiment analysis."""

from typing import Optional, Dict, Any, List
from datetime import datetime
from agents.base import BaseAgent
from agents.analysis_agent import AnalysisAgent
from agents.rules import get_trading_rules, TradingRules
from services.alpaca_market_service import AlpacaMarketService
from models.trade import TradingDecision, TradeAction
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
        """
        Initialize Decision Agent.

        Args:
            analysis_agent: AnalysisAgent for sentiment data
            market_service: AlpacaMarketService for portfolio and market data
            rules: Trading rules (uses default if not provided)
        """
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
        """
        Core decision logic.

        Args:
            symbol: Stock symbol
            aggregate: Aggregate sentiment data
            account: Account information
            position: Current position (or None)
            current_price: Current market price

        Returns:
            TradingDecision
        """
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
        """
        Calculate position size based on sentiment strength.

        Args:
            score: Sentiment score
            high_impact_count: Number of high impact articles
            analysis_count: Total number of analyses

        Returns:
            Calculated position size
        """
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
        """
        Calculate confidence level for the decision.

        Args:
            score: Absolute sentiment score
            high_impact_count: Number of high impact articles
            analysis_count: Total number of analyses

        Returns:
            Confidence level: "HIGH", "MEDIUM", or "LOW"
        """
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
        """
        Store trading decision in database.

        Args:
            decision: TradingDecision to store
        """
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
        """
        Get recent trading decisions for a symbol.

        Args:
            symbol: Stock symbol
            limit: Maximum number of decisions to return

        Returns:
            List of TradingDecision objects
        """
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
