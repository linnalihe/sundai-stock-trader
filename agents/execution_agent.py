"""Agent that executes trading decisions by placing orders."""

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

            execution.order_id = str(order.id)  # Convert UUID to string
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
                        "status": execution.status.value if execution.status else None,
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
