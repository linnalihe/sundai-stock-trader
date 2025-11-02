"""Service for fetching market data and portfolio information from Alpaca."""

from typing import Dict, Optional, Any
from alpaca.trading.client import TradingClient
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestQuoteRequest
from utils.logger import get_logger

logger = get_logger("alpaca_market_service")


class AlpacaMarketService:
    """Service for fetching market data and portfolio information from Alpaca."""

    def __init__(self, api_key: str, api_secret: str, paper: bool = True):
        """
        Initialize Alpaca Market Service.

        Args:
            api_key: Alpaca API key
            api_secret: Alpaca API secret
            paper: Whether to use paper trading (default: True)
        """
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
        """
        Get account information including buying power.

        Returns:
            Dict with account information including buying_power, cash, portfolio_value, etc.
        """
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
        """
        Get current position for a symbol.

        Args:
            symbol: Stock symbol (e.g., "AAPL")

        Returns:
            Dict with position info or None if no position exists
        """
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
        """
        Get all open positions.

        Returns:
            List of position dicts
        """
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
        """
        Get current price for a symbol.

        Args:
            symbol: Stock symbol (e.g., "AAPL")

        Returns:
            Current price (mid price between bid and ask)
        """
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
        """
        Calculate maximum affordable quantity given buying power.

        Args:
            symbol: Stock symbol
            buying_power: Available buying power
            reserve: Amount to keep in reserve (default: 0)

        Returns:
            Maximum affordable quantity
        """
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
