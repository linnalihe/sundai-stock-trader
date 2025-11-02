import asyncio
from alpaca.trading.client import TradingClient
from dotenv import load_dotenv
from utils.logger import setup_logging, get_logger
from config.settings import settings
from config.symbol_config import symbol_config
from storage.database import db
from services.alpaca_news_service import AlpacaNewsService
from services.alpaca_market_service import AlpacaMarketService
from services.llm_service import LLMService
from agents.news_agent import NewsAgent
from agents.analysis_agent import AnalysisAgent
from agents.decision_agent import DecisionAgent
from agents.execution_agent import ExecutionAgent
from models.trade import TradeAction

# Load environment variables
load_dotenv()

# Setup logging
setup_logging()
logger = get_logger("main")

def initialize_system():
    """Initialize the trading system."""
    logger.info("system_initialization_started")

    # Initialize database
    db.initialize_schema()

    # Test Alpaca connection
    trading_client = TradingClient(
        settings.alpaca_api_key,
        settings.alpaca_api_secret,
        paper=settings.paper_trading
    )

    account = trading_client.get_account()
    logger.info(
        "alpaca_connection_successful",
        buying_power=float(account.buying_power),
        paper_trading=settings.paper_trading
    )

    logger.info("system_initialization_completed")
    return trading_client

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

        # 1. Fetch news (limit to 5 articles for cost control)
        articles = await news_agent.execute(
            symbols=[symbol],
            hours_back=72,
            max_articles=5
        )

        if not articles:
            logger.info("no_articles_found", symbol=symbol)
            continue

        logger.info("articles_fetched", symbol=symbol, count=len(articles))

        # 2. Analyze articles
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
            average_score=aggregate["average_score"],
            analysis_count=aggregate["analysis_count"],
            high_impact_count=aggregate["high_impact_count"]
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
            price=decision.expected_price,
            confidence=decision.confidence
        )

        # Display decision
        print(f"\n{'='*60}")
        print(f"Trading Decision for {symbol}")
        print(f"{'='*60}")
        print(f"Action:           {decision.action}")
        print(f"Quantity:         {decision.quantity}")
        print(f"Expected Price:   ${decision.expected_price:.2f}")
        print(f"Confidence:       {decision.confidence}")
        print(f"Sentiment Score:  {decision.sentiment_score:.2f}")
        print(f"Reasoning:        {decision.reasoning}")
        print(f"{'='*60}\n")

        # 5. Execute trade (if not HOLD)
        if decision.action != TradeAction.HOLD:
            logger.info("executing_trade", symbol=symbol, action=decision.action)

            execution = await execution_agent.execute(decision, timeout=60)

            if execution:
                logger.info(
                    "trade_executed",
                    symbol=symbol,
                    status=execution.status,
                    filled_qty=execution.filled_qty
                )

                # Display execution
                print(f"\n{'='*60}")
                print(f"Trade Execution for {symbol}")
                print(f"{'='*60}")
                print(f"Order ID:         {execution.order_id or 'N/A'}")
                print(f"Status:           {execution.status.value if hasattr(execution.status, 'value') else execution.status}")
                print(f"Filled Qty:       {execution.filled_qty}")
                print(f"Avg Fill Price:   ${execution.filled_avg_price:.2f}")
                if execution.submitted_at:
                    print(f"Submitted:        {execution.submitted_at}")
                if execution.filled_at:
                    print(f"Filled:           {execution.filled_at}")
                if execution.error_message:
                    print(f"Error:            {execution.error_message}")
                print(f"{'='*60}\n")
            else:
                logger.warning("execution_returned_none", symbol=symbol)
        else:
            logger.info("hold_decision_no_execution", symbol=symbol)
            print(f"\n{'='*60}")
            print(f"HOLD Decision - No Trade Executed for {symbol}")
            print(f"{'='*60}\n")

    # Cleanup
    await llm_service.close()

    logger.info("full_pipeline_completed")

def main():
    """Main entry point."""
    try:
        initialize_system()
        logger.info("system_ready")

        # Run full pipeline (includes execution)
        asyncio.run(run_full_pipeline())

    except Exception as e:
        logger.error("system_startup_failed", error=str(e), exc_info=True)
        raise

if __name__ == "__main__":
    main()
