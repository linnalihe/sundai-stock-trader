import pytest
from datetime import datetime
from pydantic import ValidationError
from models.news import NewsArticle
from models.analysis import AnalysisResult, SentimentType, ImpactLevel
from models.trade import TradingDecision, TradeExecution, TradeAction, OrderStatus

def test_news_article_validation():
    """NewsArticle should validate fields correctly."""
    article = NewsArticle(
        id="test_1",
        title="Test Article",
        content="Test content",
        source="Reuters",
        published_at=datetime.utcnow(),
        url="https://example.com",
        symbols=["AAPL"]
    )
    assert article.id == "test_1"
    assert article.title == "Test Article"
    assert article.relevance_score == 0.0

def test_news_article_invalid_relevance_score():
    """Should reject invalid relevance scores."""
    with pytest.raises(ValidationError):
        NewsArticle(
            id="test_1",
            title="Test",
            content="Content",
            source="Reuters",
            published_at=datetime.utcnow(),
            url="https://example.com",
            symbols=["AAPL"],
            relevance_score=1.5  # > 1.0 invalid
        )

def test_analysis_result_validation():
    """AnalysisResult should validate fields correctly."""
    analysis = AnalysisResult(
        id="analysis_1",
        article_id="article_1",
        symbol="AAPL",
        sentiment=SentimentType.POSITIVE,
        sentiment_score=0.8,
        key_points=["Point 1", "Point 2"],
        impact_level=ImpactLevel.HIGH,
        reasoning="Strong positive signals"
    )
    assert analysis.sentiment == "POSITIVE"
    assert analysis.sentiment_score == 0.8

def test_analysis_invalid_sentiment_score():
    """Should reject invalid sentiment scores."""
    with pytest.raises(ValidationError):
        AnalysisResult(
            id="analysis_1",
            article_id="article_1",
            symbol="AAPL",
            sentiment=SentimentType.POSITIVE,
            sentiment_score=1.5,  # > 1.0 invalid
            impact_level=ImpactLevel.HIGH,
            reasoning="Test"
        )

def test_trading_decision_validation():
    """TradingDecision should validate fields correctly."""
    decision = TradingDecision(
        symbol="AAPL",
        action=TradeAction.BUY,
        quantity=10,
        expected_price=150.0,
        confidence="HIGH",
        reasoning="Strong buy signal",
        sentiment_score=0.75,
        high_impact_count=3,
        analysis_count=5
    )
    assert decision.action == "BUY"
    assert decision.confidence == "HIGH"
    assert decision.quantity == 10

def test_trading_decision_invalid_quantity():
    """Should reject negative quantities."""
    with pytest.raises(ValidationError):
        TradingDecision(
            symbol="AAPL",
            action=TradeAction.BUY,
            quantity=-1,  # Must be >= 0
            expected_price=150.0,
            confidence="HIGH",
            reasoning="Test",
            sentiment_score=0.5,
            high_impact_count=1,
            analysis_count=3
        )

def test_trade_execution_validation():
    """TradeExecution should validate fields correctly."""
    execution = TradeExecution(
        id="execution_1",
        decision_id="decision_1",
        symbol="AAPL",
        status=OrderStatus.FILLED,
        filled_qty=10,
        filled_avg_price=150.0
    )
    assert execution.status == "FILLED"
    assert execution.filled_qty == 10
