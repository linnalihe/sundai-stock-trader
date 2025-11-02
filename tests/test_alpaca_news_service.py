import pytest
from services.alpaca_news_service import AlpacaNewsService
from config.settings import settings

@pytest.mark.asyncio
async def test_alpaca_news_service_init():
    """Should initialize Alpaca News Service."""
    service = AlpacaNewsService(
        settings.alpaca_api_key,
        settings.alpaca_api_secret
    )
    assert service.client is not None

@pytest.mark.asyncio
async def test_fetch_news_for_aapl():
    """Should fetch news for AAPL."""
    service = AlpacaNewsService(
        settings.alpaca_api_key,
        settings.alpaca_api_secret
    )
    articles = await service.fetch_news(["AAPL"], hours_back=72, limit=10)

    assert isinstance(articles, list)
    # May be 0 if no news in last 72 hours, but should not error
    if len(articles) > 0:
        assert any("AAPL" in article.symbols for article in articles)
        assert articles[0].url is not None

@pytest.mark.asyncio
async def test_news_article_structure():
    """Fetched news should have correct structure."""
    service = AlpacaNewsService(
        settings.alpaca_api_key,
        settings.alpaca_api_secret
    )
    articles = await service.fetch_news(["AAPL"], hours_back=168, limit=5)  # 1 week

    if len(articles) > 0:
        article = articles[0]
        assert article.id is not None
        assert article.title is not None
        assert article.source is not None
        assert article.published_at is not None
        assert article.url is not None
        assert isinstance(article.symbols, list)

@pytest.mark.asyncio
async def test_fetch_news_multiple_symbols():
    """Should fetch news for multiple symbols."""
    service = AlpacaNewsService(
        settings.alpaca_api_key,
        settings.alpaca_api_secret
    )
    articles = await service.fetch_news(["AAPL", "MSFT"], hours_back=72, limit=20)

    assert isinstance(articles, list)
    # Should have news about AAPL or MSFT (if any news available)
    if len(articles) > 0:
        symbols_found = set()
        for article in articles:
            symbols_found.update(article.symbols)
        # At least one of the requested symbols should be present
        assert len(symbols_found.intersection({"AAPL", "MSFT"})) > 0

@pytest.mark.asyncio
async def test_fetch_news_respects_limit():
    """Should respect the limit parameter."""
    service = AlpacaNewsService(
        settings.alpaca_api_key,
        settings.alpaca_api_secret
    )
    articles = await service.fetch_news(["AAPL"], hours_back=168, limit=5)

    # Should not exceed limit
    assert len(articles) <= 5
