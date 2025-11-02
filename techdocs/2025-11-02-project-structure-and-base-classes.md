# Feature: Project Structure and Base Classes

**Date**: 2025-11-02
**Status**: ✅ COMPLETED
**Estimated Time**: 2-3 hours
**Actual Time**: ~2 hours

## Overview
Set up the foundational project structure with organized directories, base agent class, configuration management, and data models. This creates the framework all other features will build upon.

## What We're Building
1. Directory structure for organized code
2. Base agent class that all agents will inherit from
3. Configuration system using pydantic-settings
4. Data models for News, Analysis, and Trading
5. Structured logging setup
6. Basic database schema

## Implementation Details

### 1. Create Directory Structure
```bash
mkdir -p agents services models utils storage config tests
touch agents/__init__.py services/__init__.py models/__init__.py
touch utils/__init__.py storage/__init__.py config/__init__.py tests/__init__.py
```

### 2. Update requirements.txt
Add these dependencies:
```
python-dotenv==1.0.0
pydantic==2.5.0
pydantic-settings==2.1.0
sqlalchemy==2.0.23
structlog==23.2.0
httpx==0.25.2
alpaca-py==0.14.0
openai==1.6.1
pytest==7.4.3
pytest-asyncio==0.21.1
pyyaml==6.0.1
```

### 3. Configuration System (`config/settings.py`)
Create a Settings class that loads from .env:
- API keys (Alpaca, OpenAI, News API)
- Trading parameters (max position size, risk %)
- Agent configuration (intervals, thresholds)
- Use pydantic-settings BaseSettings

### 4. Logging Setup (`utils/logger.py`)
- Configure structlog for JSON logging
- Create `setup_logging()` function
- Create `get_logger(name)` helper function

### 5. Base Agent Class (`agents/base.py`)
- Abstract base class with `__init__(name)`
- Abstract `execute()` method
- Built-in logger for each agent
- Error handling in `__call__` method

### 6. Data Models
Create Pydantic models in `models/` directory:

**`models/news.py`**:
- `NewsArticle`: id, title, content, source, published_at, url, symbols, relevance_score

**`models/analysis.py`**:
- `SentimentType` enum: POSITIVE, NEGATIVE, NEUTRAL
- `ImpactLevel` enum: HIGH, MEDIUM, LOW
- `AnalysisResult`: id, article_id, symbol, sentiment, sentiment_score, key_points, impact_level, reasoning

**`models/trade.py`**:
- `TradeAction` enum: BUY, SELL, HOLD
- `OrderStatus` enum: PENDING, SUBMITTED, FILLED, REJECTED, CANCELLED
- `TradingDecision`: id, symbol, action, quantity, confidence, reasoning, price_limit
- `TradeExecution`: id, decision_id, order_id, symbol, status, filled_qty, filled_avg_price

### 7. Database Schema (`storage/schema.sql`)
Create tables:
- `news_articles`: Store fetched news
- `analysis_results`: Store sentiment analysis
- `trading_decisions`: Store decision logic
- `trade_executions`: Store order executions
- `system_events`: Store system logs

**`storage/database.py`**:
- Database class with SQLAlchemy connection
- `initialize_schema()` method to create tables
- `get_session()` context manager

### 8. Update main.py
- Import and setup logging
- Load settings
- Initialize database
- Test Alpaca connection
- Log successful initialization

## Success Criteria
- [ ] All directories created with `__init__.py` files
- [ ] requirements.txt updated and packages install successfully
- [ ] Settings load from .env file correctly
- [ ] Structured logging outputs JSON format
- [ ] Base agent class is abstract and cannot be instantiated directly
- [ ] All data models validate input correctly (test with invalid data)
- [ ] Database schema created successfully
- [ ] main.py runs without errors and logs to console
- [ ] Can import: `from config.settings import settings`
- [ ] Can import: `from agents.base import BaseAgent`
- [ ] Can import all models: `from models.news import NewsArticle`

## Tests

### Test File: `tests/test_config.py`
```python
def test_settings_load_from_env():
    """Settings should load from .env file"""
    assert settings.alpaca_api_key is not None
    assert settings.paper_trading == True

def test_settings_defaults():
    """Settings should have correct defaults"""
    assert settings.max_position_size == 100
    assert settings.risk_percentage == 0.02
```

### Test File: `tests/test_models.py`
```python
def test_news_article_validation():
    """NewsArticle should validate fields"""
    article = NewsArticle(
        id="test_1",
        title="Test",
        content="Content",
        source="Reuters",
        published_at=datetime.utcnow(),
        url="https://example.com",
        symbols=["AAPL"]
    )
    assert article.id == "test_1"

def test_news_article_invalid_relevance_score():
    """Should reject invalid relevance scores"""
    with pytest.raises(ValidationError):
        NewsArticle(..., relevance_score=1.5)  # > 1.0 invalid
```

### Test File: `tests/test_base_agent.py`
```python
def test_base_agent_is_abstract():
    """Cannot instantiate BaseAgent directly"""
    with pytest.raises(TypeError):
        agent = BaseAgent("test")

def test_custom_agent_requires_execute():
    """Custom agent must implement execute()"""
    class BadAgent(BaseAgent):
        pass

    with pytest.raises(TypeError):
        agent = BadAgent("bad")

@pytest.mark.asyncio
async def test_agent_logging():
    """Agent should log events"""
    class TestAgent(BaseAgent):
        async def execute(self):
            return "done"

    agent = TestAgent("test")
    result = await agent()
    assert result == "done"
```

### Test File: `tests/test_database.py`
```python
def test_database_initialization():
    """Database schema should be created"""
    db.initialize_schema()
    # Check tables exist
    with db.get_session() as session:
        result = session.execute(text("SELECT name FROM sqlite_master WHERE type='table'"))
        tables = [row[0] for row in result.fetchall()]
        assert "news_articles" in tables
        assert "analysis_results" in tables
```

### Manual Verification
```bash
# Install dependencies
pip install -r requirements.txt

# Test imports
python -c "from config.settings import settings; print(settings.paper_trading)"

# Test logging
python -c "from utils.logger import setup_logging, get_logger; setup_logging(); logger = get_logger('test'); logger.info('hello', foo='bar')"

# Test database
python -c "from storage.database import db; db.initialize_schema(); print('DB OK')"

# Run main
python main.py

# Run tests
pytest tests/ -v
```

## Files Changed
- `requirements.txt` - Add dependencies
- `config/settings.py` - New file
- `utils/logger.py` - New file
- `agents/base.py` - New file
- `models/news.py` - New file
- `models/analysis.py` - New file
- `models/trade.py` - New file
- `storage/schema.sql` - New file
- `storage/database.py` - New file
- `main.py` - Update with initialization
- `tests/test_config.py` - New file
- `tests/test_models.py` - New file
- `tests/test_base_agent.py` - New file
- `tests/test_database.py` - New file

## Dependencies
- None (this is the foundation)

## Blockers
- Need .env file with API keys configured

## Notes
- Keep it simple - this is just the foundation
- Make sure everything is importable before moving to next feature
- All tests must pass before considering this complete
