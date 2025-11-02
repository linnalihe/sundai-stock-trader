# Phase 1: Foundation & Infrastructure

## Overview
Establish the core project structure, base classes, configuration management, logging, and database schema. This phase creates the foundation that all other components will build upon.

## Timeline
**Estimated Duration**: 3-5 days

## Objectives
1. Set up scalable project structure
2. Create reusable base agent class
3. Implement configuration management
4. Set up structured logging
5. Design and implement database schema
6. Create data models with validation

## Dependencies
- Python 3.11+
- Initial requirements.txt
- .env configuration

## Implementation Tasks

### 1. Project Structure Setup
**File**: Project directories

**Work**:
```bash
Create the following directory structure:
sundai-stock-trader/
├── agents/
├── services/
├── models/
├── utils/
├── storage/
├── config/
└── tests/
```

**Success Criteria**:
- [ ] All directories created
- [ ] Each directory has `__init__.py`
- [ ] Structure matches architecture doc

---

### 2. Dependencies Installation
**File**: `requirements.txt`

**Work**:
Add and install core dependencies:
```
# Core
python-dotenv==1.0.0
pydantic==2.5.0
pydantic-settings==2.1.0

# Data
pandas==2.1.4
numpy==1.26.2

# HTTP
httpx==0.25.2

# Trading
alpaca-py==0.14.0

# LLM (choose one or both)
openai==1.6.1
anthropic==0.8.0

# Database
sqlalchemy==2.0.23

# Logging
structlog==23.2.0

# Testing
pytest==7.4.3
pytest-asyncio==0.21.1
pytest-mock==3.12.0
```

**Success Criteria**:
- [ ] requirements.txt updated
- [ ] All packages install without errors
- [ ] Virtual environment activated
- [ ] Dependencies tested with `python -c "import pkg"`

---

### 3. Configuration Management
**File**: `config/settings.py`

**Work**:
```python
from pydantic_settings import BaseSettings
from typing import Literal

class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # API Keys
    alpaca_api_key: str
    alpaca_api_secret: str
    openai_api_key: str | None = None
    anthropic_api_key: str | None = None
    news_api_key: str | None = None

    # Trading Configuration
    paper_trading: bool = True
    max_position_size: int = 100
    max_portfolio_percent: float = 0.20
    risk_percentage: float = 0.02
    max_daily_trades: int = 5
    max_daily_loss_percent: float = 0.02

    # Agent Configuration
    news_fetch_interval: int = 300  # seconds
    analysis_threshold: float = 0.6
    decision_confidence_threshold: float = 0.7

    # LLM Configuration
    llm_provider: Literal["openai", "anthropic"] = "openai"
    llm_model: str = "gpt-4-turbo-preview"
    llm_temperature: float = 0.3

    # Logging
    log_level: str = "INFO"
    log_format: str = "json"

    # Database
    database_url: str = "sqlite:///./trading.db"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

# Global settings instance
settings = Settings()
```

**Success Criteria**:
- [ ] Settings class loads from .env
- [ ] All required keys validated
- [ ] Default values work correctly
- [ ] Type validation works (try invalid values)
- [ ] Can import and use: `from config.settings import settings`

---

### 4. Structured Logging
**File**: `utils/logger.py`

**Work**:
```python
import structlog
import logging
from config.settings import settings

def setup_logging():
    """Configure structured logging for the application."""

    # Configure standard logging
    logging.basicConfig(
        format="%(message)s",
        level=getattr(logging, settings.log_level.upper()),
    )

    # Configure structlog
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.dev.set_exc_info,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.JSONRenderer() if settings.log_format == "json"
            else structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, settings.log_level.upper())
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )

def get_logger(name: str):
    """Get a logger instance for a module."""
    return structlog.get_logger(name)
```

**Success Criteria**:
- [ ] Logger outputs structured JSON logs
- [ ] Log levels work correctly (DEBUG, INFO, WARNING, ERROR)
- [ ] Timestamps in ISO format
- [ ] Context variables can be bound to logger
- [ ] Test: `logger.info("test", key="value")` outputs correctly

---

### 5. Data Models
**File**: `models/news.py`

**Work**:
```python
from pydantic import BaseModel, Field, HttpUrl
from datetime import datetime
from typing import Optional

class NewsArticle(BaseModel):
    """News article data model."""

    id: str = Field(..., description="Unique article identifier")
    title: str = Field(..., description="Article title")
    content: str = Field(..., description="Article content/summary")
    source: str = Field(..., description="News source name")
    published_at: datetime = Field(..., description="Publication timestamp")
    url: HttpUrl = Field(..., description="Article URL")
    symbols: list[str] = Field(default_factory=list, description="Related stock symbols")
    relevance_score: float = Field(default=0.0, ge=0.0, le=1.0)
    created_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        json_schema_extra = {
            "example": {
                "id": "news_123",
                "title": "Apple Reports Record Earnings",
                "content": "Apple Inc. reported record quarterly earnings...",
                "source": "Reuters",
                "published_at": "2025-11-02T10:00:00Z",
                "url": "https://example.com/article",
                "symbols": ["AAPL"],
                "relevance_score": 0.95
            }
        }
```

**File**: `models/analysis.py`

**Work**:
```python
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum

class SentimentType(str, Enum):
    POSITIVE = "POSITIVE"
    NEGATIVE = "NEGATIVE"
    NEUTRAL = "NEUTRAL"

class ImpactLevel(str, Enum):
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"

class AnalysisResult(BaseModel):
    """Analysis result data model."""

    id: str = Field(..., description="Unique analysis identifier")
    article_id: str = Field(..., description="Related article ID")
    symbol: str = Field(..., description="Stock symbol")
    sentiment: SentimentType = Field(..., description="Overall sentiment")
    sentiment_score: float = Field(..., ge=-1.0, le=1.0, description="Sentiment score")
    key_points: list[str] = Field(default_factory=list, description="Key takeaways")
    impact_level: ImpactLevel = Field(..., description="Expected impact level")
    reasoning: str = Field(..., description="Analysis reasoning")
    analyzed_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        use_enum_values = True
```

**File**: `models/trade.py`

**Work**:
```python
from pydantic import BaseModel, Field
from datetime import datetime
from typing import Optional
from enum import Enum

class TradeAction(str, Enum):
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"

class OrderStatus(str, Enum):
    PENDING = "PENDING"
    SUBMITTED = "SUBMITTED"
    FILLED = "FILLED"
    PARTIALLY_FILLED = "PARTIALLY_FILLED"
    REJECTED = "REJECTED"
    CANCELLED = "CANCELLED"

class TradingDecision(BaseModel):
    """Trading decision data model."""

    id: str = Field(..., description="Unique decision identifier")
    symbol: str = Field(..., description="Stock symbol")
    action: TradeAction = Field(..., description="Trade action")
    quantity: int = Field(..., gt=0, description="Number of shares")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Decision confidence")
    reasoning: str = Field(..., description="Decision reasoning")
    price_limit: Optional[float] = Field(None, description="Limit price")
    analysis_ids: list[str] = Field(default_factory=list, description="Related analysis IDs")
    created_at: datetime = Field(default_factory=datetime.utcnow)

    class Config:
        use_enum_values = True

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

**Success Criteria**:
- [ ] All models can be instantiated
- [ ] Validation works (test invalid data)
- [ ] Enums work correctly
- [ ] Datetime fields auto-populate
- [ ] Models can serialize to/from JSON
- [ ] Example data in docstrings works

---

### 6. Base Agent Class
**File**: `agents/base.py`

**Work**:
```python
from abc import ABC, abstractmethod
from utils.logger import get_logger
from typing import Any, Optional

class BaseAgent(ABC):
    """Base class for all agents in the system."""

    def __init__(self, name: str):
        """
        Initialize base agent.

        Args:
            name: Agent name for logging
        """
        self.name = name
        self.logger = get_logger(f"agent.{name}")
        self.logger.info("agent_initialized", agent=name)

    @abstractmethod
    async def execute(self, *args, **kwargs) -> Any:
        """
        Execute the agent's main task.

        Must be implemented by subclasses.
        """
        pass

    async def __call__(self, *args, **kwargs) -> Any:
        """Allow agent to be called directly."""
        self.logger.info("agent_execution_started", agent=self.name)
        try:
            result = await self.execute(*args, **kwargs)
            self.logger.info("agent_execution_completed", agent=self.name)
            return result
        except Exception as e:
            self.logger.error(
                "agent_execution_failed",
                agent=self.name,
                error=str(e),
                exc_info=True
            )
            raise

    def _log_event(self, event: str, **kwargs):
        """Helper method to log agent events."""
        self.logger.info(event, agent=self.name, **kwargs)
```

**Success Criteria**:
- [ ] Can create subclass of BaseAgent
- [ ] Abstract execute() method enforced
- [ ] Logging works correctly
- [ ] Error handling captures exceptions
- [ ] Can call agent with `await agent()`

---

### 7. Database Schema
**File**: `storage/schema.sql`

**Work**:
```sql
-- News Articles Table
CREATE TABLE IF NOT EXISTS news_articles (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    content TEXT NOT NULL,
    source TEXT NOT NULL,
    published_at TIMESTAMP NOT NULL,
    url TEXT NOT NULL,
    symbols TEXT NOT NULL,  -- JSON array
    relevance_score REAL DEFAULT 0.0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_news_published ON news_articles(published_at);
CREATE INDEX idx_news_symbols ON news_articles(symbols);

-- Analysis Results Table
CREATE TABLE IF NOT EXISTS analysis_results (
    id TEXT PRIMARY KEY,
    article_id TEXT NOT NULL,
    symbol TEXT NOT NULL,
    sentiment TEXT NOT NULL,
    sentiment_score REAL NOT NULL,
    key_points TEXT,  -- JSON array
    impact_level TEXT NOT NULL,
    reasoning TEXT NOT NULL,
    analyzed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (article_id) REFERENCES news_articles(id)
);

CREATE INDEX idx_analysis_symbol ON analysis_results(symbol);
CREATE INDEX idx_analysis_analyzed_at ON analysis_results(analyzed_at);

-- Trading Decisions Table
CREATE TABLE IF NOT EXISTS trading_decisions (
    id TEXT PRIMARY KEY,
    symbol TEXT NOT NULL,
    action TEXT NOT NULL,
    quantity INTEGER NOT NULL,
    confidence REAL NOT NULL,
    reasoning TEXT NOT NULL,
    price_limit REAL,
    analysis_ids TEXT,  -- JSON array
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_decisions_symbol ON trading_decisions(symbol);
CREATE INDEX idx_decisions_created_at ON trading_decisions(created_at);

-- Trade Executions Table
CREATE TABLE IF NOT EXISTS trade_executions (
    id TEXT PRIMARY KEY,
    decision_id TEXT NOT NULL,
    order_id TEXT,
    symbol TEXT NOT NULL,
    status TEXT NOT NULL,
    filled_qty INTEGER DEFAULT 0,
    filled_avg_price REAL DEFAULT 0.0,
    submitted_at TIMESTAMP,
    filled_at TIMESTAMP,
    error_message TEXT,
    FOREIGN KEY (decision_id) REFERENCES trading_decisions(id)
);

CREATE INDEX idx_executions_status ON trade_executions(status);
CREATE INDEX idx_executions_symbol ON trade_executions(symbol);

-- System Events Log
CREATE TABLE IF NOT EXISTS system_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_type TEXT NOT NULL,
    agent_name TEXT,
    details TEXT,  -- JSON
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_events_type ON system_events(event_type);
CREATE INDEX idx_events_created_at ON system_events(created_at);
```

**File**: `storage/database.py`

**Work**:
```python
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from contextlib import contextmanager
from pathlib import Path
from config.settings import settings
from utils.logger import get_logger

logger = get_logger("database")

class Database:
    """Database manager for the application."""

    def __init__(self):
        self.engine = create_engine(
            settings.database_url,
            echo=False,
            connect_args={"check_same_thread": False} if "sqlite" in settings.database_url else {}
        )
        self.SessionLocal = sessionmaker(bind=self.engine)
        logger.info("database_initialized", url=settings.database_url)

    def initialize_schema(self):
        """Initialize database schema from SQL file."""
        schema_path = Path(__file__).parent / "schema.sql"
        with open(schema_path, 'r') as f:
            schema_sql = f.read()

        with self.engine.connect() as conn:
            for statement in schema_sql.split(';'):
                if statement.strip():
                    conn.execute(text(statement))
            conn.commit()

        logger.info("database_schema_initialized")

    @contextmanager
    def get_session(self) -> Session:
        """Get a database session context manager."""
        session = self.SessionLocal()
        try:
            yield session
            session.commit()
        except Exception as e:
            session.rollback()
            logger.error("database_session_error", error=str(e))
            raise
        finally:
            session.close()

# Global database instance
db = Database()
```

**Success Criteria**:
- [ ] Database file created
- [ ] All tables created successfully
- [ ] Indexes created
- [ ] Can connect and query database
- [ ] Session context manager works
- [ ] Foreign keys enforced

---

### 8. Update Main Entry Point
**File**: `main.py`

**Work**:
```python
from alpaca.trading.client import TradingClient
import os
from dotenv import load_dotenv
from utils.logger import setup_logging, get_logger
from config.settings import settings
from storage.database import db

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

def main():
    """Main entry point."""
    try:
        trading_client = initialize_system()
        logger.info("system_ready")

        # System is ready - agents will be added in future phases

    except Exception as e:
        logger.error("system_startup_failed", error=str(e), exc_info=True)
        raise

if __name__ == "__main__":
    main()
```

**Success Criteria**:
- [ ] Script runs without errors
- [ ] Logging outputs correctly
- [ ] Database initialized
- [ ] Alpaca connection successful
- [ ] Settings loaded correctly

---

## Testing Checklist

### Unit Tests
**File**: `tests/test_models.py`
- [ ] Test all data model validation
- [ ] Test invalid data raises errors
- [ ] Test model serialization/deserialization

**File**: `tests/test_config.py`
- [ ] Test settings loading
- [ ] Test missing required env vars
- [ ] Test default values

**File**: `tests/test_database.py`
- [ ] Test database initialization
- [ ] Test session management
- [ ] Test basic CRUD operations

**File**: `tests/test_base_agent.py`
- [ ] Test agent initialization
- [ ] Test abstract method enforcement
- [ ] Test logging functionality

---

## Phase Completion Criteria

### Must Have
- [x] Project structure matches design
- [x] All dependencies installed
- [x] Configuration system working
- [x] Logging outputs structured logs
- [x] All data models defined and validated
- [x] Base agent class implemented
- [x] Database schema created and working
- [x] Main.py initializes system successfully
- [x] Basic unit tests passing

### Nice to Have
- [ ] Database migration system
- [ ] Configuration validation script
- [ ] Development setup script (setup.sh)
- [ ] Pre-commit hooks

---

## Verification Commands

```bash
# Test imports
python -c "from config.settings import settings; print(settings.paper_trading)"

# Test logging
python -c "from utils.logger import setup_logging, get_logger; setup_logging(); logger = get_logger('test'); logger.info('test', key='value')"

# Test database
python -c "from storage.database import db; db.initialize_schema()"

# Test models
python -c "from models.news import NewsArticle; print('Models OK')"

# Run main
python main.py

# Run tests
pytest tests/ -v
```

---

## Next Phase
**Phase 2: News Agent** - Implement news fetching and storage functionality
