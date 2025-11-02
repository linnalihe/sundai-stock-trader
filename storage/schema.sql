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

CREATE INDEX IF NOT EXISTS idx_news_published ON news_articles(published_at);
CREATE INDEX IF NOT EXISTS idx_news_symbols ON news_articles(symbols);

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

CREATE INDEX IF NOT EXISTS idx_analysis_symbol ON analysis_results(symbol);
CREATE INDEX IF NOT EXISTS idx_analysis_analyzed_at ON analysis_results(analyzed_at);

-- Trading Decisions Table
CREATE TABLE IF NOT EXISTS trading_decisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    action TEXT NOT NULL,
    quantity INTEGER NOT NULL,
    expected_price REAL NOT NULL,
    confidence TEXT NOT NULL,
    reasoning TEXT NOT NULL,
    sentiment_score REAL NOT NULL,
    high_impact_count INTEGER NOT NULL,
    analysis_count INTEGER NOT NULL,
    decided_at TIMESTAMP NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_decisions_symbol ON trading_decisions(symbol);
CREATE INDEX IF NOT EXISTS idx_decisions_decided_at ON trading_decisions(decided_at);

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

CREATE INDEX IF NOT EXISTS idx_executions_status ON trade_executions(status);
CREATE INDEX IF NOT EXISTS idx_executions_symbol ON trade_executions(symbol);

-- System Events Log
CREATE TABLE IF NOT EXISTS system_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    event_type TEXT NOT NULL,
    agent_name TEXT,
    details TEXT,  -- JSON
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_events_type ON system_events(event_type);
CREATE INDEX IF NOT EXISTS idx_events_created_at ON system_events(created_at);
