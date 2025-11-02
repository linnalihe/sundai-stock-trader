import pytest
import os
from sqlalchemy import text
from storage.database import Database

@pytest.fixture
def test_db():
    """Create a test database instance."""
    test_db_path = "test_trading.db"

    # Remove test database if it exists
    if os.path.exists(test_db_path):
        os.remove(test_db_path)

    # Create new test database
    db = Database(db_path=test_db_path)
    db.initialize_schema()

    yield db

    # Cleanup after test
    if os.path.exists(test_db_path):
        os.remove(test_db_path)

def test_database_initialization(test_db):
    """Database schema should be created."""
    # Check tables exist
    with test_db.get_session() as session:
        result = session.execute(
            text("SELECT name FROM sqlite_master WHERE type='table'")
        )
        tables = [row[0] for row in result.fetchall()]

        assert "news_articles" in tables
        assert "analysis_results" in tables
        assert "trading_decisions" in tables
        assert "trade_executions" in tables
        assert "system_events" in tables

def test_database_session_context(test_db):
    """Database session context manager should work."""
    with test_db.get_session() as session:
        # Insert test data
        session.execute(
            text("""
                INSERT INTO system_events (event_type, agent_name, details)
                VALUES (:event_type, :agent_name, :details)
            """),
            {
                "event_type": "test_event",
                "agent_name": "test_agent",
                "details": '{"key": "value"}'
            }
        )

    # Verify data was committed
    with test_db.get_session() as session:
        result = session.execute(
            text("SELECT COUNT(*) FROM system_events WHERE event_type = 'test_event'")
        )
        count = result.scalar()
        assert count == 1

def test_database_session_rollback(test_db):
    """Database session should rollback on error."""
    try:
        with test_db.get_session() as session:
            # This should fail due to constraint violation
            session.execute(
                text("INSERT INTO news_articles (id) VALUES (NULL)")
            )
    except Exception:
        pass

    # Session should have rolled back
    with test_db.get_session() as session:
        result = session.execute(text("SELECT COUNT(*) FROM news_articles"))
        count = result.scalar()
        assert count == 0
