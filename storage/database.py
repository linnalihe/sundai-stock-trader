from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from contextlib import contextmanager
from pathlib import Path
from config.settings import settings
from utils.logger import get_logger

logger = get_logger("database")

class Database:
    """Database manager for the application."""

    def __init__(self, db_path: str = None):
        # Use custom db_path for testing, otherwise use settings
        if db_path:
            db_url = f"sqlite:///{db_path}"
        else:
            db_url = settings.database_url

        self.engine = create_engine(
            db_url,
            echo=False,
            connect_args={"check_same_thread": False} if "sqlite" in db_url else {}
        )
        self.SessionLocal = sessionmaker(bind=self.engine)
        logger.info("database_initialized", url=db_url)

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
