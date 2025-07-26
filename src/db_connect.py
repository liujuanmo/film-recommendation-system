import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from .db_models import Base

# Database connection parameters - can be configured via environment variables
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "movie_recommendations")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "12345678")

# Create database URL
DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# Create SQLAlchemy engine
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def get_db():
    """Get database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_engine():
    """Get SQLAlchemy engine."""
    return engine


def get_session():
    """Get SQLAlchemy session."""
    return SessionLocal()


def get_connection():
    """Get raw connection for compatibility."""
    return engine.raw_connection()
