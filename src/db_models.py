from sqlalchemy import (
    Column,
    Integer,
    String,
    Text,
    ARRAY,
    ForeignKey,
    Index,
    DateTime,
    JSON,
    func,
    BigInteger,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.types import UserDefinedType
from .constants import DEFAULT_MODEL_DIMENSION

Base = declarative_base()


class Vector(UserDefinedType):
    def __init__(self, dim=None):
        self.dim = dim

    def get_col_spec(self, **kw):
        if self.dim is None:
            return "VECTOR"
        return f"VECTOR({self.dim})"

    def bind_processor(self, dialect):
        def process(value):
            if value is None:
                return None
            if isinstance(value, (list, tuple)):
                return str(value)
            return str(value)

        return process

    def result_processor(self, dialect, coltype):
        def process(value):
            if value is None:
                return None
            if isinstance(value, str):
                # Remove curly braces and split by comma
                value = value.strip("{}").split(",")
                return [float(x) for x in value]
            return value

        return process


class Movie(Base):
    __tablename__ = "movies"
    id = Column(Integer, primary_key=True, index=True)
    tconst = Column(String(20), unique=True, nullable=False, index=True)
    primary_title = Column(Text, nullable=False)
    start_year = Column(BigInteger, index=True)  # Changed from Integer to BigInteger
    genres = Column(ARRAY(String), index=True)
    directors = Column(ARRAY(String), index=True)
    actors = Column(ARRAY(String), index=True)
    embedding = Column(Vector(1181), nullable=False)

    __table_args__ = (
        Index("idx_movies_tconst", "tconst"),
        Index("idx_movies_start_year", "start_year"),
        Index("idx_movies_genres", "genres", postgresql_using="gin"),
        Index("idx_movies_directors", "directors", postgresql_using="gin"),
        Index("idx_movies_actors", "actors", postgresql_using="gin"),
        Index("idx_movies_embedding", "embedding", postgresql_using="ivfflat"),
    )


class TransformerMetadata(Base):
    __tablename__ = "transformer_metadata"
    id = Column(Integer, primary_key=True, index=True)
    metadata_type = Column(String(50), nullable=False, index=True)
    data = Column(JSON, nullable=False)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())

    __table_args__ = (Index("idx_transformer_metadata_type", "metadata_type"),)


class PersonEmbedding(Base):
    __tablename__ = "person_embeddings"
    id = Column(Integer, primary_key=True, index=True)
    person_name = Column(String(255), nullable=False)
    person_type = Column(String(20), nullable=False)  # 'director' or 'actor'
    embedding = Column(Vector(DEFAULT_MODEL_DIMENSION), nullable=False)

    __table_args__ = (
        Index("idx_person_name_type", "person_name", "person_type"),
        Index("idx_person_name_type_unique", "person_name", "person_type", unique=True),
    )
