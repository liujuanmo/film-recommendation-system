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
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import relationship
from sqlalchemy.types import UserDefinedType
from .constants import DEFAULT_MODEL_DIMENSION

Base = declarative_base()


# Custom Vector type for pgvector
class Vector(UserDefinedType):
    def __init__(self, dim=None):
        self.dim = dim

    def get_col_spec(self):
        if self.dim is None:
            return "VECTOR"
        return f"VECTOR({self.dim})"

    def bind_processor(self, dialect):
        def process(value):
            if value is None:
                return value
            if isinstance(value, list):
                return str(value)
            elif hasattr(value, "tolist"):
                return str(value.tolist())
            return str(value)

        return process

    def result_processor(self, dialect, coltype):
        def process(value):
            if value is None:
                return value
            if isinstance(value, str):
                value = value.strip("[]")
                return [float(x) for x in value.split(",")]
            return value

        return process


class Movie(Base):
    __tablename__ = "movies"
    id = Column(Integer, primary_key=True, index=True)
    tconst = Column(String(20), unique=True, nullable=False, index=True)
    primary_title = Column(Text, nullable=False)
    start_year = Column(Integer, index=True)
    genres = Column(ARRAY(String), index=True)
    overview = Column(Text)
    # Relationships
    directors = relationship("MovieDirector", back_populates="movie")
    actors = relationship("MovieActor", back_populates="movie")
    embedding = relationship("MovieEmbedding", back_populates="movie", uselist=False)


class Director(Base):
    __tablename__ = "directors"
    id = Column(Integer, primary_key=True, index=True)
    nconst = Column(String(20), unique=True, nullable=False, index=True)
    primary_name = Column(Text, nullable=False)
    # Relationships
    movies = relationship("MovieDirector", back_populates="director")


class Actor(Base):
    __tablename__ = "actors"
    id = Column(Integer, primary_key=True, index=True)
    nconst = Column(String(20), unique=True, nullable=False, index=True)
    primary_name = Column(Text, nullable=False)
    # Relationships
    movies = relationship("MovieActor", back_populates="actor")


class MovieDirector(Base):
    __tablename__ = "movie_directors"
    movie_id = Column(
        Integer, ForeignKey("movies.id", ondelete="CASCADE"), primary_key=True
    )
    director_id = Column(
        Integer, ForeignKey("directors.id", ondelete="CASCADE"), primary_key=True
    )
    # Relationships
    movie = relationship("Movie", back_populates="directors")
    director = relationship("Director", back_populates="movies")


class MovieActor(Base):
    __tablename__ = "movie_actors"
    movie_id = Column(
        Integer, ForeignKey("movies.id", ondelete="CASCADE"), primary_key=True
    )
    actor_id = Column(
        Integer, ForeignKey("actors.id", ondelete="CASCADE"), primary_key=True
    )
    # Relationships
    movie = relationship("Movie", back_populates="actors")
    actor = relationship("Actor", back_populates="movies")


class MovieEmbedding(Base):
    __tablename__ = "movie_embeddings"
    movie_id = Column(
        Integer, ForeignKey("movies.id", ondelete="CASCADE"), primary_key=True
    )
    embedding = Column(Vector(None))  # Dynamic dimension
    # Relationships
    movie = relationship("Movie", back_populates="embedding")


class TransformerMetadata(Base):
    """Store transformer metadata to avoid loading all data into memory."""

    __tablename__ = "transformer_metadata"
    id = Column(Integer, primary_key=True)
    metadata_type = Column(String(50), nullable=False, unique=True)
    data = Column(JSON, nullable=False)
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())


class PersonEmbedding(Base):
    """Store embeddings for directors and actors."""

    __tablename__ = "person_embeddings"
    id = Column(Integer, primary_key=True)
    person_name = Column(String(255), nullable=False, index=True)
    person_type = Column(String(20), nullable=False)  # 'director' or 'actor'
    embedding = Column(Vector(DEFAULT_MODEL_DIMENSION))
    __table_args__ = (
        Index("idx_person_name_type", "person_name", "person_type", unique=True),
    )


# Add indexes
Index(
    "idx_movie_embeddings_embedding",
    MovieEmbedding.embedding,
    postgresql_using="ivfflat",
    postgresql_ops={"embedding": "vector_cosine_ops"},
)
