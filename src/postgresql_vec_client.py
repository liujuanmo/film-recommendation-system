from sqlalchemy import create_engine, Column, Integer, String, Text, ARRAY, ForeignKey, Index, func, DateTime, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.sql import text
from sqlalchemy.types import UserDefinedType
import numpy as np
import pandas as pd
import os
from typing import List, Tuple, Optional, Dict, Any
import json

# Database connection parameters - can be configured via environment variables
DB_HOST = os.getenv('DB_HOST', 'localhost')
DB_PORT = os.getenv('DB_PORT', '5432')
DB_NAME = os.getenv('DB_NAME', 'movie_recommendations')
DB_USER = os.getenv('DB_USER', 'postgres')
DB_PASSWORD = os.getenv('DB_PASSWORD', '12345678')

# Create database URL
DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# Create custom Vector type for pgvector
class Vector(UserDefinedType):
    """Custom SQLAlchemy type for pgvector."""
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
            elif hasattr(value, 'tolist'):
                return str(value.tolist())
            return str(value)
        return process

    def result_processor(self, dialect, coltype):
        def process(value):
            if value is None:
                return value
            # pgvector returns vectors as strings like '[1,2,3]'
            if isinstance(value, str):
                # Remove brackets and split by comma
                value = value.strip('[]')
                return [float(x) for x in value.split(',')]
            return value
        return process

# Create SQLAlchemy engine
engine = create_engine(DATABASE_URL)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# SQLAlchemy Models
class Movie(Base):
    __tablename__ = "movies"
    
    id = Column(Integer, primary_key=True, index=True)
    tconst = Column(String(20), unique=True, nullable=False, index=True)
    primary_title = Column(Text, nullable=False)
    start_year = Column(Integer, index=True)
    genres = Column(ARRAY(String), index=True)
    title_type = Column(String(50))
    overview = Column(Text)
    keywords = Column(ARRAY(String))
    
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
    
    movie_id = Column(Integer, ForeignKey("movies.id", ondelete="CASCADE"), primary_key=True)
    director_id = Column(Integer, ForeignKey("directors.id", ondelete="CASCADE"), primary_key=True)
    
    # Relationships
    movie = relationship("Movie", back_populates="directors")
    director = relationship("Director", back_populates="movies")

class MovieActor(Base):
    __tablename__ = "movie_actors"
    
    movie_id = Column(Integer, ForeignKey("movies.id", ondelete="CASCADE"), primary_key=True)
    actor_id = Column(Integer, ForeignKey("actors.id", ondelete="CASCADE"), primary_key=True)
    
    # Relationships
    movie = relationship("Movie", back_populates="actors")
    actor = relationship("Actor", back_populates="movies")

class MovieEmbedding(Base):
    __tablename__ = "movie_embeddings"
    
    movie_id = Column(Integer, ForeignKey("movies.id", ondelete="CASCADE"), primary_key=True)
    embedding = Column(Vector(None))  # Dynamic dimension
    
    # Relationships
    movie = relationship("Movie", back_populates="embedding")

class TransformerMetadata(Base):
    """Store transformer metadata to avoid loading all data into memory."""
    __tablename__ = "transformer_metadata"
    
    id = Column(Integer, primary_key=True)
    metadata_type = Column(String(50), nullable=False, unique=True)  # 'genre_classes', 'year_stats', 'tfidf_vocab', etc.
    data = Column(JSON, nullable=False)  # Store the actual metadata as JSON
    created_at = Column(DateTime, default=func.now())
    updated_at = Column(DateTime, default=func.now(), onupdate=func.now())

class PersonEmbedding(Base):
    """Store embeddings for directors and actors."""
    __tablename__ = "person_embeddings"
    
    id = Column(Integer, primary_key=True)
    person_name = Column(String(255), nullable=False, index=True)
    person_type = Column(String(20), nullable=False)  # 'director' or 'actor'
    embedding = Column(Vector(32))  # Fixed dimension for person embeddings
    
    __table_args__ = (
        Index('idx_person_name_type', 'person_name', 'person_type', unique=True),
    )

# Add indexes
Index('idx_movie_embeddings_embedding', MovieEmbedding.embedding, postgresql_using='ivfflat', postgresql_ops={'embedding': 'vector_cosine_ops'})

def get_engine():
    """Get SQLAlchemy engine."""
    return engine

def get_session():
    """Get SQLAlchemy session."""
    return SessionLocal()

def get_connection():
    """Get raw connection for compatibility."""
    return engine.raw_connection()

def init_postgresql_tables():
    """Initialize the PostgreSQL database with all necessary tables."""
    try:
        # Enable pgvector extension
        with engine.connect() as conn:
            conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
            conn.commit()
        
        # Create all tables
        Base.metadata.create_all(bind=engine)
        print("✅ Database tables initialized successfully")
        
    except Exception as e:
        print(f"❌ Error initializing tables: {e}")
        raise

def init_embeddings_table(emb_dim: int):
    """Initialize the movie embeddings table with specific dimension."""
    try:
        # Drop existing table if dimension changed
        with engine.connect() as conn:
            conn.execute(text("DROP TABLE IF EXISTS movie_embeddings CASCADE;"))
            
            # Create the movie_embeddings table with specific vector dimension
            conn.execute(text(f"""
                CREATE TABLE movie_embeddings (
                    movie_id INTEGER PRIMARY KEY REFERENCES movies(id) ON DELETE CASCADE,
                    embedding VECTOR({emb_dim})
                );
            """))
            
            # Create the vector index
            conn.execute(text("""
                CREATE INDEX IF NOT EXISTS movie_embeddings_embedding_idx 
                ON movie_embeddings USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = 100);
            """))
            
            conn.commit()
            
        print(f"✅ Embeddings table initialized with dimension {emb_dim}")
        
    except Exception as e:
        print(f"❌ Error initializing embeddings table: {e}")
        raise

def insert_embeddings(ids: List[int], embeddings: List):
    """Insert movie embeddings into the database."""
    session = get_session()
    try:
        # Prepare data for batch insert
        embedding_objects = []
        for movie_id, emb in zip(ids, embeddings):
            if hasattr(emb, 'tolist'):
                emb = emb.tolist()
            elif isinstance(emb, np.ndarray):
                emb = emb.tolist()
            
            # Check if embedding already exists
            existing = session.query(MovieEmbedding).filter_by(movie_id=movie_id).first()
            if existing:
                existing.embedding = emb
            else:
                embedding_objects.append(MovieEmbedding(movie_id=movie_id, embedding=emb))
        
        # Add new embeddings
        if embedding_objects:
            session.add_all(embedding_objects)
        
        session.commit()
        
    except Exception as e:
        session.rollback()
        raise e
    finally:
        session.close()

def search(query_vec, top_n: int = 10) -> List[Tuple[int, float]]:
    """Search for similar movie embeddings using cosine similarity."""
    if hasattr(query_vec, 'tolist'):
        query_vec = query_vec.tolist()
    elif isinstance(query_vec, np.ndarray):
        query_vec = query_vec.tolist()
    
    # Get raw connection for direct SQL execution
    connection = engine.raw_connection()
    try:
        with connection.cursor() as cursor:
            # Convert query vector to string format for pgvector
            query_vec_str = str(query_vec)
            
            # Use cosine distance for similarity search
            cursor.execute("""
                SELECT movie_id, (embedding <=> %s::vector) AS distance
                FROM movie_embeddings
                ORDER BY embedding <=> %s::vector
                LIMIT %s;
            """, (query_vec_str, query_vec_str, top_n))
            
            return cursor.fetchall()
        
    except Exception as e:
        raise e
    finally:
        connection.close()

# New functions for transformer metadata
def store_transformer_metadata(metadata_type: str, data: dict):
    """Store transformer metadata in the database."""
    session = get_session()
    try:
        # Check if metadata already exists
        existing = session.query(TransformerMetadata).filter_by(metadata_type=metadata_type).first()
        if existing:
            existing.data = data
            existing.updated_at = func.now()
        else:
            metadata = TransformerMetadata(metadata_type=metadata_type, data=data)
            session.add(metadata)
        
        session.commit()
    except Exception as e:
        session.rollback()
        raise e
    finally:
        session.close()

def get_transformer_metadata(metadata_type: str) -> dict:
    """Retrieve transformer metadata from the database."""
    session = get_session()
    try:
        metadata = session.query(TransformerMetadata).filter_by(metadata_type=metadata_type).first()
        return metadata.data if metadata else None
    finally:
        session.close()

def store_person_embeddings(person_embeddings: dict, person_type: str):
    """Store director or actor embeddings in the database."""
    session = get_session()
    try:
        embedding_objects = []
        for person_name, embedding in person_embeddings.items():
            if hasattr(embedding, 'tolist'):
                embedding = embedding.tolist()
            
            # Check if embedding already exists
            existing = session.query(PersonEmbedding).filter_by(
                person_name=person_name, person_type=person_type
            ).first()
            
            if existing:
                existing.embedding = embedding
            else:
                embedding_objects.append(PersonEmbedding(
                    person_name=person_name,
                    person_type=person_type,
                    embedding=embedding
                ))
        
        if embedding_objects:
            session.add_all(embedding_objects)
        
        session.commit()
    except Exception as e:
        session.rollback()
        raise e
    finally:
        session.close()

def get_person_embeddings(person_names: List[str], person_type: str) -> dict:
    """Get embeddings for a list of persons from the database."""
    session = get_session()
    try:
        embeddings = session.query(PersonEmbedding).filter(
            PersonEmbedding.person_name.in_(person_names),
            PersonEmbedding.person_type == person_type
        ).all()
        
        result = {}
        for emb in embeddings:
            result[emb.person_name] = emb.embedding
        
        return result
    finally:
        session.close()

def get_embedding_count() -> int:
    """Get the count of embeddings in the database."""
    session = get_session()
    try:
        count = session.query(MovieEmbedding).count()
        return count
    finally:
        session.close()

def table_exists(table_name: str = 'movie_embeddings') -> bool:
    """Check if a table exists."""
    session = get_session()
    try:
        result = session.execute(text("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = :table_name
            );
        """), {'table_name': table_name})
        
        exists = result.fetchone()[0]
        return exists
    finally:
        session.close()

def get_movie_count() -> int:
    """Get the count of movies in the database."""
    session = get_session()
    try:
        count = session.query(Movie).count()
        return count
    finally:
        session.close()

def load_movies_data(movies_df):
    """Load movies data into the database."""
    session = get_session()
    try:
        for _, row in movies_df.iterrows():
            genres = row['genres'] if isinstance(row['genres'], list) else []
            keywords = row.get('keywords', []) if isinstance(row.get('keywords'), list) else []
            
            # Check if movie already exists
            existing = session.query(Movie).filter_by(tconst=row['tconst']).first()
            if existing:
                # Update existing movie
                existing.primary_title = row['primaryTitle']
                existing.start_year = int(row['startYear']) if pd.notna(row['startYear']) else None
                existing.genres = genres
                existing.title_type = row.get('titleType', 'movie')
                existing.overview = row.get('overview', '')
                existing.keywords = keywords
            else:
                # Create new movie
                movie = Movie(
                    tconst=row['tconst'],
                    primary_title=row['primaryTitle'],
                    start_year=int(row['startYear']) if pd.notna(row['startYear']) else None,
                    genres=genres,
                    title_type=row.get('titleType', 'movie'),
                    overview=row.get('overview', ''),
                    keywords=keywords
                )
                session.add(movie)
        
        session.commit()
        
    except Exception as e:
        session.rollback()
        raise e
    finally:
        session.close()

def load_directors_data(directors_df):
    """Load directors data into the database."""
    session = get_session()
    try:
        for _, row in directors_df.iterrows():
            # Check if director already exists
            existing = session.query(Director).filter_by(nconst=row['nconst']).first()
            if existing:
                existing.primary_name = row['primaryName']
            else:
                director = Director(
                    nconst=row['nconst'],
                    primary_name=row['primaryName']
                )
                session.add(director)
        
        session.commit()
        
    except Exception as e:
        session.rollback()
        raise e
    finally:
        session.close()

def load_actors_data(actors_df):
    """Load actors data into the database."""
    session = get_session()
    try:
        for _, row in actors_df.iterrows():
            # Check if actor already exists
            existing = session.query(Actor).filter_by(nconst=row['nconst']).first()
            if existing:
                existing.primary_name = row['primaryName']
            else:
                actor = Actor(
                    nconst=row['nconst'],
                    primary_name=row['primaryName']
                )
                session.add(actor)
        
        session.commit()
        
    except Exception as e:
        session.rollback()
        raise e
    finally:
        session.close()

def load_movie_directors(movie_directors_data):
    """Load movie-director relationships."""
    session = get_session()
    try:
        for tconst, nconst in movie_directors_data:
            # Get movie and director IDs
            movie = session.query(Movie).filter_by(tconst=tconst).first()
            director = session.query(Director).filter_by(nconst=nconst).first()
            
            if movie and director:
                # Check if relationship already exists
                existing = session.query(MovieDirector).filter_by(
                    movie_id=movie.id, 
                    director_id=director.id
                ).first()
                
                if not existing:
                    movie_director = MovieDirector(
                        movie_id=movie.id,
                        director_id=director.id
                    )
                    session.add(movie_director)
        
        session.commit()
        
    except Exception as e:
        session.rollback()
        raise e
    finally:
        session.close()

def load_movie_actors(movie_actors_data):
    """Load movie-actor relationships."""
    session = get_session()
    try:
        for tconst, nconst in movie_actors_data:
            # Get movie and actor IDs
            movie = session.query(Movie).filter_by(tconst=tconst).first()
            actor = session.query(Actor).filter_by(nconst=nconst).first()
            
            if movie and actor:
                # Check if relationship already exists
                existing = session.query(MovieActor).filter_by(
                    movie_id=movie.id, 
                    actor_id=actor.id
                ).first()
                
                if not existing:
                    movie_actor = MovieActor(
                        movie_id=movie.id,
                        actor_id=actor.id
                    )
                    session.add(movie_actor)
        
        session.commit()
        
    except Exception as e:
        session.rollback()
        raise e
    finally:
        session.close()

def get_movies_with_details() -> List[Dict[str, Any]]:
    """Get all movies with their directors and actors."""
    session = get_session()
    try:
        # Use SQLAlchemy to get movies with details
        query = session.query(
            Movie.id,
            Movie.tconst,
            Movie.primary_title,
            Movie.start_year,
            Movie.genres,
            Movie.title_type,
            Movie.overview,
            Movie.keywords
        ).filter(
            Movie.title_type == 'movie'
        ).filter(
            Movie.genres != None
        )
        
        movies = query.all()
        result = []
        
        for movie in movies:
            # Get directors for this movie
            directors = session.query(Director.primary_name).join(
                MovieDirector, Director.id == MovieDirector.director_id
            ).filter(MovieDirector.movie_id == movie.id).all()
            
            # Get actors for this movie
            actors = session.query(Actor.primary_name).join(
                MovieActor, Actor.id == MovieActor.actor_id
            ).filter(MovieActor.movie_id == movie.id).all()
            
            result.append({
                'id': movie.id,
                'tconst': movie.tconst,
                'primary_title': movie.primary_title,
                'start_year': movie.start_year,
                'genres': movie.genres or [],
                'title_type': movie.title_type,
                'overview': movie.overview,
                'keywords': movie.keywords or [],
                'directors': [d[0] for d in directors],
                'cast': [a[0] for a in actors]
            })
        
        return result
        
    except Exception as e:
        raise e
    finally:
        session.close() 

def get_movie_details_by_ids(movie_ids: List[int]) -> List[dict]:
    """Get movie details for specific movie IDs."""
    session = get_session()
    try:
        movies = session.query(Movie).filter(Movie.id.in_(movie_ids)).all()
        
        result = []
        for movie in movies:
            # Get directors
            directors = [md.director.primary_name for md in movie.directors]
            # Get actors
            actors = [ma.actor.primary_name for ma in movie.actors]
            
            result.append({
                'id': movie.id,
                'tconst': movie.tconst,
                'primary_title': movie.primary_title,
                'start_year': movie.start_year,
                'genres': movie.genres or [],
                'title_type': movie.title_type,
                'overview': movie.overview,
                'keywords': movie.keywords or [],
                'directors': directors,
                'cast': actors
            })
        
        return result
    finally:
        session.close() 