from sqlalchemy import (
    func,
    text,
)
from sqlalchemy.dialects.postgresql import insert as pg_insert
import numpy as np
import pandas as pd
import os
from typing import List, Tuple, Optional, Dict, Any
import json
from .constants import DEFAULT_MODEL_DIMENSION
from .db_models import (
    Base,
    Movie,
    TransformerMetadata,
    PersonEmbedding,
)
from .db_connect import get_engine, get_session, get_connection


def create_tables():
    Base.metadata.create_all(bind=get_engine())


def init_embeddings_table(emb_dim: int):
    conn = get_connection()
    try:
        with conn.cursor() as cursor:
            cursor.execute("DROP TABLE IF EXISTS movie_embeddings CASCADE;")

            cursor.execute(
                f"""
            CREATE TABLE movie_embeddings (
                movie_id INTEGER PRIMARY KEY REFERENCES movies(id) ON DELETE CASCADE,
                embedding VECTOR({emb_dim})
            );
        """
            )

            cursor.execute(
                """
            CREATE INDEX IF NOT EXISTS movie_embeddings_embedding_idx 
            ON movie_embeddings USING ivfflat (embedding vector_cosine_ops)
            WITH (lists = 100);
        """
            )

            cursor.execute("DROP TABLE IF EXISTS person_embeddings CASCADE;")

            # Create the person_embeddings table with correct dimensions for sentence transformers
            cursor.execute(
                f"""
            CREATE TABLE person_embeddings (
                id SERIAL PRIMARY KEY,
                person_name VARCHAR(255) NOT NULL,
                person_type VARCHAR(20) NOT NULL,
                embedding VECTOR({DEFAULT_MODEL_DIMENSION})
            );
        """
            )

            # Create indexes
            cursor.execute(
                """
            CREATE INDEX IF NOT EXISTS idx_person_name_type 
            ON person_embeddings (person_name, person_type);
        """
            )

            cursor.execute(
                """
            CREATE UNIQUE INDEX IF NOT EXISTS idx_person_name_type_unique 
            ON person_embeddings (person_name, person_type);
        """
            )

        conn.commit()
    finally:
        conn.close()


def insert_embeddings(ids: List[int], embeddings: List):
    """Insert movie embeddings into the database."""
    session = get_session()
    try:
        # Prepare data for batch insert
        embedding_objects = []
        for movie_id, emb in zip(ids, embeddings):
            if hasattr(emb, "tolist"):
                emb = emb.tolist()
            elif isinstance(emb, np.ndarray):
                emb = emb.tolist()

            # Check if embedding already exists
            existing = (
                session.query(MovieEmbedding).filter_by(movie_id=movie_id).first()
            )
            if existing:
                existing.embedding = emb
            else:
                embedding_objects.append(
                    MovieEmbedding(movie_id=movie_id, embedding=emb)
                )

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
    if hasattr(query_vec, "tolist"):
        query_vec = query_vec.tolist()
    elif isinstance(query_vec, np.ndarray):
        query_vec = query_vec.tolist()

    # Get raw connection for direct SQL execution
    connection = get_connection()
    try:
        with connection.cursor() as cursor:
            # Convert query vector to string format for pgvector
            query_vec_str = str(query_vec)

            # Use cosine distance for similarity search
            cursor.execute(
                """
                SELECT movie_id, (embedding <=> %s::vector) AS distance
                FROM movie_embeddings
                ORDER BY embedding <=> %s::vector
                LIMIT %s;
            """,
                (query_vec_str, query_vec_str, top_n),
            )

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
        existing = (
            session.query(TransformerMetadata)
            .filter_by(metadata_type=metadata_type)
            .first()
        )
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
        metadata = (
            session.query(TransformerMetadata)
            .filter_by(metadata_type=metadata_type)
            .first()
        )
        return metadata.data if metadata else None
    finally:
        session.close()


def store_person_embeddings(person_embeddings: dict, person_type: str):
    """Store director or actor embeddings in the database."""
    session = get_session()
    try:
        embedding_objects = []
        for person_name, embedding in person_embeddings.items():
            if hasattr(embedding, "tolist"):
                embedding = embedding.tolist()

            # Check if embedding already exists
            existing = (
                session.query(PersonEmbedding)
                .filter_by(person_name=person_name, person_type=person_type)
                .first()
            )

            if existing:
                existing.embedding = embedding
            else:
                embedding_objects.append(
                    PersonEmbedding(
                        person_name=person_name,
                        person_type=person_type,
                        embedding=embedding,
                    )
                )

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
        embeddings = (
            session.query(PersonEmbedding)
            .filter(
                PersonEmbedding.person_name.in_(person_names),
                PersonEmbedding.person_type == person_type,
            )
            .all()
        )

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


def table_exists(table_name: str = "movie_embeddings") -> bool:
    """Check if a table exists."""
    session = get_session()
    try:
        result = session.execute(
            text(
                """
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = :table_name
            );
        """
            ),
            {"table_name": table_name},
        )

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


def insert_movies(movies_df):
    """Load movies data into the database."""
    session = get_session()
    try:
        for _, row in movies_df.iterrows():
            genres = row["genres"] if isinstance(row["genres"], list) else []

            # Check if movie already exists
            existing = session.query(Movie).filter_by(tconst=row["tconst"]).first()
            if existing:
                # Update existing movie
                existing.primary_title = row["primaryTitle"]
                existing.start_year = (
                    int(row["startYear"]) if pd.notna(row["startYear"]) else None
                )
                existing.genres = genres
            else:
                # Create new movie
                movie = Movie(
                    tconst=row["tconst"],
                    primary_title=row["primaryTitle"],
                    start_year=(
                        int(row["startYear"]) if pd.notna(row["startYear"]) else None
                    ),
                    genres=genres,
                )
                session.add(movie)

        session.commit()

    except Exception as e:
        session.rollback()
        raise e
    finally:
        session.close()


def insert_directors(directors_df):
    """Load directors data into the database."""
    session = get_session()
    try:
        for _, row in directors_df.iterrows():
            # Check if director already exists
            existing = session.query(Director).filter_by(nconst=row["nconst"]).first()
            if existing:
                existing.primary_name = row["primaryName"]
            else:
                director = Director(
                    nconst=row["nconst"], primary_name=row["primaryName"]
                )
                session.add(director)

        session.commit()

    except Exception as e:
        session.rollback()
        raise e
    finally:
        session.close()


def insert_actors(actors_df):
    """Load actors data into the database."""
    session = get_session()
    try:
        for _, row in actors_df.iterrows():
            # Check if actor already exists
            existing = session.query(Actor).filter_by(nconst=row["nconst"]).first()
            if existing:
                existing.primary_name = row["primaryName"]
            else:
                actor = Actor(nconst=row["nconst"], primary_name=row["primaryName"])
                session.add(actor)

        session.commit()

    except Exception as e:
        session.rollback()
        raise e
    finally:
        session.close()


def insert_movie_directors(movie_directors_data):
    """Load movie-director relationships."""
    session = get_session()
    try:
        for tconst, nconst in movie_directors_data:
            # Get movie and director IDs
            movie = session.query(Movie).filter_by(tconst=tconst).first()
            director = session.query(Director).filter_by(nconst=nconst).first()

            if movie and director:
                # Check if relationship already exists
                existing = (
                    session.query(MovieDirector)
                    .filter_by(movie_id=movie.id, director_id=director.id)
                    .first()
                )

                if not existing:
                    movie_director = MovieDirector(
                        movie_id=movie.id, director_id=director.id
                    )
                    session.add(movie_director)

        session.commit()

    except Exception as e:
        session.rollback()
        raise e
    finally:
        session.close()


def insert_movie_actors(movie_actors_data):
    """Load movie-actor relationships."""
    session = get_session()
    try:
        for tconst, nconst in movie_actors_data:
            # Get movie and actor IDs
            movie = session.query(Movie).filter_by(tconst=tconst).first()
            actor = session.query(Actor).filter_by(nconst=nconst).first()

            if movie and actor:
                # Check if relationship already exists
                existing = (
                    session.query(MovieActor)
                    .filter_by(movie_id=movie.id, actor_id=actor.id)
                    .first()
                )

                if not existing:
                    movie_actor = MovieActor(movie_id=movie.id, actor_id=actor.id)
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
        ).filter(Movie.genres != None)

        movies = query.all()
        result = []

        for movie in movies:
            # Get directors for this movie
            directors = (
                session.query(Director.primary_name)
                .join(MovieDirector, Director.id == MovieDirector.director_id)
                .filter(MovieDirector.movie_id == movie.id)
                .all()
            )

            # Get actors for this movie
            actors = (
                session.query(Actor.primary_name)
                .join(MovieActor, Actor.id == MovieActor.actor_id)
                .filter(MovieActor.movie_id == movie.id)
                .all()
            )

            result.append(
                {
                    "id": movie.id,
                    "tconst": movie.tconst,
                    "primary_title": movie.primary_title,
                    "start_year": movie.start_year,
                    "genres": movie.genres or [],
                    "directors": [d[0] for d in directors],
                    "cast": [a[0] for a in actors],
                }
            )

        return result

    except Exception as e:
        raise e
    finally:
        session.close()


def get_year_min_max():
    session = get_session()
    try:
        min_year = session.query(func.min(Movie.start_year)).scalar()
        max_year = session.query(func.max(Movie.start_year)).scalar()
        return min_year, max_year
    finally:
        session.close()


def get_all_unique_genres():
    session = get_session()
    try:
        genres = session.query(Movie.genres).all()
        genre_set = set()
        for row in genres:
            if row[0]:
                genre_set.update(row[0])
        return sorted(g for g in genre_set if g)
    finally:
        session.close()
