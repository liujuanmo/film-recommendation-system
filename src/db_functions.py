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
    Director,
    Actor,
    MovieDirector,
    MovieActor,
    MovieEmbedding,
    TransformerMetadata,
    PersonEmbedding,
    Vector,
)
from .db_connect import get_engine, get_session, get_connection


def enable_pgvector_and_create_tables():
    with get_connection() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
        conn.commit()

    Base.metadata.create_all(bind=get_engine())


def init_embeddings_table(emb_dim: int):
    with get_connection() as conn:
        conn.execute(text("DROP TABLE IF EXISTS movie_embeddings CASCADE;"))

        conn.execute(
            text(
                f"""
            CREATE TABLE movie_embeddings (
                movie_id INTEGER PRIMARY KEY REFERENCES movies(id) ON DELETE CASCADE,
                embedding VECTOR({emb_dim})
            );
        """
            )
        )

        conn.execute(
            text(
                """
            CREATE INDEX IF NOT EXISTS movie_embeddings_embedding_idx 
            ON movie_embeddings USING ivfflat (embedding vector_cosine_ops)
            WITH (lists = 100);
        """
            )
        )

        conn.execute(text("DROP TABLE IF EXISTS person_embeddings CASCADE;"))

        # Create the person_embeddings table with correct dimensions for sentence transformers
        conn.execute(
            text(
                f"""
            CREATE TABLE person_embeddings (
                id SERIAL PRIMARY KEY,
                person_name VARCHAR(255) NOT NULL,
                person_type VARCHAR(20) NOT NULL,
                embedding VECTOR({DEFAULT_MODEL_DIMENSION})
            );
        """
            )
        )

        # Create indexes
        conn.execute(
            text(
                """
            CREATE INDEX IF NOT EXISTS idx_person_name_type 
            ON person_embeddings (person_name, person_type);
        """
            )
        )

        conn.execute(
            text(
                """
            CREATE UNIQUE INDEX IF NOT EXISTS idx_person_name_type_unique 
            ON person_embeddings (person_name, person_type);
        """
            )
        )

        conn.commit()


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
                existing.overview = row.get("overview", "")
            else:
                # Create new movie
                movie = Movie(
                    tconst=row["tconst"],
                    primary_title=row["primaryTitle"],
                    start_year=(
                        int(row["startYear"]) if pd.notna(row["startYear"]) else None
                    ),
                    genres=genres,
                    overview=row.get("overview", ""),
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
            Movie.overview,
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
                    "overview": movie.overview,
                    "directors": [d[0] for d in directors],
                    "cast": [a[0] for a in actors],
                }
            )

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

            result.append(
                {
                    "id": movie.id,
                    "tconst": movie.tconst,
                    "primary_title": movie.primary_title,
                    "start_year": movie.start_year,
                    "genres": movie.genres or [],
                    "overview": movie.overview,
                    "directors": directors,
                    "cast": actors,
                }
            )

        return result
    finally:
        session.close()


def stream_movie_features(batch_size=1000):
    """Yield batches of movie features (genres, directors, cast, years) for embedding computation."""
    session = get_session()
    try:
        total = session.query(Movie).filter(Movie.genres != None).count()
        for offset in range(0, total, batch_size):
            # Get movies with directors and actors in a single query using joins
            movies_with_details = (
                session.query(
                    Movie.id,
                    Movie.genres,
                    Movie.start_year,
                    Director.primary_name.label("director_name"),
                    Actor.primary_name.label("actor_name"),
                )
                .outerjoin(MovieDirector, Movie.id == MovieDirector.movie_id)
                .outerjoin(Director, MovieDirector.director_id == Director.id)
                .outerjoin(MovieActor, Movie.id == MovieActor.movie_id)
                .outerjoin(Actor, MovieActor.actor_id == Actor.id)
                .filter(Movie.genres != None)
                .order_by(Movie.id)
                .offset(offset)
                .limit(
                    batch_size * 10
                )  # Multiply by 10 to account for multiple directors/actors per movie
                .all()
            )

            # Group by movie
            movie_dict = {}
            for row in movies_with_details:
                movie_id = row.id
                if movie_id not in movie_dict:
                    movie_dict[movie_id] = {
                        "genres": row.genres or [],
                        "directors": set(),
                        "cast": set(),
                        "start_year": row.start_year,
                    }

                if row.director_name:
                    movie_dict[movie_id]["directors"].add(row.director_name)
                if row.actor_name:
                    movie_dict[movie_id]["cast"].add(row.actor_name)

            # Convert sets to lists and create batch
            batch = []
            for movie_data in movie_dict.values():
                batch.append(
                    {
                        "genres": movie_data["genres"],
                        "directors": list(movie_data["directors"]),
                        "cast": list(movie_data["cast"]),
                        "start_year": movie_data["start_year"],
                    }
                )

            if batch:
                yield batch
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


def bulk_insert_movies(movies_df):
    """Bulk insert movies data into the database, ignoring duplicates by tconst."""
    session = get_session()
    try:
        # Prepare data for insert
        values = []
        for _, row in movies_df.iterrows():
            genres = row.get("genres", [])
            if isinstance(genres, str):
                genres = genres.split(",") if genres else []
            values.append(
                {
                    "tconst": row["tconst"],
                    "primary_title": row["primaryTitle"],
                    "start_year": row.get("startYear"),
                    "genres": genres,
                    "overview": row.get("overview", row["primaryTitle"]),
                }
            )
        if not values:
            return
        insert_stmt = (
            pg_insert(Movie.__table__)
            .values(values)
            .on_conflict_do_nothing(index_elements=["tconst"])
        )
        session.execute(insert_stmt)
        session.commit()
        print(f"✅ Bulk inserted {len(values)} movies (duplicates ignored)")
    except Exception as e:
        session.rollback()
        print(f"❌ Error bulk inserting movies: {e}")
        raise
    finally:
        session.close()


def bulk_insert_directors(directors_df):
    """Bulk insert directors data into the database, ignoring duplicates by nconst."""
    session = get_session()
    try:
        values = []
        for _, row in directors_df.iterrows():
            values.append(
                {
                    "nconst": row["nconst"],
                    "primary_name": row["primaryName"],
                }
            )

        if not values:
            return
        insert_stmt = (
            pg_insert(Director.__table__)
            .values(values)
            .on_conflict_do_nothing(index_elements=["nconst"])
        )
        session.execute(insert_stmt)
        session.commit()
        print(f"✅ Bulk inserted {len(values)} directors (duplicates ignored)")
    except Exception as e:
        session.rollback()
        print(f"❌ Error bulk inserting directors: {e}")
        raise
    finally:
        session.close()


def bulk_insert_actors(actors_df):
    """Bulk insert actors data into the database, ignoring duplicates by nconst."""
    session = get_session()
    try:
        values = []
        for _, row in actors_df.iterrows():
            values.append(
                {
                    "nconst": row["nconst"],
                    "primary_name": row["primaryName"],
                }
            )
        if not values:
            return
        insert_stmt = (
            pg_insert(Actor.__table__)
            .values(values)
            .on_conflict_do_nothing(index_elements=["nconst"])
        )
        session.execute(insert_stmt)
        session.commit()
        print(f"✅ Bulk inserted {len(values)} actors (duplicates ignored)")
    except Exception as e:
        session.rollback()
        print(f"❌ Error bulk inserting actors: {e}")
        raise
    finally:
        session.close()


def bulk_insert_movie_directors(movie_directors_data):
    """Bulk insert movie-director relationships, ignoring duplicates."""
    session = get_session()
    try:
        # Get movie and director IDs
        movies = session.query(Movie.tconst, Movie.id).all()
        movie_ids = {m.tconst: m.id for m in movies}
        directors = session.query(Director.nconst, Director.id).all()
        director_ids = {d.nconst: d.id for d in directors}
        values = []
        for tconst, director_nconst in movie_directors_data:
            movie_id = movie_ids.get(tconst)
            director_id = director_ids.get(director_nconst)
            if movie_id and director_id:
                values.append(
                    {
                        "movie_id": movie_id,
                        "director_id": director_id,
                    }
                )
        if not values:
            return
        insert_stmt = (
            pg_insert(MovieDirector.__table__)
            .values(values)
            .on_conflict_do_nothing(index_elements=["movie_id", "director_id"])
        )
        session.execute(insert_stmt)
        session.commit()
        print(
            f"✅ Bulk inserted {len(values)} movie-director relationships (duplicates ignored)"
        )
    except Exception as e:
        session.rollback()
        print(f"❌ Error bulk inserting movie-director relationships: {e}")
        raise
    finally:
        session.close()


def bulk_insert_movie_actors(movie_actors_data):
    """Bulk insert movie-actor relationships, ignoring duplicates."""
    session = get_session()
    try:
        movies = session.query(Movie.tconst, Movie.id).all()
        movie_ids = {m.tconst: m.id for m in movies}
        actors = session.query(Actor.nconst, Actor.id).all()
        actor_ids = {a.nconst: a.id for a in actors}
        values = []
        for tconst, actor_nconst in movie_actors_data:
            movie_id = movie_ids.get(tconst)
            actor_id = actor_ids.get(actor_nconst)
            if movie_id and actor_id:
                values.append(
                    {
                        "movie_id": movie_id,
                        "actor_id": actor_id,
                    }
                )
        if not values:
            return
        insert_stmt = (
            pg_insert(MovieActor.__table__)
            .values(values)
            .on_conflict_do_nothing(index_elements=["movie_id", "actor_id"])
        )
        session.execute(insert_stmt)
        session.commit()
        print(
            f"✅ Bulk inserted {len(values)} movie-actor relationships (duplicates ignored)"
        )
    except Exception as e:
        session.rollback()
        print(f"❌ Error bulk inserting movie-actor relationships: {e}")
        raise
    finally:
        session.close()
