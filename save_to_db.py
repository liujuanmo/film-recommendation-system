import os
import pandas as pd
import numpy as np
import json
from src.db_models import Base, Movie
from src.db_functions import create_tables, get_session, get_engine
from src.db_connect import get_connection
from sqlalchemy.orm import sessionmaker
from datetime import datetime


def load_merged_csv(csv_file):
    """Load the merged CSV file."""
    print(f"🔄 Loading {csv_file}...")

    try:
        df = pd.read_csv(csv_file)
        print(f"✅ Loaded {len(df)} movies from CSV")
        return df

    except Exception as e:
        print(f"❌ Error loading CSV: {e}")
        return None


def load_embeddings_npz(npz_file):
    """Load the embeddings from NPZ file."""
    print(f"🔄 Loading {npz_file}...")

    try:
        data = np.load(npz_file)
        movie_ids = data["movie_ids"]
        embeddings = data["embeddings"]

        # Create a dictionary mapping tconst to embedding
        embedding_dict = dict(zip(movie_ids, embeddings))

        print(f"✅ Loaded {len(embedding_dict)} embeddings")
        return embedding_dict

    except Exception as e:
        print(f"❌ Error loading embeddings: {e}")
        return None


def save_to_database(movies_df, embeddings_dict):
    """Save movies and embeddings to database using existing models."""
    print("💾 Saving to database...")

    # Create session
    session = get_session()

    try:
        batch_size = 1000
        total_movies = len(movies_df)

        for i in range(0, total_movies, batch_size):
            batch_df = movies_df.iloc[i : i + batch_size]

            for _, row in batch_df.iterrows():
                tconst = row["tconst"]

                # Get embedding for this movie
                embedding = embeddings_dict.get(tconst, None)

                # Convert pipe-separated strings back to lists
                genres = (
                    row["genres"].split("|")
                    if pd.notna(row["genres"]) and row["genres"] != ""
                    else []
                )
                directors = (
                    row["directors"].split("|")
                    if pd.notna(row["directors"]) and row["directors"] != ""
                    else []
                )
                actors = (
                    row["actors"].split("|")
                    if pd.notna(row["actors"]) and row["actors"] != ""
                    else []
                )

                # Check if movie already exists
                existing_movie = session.query(Movie).filter_by(tconst=tconst).first()

                if existing_movie:
                    # Update existing movie
                    existing_movie.primary_title = row["primaryTitle"]
                    existing_movie.start_year = (
                        int(row["startYear"]) if pd.notna(row["startYear"]) else None
                    )
                    existing_movie.genres = genres
                    existing_movie.directors = directors
                    existing_movie.actors = actors
                    if embedding is not None:
                        # Convert numpy array to list for pgvector
                        existing_movie.embedding = embedding.tolist()
                else:
                    # Create new movie
                    movie_data = {
                        "tconst": tconst,
                        "primary_title": row["primaryTitle"],
                        "start_year": (
                            int(row["startYear"])
                            if pd.notna(row["startYear"])
                            else None
                        ),
                        "genres": genres,
                        "directors": directors,
                        "actors": actors,
                    }

                    if embedding is not None:
                        # Convert numpy array to list for pgvector
                        movie_data["embedding"] = embedding.tolist()

                    new_movie = Movie(**movie_data)
                    session.add(new_movie)

            # Commit batch
            session.commit()

            batch_num = (i // batch_size) + 1
            total_batches = (total_movies + batch_size - 1) // batch_size
            print(
                f"   Batch {batch_num}/{total_batches} completed ({len(batch_df)} movies)"
            )

        print(f"✅ Successfully saved {total_movies} movies to database")

    except Exception as e:
        session.rollback()
        print(f"❌ Error saving to database: {e}")
        return False

    finally:
        session.close()

    return True


def save_to_database_raw_sql(movies_df, embeddings_dict):
    """Save movies and embeddings using raw SQL to handle vector type properly."""
    print("💾 Saving to database using raw SQL...")

    conn = get_connection()

    try:
        batch_size = 1000
        total_movies = len(movies_df)

        for i in range(0, total_movies, batch_size):
            batch_df = movies_df.iloc[i : i + batch_size]

            with conn.cursor() as cursor:
                for _, row in batch_df.iterrows():
                    tconst = row["tconst"]

                    # Get embedding for this movie
                    embedding = embeddings_dict.get(tconst, None)

                    # Convert pipe-separated strings back to lists
                    genres = (
                        row["genres"].split("|")
                        if pd.notna(row["genres"]) and row["genres"] != ""
                        else []
                    )
                    directors = (
                        row["directors"].split("|")
                        if pd.notna(row["directors"]) and row["directors"] != ""
                        else []
                    )
                    actors = (
                        row["actors"].split("|")
                        if pd.notna(row["actors"]) and row["actors"] != ""
                        else []
                    )

                    # Check if movie already exists
                    cursor.execute("SELECT id FROM movies WHERE tconst = %s", (tconst,))
                    existing = cursor.fetchone()

                    if existing:
                        # Update existing movie
                        if embedding is not None:
                            cursor.execute(
                                """
                                UPDATE movies 
                                SET primary_title = %s, start_year = %s, genres = %s, 
                                    directors = %s, actors = %s, embedding = %s::vector
                                WHERE tconst = %s
                            """,
                                (
                                    row["primaryTitle"],
                                    (
                                        int(row["startYear"])
                                        if pd.notna(row["startYear"])
                                        else None
                                    ),
                                    genres,
                                    directors,
                                    actors,
                                    embedding.tolist(),
                                    tconst,
                                ),
                            )
                        else:
                            cursor.execute(
                                """
                                UPDATE movies 
                                SET primary_title = %s, start_year = %s, genres = %s, 
                                    directors = %s, actors = %s
                                WHERE tconst = %s
                            """,
                                (
                                    row["primaryTitle"],
                                    (
                                        int(row["startYear"])
                                        if pd.notna(row["startYear"])
                                        else None
                                    ),
                                    genres,
                                    directors,
                                    actors,
                                    tconst,
                                ),
                            )
                    else:
                        # Insert new movie
                        if embedding is not None:
                            cursor.execute(
                                """
                                INSERT INTO movies (tconst, primary_title, start_year, genres, directors, actors, embedding)
                                VALUES (%s, %s, %s, %s, %s, %s, %s::vector)
                            """,
                                (
                                    tconst,
                                    row["primaryTitle"],
                                    (
                                        int(row["startYear"])
                                        if pd.notna(row["startYear"])
                                        else None
                                    ),
                                    genres,
                                    directors,
                                    actors,
                                    embedding.tolist(),
                                ),
                            )
                        else:
                            cursor.execute(
                                """
                                INSERT INTO movies (tconst, primary_title, start_year, genres, directors, actors)
                                VALUES (%s, %s, %s, %s, %s, %s)
                            """,
                                (
                                    tconst,
                                    row["primaryTitle"],
                                    (
                                        int(row["startYear"])
                                        if pd.notna(row["startYear"])
                                        else None
                                    ),
                                    genres,
                                    directors,
                                    actors,
                                ),
                            )

            batch_num = (i // batch_size) + 1
            total_batches = (total_movies + batch_size - 1) // batch_size
            print(
                f"   Batch {batch_num}/{total_batches} completed ({len(batch_df)} movies)"
            )

        conn.commit()
        print(f"✅ Successfully saved {total_movies} movies to database")

    except Exception as e:
        conn.rollback()
        print(f"❌ Error saving to database: {e}")
        return False

    finally:
        conn.close()

    return True


def verify_database_data():
    """Verify the data was saved correctly."""
    print("🔍 Verifying database data...")

    try:
        session = get_session()

        # Count total movies
        total_count = session.query(Movie).count()
        print(f"   Total movies in database: {total_count}")

        # Count movies with embeddings
        embedding_count = (
            session.query(Movie).filter(Movie.embedding.isnot(None)).count()
        )
        print(f"   Movies with embeddings: {embedding_count}")

        # Sample data
        sample_movies = (
            session.query(Movie.tconst, Movie.primary_title, Movie.start_year)
            .limit(5)
            .all()
        )
        print("   Sample data:")
        for movie in sample_movies:
            print(f"     - {movie.tconst}: {movie.primary_title} ({movie.start_year})")

        session.close()
        print("✅ Database verification completed")

    except Exception as e:
        print(f"❌ Error verifying database: {e}")


def main():
    """Main function to load data and save to database."""
    print("🎬 Movie Data Database Storage Pipeline")
    print("=" * 50)

    # File paths
    csv_file = "work_data/merged.csv"
    npz_file = "movie_embeddings.npz"

    # Check if files exist
    if not os.path.exists(csv_file):
        print(f"❌ CSV file not found: {csv_file}")
        return

    if not os.path.exists(npz_file):
        print(f"❌ NPZ file not found: {npz_file}")
        return

    # Create database tables using existing function
    print("🗄️  Setting up database...")
    try:
        create_tables()
        print("   ✅ Database tables created/verified")
    except Exception as e:
        print(f"❌ Error creating tables: {e}")
        print("💡 Please ensure PostgreSQL is running and database exists")
        return

    # Load data
    movies_df = load_merged_csv(csv_file)
    if movies_df is None:
        return

    embeddings_dict = load_embeddings_npz(npz_file)
    if embeddings_dict is None:
        return

    # Save to database using raw SQL for proper vector handling
    if save_to_database_raw_sql(movies_df, embeddings_dict):
        # Verify data
        verify_database_data()

        print("\n🎉 Database storage completed successfully!")
        print(f"📊 Total movies processed: {len(movies_df)}")
        print(f"🎯 Embeddings included: {len(embeddings_dict)}")
        print("🗄️  Data saved to 'movies' table using existing models")
    else:
        print("❌ Database storage failed")


if __name__ == "__main__":
    main()
