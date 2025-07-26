import psycopg2
import psycopg2.extras
import numpy as np
import pandas as pd
import os
from typing import List, Tuple, Optional
import json

# Database connection parameters - can be configured via environment variables
DB_CONFIG = {
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': os.getenv('DB_PORT', '5432'),
    'database': os.getenv('DB_NAME', 'movie_recommendations'),
    'user': os.getenv('DB_USER', 'postgres'),
    'password': os.getenv('DB_PASSWORD', '12345678')
}

def get_connection():
    """Create and return a PostgreSQL connection."""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        return conn
    except psycopg2.OperationalError as e:
        print(f"Error connecting to PostgreSQL: {e}")
        print("Please make sure PostgreSQL is running and the database exists.")
        print("You can create the database with: createdb movie_recommendations")
        raise

def init_postgresql_tables(conn):
    """Initialize the PostgreSQL database with all necessary tables."""
    with conn.cursor() as cur:
        # Enable pgvector extension
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        
        # Create movies table for basic movie information
        cur.execute("""
            CREATE TABLE IF NOT EXISTS movies (
                id SERIAL PRIMARY KEY,
                tconst VARCHAR(20) UNIQUE NOT NULL,
                primary_title TEXT NOT NULL,
                start_year INTEGER,
                genres TEXT[],
                title_type VARCHAR(50),
                overview TEXT,
                keywords TEXT[]
            );
        """)
        
        # Create directors table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS directors (
                id SERIAL PRIMARY KEY,
                nconst VARCHAR(20) UNIQUE NOT NULL,
                primary_name TEXT NOT NULL
            );
        """)
        
        # Create actors table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS actors (
                id SERIAL PRIMARY KEY,
                nconst VARCHAR(20) UNIQUE NOT NULL,
                primary_name TEXT NOT NULL
            );
        """)
        
        # Create movie_directors junction table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS movie_directors (
                movie_id INTEGER REFERENCES movies(id) ON DELETE CASCADE,
                director_id INTEGER REFERENCES directors(id) ON DELETE CASCADE,
                PRIMARY KEY (movie_id, director_id)
            );
        """)
        
        # Create movie_actors junction table
        cur.execute("""
            CREATE TABLE IF NOT EXISTS movie_actors (
                movie_id INTEGER REFERENCES movies(id) ON DELETE CASCADE,
                actor_id INTEGER REFERENCES actors(id) ON DELETE CASCADE,
                PRIMARY KEY (movie_id, actor_id)
            );
        """)
        
        # Create indexes for better query performance
        cur.execute("CREATE INDEX IF NOT EXISTS idx_movies_tconst ON movies(tconst);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_movies_start_year ON movies(start_year);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_movies_genres ON movies USING GIN(genres);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_directors_nconst ON directors(nconst);")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_actors_nconst ON actors(nconst);")
        
    conn.commit()

def init_embeddings_table(conn, emb_dim: int):
    """Initialize the movie embeddings table."""
    with conn.cursor() as cur:
        # Create the movie_embeddings table
        cur.execute(f"""
            CREATE TABLE IF NOT EXISTS movie_embeddings (
                movie_id INTEGER PRIMARY KEY REFERENCES movies(id) ON DELETE CASCADE,
                embedding vector({emb_dim})
            );
        """)
        
        # Create an index for faster similarity search
        cur.execute("""
            CREATE INDEX IF NOT EXISTS movie_embeddings_embedding_idx 
            ON movie_embeddings USING ivfflat (embedding vector_cosine_ops)
            WITH (lists = 100);
        """)
        
    conn.commit()

def insert_embeddings(conn, ids: List[int], embeddings: List):
    """Insert movie embeddings into the database."""
    with conn.cursor() as cur:
        # Prepare data for batch insert
        data = []
        for movie_id, emb in zip(ids, embeddings):
            if hasattr(emb, 'tolist'):
                emb = emb.tolist()
            elif isinstance(emb, np.ndarray):
                emb = emb.tolist()
            data.append((movie_id, emb))
        
        # Use execute_values for efficient batch insert
        psycopg2.extras.execute_values(
            cur,
            "INSERT INTO movie_embeddings (movie_id, embedding) VALUES %s ON CONFLICT (movie_id) DO UPDATE SET embedding = EXCLUDED.embedding",
            data,
            template=None,
            page_size=1000
        )
    
    conn.commit()

def search(conn, query_vec, top_n: int = 10) -> List[Tuple[int, float]]:
    """Search for similar movie embeddings using cosine similarity."""
    if hasattr(query_vec, 'tolist'):
        query_vec = query_vec.tolist()
    elif isinstance(query_vec, np.ndarray):
        query_vec = query_vec.tolist()
    
    with conn.cursor() as cur:
        # Use cosine distance for similarity search
        cur.execute("""
            SELECT movie_id, (embedding <=> %s::vector) AS distance
            FROM movie_embeddings
            ORDER BY embedding <=> %s::vector
            LIMIT %s;
        """, (query_vec, query_vec, top_n))
        
        results = cur.fetchall()
    
    return results

def get_embedding_count(conn) -> int:
    """Get the count of embeddings in the database."""
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM movie_embeddings;")
        count = cur.fetchone()[0]
    return count

def table_exists(conn, table_name: str = 'movie_embeddings') -> bool:
    """Check if a table exists."""
    with conn.cursor() as cur:
        cur.execute("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_name = %s
            );
        """, (table_name,))
        exists = cur.fetchone()[0]
    return exists

def get_movie_count(conn) -> int:
    """Get the count of movies in the database."""
    with conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM movies;")
        count = cur.fetchone()[0]
    return count

def load_movies_data(conn, movies_df):
    """Load movies data into the database."""
    with conn.cursor() as cur:
        # Prepare data for batch insert
        data = []
        for _, row in movies_df.iterrows():
            genres = row['genres'] if isinstance(row['genres'], list) else []
            keywords = row.get('keywords', []) if isinstance(row.get('keywords'), list) else []
            data.append((
                row['tconst'],
                row['primaryTitle'],
                int(row['startYear']) if pd.notna(row['startYear']) else None,
                genres,
                row.get('titleType', 'movie'),
                row.get('overview', ''),
                keywords
            ))
        
        # Batch insert movies
        psycopg2.extras.execute_values(
            cur,
            """INSERT INTO movies (tconst, primary_title, start_year, genres, title_type, overview, keywords) 
               VALUES %s ON CONFLICT (tconst) DO UPDATE SET
               primary_title = EXCLUDED.primary_title,
               start_year = EXCLUDED.start_year,
               genres = EXCLUDED.genres,
               title_type = EXCLUDED.title_type,
               overview = EXCLUDED.overview,
               keywords = EXCLUDED.keywords""",
            data,
            template=None,
            page_size=1000
        )
    
    conn.commit()

def load_directors_data(conn, directors_df):
    """Load directors data into the database."""
    with conn.cursor() as cur:
        # Prepare data for batch insert
        data = [(row['nconst'], row['primaryName']) for _, row in directors_df.iterrows()]
        
        # Batch insert directors
        psycopg2.extras.execute_values(
            cur,
            """INSERT INTO directors (nconst, primary_name) VALUES %s 
               ON CONFLICT (nconst) DO UPDATE SET primary_name = EXCLUDED.primary_name""",
            data,
            template=None,
            page_size=1000
        )
    
    conn.commit()

def load_actors_data(conn, actors_df):
    """Load actors data into the database."""
    with conn.cursor() as cur:
        # Prepare data for batch insert
        data = [(row['nconst'], row['primaryName']) for _, row in actors_df.iterrows()]
        
        # Batch insert actors
        psycopg2.extras.execute_values(
            cur,
            """INSERT INTO actors (nconst, primary_name) VALUES %s 
               ON CONFLICT (nconst) DO UPDATE SET primary_name = EXCLUDED.primary_name""",
            data,
            template=None,
            page_size=1000
        )
    
    conn.commit()

def load_movie_directors(conn, movie_directors_data):
    """Load movie-director relationships."""
    with conn.cursor() as cur:
        # Batch insert movie-director relationships
        psycopg2.extras.execute_values(
            cur,
            """INSERT INTO movie_directors (movie_id, director_id) 
               SELECT m.id, d.id FROM (VALUES %s) AS v(tconst, nconst)
               JOIN movies m ON m.tconst = v.tconst
               JOIN directors d ON d.nconst = v.nconst
               ON CONFLICT DO NOTHING""",
            movie_directors_data,
            template=None,
            page_size=1000
        )
    
    conn.commit()

def load_movie_actors(conn, movie_actors_data):
    """Load movie-actor relationships."""
    with conn.cursor() as cur:
        # Batch insert movie-actor relationships
        psycopg2.extras.execute_values(
            cur,
            """INSERT INTO movie_actors (movie_id, actor_id) 
               SELECT m.id, a.id FROM (VALUES %s) AS v(tconst, nconst)
               JOIN movies m ON m.tconst = v.tconst
               JOIN actors a ON a.nconst = v.nconst
               ON CONFLICT DO NOTHING""",
            movie_actors_data,
            template=None,
            page_size=1000
        )
    
    conn.commit()

def get_movies_with_details(conn):
    """Get all movies with their directors and actors."""
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("""
            SELECT 
                m.id,
                m.tconst,
                m.primary_title,
                m.start_year,
                m.genres,
                m.title_type,
                m.overview,
                m.keywords,
                COALESCE(array_agg(DISTINCT d.primary_name) FILTER (WHERE d.primary_name IS NOT NULL), '{}') as directors,
                COALESCE(array_agg(DISTINCT a.primary_name) FILTER (WHERE a.primary_name IS NOT NULL), '{}') as cast
            FROM movies m
            LEFT JOIN movie_directors md ON m.id = md.movie_id
            LEFT JOIN directors d ON md.director_id = d.id
            LEFT JOIN movie_actors ma ON m.id = ma.movie_id
            LEFT JOIN actors a ON ma.actor_id = a.id
            WHERE m.title_type = 'movie' AND array_length(m.genres, 1) > 0
            GROUP BY m.id, m.tconst, m.primary_title, m.start_year, m.genres, m.title_type, m.overview, m.keywords
            ORDER BY m.id;
        """)
        
        return cur.fetchall() 