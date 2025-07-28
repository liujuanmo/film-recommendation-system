import numpy as np
from .db_functions import (
    get_engine,
    get_session,
    get_connection,
    get_movie_count,
    get_transformer_metadata,
)
from .constants import get_text_model
from typing import List, Dict, Optional
import pandas as pd
from .db_models import Movie


class MovieRecommender:
    def __init__(self):
        """Initialize the movie recommender system."""
        # Establish PostgreSQL connection
        try:
            self.pg_engine = get_engine()
            self.session = get_session()
        except Exception as e:
            raise Exception(
                f"Failed to connect to PostgreSQL: {e}\n"
                "Please ensure:\n"
                "1. PostgreSQL is running\n"
                "2. Database exists\n"
                "3. Connection settings are correct"
            )

        self._check_data()
        self._load_metadata()
        self._is_initialized = True

    def _check_data(self):
        """Check if movie data exists in PostgreSQL."""
        print("🔍 Checking movie data in PostgreSQL...")

        try:
            movie_count = get_movie_count()
            if movie_count == 0:
                raise Exception(
                    "Movie table exists but is empty!\n"
                    "Please run the data pipeline first to load movies and embeddings."
                )
            print(f"✅ Found {movie_count} movies in database")
        except Exception as e:
            raise Exception(f"Error checking movie data: {e}")

    def _load_metadata(self):
        """Load transformer metadata from database."""
        print("📊 Loading transformer metadata...")

        try:
            # Load genre classes
            self.genre_classes = get_transformer_metadata("genre_classes")
            if not self.genre_classes:
                print("⚠️  Genre classes not found, using empty list")
                self.genre_classes = []

            # Load year normalization parameters
            year_stats = get_transformer_metadata("year_stats")
            if year_stats:
                self.year_min = year_stats.get("min", 1900)
                self.year_max = year_stats.get("max", 2024)
            else:
                print("⚠️  Year stats not found, using defaults")
                self.year_min = 1900
                self.year_max = 2024

            # Load text embedding metadata
            self.text_metadata = get_transformer_metadata("text_metadata")
            if self.text_metadata:
                self.text_embedding_dim = self.text_metadata.get("embedding_dim", 384)
            else:
                print("⚠️  Text metadata not found, using defaults")
                self.text_embedding_dim = 384

            # Load person embedding metadata
            self.person_metadata = get_transformer_metadata("person_metadata")
            if self.person_metadata:
                self.person_embedding_dim = self.person_metadata.get(
                    "embedding_dim", 384
                )
            else:
                print("⚠️  Person metadata not found, using defaults")
                self.person_embedding_dim = 384

            # Initialize sentence transformer models (using global cached model)
            self.text_model = get_text_model()
            self.person_model = (
                get_text_model()
            )  # Same model for both text and person embeddings

            # Get actual embedding structure from database
            self._analyze_embedding_structure()

            print(f"✅ Loaded transformer model: all-MiniLM-L6-v2")
            print(f"✅ Text embedding dimension: {self.text_embedding_dim}")
            print(f"✅ Person embedding dimension: {self.person_embedding_dim}")

        except Exception as e:
            print(f"⚠️  Error loading metadata: {e}")
            # Use defaults if metadata loading fails
            self.genre_classes = []
            self.year_min = 1900
            self.year_max = 2024
            self.text_embedding_dim = 384
            self.person_embedding_dim = 384
            self.text_model = get_text_model()
            self.person_model = get_text_model()
            self._analyze_embedding_structure()

    def _analyze_embedding_structure(self):
        """Analyze the actual embedding structure from the database."""
        try:
            conn = get_connection()
            cursor = conn.cursor()

            # Get a sample embedding to understand the structure
            cursor.execute(
                """
                SELECT embedding FROM movies 
                WHERE embedding IS NOT NULL 
                LIMIT 1
            """
            )

            result = cursor.fetchone()
            if result:
                embedding_data = result[0]

                # Parse the string representation
                if isinstance(embedding_data, str):
                    embedding_str = embedding_data.strip("[]")
                    embedding_list = [
                        float(x.strip()) for x in embedding_str.split(",")
                    ]
                    embedding_array = np.array(embedding_list)
                else:
                    embedding_array = np.array(embedding_data)

                # Analyze structure
                binary_indices = []
                continuous_indices = []

                for i, val in enumerate(embedding_array):
                    if val in [0, 1]:
                        binary_indices.append(i)
                    else:
                        continuous_indices.append(i)
                        if len(continuous_indices) == 1:
                            break

                # Set the actual dimensions
                self.genre_count = len(binary_indices)
                self.continuous_dim = len(embedding_array) - len(binary_indices)
                self.total_dim = len(embedding_array)

                print(f"📊 Analyzed embedding structure:")
                print(f"   - Genre categories: {self.genre_count}")
                print(f"   - Continuous dimensions: {self.continuous_dim}")
                print(f"   - Total dimensions: {self.total_dim}")

                # Update genre classes if not loaded from metadata
                if not self.genre_classes:
                    self.genre_classes = [f"genre_{i}" for i in range(self.genre_count)]

            conn.close()

        except Exception as e:
            print(f"⚠️  Error analyzing embedding structure: {e}")
            # Use defaults
            self.genre_count = 28  # Based on analysis
            self.continuous_dim = 1153  # Based on analysis
            self.total_dim = 1181  # Based on analysis

    def _build_genre_vector(self, user_genres: List[str]) -> np.ndarray:
        """Build genre vector using stored genre classes."""
        if not user_genres or not self.genre_classes:
            return np.zeros(len(self.genre_classes), dtype=np.float32)

        genre_vec = np.zeros(len(self.genre_classes), dtype=np.float32)
        for genre in user_genres:
            if genre in self.genre_classes:
                idx = self.genre_classes.index(genre)
                genre_vec[idx] = 1.0

        return genre_vec

    def _build_year_vector(self, year: Optional[int]) -> np.ndarray:
        """Build normalized year vector."""
        if not year:
            return np.array([0.0], dtype=np.float32)

        year_val = float(year)
        if self.year_max > self.year_min:
            year_norm = (year_val - self.year_min) / (self.year_max - self.year_min)
        else:
            year_norm = 0.0

        return np.array([year_norm], dtype=np.float32)

    def _build_person_vector(self, person_names: List[str]) -> np.ndarray:
        """Build person vector using sentence transformer."""
        if not person_names:
            return np.zeros(self.person_embedding_dim, dtype=np.float32)

        # Use sentence transformer to encode person names
        person_texts = [str(name) for name in person_names]
        embeddings = self.person_model.encode(person_texts)

        # Average the embeddings
        return np.mean(embeddings, axis=0).astype(np.float32)

    def _build_text_vector(self, text_input: str) -> np.ndarray:
        """Build text vector using sentence transformer."""
        if not text_input or not text_input.strip():
            return np.zeros(self.text_embedding_dim, dtype=np.float32)

        # Use sentence transformer to encode text
        embedding = self.text_model.encode([text_input])
        return embedding[0].astype(np.float32)

    def _build_query_embedding(
        self,
        query: str,
        genres: List[str] = None,
        year: int = None,
        directors: List[str] = None,
        actors: List[str] = None,
    ) -> np.ndarray:
        """Build complete query embedding combining all features."""
        # Build individual feature vectors
        genre_vec = self._build_genre_vector(genres or [])
        year_vec = self._build_year_vector(year)
        director_vec = self._build_person_vector(directors or [])
        actor_vec = self._build_person_vector(actors or [])
        text_vec = self._build_text_vector(query)

        # Combine all features to match database structure
        # Structure: [genres] + [year] + [directors] + [actors] + [text]
        query_embedding = np.hstack(
            [
                genre_vec,  # Genre one-hot vectors (28 dimensions)
                year_vec,  # Normalized year (1 dimension)
                director_vec,  # Director embeddings (384 dimensions)
                actor_vec,  # Actor embeddings (384 dimensions)
                text_vec,  # Text features (384 dimensions)
            ]
        ).astype(np.float32)

        # Ensure the embedding matches the expected dimension
        expected_dim = (
            self.genre_count + 1 + 384 + 384 + 384
        )  # 28 + 1 + 384 + 384 + 384 = 1181
        if len(query_embedding) != expected_dim:
            print(
                f"⚠️  Warning: Query embedding dimension {len(query_embedding)} doesn't match expected {expected_dim}"
            )
            # Pad or truncate to match expected dimension
            if len(query_embedding) < expected_dim:
                padding = np.zeros(
                    expected_dim - len(query_embedding), dtype=np.float32
                )
                query_embedding = np.hstack([query_embedding, padding])
            else:
                query_embedding = query_embedding[:expected_dim]

        return query_embedding

    def _vector_search(
        self, query_embedding: np.ndarray, limit: int = 10
    ) -> List[Dict]:
        """Search for similar movies using vector similarity."""
        try:
            conn = get_connection()
            with conn.cursor() as cursor:
                # Use cosine similarity to find similar movies
                cursor.execute(
                    """
                    SELECT 
                        m.id,
                        m.tconst,
                        m.primary_title,
                        m.start_year,
                        m.genres,
                        m.directors,
                        m.actors,
                        1 - (m.embedding <=> %s::vector) as similarity
                    FROM movies m
                    WHERE m.embedding IS NOT NULL
                    ORDER BY m.embedding <=> %s::vector
                    LIMIT %s
                """,
                    (query_embedding.tolist(), query_embedding.tolist(), limit),
                )

                results = cursor.fetchall()

                recommendations = []
                for row in results:
                    recommendations.append(
                        {
                            "id": row[0],
                            "tconst": row[1],
                            "title": row[2],
                            "year": row[3],
                            "genres": row[4] if row[4] else [],
                            "directors": row[5] if row[5] else [],
                            "actors": row[6] if row[6] else [],
                            "similarity": float(row[7]),
                        }
                    )

                return recommendations

        except Exception as e:
            print(f"❌ Error in vector search: {e}")
            return []

    def recommend_by_text(self, query: str, limit: int = 10) -> List[Dict]:
        """Get movie recommendations based on natural language query."""
        if not self._is_initialized:
            raise RuntimeError("Recommender not initialized")

        print(f"🔍 Searching for movies similar to: '{query}'")

        # Build query embedding
        query_embedding = self._build_query_embedding(query)

        # Search for similar movies
        recommendations = self._vector_search(query_embedding, limit)

        print(f"✅ Found {len(recommendations)} recommendations")
        return recommendations

    def recommend_by_filters(
        self,
        genres: List[str] = None,
        year: int = None,
        directors: List[str] = None,
        actors: List[str] = None,
        limit: int = 10,
    ) -> List[Dict]:
        """Get movie recommendations based on specific filters."""
        if not self._is_initialized:
            raise RuntimeError("Recommender not initialized")

        print(f"🔍 Searching for movies with filters:")
        if genres:
            print(f"   Genres: {genres}")
        if year:
            print(f"   Year: {year}")
        if directors:
            print(f"   Directors: {directors}")
        if actors:
            print(f"   Actors: {actors}")

        # Build query embedding from filters
        query_embedding = self._build_query_embedding(
            query="",  # Empty text query
            genres=genres,
            year=year,
            directors=directors,
            actors=actors,
        )

        # Search for similar movies
        recommendations = self._vector_search(query_embedding, limit)

        print(f"✅ Found {len(recommendations)} recommendations")
        return recommendations

    def recommend_similar_to_movie(self, movie: dict, limit: int = 10) -> List[Dict]:
        """Get movie recommendations similar to a specific movie."""
        if not self._is_initialized:
            raise RuntimeError("Recommender not initialized")

        try:
            conn = get_connection()
            with conn.cursor() as cursor:
                # Get the target movie's embedding
                cursor.execute(
                    """
                    SELECT embedding FROM movies 
                    WHERE tconst = %s AND embedding IS NOT NULL
                """,
                    (movie["tconst"],),
                )

                result = cursor.fetchone()
                if not result:
                    print(f"❌ Movie {movie['tconst']} not found or has no embedding")
                    return []

                target_embedding = result[0]

                # Find similar movies
                cursor.execute(
                    """
                    SELECT 
                        m.id,
                        m.tconst,
                        m.primary_title,
                        m.start_year,
                        m.genres,
                        m.directors,
                        m.actors,
                        1 - (m.embedding <=> %s::vector) as similarity
                    FROM movies m
                    WHERE m.embedding IS NOT NULL AND m.tconst != %s
                    ORDER BY m.embedding <=> %s::vector
                    LIMIT %s
                """,
                    (target_embedding, movie["tconst"], target_embedding, limit),
                )

                results = cursor.fetchall()

                recommendations = []
                for row in results:
                    recommendations.append(
                        {
                            "id": row[0],
                            "tconst": row[1],
                            "title": row[2],
                            "year": row[3],
                            "genres": row[4] if row[4] else [],
                            "directors": row[5] if row[5] else [],
                            "actors": row[6] if row[6] else [],
                            "similarity": float(row[7]),
                        }
                    )

                print(f"✅ Found {len(recommendations)} similar movies")
                return recommendations

        except Exception as e:
            print(f"❌ Error finding similar movies: {e}")
            return []

    def get_movie_details(self, movie_id: str) -> Optional[Dict]:
        """Get detailed information about a specific movie."""
        try:
            conn = get_connection()
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT 
                        id, tconst, primary_title, start_year, 
                        genres, directors, actors
                    FROM movies 
                    WHERE tconst = %s
                """,
                    (movie_id,),
                )

                result = cursor.fetchone()
                if result:
                    return {
                        "id": result[0],
                        "tconst": result[1],
                        "title": result[2],
                        "year": result[3],
                        "genres": result[4] if result[4] else [],
                        "directors": result[5] if result[5] else [],
                        "actors": result[6] if result[6] else [],
                    }
                else:
                    return None

        except Exception as e:
            print(f"❌ Error getting movie details: {e}")
            return None

    def search_movies(self, search_term: str, limit: int = 10) -> List[Dict]:
        """Search movies by title using text search."""
        try:
            conn = get_connection()
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT 
                        id, tconst, primary_title, start_year, 
                        genres, directors, actors
                    FROM movies 
                    WHERE primary_title ILIKE %s
                    ORDER BY start_year DESC
                    LIMIT %s
                """,
                    (f"%{search_term}%", limit),
                )

                results = cursor.fetchall()

                movies = []
                for row in results:
                    movies.append(
                        {
                            "id": row[0],
                            "tconst": row[1],
                            "title": row[2],
                            "year": row[3],
                            "genres": row[4] if row[4] else [],
                            "directors": row[5] if row[5] else [],
                            "actors": row[6] if row[6] else [],
                        }
                    )

                return movies

        except Exception as e:
            print(f"❌ Error searching movies: {e}")
            return []

    def get_popular_genres(self, limit: int = 10) -> List[Dict]:
        """Get most popular genres in the database."""
        try:
            conn = get_connection()
            with conn.cursor() as cursor:
                cursor.execute(
                    """
                    SELECT 
                        unnest(genres) as genre,
                        COUNT(*) as count
                    FROM movies 
                    WHERE genres IS NOT NULL AND array_length(genres, 1) > 0
                    GROUP BY genre
                    ORDER BY count DESC
                    LIMIT %s
                """,
                    (limit,),
                )

                results = cursor.fetchall()

                genres = []
                for row in results:
                    genres.append({"genre": row[0], "count": row[1]})

                return genres

        except Exception as e:
            print(f"❌ Error getting popular genres: {e}")
            return []

    def __del__(self):
        """Cleanup when the recommender is destroyed."""
        if hasattr(self, "session"):
            self.session.close()


# Example usage functions
def create_recommender() -> MovieRecommender:
    """Create and return a configured movie recommender."""
    return MovieRecommender()


def example_recommendations():
    """Example of how to use the recommender."""
    try:
        recommender = create_recommender()

        print("\n🎬 Movie Recommendation Examples")
        print("=" * 40)

        # Example 1: Text-based recommendations
        print("\n1. Text-based recommendations:")
        recommendations = recommender.recommend_by_text(
            "action movies with explosions", limit=5
        )
        for i, movie in enumerate(recommendations, 1):
            print(
                f"   {i}. {movie['title']} ({movie['year']}) - Similarity: {movie['similarity']:.3f}"
            )

        # Example 2: Filter-based recommendations
        print("\n2. Filter-based recommendations:")
        recommendations = recommender.recommend_by_filters(
            genres=["Comedy", "Romance"], year=2020, limit=5
        )
        for i, movie in enumerate(recommendations, 1):
            print(
                f"   {i}. {movie['title']} ({movie['year']}) - Similarity: {movie['similarity']:.3f}"
            )

        # Example 3: Popular genres
        print("\n3. Popular genres:")
        genres = recommender.get_popular_genres(limit=5)
        for genre in genres:
            print(f"   {genre['genre']}: {genre['count']} movies")

    except Exception as e:
        print(f"❌ Error in example: {e}")


if __name__ == "__main__":
    example_recommendations()
