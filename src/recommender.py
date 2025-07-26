import numpy as np
from .postgresql_vec_client import (
    get_engine, search as postgresql_vec_search, get_embedding_count, table_exists,
    get_movie_count, get_transformer_metadata, get_person_embeddings, get_movie_details_by_ids
)

EMBED_DIM = 32

class MovieRecommender:
    def __init__(self):
        """Lightweight database-first recommender initialization."""
        # Establish PostgreSQL connection
        try:
            self.pg_engine = get_engine()
        except Exception as e:
            raise Exception(
                f"Failed to connect to PostgreSQL: {e}\n"
                "Please ensure:\n"
                "1. PostgreSQL is running\n"
                "2. Database 'movie_recommendations' exists\n"
                "3. Connection settings are correct (check environment variables)"
            )
        
        self._check_data()
        self._check_embeddings()
        self._load_metadata()

    def _check_data(self):
        """Check if movie data exists in PostgreSQL."""
        print("Checking movie data in PostgreSQL...")
        
        if not table_exists('movies'):
            raise Exception(
                "No movie data found in PostgreSQL!\n"
                "Please run 'python load_data.py' first to load IMDB data into the database."
            )
        
        movie_count = get_movie_count()
        if movie_count == 0:
            raise Exception(
                "Movie table exists but is empty!\n"
                "Please run 'python load_data.py' first to load IMDB data into the database."
            )
        
        print(f"✓ Found {movie_count} movies in database.")

    def _check_embeddings(self):
        """Check if embeddings and metadata exist in PostgreSQL."""
        print("Checking embeddings in PostgreSQL...")
        
        if not table_exists('movie_embeddings'):
            raise Exception(
                "Movie embeddings table not found!\n"
                "Please run 'python load_embeddings.py' to compute and store embeddings first."
            )
        
        count = get_embedding_count()
        if count == 0:
            raise Exception(
                "Movie embeddings table exists but is empty!\n"
                "Please run 'python load_embeddings.py' to compute and store embeddings first."
            )
        
        print(f"✓ Found {count} movie embeddings in database.")

    def _load_metadata(self):
        """Load lightweight transformer metadata from database."""
        print("Loading transformer metadata from database...")
        
        # Load genre classes
        self.genre_classes = get_transformer_metadata('genre_classes')
        if not self.genre_classes:
            raise Exception("Genre classes metadata not found! Please run 'python load_embeddings.py' first.")
        
        # Load year normalization parameters
        year_stats = get_transformer_metadata('year_stats')
        if not year_stats:
            raise Exception("Year statistics metadata not found! Please run 'python load_embeddings.py' first.")
        
        self.year_min = year_stats['min']
        self.year_max = year_stats['max']
        
        # Load TF-IDF metadata
        self.tfidf_metadata = get_transformer_metadata('tfidf_metadata')
        if not self.tfidf_metadata:
            raise Exception("TF-IDF metadata not found! Please run 'python load_embeddings.py' first.")
        
        print("✓ Loaded transformer metadata from database.")

    def _build_genre_vector(self, user_genres):
        """Build genre vector using stored genre classes."""
        if not user_genres:
            return np.zeros(len(self.genre_classes), dtype=np.float32)
        
        genre_vec = np.zeros(len(self.genre_classes), dtype=np.float32)
        for genre in user_genres:
            if genre in self.genre_classes:
                idx = self.genre_classes.index(genre)
                genre_vec[idx] = 1.0
        
        return genre_vec

    def _build_year_vector(self, year):
        """Build normalized year vector."""
        if not year:
            return np.array([0.0], dtype=np.float32)
        
        year_val = float(year)
        if self.year_max > self.year_min:
            year_norm = (year_val - self.year_min) / (self.year_max - self.year_min)
        else:
            year_norm = 0.0
        
        return np.array([year_norm], dtype=np.float32)

    def _build_person_vector(self, person_names, person_type):
        """Build person vector using database lookups."""
        if not person_names:
            return np.zeros(EMBED_DIM, dtype=np.float32)
        
        # Get embeddings from database
        embeddings = get_person_embeddings(person_names, person_type)
        
        if not embeddings:
            return np.zeros(EMBED_DIM, dtype=np.float32)
        
        # Average the embeddings
        vectors = [np.array(emb) for emb in embeddings.values()]
        return np.mean(vectors, axis=0).astype(np.float32)

    def _build_text_vector(self, text_input):
        """Build text vector using stored TF-IDF metadata."""
        if not text_input or not text_input.strip():
            return np.zeros(len(self.tfidf_metadata['vocabulary']), dtype=np.float32)
        
        # Simple TF-IDF transformation using stored vocabulary and IDF
        words = text_input.lower().split()
        vocabulary = self.tfidf_metadata['vocabulary']
        idf_weights = self.tfidf_metadata['idf_weights']
        
        # Calculate term frequencies
        word_counts = {}
        for word in words:
            word_counts[word] = word_counts.get(word, 0) + 1
        
        # Build TF-IDF vector
        text_vec = np.zeros(len(vocabulary), dtype=np.float32)
        total_words = len(words)
        
        for word, count in word_counts.items():
            if word in vocabulary:
                idx = vocabulary.index(word)
                tf = count / total_words if total_words > 0 else 0
                idf = idf_weights[idx]
                text_vec[idx] = tf * idf
        
        return text_vec

    def recommend(self, genres=None, year=None, directors=None, cast=None, keywords=None, overview=None, title=None, top_n=10):
        """Generate recommendations using database-only operations."""
        
        # Build query vector components
        genre_vec = self._build_genre_vector(genres)
        year_vec = self._build_year_vector(year)
        director_vec = self._build_person_vector(directors, 'director')
        cast_vec = self._build_person_vector(cast, 'actor')
        
        # Build text input
        text_input = ''
        if title:
            text_input += title + ' '
        if overview:
            text_input += overview + ' '
        if keywords:
            text_input += ' '.join(keywords) + ' '
        
        text_vec = self._build_text_vector(text_input)
        
        # Combine all features into query vector
        query_vector = np.hstack([
            genre_vec,
            year_vec,
            director_vec,
            cast_vec,
            text_vec
        ]).astype(np.float32)
        
        # Perform similarity search
        results = postgresql_vec_search(query_vector, top_n=top_n)
        
        if not results:
            return []
        
        # Get movie details for results
        movie_ids = [movie_id for movie_id, _ in results]
        movies_data = get_movie_details_by_ids(movie_ids)
        
        # Convert to expected format
        out = []
        for movie in movies_data:
            out.append({
                'title': movie['primary_title'],
                'year': str(int(movie['start_year'])) if movie['start_year'] else 'N/A',
                'genres': ','.join(movie['genres']) if movie['genres'] else 'N/A',
                'directors': ','.join(movie['directors']) if movie['directors'] else 'N/A',
                'cast': ','.join(movie['cast']) if movie['cast'] else 'N/A'
            })
        
        return out 